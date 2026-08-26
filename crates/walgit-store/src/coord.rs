//! Coordination primitives (CAS loop, Lease) built on [`crate::ObjectStore`].
//!
//! Every read of a repo starts with a freshness check on `manifest.pb`
//! (conditional GET). Mutations go through [`cas_update`], a generic
//! read-modify-write loop that re-reads on `PreconditionFailed` and backs off
//! on `Retryable`. Leases are protobuf objects under `leases/` acquired by
//! `Create` or by `Update` over an expired lease, renewed by CAS heartbeat.
//!
//! Lease keys are never physically deleted. Release is a CAS write of an
//! already-expired tombstone. This matters for S3-compatible stores: their
//! `HEAD` + `DELETE` emulation is not atomic, so a stale holder could delete a
//! newer lease after a successor had reclaimed the key.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use prost::Message;
use tokio::sync::Mutex;

use crate::{DynStore, ObjectMeta, ObjectStore, ObjectStoreExt, PutMode, StoreError, Version};
use walgit_proto::time;
use walgit_proto::v1::Lease;

/// Small clock-skew grace: only steal a lease once this much past its expiry,
/// so a holder whose clock is slightly ahead does not have its lease ripped
/// away while it is still legitimately active.
const LEASE_SKEW_TOLERANCE: Duration = Duration::from_secs(2);

#[derive(Debug, thiserror::Error)]
pub enum CoordError {
    #[error("store error: {0}")]
    Store(#[from] StoreError),
    #[error("protobuf decode error: {0}")]
    Decode(#[from] prost::DecodeError),
    #[error("operation aborted")]
    Aborted,
    #[error("lease lost")]
    LeaseLost,
    #[error("retries exhausted on {key} after {attempts} attempts")]
    RetriesExhausted { key: String, attempts: u32 },
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

// ---------------------------------------------------------------------------
// Generic CAS loop
// ---------------------------------------------------------------------------

/// Generic read-modify-write CAS loop on a protobuf object.
///
/// `f(None)` is called when the object is absent. Returning `None` from `f`
/// aborts with `Ok(None)`. Returning `Some(new)` writes `new` with `Create` if
/// the object was absent or `Update(version)` if it existed. On
/// `PreconditionFailed` the loop re-reads and retries (counted); on `Retryable`
/// it sleeps with jittered backoff then retries. After `max_retries` retries
/// the error is [`CoordError::RetriesExhausted`].
pub async fn cas_update<T, F>(
    store: &dyn ObjectStore,
    key: &str,
    max_retries: u32,
    mut f: F,
) -> Result<Option<(ObjectMeta, T)>, CoordError>
where
    T: prost::Message + Default,
    F: FnMut(Option<&T>) -> Result<Option<T>, CoordError>,
{
    let mut attempts: u32 = 0;
    loop {
        let current = get_message::<T>(store, key).await?;
        let new = match f(current.as_ref().map(|(_, t)| t)) {
            Ok(None) => return Ok(None),
            Ok(Some(new)) => new,
            Err(e) => return Err(e),
        };
        let mode = match &current {
            Some((meta, _)) => PutMode::Update(meta.version.clone()),
            None => PutMode::Create,
        };
        let encoded = new.encode_to_vec();
        match store.put_bytes(key, encoded, mode).await {
            Ok(meta) => return Ok(Some((meta, new))),
            Err(StoreError::PreconditionFailed { .. }) => {
                attempts += 1;
                if attempts > max_retries {
                    return Err(CoordError::RetriesExhausted {
                        key: key.to_string(),
                        attempts,
                    });
                }
                // re-read on next iteration
            }
            Err(StoreError::Retryable(_)) => {
                attempts += 1;
                if attempts > max_retries {
                    return Err(CoordError::RetriesExhausted {
                        key: key.to_string(),
                        attempts,
                    });
                }
                let d = crate::util::backoff(
                    attempts - 1,
                    Duration::from_millis(5),
                    Duration::from_millis(100),
                );
                tokio::time::sleep(d).await;
            }
            Err(e) => return Err(CoordError::Store(e)),
        }
    }
}

/// Read a protobuf object with its version. `Ok(None)` if absent.
pub async fn get_message<T>(
    store: &dyn ObjectStore,
    key: &str,
) -> Result<Option<(ObjectMeta, T)>, CoordError>
where
    T: prost::Message + Default,
{
    match store.get_bytes(key).await? {
        None => Ok(None),
        Some((meta, bytes)) => {
            let msg = T::decode(bytes)?;
            Ok(Some((meta, msg)))
        }
    }
}

/// Read a protobuf object only if its version changed since `known`.
/// `Ok(None)` if unchanged or absent.
pub async fn get_message_if_changed<T>(
    store: &dyn ObjectStore,
    key: &str,
    known: &Version,
) -> Result<Option<(ObjectMeta, T)>, CoordError>
where
    T: prost::Message + Default,
{
    match store.get_if_changed(key, known).await {
        Err(StoreError::NotFound { .. }) => Ok(None),
        Err(e) => Err(CoordError::Store(e)),
        Ok(None) => Ok(None),
        Ok(Some((meta, bytes))) => {
            let msg = T::decode(bytes)?;
            Ok(Some((meta, msg)))
        }
    }
}

// ---------------------------------------------------------------------------
// Lease
// ---------------------------------------------------------------------------

fn make_lease(holder: &str, purpose: &str, now: SystemTime, ttl: Duration, epoch: u64) -> Lease {
    Lease {
        holder: holder.to_string(),
        purpose: purpose.to_string(),
        acquired_at: Some(time::from_system(now)),
        expires_at: Some(time::from_system(now + ttl)),
        epoch,
    }
}

fn released_lease(purpose: &str, epoch: u64) -> Lease {
    Lease {
        holder: String::new(),
        purpose: purpose.to_string(),
        acquired_at: None,
        expires_at: Some(time::from_system(UNIX_EPOCH)),
        epoch,
    }
}

async fn release_lease(
    store: &dyn ObjectStore,
    key: &str,
    purpose: &str,
    version: Version,
    epoch: u64,
) -> Result<(), CoordError> {
    let tombstone = released_lease(purpose, epoch).encode_to_vec();
    match store
        .put_bytes(key, tombstone, PutMode::Update(version))
        .await
    {
        Ok(_) | Err(StoreError::PreconditionFailed { .. }) | Err(StoreError::NotFound { .. }) => {
            Ok(())
        }
        Err(e) => Err(CoordError::Store(e)),
    }
}

/// A held lease. Drop performs a best-effort release when inside a Tokio
/// runtime; call [`LeaseGuard::release`] for a confirmed release.
pub struct LeaseGuard {
    store: DynStore,
    key: String,
    holder: String,
    purpose: String,
    version: Version,
    expires_at: SystemTime,
    epoch: u64,
    /// Stable fencing token for this lease acquisition. Heartbeats advance
    /// the lease's liveness epoch, but must not change the token a downstream
    /// writer uses to fence stale work from this owner.
    fence_token: u64,
    /// Set by `release` / `Drop` so the other path is a no-op. Also read by the
    /// heartbeat task to know when to stop.
    released: AtomicBool,
}

impl LeaseGuard {
    fn new(
        store: DynStore,
        key: &str,
        holder: &str,
        purpose: &str,
        version: Version,
        now: SystemTime,
        ttl: Duration,
        epoch: u64,
    ) -> Self {
        LeaseGuard {
            store,
            key: key.to_string(),
            holder: holder.to_string(),
            purpose: purpose.to_string(),
            version,
            expires_at: now + ttl,
            epoch,
            fence_token: epoch,
            released: AtomicBool::new(false),
        }
    }

    /// CAS-extend `expires_at` and increment `epoch`. A `PreconditionFailed`
    /// (someone stole the lease) returns [`CoordError::LeaseLost`].
    pub async fn heartbeat(&mut self, ttl: Duration) -> Result<(), CoordError> {
        let now = SystemTime::now();
        self.epoch += 1;
        let lease = make_lease(&self.holder, &self.purpose, now, ttl, self.epoch);
        let encoded = lease.encode_to_vec();
        match self
            .store
            .put_bytes(&self.key, encoded, PutMode::Update(self.version.clone()))
            .await
        {
            Ok(meta) => {
                self.version = meta.version;
                self.expires_at = now + ttl;
                Ok(())
            }
            Err(StoreError::PreconditionFailed { .. }) => Err(CoordError::LeaseLost),
            Err(e) => Err(CoordError::Store(e)),
        }
    }

    /// Renew only when the local validity window is nearly exhausted. A store
    /// response may itself arrive after the renewed expiry, so retry the CAS
    /// once and report `false` unless a usable window is confirmed.
    pub async fn renew_if_needed(&mut self, ttl: Duration) -> Result<bool, CoordError> {
        let margin = ttl / 3;
        for _ in 0..2 {
            if self.expires_at > SystemTime::now() + margin {
                return Ok(true);
            }
            match self.heartbeat(ttl).await {
                Ok(()) => {}
                Err(CoordError::LeaseLost) => return Ok(false),
                Err(e) => return Err(e),
            }
        }
        Ok(self.expires_at > SystemTime::now() + margin)
    }

    /// Confirmed CAS release. Consumes the guard. `Ok(())` even if the lease
    /// was already stolen (the desired end state — no one holds it in our
    /// name).
    pub async fn release(self) -> Result<(), CoordError> {
        self.released.store(true, Ordering::SeqCst);
        release_lease(
            self.store.as_ref(),
            &self.key,
            &self.purpose,
            self.version.clone(),
            self.epoch,
        )
        .await
    }

    /// Spawn a background task that calls [`heartbeat`](Self::heartbeat) every
    /// `every` with `ttl` until the guard is released or the lease is lost.
    ///
    /// Called as `LeaseGuard::spawn_heartbeat(guard, every, ttl)`. (Rust does
    /// not allow `Arc<Mutex<Self>>` as a `self` receiver, so this is an
    /// associated function rather than a method.)
    pub fn spawn_heartbeat(
        guard: Arc<Mutex<Self>>,
        every: Duration,
        ttl: Duration,
    ) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(every).await;
                let mut g = guard.lock().await;
                if g.released.load(Ordering::SeqCst) {
                    break;
                }
                match g.heartbeat(ttl).await {
                    Ok(()) => {}
                    Err(CoordError::LeaseLost) => {
                        g.released.store(true, Ordering::SeqCst);
                        tracing::warn!(key = %g.key, "lease lost during heartbeat");
                        break;
                    }
                    Err(e) => {
                        // Transient store error: keep trying; the store may recover.
                        tracing::debug!(key = %g.key, error = %e, "heartbeat transient error");
                    }
                }
            }
        })
    }

    pub fn holder(&self) -> &str {
        &self.holder
    }

    pub fn expires_at(&self) -> SystemTime {
        self.expires_at
    }

    /// Stable token identifying this lease acquisition. It increases only
    /// when a successor takes over the lease, not on heartbeat renewals.
    pub fn fencing_token(&self) -> u64 {
        self.fence_token
    }
}

impl Drop for LeaseGuard {
    fn drop(&mut self) {
        if self.released.swap(true, Ordering::SeqCst) {
            return;
        }
        // Best-effort release when inside a Tokio runtime.
        let store = self.store.clone();
        let key = self.key.clone();
        let purpose = self.purpose.clone();
        let version = self.version.clone();
        let epoch = self.epoch;
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            let _ = handle.spawn(async move {
                let _ = release_lease(store.as_ref(), &key, &purpose, version, epoch).await;
            });
        }
    }
}

/// Try to acquire a lease. Returns `Ok(Some(guard))` on success, `Ok(None)` if
/// the lease is held by someone else and not expired.
pub async fn try_acquire(
    store: DynStore,
    key: &str,
    holder: &str,
    purpose: &str,
    ttl: Duration,
) -> Result<Option<LeaseGuard>, CoordError> {
    let now = SystemTime::now();
    let current = get_message::<Lease>(store.as_ref(), key).await?;
    match current {
        None => {
            let lease = make_lease(holder, purpose, now, ttl, 0);
            let encoded = lease.encode_to_vec();
            match store.put_bytes(key, encoded, PutMode::Create).await {
                Ok(meta) => Ok(Some(LeaseGuard::new(
                    store,
                    key,
                    holder,
                    purpose,
                    meta.version,
                    now,
                    ttl,
                    0,
                ))),
                Err(StoreError::PreconditionFailed { .. }) => Ok(None),
                Err(e) => Err(CoordError::Store(e)),
            }
        }
        Some((meta, existing)) => {
            let expires_at = existing
                .expires_at
                .as_ref()
                .map(time::to_system)
                .unwrap_or(UNIX_EPOCH);
            if now >= expires_at + LEASE_SKEW_TOLERANCE {
                let epoch = existing.epoch + 1;
                let lease = make_lease(holder, purpose, now, ttl, epoch);
                let encoded = lease.encode_to_vec();
                match store
                    .put_bytes(key, encoded, PutMode::Update(meta.version.clone()))
                    .await
                {
                    Ok(new_meta) => Ok(Some(LeaseGuard::new(
                        store,
                        key,
                        holder,
                        purpose,
                        new_meta.version,
                        now,
                        ttl,
                        epoch,
                    ))),
                    Err(StoreError::PreconditionFailed { .. }) => Ok(None),
                    Err(e) => Err(CoordError::Store(e)),
                }
            } else {
                Ok(None)
            }
        }
    }
}

/// Acquire a lease, polling with jittered backoff until `wait_up_to` elapses.
/// Returns `Ok(Some(guard))` on success, `Ok(None)` on timeout.
pub async fn acquire(
    store: DynStore,
    key: &str,
    holder: &str,
    purpose: &str,
    ttl: Duration,
    wait_up_to: Duration,
) -> Result<Option<LeaseGuard>, CoordError> {
    let deadline = tokio::time::Instant::now() + wait_up_to;
    let mut attempt: u32 = 0;
    loop {
        if let Some(guard) = try_acquire(store.clone(), key, holder, purpose, ttl).await? {
            return Ok(Some(guard));
        }
        let now = tokio::time::Instant::now();
        if now >= deadline {
            return Ok(None);
        }
        let sleep = crate::util::backoff(
            attempt,
            Duration::from_millis(10),
            Duration::from_millis(200),
        );
        let remaining = deadline.saturating_duration_since(now);
        tokio::time::sleep(sleep.min(remaining)).await;
        attempt += 1;
    }
}

/// A process-stable identity: explicit `WALGIT_INSTANCE_NAME`/`WALGIT_INSTANCE_ID`,
/// else `HOSTNAME`/pid, else a random UUID. Computed once and cached.
pub fn instance_id() -> &'static str {
    static ID: LazyLock<String> = LazyLock::new(|| {
        if let (Ok(name), Ok(inst)) = (
            std::env::var("WALGIT_INSTANCE_NAME"),
            std::env::var("WALGIT_INSTANCE_ID"),
        ) && !name.is_empty()
            && !inst.is_empty()
        {
            return format!("{name}/{inst}");
        }
        if let Ok(h) = std::env::var("HOSTNAME")
            && !h.is_empty()
        {
            return format!("{h}/{}", std::process::id());
        }
        uuid::Uuid::new_v4().to_string()
    });
    &ID
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fault::{FaultPlan, FaultStore};
    use crate::memory::MemoryStore;
    use walgit_proto::v1::RepoCatalog;

    /// Reproduces the S3 conditional-delete implementation: it observes the
    /// expected version, pauses, then performs an unconditional delete. The
    /// pause lets a successor CAS-steal an expired lease in the middle of the
    /// stale holder's release.
    struct HeadThenDeleteStore {
        inner: Arc<MemoryStore>,
        head_observed: Arc<tokio::sync::Notify>,
        allow_delete: Arc<tokio::sync::Notify>,
    }

    impl HeadThenDeleteStore {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                inner: Arc::new(MemoryStore::new()),
                head_observed: Arc::new(tokio::sync::Notify::new()),
                allow_delete: Arc::new(tokio::sync::Notify::new()),
            })
        }
    }

    #[async_trait::async_trait]
    impl ObjectStore for HeadThenDeleteStore {
        fn backend(&self) -> &'static str {
            "s3-race"
        }

        async fn get(&self, key: &str, opts: crate::GetOptions) -> crate::Result<crate::GetResult> {
            self.inner.get(key, opts).await
        }

        async fn head(&self, key: &str) -> crate::Result<Option<crate::ObjectMeta>> {
            self.inner.head(key).await
        }

        async fn put(
            &self,
            key: &str,
            body: crate::PutBody,
            opts: crate::PutOptions,
        ) -> crate::Result<crate::ObjectMeta> {
            self.inner.put(key, body, opts).await
        }

        async fn delete(&self, key: &str, if_version: Option<crate::Version>) -> crate::Result<()> {
            if let Some(want) = if_version {
                let current = self
                    .inner
                    .head(key)
                    .await?
                    .ok_or_else(|| crate::StoreError::NotFound { key: key.into() })?;
                if current.version != want {
                    return Err(crate::StoreError::PreconditionFailed {
                        key: key.into(),
                        current: Some(current.version),
                    });
                }
                self.head_observed.notify_one();
                self.allow_delete.notified().await;
            }
            self.inner.delete(key, None).await
        }

        fn list(
            &self,
            prefix: &str,
            start_after: Option<&str>,
        ) -> crate::BoxStream<'static, crate::Result<crate::ObjectMeta>> {
            self.inner.list(prefix, start_after)
        }

        async fn list_prefixes(&self, prefix: &str) -> crate::Result<Vec<String>> {
            self.inner.list_prefixes(prefix).await
        }
    }

    fn dyn_store() -> DynStore {
        MemoryStore::shared() as DynStore
    }

    #[tokio::test]
    async fn cas_update_convergence_64_incrementers() {
        let store = dyn_store();
        let key = "counter.pb";
        const N: u32 = 64;

        let mut handles = Vec::new();
        for i in 0..N {
            let s = store.clone();
            let k = key.to_string();
            handles.push(tokio::spawn(async move {
                let tag = format!("repo-{i}");
                cas_update::<RepoCatalog, _>(s.as_ref(), &k, 500, |current| {
                    let mut cat = current.cloned().unwrap_or_default();
                    cat.repos.push(tag.clone());
                    Ok(Some(cat))
                })
                .await
                .unwrap();
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
        let (_, cat) = get_message::<RepoCatalog>(store.as_ref(), key)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(cat.repos.len(), N as usize);
    }

    #[tokio::test]
    async fn cas_and_lease_exclusivity_survive_delayed_responses() {
        let truth: DynStore = MemoryStore::shared();
        let link = FaultStore::new(truth, "delayed", 0xD1A5);
        link.set(FaultPlan {
            delay: Some((Duration::from_millis(2), Duration::from_millis(10))),
            delay_after: Some(Duration::from_millis(8)),
            ..Default::default()
        });
        let store: DynStore = link;

        let mut cas_handles = Vec::new();
        for i in 0..32u32 {
            let s = store.clone();
            cas_handles.push(tokio::spawn(async move {
                let tag = format!("delayed-{i}");
                cas_update::<RepoCatalog, _>(s.as_ref(), "delayed-counter.pb", 500, |current| {
                    let mut cat = current.cloned().unwrap_or_default();
                    cat.repos.push(tag.clone());
                    Ok(Some(cat))
                })
                .await
            }));
        }
        for h in cas_handles {
            h.await.unwrap().unwrap();
        }
        let (_, cat) = get_message::<RepoCatalog>(store.as_ref(), "delayed-counter.pb")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(cat.repos.len(), 32);

        let mut lease_handles = Vec::new();
        for i in 0..32u32 {
            let s = store.clone();
            lease_handles.push(tokio::spawn(async move {
                try_acquire(
                    s,
                    "leases/delayed.pb",
                    &format!("h{i}"),
                    "delayed",
                    Duration::from_secs(60),
                )
                .await
            }));
        }
        let mut acquired = 0;
        for h in lease_handles {
            if h.await.unwrap().unwrap().is_some() {
                acquired += 1;
            }
        }
        assert_eq!(acquired, 1);
    }

    #[tokio::test]
    async fn delayed_lease_heartbeat_response_is_not_reported_live() {
        let truth: DynStore = MemoryStore::shared();
        let link = FaultStore::new(truth, "lease-delay", 0xD1A5);
        let key = "leases/delayed-heartbeat.pb";
        let mut guard = try_acquire(
            link.clone() as DynStore,
            key,
            "h1",
            "delayed",
            Duration::from_millis(5),
        )
        .await
        .unwrap()
        .unwrap();
        link.set(FaultPlan {
            delay_after: Some(Duration::from_millis(30)),
            delay_after_ops: Some(vec!["put".into()]),
            only_keys: Some(vec![key.into()]),
            ..FaultPlan::default()
        });
        tokio::time::sleep(Duration::from_millis(10)).await;

        assert!(
            !guard
                .renew_if_needed(Duration::from_millis(5))
                .await
                .unwrap(),
            "a late heartbeat response must not claim a usable lease window"
        );
    }

    #[tokio::test]
    async fn cas_update_abort_returns_none() {
        let store = dyn_store();
        let key = "abort.pb";
        let res = cas_update::<RepoCatalog, _>(store.as_ref(), key, 10, |_| Ok(None))
            .await
            .unwrap();
        assert!(res.is_none());
        // object was never created
        assert!(store.get_bytes(key).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn lease_exclusivity_32_concurrent() {
        let store = dyn_store();
        let key = "leases/excl.pb";
        const N: u32 = 32;

        let mut handles = Vec::new();
        for i in 0..N {
            let s = store.clone();
            let k = key.to_string();
            handles.push(tokio::spawn(async move {
                let holder = format!("h{i}");
                try_acquire(s, &k, &holder, "test", Duration::from_secs(60)).await
            }));
        }
        let mut successes = 0;
        for h in handles {
            if h.await.unwrap().unwrap().is_some() {
                successes += 1;
            }
        }
        assert_eq!(successes, 1);
    }

    #[tokio::test]
    async fn lease_steal_after_expiry() {
        let store = dyn_store();
        let key = "leases/steal.pb";

        let g1 = try_acquire(store.clone(), key, "h1", "test", Duration::from_millis(50))
            .await
            .unwrap()
            .unwrap();
        // Wait past expiry + skew tolerance.
        tokio::time::sleep(LEASE_SKEW_TOLERANCE + Duration::from_millis(100)).await;

        let g2 = try_acquire(store.clone(), key, "h2", "test", Duration::from_secs(30))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(g2.holder(), "h2");

        // h1's CAS should now fail.
        let mut g1 = g1;
        let res = g1.heartbeat(Duration::from_secs(30)).await;
        assert!(matches!(res, Err(CoordError::LeaseLost)));
    }

    #[tokio::test]
    async fn lease_heartbeat_keeps_it() {
        let store = dyn_store();
        let key = "leases/hb.pb";
        let ttl = Duration::from_millis(100);

        let g = try_acquire(store.clone(), key, "h1", "test", ttl)
            .await
            .unwrap()
            .unwrap();
        let g = Arc::new(Mutex::new(g));
        let handle = LeaseGuard::spawn_heartbeat(g.clone(), Duration::from_millis(20), ttl);

        // Wait well past the original ttl.
        tokio::time::sleep(Duration::from_millis(400)).await;

        // Lease should still be held.
        let res = try_acquire(store.clone(), key, "h2", "test", Duration::from_secs(30))
            .await
            .unwrap();
        assert!(res.is_none());

        // Stop heartbeat and release.
        {
            let guard = g.lock().await;
            guard.released.store(true, Ordering::SeqCst);
        }
        handle.await.unwrap();
        let guard = Arc::try_unwrap(g).ok().expect("single ref").into_inner();
        guard.release().await.unwrap();
    }

    #[tokio::test]
    async fn fencing_token_is_stable_across_heartbeats() {
        let store = dyn_store();
        let key = "leases/fence-token.pb";
        let mut guard = try_acquire(store, key, "h1", "test", Duration::from_secs(30))
            .await
            .unwrap()
            .unwrap();
        let token = guard.fencing_token();
        guard.heartbeat(Duration::from_secs(30)).await.unwrap();
        assert_eq!(guard.fencing_token(), token);
        guard.release().await.unwrap();
    }

    #[tokio::test]
    async fn lease_release_frees_it() {
        let store = dyn_store();
        let key = "leases/rel.pb";

        let g = try_acquire(store.clone(), key, "h1", "test", Duration::from_secs(30))
            .await
            .unwrap()
            .unwrap();
        g.release().await.unwrap();

        let g2 = try_acquire(store.clone(), key, "h2", "test", Duration::from_secs(30))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(g2.holder(), "h2");
    }

    #[tokio::test]
    async fn stale_owner_release_cannot_delete_reclaimed_lease() {
        let race = HeadThenDeleteStore::new();
        let store: DynStore = race.clone();
        let key = "leases/stale-release.pb";

        let g1 = try_acquire(store.clone(), key, "h1", "test", Duration::from_millis(20))
            .await
            .unwrap()
            .unwrap();
        tokio::time::sleep(LEASE_SKEW_TOLERANCE + Duration::from_millis(50)).await;

        let release = tokio::spawn(async move { g1.release().await });
        let observed_unsafe_delete =
            tokio::time::timeout(Duration::from_millis(250), race.head_observed.notified())
                .await
                .is_ok();

        // The unfixed S3 path pauses here after HEAD. Let the successor steal
        // the expired lease, then allow the stale DELETE to run.
        let g2 = if observed_unsafe_delete {
            let g2 = try_acquire(store.clone(), key, "h2", "test", Duration::from_secs(30))
                .await
                .unwrap()
                .unwrap();
            race.allow_delete.notify_one();
            g2
        } else {
            // The safe path released with a conditional PUT and never called
            // delete; it should make the key immediately reusable.
            try_acquire(store.clone(), key, "h2", "test", Duration::from_secs(30))
                .await
                .unwrap()
                .unwrap()
        };

        release.await.unwrap().unwrap();

        let mut g2 = g2;
        g2.heartbeat(Duration::from_secs(30)).await.unwrap();
    }

    #[tokio::test]
    async fn lease_lost_after_external_steal() {
        let store = dyn_store();
        let key = "leases/lost.pb";
        let ttl = Duration::from_millis(50);

        let mut g1 = try_acquire(store.clone(), key, "h1", "test", ttl)
            .await
            .unwrap()
            .unwrap();
        // Wait for expiry + skew, then steal from outside.
        tokio::time::sleep(LEASE_SKEW_TOLERANCE + Duration::from_millis(100)).await;
        let _g2 = try_acquire(store.clone(), key, "h2", "test", Duration::from_secs(30))
            .await
            .unwrap()
            .unwrap();

        let res = g1.heartbeat(Duration::from_secs(30)).await;
        assert!(matches!(res, Err(CoordError::LeaseLost)));
    }

    #[tokio::test]
    async fn acquire_waits_then_succeeds() {
        let store = dyn_store();
        let key = "leases/acquire.pb";

        let g1 = try_acquire(store.clone(), key, "h1", "test", Duration::from_millis(50))
            .await
            .unwrap()
            .unwrap();
        // g1 holds with 50ms ttl + 2s skew tolerance ≈ 2050ms before stealable.
        // acquire with wait_up_to = 3s should eventually get it.
        let g2 = acquire(
            store.clone(),
            key,
            "h2",
            "test",
            Duration::from_secs(30),
            Duration::from_secs(3),
        )
        .await
        .unwrap()
        .expect("should acquire within 3s");
        assert_eq!(g2.holder(), "h2");
        // g1 is now stale; drop it (best-effort release will no-op on PreconditionFailed).
        drop(g1);
    }

    #[tokio::test]
    async fn acquire_times_out() {
        let store = dyn_store();
        let key = "leases/timeout.pb";

        let _g1 = try_acquire(store.clone(), key, "h1", "test", Duration::from_secs(60))
            .await
            .unwrap()
            .unwrap();
        let g2 = acquire(
            store.clone(),
            key,
            "h2",
            "test",
            Duration::from_secs(30),
            Duration::from_millis(200),
        )
        .await
        .unwrap();
        assert!(g2.is_none());
    }

    #[tokio::test]
    async fn get_message_if_changed_works() {
        let store = dyn_store();
        let key = "catalog.pb";

        // Absent => None.
        let res = get_message_if_changed::<RepoCatalog>(store.as_ref(), key, &Version::new("0"))
            .await
            .unwrap();
        assert!(res.is_none());

        // Create.
        let (meta, _) = cas_update::<RepoCatalog, _>(store.as_ref(), key, 10, |current| {
            assert!(current.is_none());
            let mut c = RepoCatalog::default();
            c.repos.push("a".into());
            Ok(Some(c))
        })
        .await
        .unwrap()
        .unwrap();

        // Same version => None (unchanged).
        let res = get_message_if_changed::<RepoCatalog>(store.as_ref(), key, &meta.version)
            .await
            .unwrap();
        assert!(res.is_none());

        // Different version => Some.
        let res = get_message_if_changed::<RepoCatalog>(store.as_ref(), key, &Version::new("0"))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(res.1.repos, vec!["a".to_string()]);
    }

    #[test]
    fn instance_id_is_stable() {
        let a = instance_id();
        let b = instance_id();
        assert_eq!(a, b);
        assert!(!a.is_empty());
    }
}
