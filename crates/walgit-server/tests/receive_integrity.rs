//! Acknowledged refs must remain cloneable after the writer loses its cache.
mod harness;

use anyhow::Result;
use harness::{Server, TestRepo, git_in};
use sha1::{Digest, Sha1};

fn writer_auth(config: &mut walgit_config::Config) {
    config.server.auth.mode = walgit_config::AuthMode::Token;
    config.server.auth.anonymous_read = false;
    config.server.auth.tokens = vec![walgit_config::StaticToken {
        principal: "writer".into(),
        token: "integrity-test".into(),
        token_env: None,
        write: true,
        admin: false,
    }];
    config.bundles.advertise = false;
    config.wal.prefetch_packs = false;
}

fn empty_pack() -> Vec<u8> {
    // Independently encode the Git wire format, including the checksum.
    let header = b"PACK\0\0\0\x02\0\0\0\0";
    let mut pack = header.to_vec();
    pack.extend_from_slice(&Sha1::digest(header));
    pack
}

async fn push_with_pack(
    server: &Server,
    old: &str,
    new: &str,
    name: &str,
    pack: &[u8],
) -> Result<String> {
    let command = format!("{old} {new} {name}\0report-status\n");
    let mut body = format!("{:04x}{command}0000", command.len() + 4).into_bytes();
    body.extend_from_slice(pack);
    let response = reqwest::Client::new()
        .post(format!("{}/t/r.git/git-receive-pack", server.base_url))
        .bearer_auth("integrity-test")
        .header("Content-Type", "application/x-git-receive-pack-request")
        .body(body)
        .send()
        .await?;
    assert_eq!(response.status(), reqwest::StatusCode::OK);
    Ok(response.text().await?)
}

async fn seeded_server() -> Result<(Server, TestRepo, String)> {
    let server = Server::start_with_tweak(writer_auth).await?;
    let response = reqwest::Client::new()
        .put(format!("{}/t/r.git", server.base_url))
        .bearer_auth("integrity-test")
        .send()
        .await?;
    assert!(response.status().is_success());
    let source = TestRepo::synthetic(2, 2)?;
    git_in(&source, &["commit", "--allow-empty", "-m", "seed"])?;
    git_in(&source, &["branch", "-M", "main"])?;
    git_in(
        &source,
        &[
            "-c",
            "http.extraHeader=Authorization: Bearer integrity-test",
            "push",
            &server.repo_url("t", "r"),
            "main",
        ],
    )?;
    let head = git_in(&source, &["rev-parse", "HEAD"])?;
    Ok((server, source, head.trim().to_owned()))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn ref_only_push_cannot_publish_missing_objects() -> Result<()> {
    let (server, _source, head) = seeded_server().await?;
    let null = "0".repeat(40);
    let missing = "a".repeat(40);
    let log_before = server.read_log("t", "r").await?;

    let empty_pack = empty_pack();
    for pack in [empty_pack.as_slice(), &[]] {
        for (old, name) in [(&null, "refs/heads/ghost"), (&head, "refs/heads/main")] {
            let report = push_with_pack(&server, old, &missing, name, pack).await?;
            assert!(
                report.contains(&format!("ng {name} connectivity:")),
                "missing target was not refused (pack_bytes={}): {report:?}",
                pack.len()
            );
        }
    }
    assert_eq!(
        server.read_log("t", "r").await?,
        log_before,
        "refused pushes must not append durable transactions"
    );

    // The later reader has no local cache from the writer. Stock Git must
    // reconstruct the original repository entirely from durable state.
    let cold = server.start_sibling_with(writer_auth).await?;
    drop(server);
    let clone = tempfile::tempdir()?;
    git_in(
        clone.path(),
        &[
            "-c",
            "http.extraHeader=Authorization: Bearer integrity-test",
            "clone",
            "--mirror",
            &cold.repo_url("t", "r"),
            ".",
        ],
    )?;
    git_in(clone.path(), &["fsck", "--full"])?;
    assert_eq!(
        git_in(clone.path(), &["show-ref"])?.trim(),
        format!("{head} refs/heads/main")
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn rejected_pack_cannot_supply_objects_to_a_later_push() -> Result<()> {
    let (server, source, head) = seeded_server().await?;
    git_in(&source, &["commit", "--allow-empty", "-m", "unpublished"])?;
    let unpublished = git_in(&source, &["rev-parse", "HEAD"])?;
    let packs = tempfile::tempdir()?;
    let prefix = packs.path().join("rejected");
    let checksum = git_in(
        &source,
        &["pack-objects", "--all", prefix.to_str().unwrap()],
    )?;
    let pack = std::fs::read(prefix.with_file_name(format!("rejected-{}.pack", checksum.trim())))?;
    let null = "0".repeat(40);

    // The valid pack is ingested before the stale ref update is refused.
    let refused =
        push_with_pack(&server, &null, unpublished.trim(), "refs/heads/main", &pack).await?;
    assert!(refused.contains("ng refs/heads/main"), "{refused:?}");
    let log_before = server.read_log("t", "r").await?;
    let empty_pack = empty_pack();
    for pack in [empty_pack.as_slice(), &[]] {
        let report =
            push_with_pack(&server, &null, unpublished.trim(), "refs/heads/ghost", pack).await?;
        assert!(
            report.contains("ng refs/heads/ghost connectivity:"),
            "cache-only object was accepted: {report:?}"
        );
    }
    assert_eq!(server.read_log("t", "r").await?, log_before);
    let cold = server.start_sibling_with(writer_auth).await?;
    drop(server);
    let clone = tempfile::tempdir()?;
    git_in(
        clone.path(),
        &[
            "-c",
            "http.extraHeader=Authorization: Bearer integrity-test",
            "clone",
            "--mirror",
            &cold.repo_url("t", "r"),
            ".",
        ],
    )?;
    git_in(clone.path(), &["fsck", "--full"])?;
    assert_eq!(
        git_in(clone.path(), &["show-ref"])?.trim(),
        format!("{head} refs/heads/main")
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn ref_only_push_preserves_existing_objects_and_deletions() -> Result<()> {
    let (server, _source, head) = seeded_server().await?;
    let null = "0".repeat(40);
    let empty_pack = empty_pack();
    for pack in [empty_pack.as_slice(), &[]] {
        let name = if !pack.is_empty() {
            "refs/heads/packed"
        } else {
            "refs/heads/unpacked"
        };
        let create = push_with_pack(&server, &null, &head, name, pack).await?;
        assert!(create.contains(&format!("ok {name}\n")), "{create:?}");
        let cold = server.start_sibling_with(writer_auth).await?;
        let refs = git_in(
            &_source,
            &[
                "-c",
                "http.extraHeader=Authorization: Bearer integrity-test",
                "ls-remote",
                &cold.repo_url("t", "r"),
                name,
            ],
        )?;
        assert_eq!(refs.trim(), format!("{head}\t{name}"));
        let delete = push_with_pack(&server, &head, &null, name, &[]).await?;
        assert!(delete.contains(&format!("ok {name}\n")), "{delete:?}");
    }
    let cold = server.start_sibling_with(writer_auth).await?;
    drop(server);
    let clone = tempfile::tempdir()?;
    git_in(
        clone.path(),
        &[
            "-c",
            "http.extraHeader=Authorization: Bearer integrity-test",
            "clone",
            "--mirror",
            &cold.repo_url("t", "r"),
            ".",
        ],
    )?;
    git_in(clone.path(), &["fsck", "--full"])?;
    assert_eq!(
        git_in(clone.path(), &["show-ref"])?.trim(),
        format!("{head} refs/heads/main")
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unrelated_unmaterialized_packs_do_not_block_valid_ref_updates() -> Result<()> {
    let (server, source, head) = seeded_server().await?;
    let handle = server
        .state
        .registry
        .open(&walgit_git::RepoId::new("t", "r")?)
        .await?;
    let _request_guard = handle.sync_full().await?;
    let sibling = server.start_sibling_with(writer_auth).await?;
    git_in(&source, &["commit", "--allow-empty", "-m", "sibling ref"])?;
    git_in(
        &source,
        &[
            "-c",
            "http.extraHeader=Authorization: Bearer integrity-test",
            "push",
            &sibling.repo_url("t", "r"),
            "HEAD:refs/heads/sibling",
        ],
    )?;
    // A concurrent advertisement can advance refs while an admitted receive
    // request still holds its guard. It does not download the sibling's pack.
    let _refs_guard = handle.sync_refs().await?;
    assert!(!handle.packs_ready());
    let tip = gix_hash::ObjectId::from_hex(head.as_bytes())?;
    handle.check_push_connectivity(&[tip], None).await?;
    Ok(())
}
