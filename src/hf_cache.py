"""
hf_cache.py — tiny Hugging Face cache tool.

Two commands over the local HF hub cache (`~/.cache/huggingface/hub`, or `$HF_HOME`):

    python src/hf_cache.py ls                 # list every cached repo-id + size (largest first)
    python src/hf_cache.py rm <repo_id> ...   # delete a repo (all its revisions) from the cache

Thin wrapper over `huggingface_hub.scan_cache_dir()` — it owns the cache layout
(deduplicated blobs symlinked into snapshots), so we never touch files by hand.
"""

from __future__ import annotations

import argparse
import sys

from huggingface_hub import scan_cache_dir


def cmd_ls() -> int:
    info = scan_cache_dir()
    repos = sorted(info.repos, key=lambda r: r.size_on_disk, reverse=True)
    if not repos:
        print("(cache is empty)")
        return 0
    width = max(len(r.repo_id) for r in repos)
    for r in repos:
        print(f"{r.repo_id:<{width}}  {r.size_on_disk_str:>10}  [{r.repo_type}]")
    print("-" * (width + 24))
    print(f"{len(repos)} repos  ·  {info.size_on_disk_str} total")
    return 0


def cmd_rm(repo_ids: list[str], yes: bool) -> int:
    info = scan_cache_dir()
    by_id = {r.repo_id: r for r in info.repos}

    missing = [rid for rid in repo_ids if rid not in by_id]
    if missing:
        print(f"not in cache: {', '.join(missing)}", file=sys.stderr)
        print(f"available: {', '.join(sorted(by_id)) or '(none)'}", file=sys.stderr)
        return 1

    hashes = [rev.commit_hash for rid in repo_ids for rev in by_id[rid].revisions]
    strategy = info.delete_revisions(*hashes)
    print(f"about to delete: {', '.join(repo_ids)}")
    print(f"frees ~{strategy.expected_freed_size_str}")
    if not yes:
        if input("proceed? [y/N] ").strip().lower() not in ("y", "yes"):
            print("aborted.")
            return 1
    strategy.execute()
    print("done.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="list / remove Hugging Face cached repos")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ls", help="list cached repo-ids and sizes")
    rm = sub.add_parser("rm", help="remove repo(s) from the cache")
    rm.add_argument("repo_id", nargs="+", help="repo id(s), e.g. google/gemma-4-e2b-it")
    rm.add_argument("-y", "--yes", action="store_true", help="skip the confirmation prompt")
    args = p.parse_args()

    if args.cmd == "ls":
        return cmd_ls()
    return cmd_rm(args.repo_id, args.yes)


if __name__ == "__main__":
    raise SystemExit(main())
