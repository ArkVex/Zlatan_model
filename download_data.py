"""Download IO-VNBD smartphone (S-*) CSV files from the GitHub LFS media host.

The repo stores CSVs via Git LFS, so the normal raw.githubusercontent URL returns
only a pointer file. The media.githubusercontent.com/media/... host returns the
real content.
"""
import json
import os
import sys
import time
from urllib.parse import quote
from urllib.request import urlopen, Request

REPO = "onyekpeu/IO-VNBD"
BRANCH = "master"
MEDIA_BASE = f"https://media.githubusercontent.com/media/{REPO}/{BRANCH}/"
OUT_DIR = os.path.join(os.path.dirname(__file__), "data")

# How many files to pull this run (spread across categories). Set to None for all 144.
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 40


def pick_subset(paths, limit):
    """Spread the selection across scenario categories for variety."""
    if limit is None or limit >= len(paths):
        return paths
    by_cat = {}
    for p in paths:
        parts = p.split("/")
        cat = parts[2] if len(parts) > 2 else "?"
        by_cat.setdefault(cat, []).append(p)
    picked, i = [], 0
    cats = list(by_cat.values())
    while len(picked) < limit:
        added = False
        for lst in cats:
            if i < len(lst):
                picked.append(lst[i])
                added = True
                if len(picked) >= limit:
                    break
        if not added:
            break
        i += 1
    return picked


def download(path):
    # URL-encode each path segment (spaces, parentheses) but keep the slashes.
    url = MEDIA_BASE + quote(path)
    local = os.path.join(OUT_DIR, path.split("/")[-1])
    if os.path.exists(local) and os.path.getsize(local) > 1000:
        return "cached"
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=120) as r:
        data = r.read()
    # Guard: an LFS pointer starts with this text and is tiny.
    if data[:40].startswith(b"version https://git-lfs"):
        return "pointer-only(FAIL)"
    with open(local, "wb") as f:
        f.write(data)
    return f"{len(data)//1024} KB"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    paths = json.load(open(os.path.join(os.path.dirname(__file__), "sfile_paths.json")))
    subset = pick_subset(paths, LIMIT)
    print(f"Downloading {len(subset)} of {len(paths)} S-files -> {OUT_DIR}")
    ok = 0
    for i, p in enumerate(subset, 1):
        name = p.split("/")[-1]
        try:
            status = download(p)
            print(f"[{i}/{len(subset)}] {name}: {status}")
            if "FAIL" not in status:
                ok += 1
        except Exception as e:
            print(f"[{i}/{len(subset)}] {name}: ERROR {e}")
        time.sleep(0.1)
    print(f"Done. {ok}/{len(subset)} files ready in {OUT_DIR}")


if __name__ == "__main__":
    main()
