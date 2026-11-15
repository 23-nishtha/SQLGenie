"""Render build-time step: ensures database/olist.db exists and matches its
known SHA-256 checksum before the app is allowed to start.

Not used in local development. Your local database/olist.db is built
directly (see database/build_db.py) and this script is a no-op if it's
already present and correct -- it's safe to run locally too, it just
won't need to do anything.

Why this exists: database/olist.db is ~183 MB. GitHub hard-blocks any file
over 100 MB pushed without Git LFS, which this project deliberately avoids
(see the README's deployment section for why). Instead, the built .db file
is uploaded as a GitHub Release asset -- outside git history entirely --
and downloaded here at Render BUILD time, then verified against the
checksum committed in database/olist.db.sha256 before the build is allowed
to succeed. football.db (~9 MB) and movies.db (~0.5 MB) are small enough
to commit directly and need no such step.

Configure via:
  OLIST_DB_URL -- direct download URL for the olist.db Release asset.
                  Required ONLY when the file isn't already present and
                  checksum-valid (i.e. on Render; never needed locally).
                  Set this in Render's dashboard environment variables,
                  never commit it as a secret-bearing value (it isn't
                  secret, but it also isn't known until you've actually
                  created the Release -- see the README).

Exits non-zero (failing the Render build) if the file is missing, the URL
isn't configured, the download fails, or the checksum doesn't match --
this app must never silently start up with a missing or corrupted
built-in dataset.
"""
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "database" / "olist.db"
CHECKSUM_PATH = PROJECT_ROOT / "database" / "olist.db.sha256"
_CHUNK_SIZE = 1024 * 1024


def _read_expected_checksum() -> str:
    text = CHECKSUM_PATH.read_text().strip()
    # Standard `sha256sum` output is "<hash>  <filename>"; a bare hash with
    # nothing else on the line also works.
    return text.split()[0].lower()


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    print(f"Downloading Olist database from {url} ...")
    request = urllib.request.Request(url, headers={"User-Agent": "sqlgenie-build-script"})
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as out:
        while chunk := response.read(_CHUNK_SIZE):
            out.write(chunk)


def main() -> int:
    expected = _read_expected_checksum()

    if DB_PATH.exists():
        actual = _sha256_of(DB_PATH)
        if actual == expected:
            print(f"{DB_PATH} already present and checksum-verified -- skipping download.")
            return 0
        print(
            f"{DB_PATH} exists but its checksum does not match "
            f"{CHECKSUM_PATH.name} (expected {expected}, got {actual}) -- re-downloading."
        )
        DB_PATH.unlink()

    url = os.environ.get("OLIST_DB_URL")
    if not url:
        print(
            "ERROR: database/olist.db is missing and OLIST_DB_URL is not set. "
            "Set OLIST_DB_URL to the Olist database's GitHub Release asset "
            "download URL in Render's environment variables (see README.md's "
            "deployment section).",
            file=sys.stderr,
        )
        return 1

    tmp_path = DB_PATH.with_name(DB_PATH.name + ".partial")
    try:
        _download(url, tmp_path)
    except Exception as exc:
        print(f"ERROR: could not download the Olist database: {exc}", file=sys.stderr)
        tmp_path.unlink(missing_ok=True)
        return 1

    actual = _sha256_of(tmp_path)
    if actual != expected:
        tmp_path.unlink(missing_ok=True)
        print(
            f"ERROR: downloaded Olist database failed checksum verification "
            f"(expected {expected}, got {actual}). Refusing to deploy a "
            "corrupted or unexpected database file.",
            file=sys.stderr,
        )
        return 1

    tmp_path.rename(DB_PATH)
    print(f"Olist database downloaded and verified ({DB_PATH}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
