"""Day 7: turns one uploaded CSV into one read-only-queryable SQLite table.

Scope, deliberately: ONE CSV -> ONE SQLite table. No multi-file datasets, no
relationships between uploads, no database server connections — all
explicitly out of scope for this first version.

Security model
----------------
Every requirement below maps to one specific step, so each is easy to point
at and explain on its own:

  1. Only .csv is accepted           -> extension check, before anything else is read.
  2. A size limit is enforced        -> _read_upload_bytes() aborts mid-stream,
                                         never buffers more than the limit.
  3. Nothing in the file is executed -> only pandas.read_csv touches its bytes;
                                         no eval, no formula evaluation, no exec.
  4. The filename is never trusted   -> only ever used as free TEXT for a
     as a SQL identifier                display name and as *input* to a
                                         sanitizer; the sanitizer's OUTPUT,
                                         never the filename itself, becomes
                                         the SQL table name.
  5. Column names are sanitized too  -> a CSV header is exactly as
                                         attacker-controlled as the filename;
                                         both go through the same sanitizer.
  6. Cell values are never SQL text  -> INSERT uses `?` parameter placeholders;
                                         a value can never inject SQL no matter
                                         what it contains, so sanitization only
                                         needs to cover identifiers, not data.
  7. Read-only afterward             -> this module opens the ONLY read-write
                                         connection the file will ever have,
                                         to build it, then closes it. From
                                         then on backend/db.py opens it
                                         exactly like olist.db: mode=ro +
                                         PRAGMA query_only.
  8. No path ever reaches the client -> the on-disk location is a server-
                                         generated UUID folder under
                                         settings.UPLOADS_DIR, never derived
                                         from the filename, and the caller
                                         (backend/main.py) only ever returns
                                         the fields of CsvUploadResult that
                                         aren't a filesystem path.
"""
import io
import re
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from backend.config import settings
from backend.dataset import ThemeKey
from backend.domain_detection import detect_domain_and_theme

ALLOWED_EXTENSION = ".csv"
MAX_COLUMNS = 200          # sanity cap, independent of the byte-size limit
_CHUNK_SIZE = 64 * 1024    # read in 64 KB chunks while enforcing the size cap
_SAMPLE_VALUE_LIMIT = 30   # cell values shown to domain_detection.py


class CsvValidationError(ValueError):
    """A clear, user-facing reason the upload was rejected. Never wraps a
    raw traceback — see build_dataset_from_csv()'s except clause."""


@dataclass
class CsvUploadResult:
    """Everything backend/main.py needs to register the new dataset.
    `database_path` is here because dataset.py's registration needs it —
    main.py must NOT put it in any response it returns to the client."""
    dataset_id: str
    table_name: str
    database_path: Path
    row_count: int
    column_names: list[str]     # the sanitized names actually used in the table
    domain: str
    theme: ThemeKey
    display_name: str           # a cleaned-up filename, for display only


def _sanitize_identifier(raw: str, fallback: str) -> str:
    """Turn arbitrary text (a filename or a CSV header cell) into a safe
    SQL identifier: lowercase, only [a-z0-9_], never starts with a digit or
    collides with SQLite's reserved `sqlite_` prefix, never empty."""
    text = re.sub(r"[^a-z0-9_]+", "_", raw.strip().lower())
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = fallback
    if text[0].isdigit():
        text = f"{fallback}_{text}"
    if text.startswith("sqlite_"):
        text = f"t_{text}"
    return text[:63]


def _dedupe(names: list[str]) -> list[str]:
    """Make a list of sanitized names unique (two columns that sanitize to
    the same name, e.g. "Price ($)" and "price!!", must not collide)."""
    used: set[str] = set()
    result = []
    for name in names:
        candidate, n = name, 2
        while candidate in used:
            candidate = f"{name}_{n}"
            n += 1
        used.add(candidate)
        result.append(candidate)
    return result


def sanitize_table_name(filename: str) -> str:
    # Path(...).stem also strips any directory components a browser might
    # (unusually) send — defense in depth, even though this value is never
    # used to build a filesystem path.
    return _sanitize_identifier(Path(filename).stem, fallback="uploaded_table")


def _clean_display_name(filename: str) -> str:
    stem = re.sub(r"[_\-]+", " ", Path(filename).stem).strip()
    return (stem or "Uploaded dataset")[:80]


def _read_upload_bytes(file_obj, max_bytes: int) -> bytes:
    """Read `file_obj` (a standard Python file-like object) in bounded
    chunks, aborting as soon as the size limit is exceeded — so a
    malicious/oversized upload is never fully buffered in memory even
    transiently, regardless of what Content-Length claimed."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = file_obj.read(_CHUNK_SIZE)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise CsvValidationError(
                f"File is larger than the {max_bytes // (1024 * 1024)} MB upload limit."
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _infer_column_types(df: pd.DataFrame) -> dict[str, str]:
    """INTEGER / REAL / TEXT per column, same heuristic as Day 1's
    database/build_db.py: try numeric, otherwise TEXT."""
    types: dict[str, str] = {}
    for col in df.columns:
        values = df[col].dropna()
        if values.empty:
            types[col] = "TEXT"
            continue
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.notna().all():
            types[col] = "INTEGER" if (numeric % 1 == 0).all() else "REAL"
        else:
            types[col] = "TEXT"
    return types


def _convert_cell(value, sqlite_type: str):
    if pd.isna(value):
        return None
    if sqlite_type == "INTEGER":
        return int(float(value))
    if sqlite_type == "REAL":
        return float(value)
    return value


def _collect_sample_values(df: pd.DataFrame, limit: int) -> list[str]:
    flat = df.head(5).astype(str).to_numpy().flatten().tolist()
    return [v for v in flat if v and v.lower() != "nan"][:limit]


def build_dataset_from_csv(filename: str, file_obj, max_bytes: int | None = None) -> CsvUploadResult:
    """Validate, sanitize and convert one uploaded CSV into its own SQLite
    database. Raises CsvValidationError (safe to show the user) for any
    rejected input; never lets a raw exception from pandas/sqlite3 escape.

    `file_obj` is a standard Python file-like object (e.g. FastAPI's
    `UploadFile.file`) — read synchronously, in bounded chunks.
    """
    if not filename or not filename.lower().endswith(ALLOWED_EXTENSION):
        raise CsvValidationError("Only .csv files are supported.")

    max_bytes = settings.CSV_UPLOAD_MAX_BYTES if max_bytes is None else max_bytes
    raw_bytes = _read_upload_bytes(file_obj, max_bytes)
    if not raw_bytes:
        raise CsvValidationError("The uploaded file is empty.")

    try:
        df = pd.read_csv(io.BytesIO(raw_bytes), dtype=str)
    except Exception as exc:
        raise CsvValidationError(f"Could not parse this file as CSV: {exc}") from exc

    if df.shape[1] == 0:
        raise CsvValidationError("The CSV has no columns.")
    if df.shape[1] > MAX_COLUMNS:
        raise CsvValidationError(f"The CSV has too many columns (limit is {MAX_COLUMNS}).")
    if df.shape[0] == 0:
        raise CsvValidationError("The CSV has no data rows.")

    sanitized_columns = _dedupe(
        [_sanitize_identifier(str(col), f"col_{i + 1}") for i, col in enumerate(df.columns)]
    )
    df.columns = sanitized_columns
    column_types = _infer_column_types(df)
    table_name = sanitize_table_name(filename)

    # Server-generated identifiers only — the upload's on-disk location
    # never depends on the client-supplied filename.
    upload_uuid = uuid.uuid4().hex
    upload_dir = settings.UPLOADS_DIR / upload_uuid
    upload_dir.mkdir(parents=True, exist_ok=True)
    database_path = upload_dir / "dataset.db"

    # The ONE read-write connection this file will ever have. Identifiers
    # are sanitized (above) and double-quoted; cell values go through `?`
    # placeholders, never string-formatted into SQL.
    conn = sqlite3.connect(database_path)
    try:
        columns_sql = ", ".join(f'"{col}" {column_types[col]}' for col in sanitized_columns)
        conn.execute(f'CREATE TABLE "{table_name}" ({columns_sql})')

        quoted_columns = ", ".join(f'"{col}"' for col in sanitized_columns)
        placeholders = ", ".join("?" for _ in sanitized_columns)
        insert_sql = f'INSERT INTO "{table_name}" ({quoted_columns}) VALUES ({placeholders})'
        rows = [
            tuple(_convert_cell(value, column_types[col]) for col, value in zip(sanitized_columns, row))
            for row in df.itertuples(index=False, name=None)
        ]
        conn.executemany(insert_sql, rows)
        conn.commit()
    finally:
        conn.close()

    domain, theme = detect_domain_and_theme(
        filename=filename,
        column_names=sanitized_columns,
        sample_values=_collect_sample_values(df, _SAMPLE_VALUE_LIMIT),
    )

    return CsvUploadResult(
        dataset_id=f"upload_{upload_uuid[:12]}",
        table_name=table_name,
        database_path=database_path,
        row_count=len(df),
        column_names=sanitized_columns,
        domain=domain,
        theme=theme,
        display_name=_clean_display_name(filename),
    )
