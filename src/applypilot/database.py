"""ApplyPilot database layer: schema, migrations, stats, and connection helpers.

Single source of truth for the jobs table schema. All columns from every
pipeline stage are created up front so any stage can run independently
without migration ordering issues.
"""

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from applypilot.config import DB_PATH

# Thread-local connection storage — each thread gets its own connection
# (required for SQLite thread safety with parallel workers)
_local = threading.local()


def get_connection(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Get a thread-local cached SQLite connection with WAL mode enabled.

    Each thread gets its own connection (required for SQLite thread safety).
    Connections are cached and reused within the same thread.

    Args:
        db_path: Override the default DB_PATH. Useful for testing.

    Returns:
        sqlite3.Connection configured with WAL mode and row factory.
    """
    path = str(db_path or DB_PATH)

    if not hasattr(_local, 'connections'):
        _local.connections = {}

    conn = _local.connections.get(path)
    if conn is not None:
        try:
            conn.execute("SELECT 1")
            return conn
        except sqlite3.ProgrammingError:
            pass

    conn = sqlite3.connect(path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    conn.row_factory = sqlite3.Row
    _local.connections[path] = conn
    return conn


def close_connection(db_path: Path | str | None = None) -> None:
    """Close the cached connection for the current thread."""
    path = str(db_path or DB_PATH)
    if hasattr(_local, 'connections'):
        conn = _local.connections.pop(path, None)
        if conn is not None:
            conn.close()


def init_db(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Create the full jobs table with all columns from every pipeline stage.

    This is idempotent -- safe to call on every startup. Uses CREATE TABLE IF NOT EXISTS
    so it won't destroy existing data.

    Schema columns by stage:
      - Discovery:  url, title, salary, description, location, site, strategy, discovered_at
      - Enrichment: full_description, application_url, detail_scraped_at, detail_error
      - Scoring:    fit_score, score_reasoning, scored_at plus local/cloud metadata
      - Tailoring:  tailored_resume_path, tailored_at, tailor_attempts
      - Cover:      cover_letter_path, cover_letter_at, cover_attempts
      - Apply:      applied_at, apply_status, apply_error, apply_attempts,
                   agent_id, last_attempted_at, apply_duration_ms, apply_task_id,
                   verification_confidence

    Args:
        db_path: Override the default DB_PATH.

    Returns:
        sqlite3.Connection with the schema initialized.
    """
    path = db_path or DB_PATH

    # Ensure parent directory exists
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    conn = get_connection(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            -- Discovery stage (smart_extract / job_search)
            url                   TEXT PRIMARY KEY,
            title                 TEXT,
            salary                TEXT,
            description           TEXT,
            location              TEXT,
            site                  TEXT,
            strategy              TEXT,
            discovered_at         TEXT,

            -- Enrichment stage (detail_scraper)
            full_description      TEXT,
            application_url       TEXT,
            application_url_checked_at TEXT,
            detail_scraped_at     TEXT,
            detail_error          TEXT,

            -- Scoring stage (job_scorer)
            fit_score             INTEGER,
            score_reasoning       TEXT,
            scored_at             TEXT,
            local_fit_score       INTEGER,
            local_score_reasoning TEXT,
            local_scored_at       TEXT,
            local_score_error     TEXT,
            cloud_fit_score       INTEGER,
            cloud_score_reasoning TEXT,
            cloud_validated_at    TEXT,
            cloud_validation_status TEXT,
            cloud_validation_error TEXT,

            -- Tailoring stage (resume tailor)
            tailored_resume_path  TEXT,
            tailored_at           TEXT,
            tailor_attempts       INTEGER DEFAULT 0,

            -- Cover letter stage
            cover_letter_path     TEXT,
            cover_letter_at       TEXT,
            cover_attempts        INTEGER DEFAULT 0,

            -- Application stage
            applied_at            TEXT,
            apply_status          TEXT,
            apply_error           TEXT,
            apply_attempts        INTEGER DEFAULT 0,
            agent_id              TEXT,
            last_attempted_at     TEXT,
            apply_duration_ms     INTEGER,
            apply_task_id         TEXT,
            verification_confidence TEXT
        )
    """)
    conn.commit()

    # Run migrations for any columns added after initial schema
    ensure_columns(conn)

    return conn


# Complete column registry: column_name -> SQL type with optional default.
# This is the single source of truth. Adding a column here is all that's needed
# for it to appear in both new databases and migrated ones.
_ALL_COLUMNS: dict[str, str] = {
    # Discovery
    "url": "TEXT PRIMARY KEY",
    "title": "TEXT",
    "salary": "TEXT",
    "description": "TEXT",
    "location": "TEXT",
    "site": "TEXT",
    "strategy": "TEXT",
    "discovered_at": "TEXT",
    # Enrichment
    "full_description": "TEXT",
    "application_url": "TEXT",
    "application_url_checked_at": "TEXT",
    "detail_scraped_at": "TEXT",
    "detail_error": "TEXT",
    # Scoring
    "fit_score": "INTEGER",
    "score_reasoning": "TEXT",
    "scored_at": "TEXT",
    "local_fit_score": "INTEGER",
    "local_score_reasoning": "TEXT",
    "local_scored_at": "TEXT",
    "local_score_error": "TEXT",
    "cloud_fit_score": "INTEGER",
    "cloud_score_reasoning": "TEXT",
    "cloud_validated_at": "TEXT",
    "cloud_validation_status": "TEXT",
    "cloud_validation_error": "TEXT",
    # Tailoring
    "tailored_resume_path": "TEXT",
    "tailored_at": "TEXT",
    "tailor_attempts": "INTEGER DEFAULT 0",
    # Cover letter
    "cover_letter_path": "TEXT",
    "cover_letter_at": "TEXT",
    "cover_attempts": "INTEGER DEFAULT 0",
    # Application
    "applied_at": "TEXT",
    "apply_status": "TEXT",
    "apply_error": "TEXT",
    "apply_attempts": "INTEGER DEFAULT 0",
    "agent_id": "TEXT",
    "last_attempted_at": "TEXT",
    "apply_duration_ms": "INTEGER",
    "apply_task_id": "TEXT",
    "verification_confidence": "TEXT",
}


def ensure_columns(conn: sqlite3.Connection | None = None) -> list[str]:
    """Add any missing columns to the jobs table (forward migration).

    Reads the current table schema via PRAGMA table_info and compares against
    the full column registry. Any missing columns are added with ALTER TABLE.

    This makes it safe to upgrade the database from any previous version --
    columns are only added, never removed or renamed.

    Args:
        conn: Database connection. Uses get_connection() if None.

    Returns:
        List of column names that were added (empty if schema was already current).
    """
    if conn is None:
        conn = get_connection()

    existing = {row[1] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()}
    added = []

    for col, dtype in _ALL_COLUMNS.items():
        if col not in existing:
            # PRIMARY KEY columns can't be added via ALTER TABLE, but url
            # is always created with the table itself so this is safe
            if "PRIMARY KEY" in dtype:
                continue
            conn.execute(f"ALTER TABLE jobs ADD COLUMN {col} {dtype}")
            added.append(col)

    if added:
        conn.commit()

    return added


def get_stats(conn: sqlite3.Connection | None = None) -> dict:
    """Return job counts by pipeline stage.

    Provides a snapshot of how many jobs are at each stage, useful for
    dashboard display and pipeline progress tracking.

    Args:
        conn: Database connection. Uses get_connection() if None.

    Returns:
        Dictionary with keys:
            total, by_site, pending_detail, with_description,
            scored, unscored, tailored, untailored_eligible,
            with_cover_letter, applied, score_distribution
    """
    if conn is None:
        conn = get_connection()

    stats: dict = {}

    # Total jobs
    stats["total"] = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]

    # By site breakdown
    rows = conn.execute(
        "SELECT site, COUNT(*) as cnt FROM jobs GROUP BY site ORDER BY cnt DESC"
    ).fetchall()
    stats["by_site"] = [(row[0], row[1]) for row in rows]

    # Enrichment stage
    stats["pending_detail"] = conn.execute(
        "SELECT COUNT(*) FROM jobs "
        "WHERE detail_scraped_at IS NULL "
        "OR ((application_url IS NULL OR application_url = '') "
        "AND application_url_checked_at IS NULL)"
    ).fetchone()[0]

    stats["with_description"] = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE full_description IS NOT NULL"
    ).fetchone()[0]

    stats["detail_errors"] = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE detail_error IS NOT NULL"
    ).fetchone()[0]

    # Scoring stage
    stats["scored"] = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE fit_score IS NOT NULL"
    ).fetchone()[0]

    stats["unscored"] = conn.execute(
        "SELECT COUNT(*) FROM jobs "
        "WHERE full_description IS NOT NULL AND fit_score IS NULL"
    ).fetchone()[0]

    # Score distribution
    dist_rows = conn.execute(
        "SELECT fit_score, COUNT(*) as cnt FROM jobs "
        "WHERE fit_score IS NOT NULL "
        "GROUP BY fit_score ORDER BY fit_score DESC"
    ).fetchall()
    stats["score_distribution"] = [(row[0], row[1]) for row in dist_rows]

    # Tailoring stage
    stats["tailored"] = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE tailored_resume_path IS NOT NULL"
    ).fetchone()[0]

    stats["untailored_eligible"] = conn.execute(
        "SELECT COUNT(*) FROM jobs "
        "WHERE fit_score >= 7 AND full_description IS NOT NULL "
        "AND tailored_resume_path IS NULL"
    ).fetchone()[0]

    stats["tailor_exhausted"] = conn.execute(
        "SELECT COUNT(*) FROM jobs "
        "WHERE COALESCE(tailor_attempts, 0) >= 5 "
        "AND tailored_resume_path IS NULL"
    ).fetchone()[0]

    # Cover letter stage
    stats["with_cover_letter"] = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE cover_letter_path IS NOT NULL"
    ).fetchone()[0]

    stats["cover_exhausted"] = conn.execute(
        "SELECT COUNT(*) FROM jobs "
        "WHERE COALESCE(cover_attempts, 0) >= 5 "
        "AND (cover_letter_path IS NULL OR cover_letter_path = '')"
    ).fetchone()[0]

    # Application stage
    stats["applied"] = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE applied_at IS NOT NULL"
    ).fetchone()[0]

    stats["apply_errors"] = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE apply_error IS NOT NULL"
    ).fetchone()[0]

    stats["ready_to_apply"] = conn.execute(
        "SELECT COUNT(*) FROM jobs "
        "WHERE tailored_resume_path IS NOT NULL "
        "AND applied_at IS NULL "
        "AND (application_url IS NOT NULL OR url IS NOT NULL)"
    ).fetchone()[0]

    return stats


def normalize_job_id(job_id: str) -> str:
    """Normalize a user-provided URL-ish job identifier for matching."""
    value = (job_id or "").strip()
    if not value:
        return ""
    try:
        parts = urlsplit(value)
    except ValueError:
        return value.rstrip("/")
    if not parts.scheme or not parts.netloc:
        return value.rstrip("/")
    netloc = parts.netloc.lower()
    path = parts.path.rstrip("/")
    return urlunsplit((parts.scheme.lower(), netloc, path, "", ""))


def _job_id_like(job_id: str) -> str:
    normalized = normalize_job_id(job_id)
    return f"%{normalized}%"


def _rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict]:
    if not rows:
        return []
    columns = rows[0].keys()
    return [dict(zip(columns, row)) for row in rows]


def resolve_job_id(conn: sqlite3.Connection, job_id: str) -> dict | None:
    """Resolve a CLI job ID to a single job row.

    The current schema uses ``url`` as the primary key. This helper also accepts
    application URLs and normalized partial URL strings for terminal ergonomics.
    """
    normalized = normalize_job_id(job_id)
    if not normalized:
        return None
    like = _job_id_like(normalized)
    row = conn.execute(
        """
        SELECT * FROM jobs
        WHERE url = ?
           OR application_url = ?
           OR url LIKE ?
           OR application_url LIKE ?
        ORDER BY
            CASE
                WHEN url = ? THEN 0
                WHEN application_url = ? THEN 1
                ELSE 2
            END,
            discovered_at DESC
        LIMIT 1
        """,
        (job_id, job_id, like, like, job_id, job_id),
    ).fetchone()
    return dict(row) if row else None


def resolve_job_ids(conn: sqlite3.Connection, job_ids: list[str]) -> list[str]:
    """Resolve many CLI job IDs to canonical primary-key URLs."""
    urls: list[str] = []
    seen: set[str] = set()
    missing: list[str] = []
    for job_id in job_ids:
        row = resolve_job_id(conn, job_id)
        if not row:
            missing.append(job_id)
            continue
        url = row["url"]
        if url not in seen:
            urls.append(url)
            seen.add(url)
    if missing:
        raise ValueError(f"No matching job found for: {', '.join(missing)}")
    return urls


def read_ids_file(path: Path | str) -> list[str]:
    """Read newline-delimited job IDs, ignoring blanks and comments."""
    ids_path = Path(path).expanduser()
    values: list[str] = []
    for line in ids_path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            values.append(value)
    return values


def job_id_filter_sql(job_ids: list[str] | None, column: str = "url") -> tuple[str, list[str]]:
    """Return an SQL ``IN`` clause fragment and parameters for canonical URLs."""
    if not job_ids:
        return "", []
    placeholders = ",".join("?" for _ in job_ids)
    return f" AND {column} IN ({placeholders})", list(job_ids)


def _stage_key(stage: str) -> str:
    return (stage or "all").strip().lower().replace("-", "_")


def add_job(
    conn: sqlite3.Connection,
    url: str,
    title: str | None = None,
    site: str | None = None,
    location: str | None = None,
) -> bool:
    """Insert one manually discovered job. Returns True when inserted."""
    now = datetime.now(timezone.utc).isoformat()
    try:
        conn.execute(
            """
            INSERT INTO jobs (url, title, site, location, strategy, discovered_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (url, title or url, site or "manual", location, "manual", now),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False


def mark_job_status(
    conn: sqlite3.Connection,
    url: str,
    status: str,
    reason: str | None = None,
) -> int:
    """Set a manual apply status for a job and return affected row count."""
    now = datetime.now(timezone.utc).isoformat()
    normalized_status = status.strip().lower()
    if normalized_status == "applied":
        cursor = conn.execute(
            """
            UPDATE jobs
            SET apply_status = 'applied',
                applied_at = ?,
                apply_error = NULL,
                agent_id = NULL
            WHERE url = ?
            """,
            (now, url),
        )
    elif normalized_status == "manual":
        cursor = conn.execute(
            """
            UPDATE jobs
            SET apply_status = 'manual',
                apply_error = ?,
                agent_id = NULL
            WHERE url = ?
            """,
            (reason or "manual", url),
        )
    elif normalized_status == "failed":
        cursor = conn.execute(
            """
            UPDATE jobs
            SET apply_status = 'failed',
                apply_error = ?,
                apply_attempts = 99,
                agent_id = NULL
            WHERE url = ?
            """,
            (reason or "manual", url),
        )
    else:
        raise ValueError("status must be applied, failed, or manual")
    conn.commit()
    return cursor.rowcount


def reset_jobs_for_stage(
    conn: sqlite3.Connection,
    stage: str,
    urls: list[str],
    include_applied: bool = False,
) -> int:
    """Clear stage output for selected jobs before a forced rerun."""
    if not urls:
        return 0
    placeholders = ",".join("?" for _ in urls)
    key = _stage_key(stage)
    params: list = list(urls)
    if key == "enrich":
        sql = (
            "UPDATE jobs SET full_description = NULL, application_url_checked_at = NULL, "
            "detail_scraped_at = NULL, detail_error = NULL WHERE url IN "
            f"({placeholders})"
        )
    elif key == "score":
        sql = (
            "UPDATE jobs SET fit_score = NULL, score_reasoning = NULL, scored_at = NULL, "
            "local_fit_score = NULL, local_score_reasoning = NULL, local_scored_at = NULL, "
            "local_score_error = NULL, cloud_fit_score = NULL, cloud_score_reasoning = NULL, "
            "cloud_validated_at = NULL, cloud_validation_status = NULL, cloud_validation_error = NULL "
            f"WHERE url IN ({placeholders})"
        )
    elif key == "tailor":
        sql = (
            "UPDATE jobs SET tailored_resume_path = NULL, tailored_at = NULL, tailor_attempts = 0 "
            f"WHERE url IN ({placeholders})"
        )
    elif key == "cover":
        sql = (
            "UPDATE jobs SET cover_letter_path = NULL, cover_letter_at = NULL, cover_attempts = 0 "
            f"WHERE url IN ({placeholders})"
        )
    elif key == "apply":
        sql = (
            "UPDATE jobs SET apply_status = NULL, apply_error = NULL, apply_attempts = 0, "
            "agent_id = NULL, last_attempted_at = NULL "
            f"WHERE url IN ({placeholders})"
        )
        if not include_applied:
            sql += " AND applied_at IS NULL"
    else:
        raise ValueError(f"Cannot force-reset unknown stage: {stage}")
    cursor = conn.execute(sql, params)
    conn.commit()
    return cursor.rowcount


def delete_jobs(conn: sqlite3.Connection, urls: list[str]) -> int:
    """Delete selected jobs from the discovery database."""
    if not urls:
        return 0
    placeholders = ",".join("?" for _ in urls)
    cursor = conn.execute(f"DELETE FROM jobs WHERE url IN ({placeholders})", urls)
    conn.commit()
    return cursor.rowcount


def reset_manual_statuses(
    conn: sqlite3.Connection,
    failed: bool = False,
    manual: bool = False,
    in_progress: bool = False,
) -> int:
    """Reset selected non-applied apply statuses for retry."""
    statuses: list[str] = []
    if failed:
        statuses.append("failed")
    if manual:
        statuses.append("manual")
    if in_progress:
        statuses.append("in_progress")
    if not statuses:
        return 0
    placeholders = ",".join("?" for _ in statuses)
    cursor = conn.execute(
        f"""
        UPDATE jobs
        SET apply_status = NULL,
            apply_error = NULL,
            apply_attempts = 0,
            agent_id = NULL,
            last_attempted_at = NULL
        WHERE apply_status IN ({placeholders})
        """,
        statuses,
    )
    conn.commit()
    return cursor.rowcount


def store_jobs(conn: sqlite3.Connection, jobs: list[dict],
               site: str, strategy: str) -> tuple[int, int]:
    """Store discovered jobs, skipping duplicates by URL.

    Args:
        conn: Database connection.
        jobs: List of job dicts with keys: url, title, salary, description, location.
        site: Source site name (e.g. "RemoteOK", "Dice").
        strategy: Extraction strategy used (e.g. "json_ld", "api_response", "css_selectors").

    Returns:
        Tuple of (new_count, duplicate_count).
    """
    now = datetime.now(timezone.utc).isoformat()
    new = 0
    existing = 0

    for job in jobs:
        url = job.get("url")
        if not url:
            continue
        try:
            conn.execute(
                "INSERT INTO jobs (url, title, salary, description, location, site, strategy, discovered_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (url, job.get("title"), job.get("salary"), job.get("description"),
                 job.get("location"), site, strategy, now),
            )
            new += 1
        except sqlite3.IntegrityError:
            existing += 1

    conn.commit()
    return new, existing


def get_jobs_by_stage(conn: sqlite3.Connection | None = None,
                      stage: str = "discovered",
                      min_score: int | None = None,
                      limit: int = 100,
                      job_ids: list[str] | None = None) -> list[dict]:
    """Fetch jobs filtered by pipeline stage.

    Args:
        conn: Database connection. Uses get_connection() if None.
        stage: One of "discovered", "enriched", "scored", "tailored", "applied".
        min_score: Minimum fit_score filter (only relevant for scored+ stages).
        limit: Maximum number of rows to return.

    Returns:
        List of job dicts.
    """
    if conn is None:
        conn = get_connection()

    stage = _stage_key(stage)
    if stage == "all":
        stage = "discovered"

    conditions = {
        "discovered": "1=1",
        "pending_detail": (
            "detail_scraped_at IS NULL "
            "OR ((application_url IS NULL OR application_url = '') "
            "AND application_url_checked_at IS NULL)"
        ),
        "enriched": "full_description IS NOT NULL",
        "pending_score": "full_description IS NOT NULL AND fit_score IS NULL",
        "scored": "fit_score IS NOT NULL",
        "pending_tailor": (
            "fit_score >= ? AND full_description IS NOT NULL "
            "AND tailored_resume_path IS NULL AND COALESCE(tailor_attempts, 0) < 5"
        ),
        "tailored": "tailored_resume_path IS NOT NULL",
        "pending_apply": (
            "tailored_resume_path IS NOT NULL AND applied_at IS NULL "
            "AND (application_url IS NOT NULL OR url IS NOT NULL)"
        ),
        "manual": "apply_status = 'manual'",
        "failed": "apply_status = 'failed'",
        "applied": "applied_at IS NOT NULL",
    }

    where = conditions.get(stage, "1=1")
    params: list = []

    if "?" in where and min_score is not None:
        params.append(min_score)
    elif "?" in where:
        params.append(7)  # default min_score

    if min_score is not None and "fit_score" not in where and stage in ("scored", "tailored", "applied"):
        where += " AND fit_score >= ?"
        params.append(min_score)

    if job_ids:
        id_filter, id_params = job_id_filter_sql(job_ids)
        where += id_filter
        params.extend(id_params)

    query = f"SELECT * FROM jobs WHERE {where} ORDER BY fit_score DESC NULLS LAST, discovered_at DESC"
    if limit > 0:
        query += " LIMIT ?"
        params.append(limit)

    rows = conn.execute(query, params).fetchall()

    # Convert sqlite3.Row objects to dicts
    if rows:
        columns = rows[0].keys()
        return [dict(zip(columns, row)) for row in rows]
    return []
