import json
import sys
from pathlib import Path

from typer.testing import CliRunner


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _configure_app_dir(monkeypatch, tmp_path):
    from applypilot import config
    from applypilot import database

    app_dir = tmp_path / "applypilot"
    monkeypatch.setattr(config, "APP_DIR", app_dir)
    monkeypatch.setattr(config, "DB_PATH", app_dir / "applypilot.db")
    monkeypatch.setattr(config, "PROFILE_PATH", app_dir / "profile.json")
    monkeypatch.setattr(config, "RESUME_PATH", app_dir / "resume.txt")
    monkeypatch.setattr(config, "RESUME_PDF_PATH", app_dir / "resume.pdf")
    monkeypatch.setattr(config, "SEARCH_CONFIG_PATH", app_dir / "searches.yaml")
    monkeypatch.setattr(config, "ENV_PATH", app_dir / ".env")
    monkeypatch.setattr(config, "TAILORED_DIR", app_dir / "tailored_resumes")
    monkeypatch.setattr(config, "COVER_LETTER_DIR", app_dir / "cover_letters")
    monkeypatch.setattr(config, "LOG_DIR", app_dir / "logs")
    monkeypatch.setattr(config, "CHROME_WORKER_DIR", app_dir / "chrome-workers")
    monkeypatch.setattr(config, "APPLY_WORKER_DIR", app_dir / "apply-workers")
    monkeypatch.setattr(database, "DB_PATH", app_dir / "applypilot.db")
    database.close_connection(app_dir / "applypilot.db")
    return app_dir


def _insert_job(conn, url="https://example.com/jobs/1", **overrides):
    values = {
        "url": url,
        "title": "Partner Manager",
        "site": "ExampleCo",
        "location": "Remote",
        "full_description": "Build partner programs.",
        "application_url": "https://apply.example.com/1",
        "fit_score": 8,
        "tailored_resume_path": "/tmp/resume.txt",
        "cover_letter_path": "/tmp/cover.txt",
    }
    values.update(overrides)
    columns = ", ".join(values)
    placeholders = ", ".join("?" for _ in values)
    conn.execute(
        f"INSERT INTO jobs ({columns}) VALUES ({placeholders})",
        list(values.values()),
    )
    conn.commit()


def test_jobs_list_show_add_and_ids(monkeypatch, tmp_path):
    app_dir = _configure_app_dir(monkeypatch, tmp_path)

    from applypilot.cli import app
    from applypilot.database import get_connection, init_db

    conn = init_db(app_dir / "applypilot.db")
    _insert_job(conn)

    runner = CliRunner()
    result = runner.invoke(app, ["jobs", "list", "pending-apply", "--format", "json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data[0]["url"] == "https://example.com/jobs/1"

    result = runner.invoke(app, ["jobs", "show", "apply.example.com/1", "--format", "json"])
    assert result.exit_code == 0
    assert json.loads(result.output)["application_url"] == "https://apply.example.com/1"

    result = runner.invoke(
        app,
        ["jobs", "add", "https://example.com/jobs/2", "--title", "Engineer", "--site", "ExampleCo"],
    )
    assert result.exit_code == 0
    assert get_connection().execute(
        "SELECT title FROM jobs WHERE url = ?",
        ("https://example.com/jobs/2",),
    ).fetchone()[0] == "Engineer"

    result = runner.invoke(app, ["jobs", "ids", "pending-apply"])
    assert result.exit_code == 0
    assert "https://example.com/jobs/1" in result.output


def test_jobs_reset_score_and_remove(monkeypatch, tmp_path):
    app_dir = _configure_app_dir(monkeypatch, tmp_path)

    from applypilot.cli import app
    from applypilot.database import get_connection, init_db

    conn = init_db(app_dir / "applypilot.db")
    _insert_job(conn, score_reasoning="old score")

    runner = CliRunner()
    result = runner.invoke(app, ["jobs", "reset", "score", "https://example.com/jobs/1", "--yes"])
    assert result.exit_code == 0
    row = get_connection().execute(
        "SELECT fit_score, score_reasoning, scored_at FROM jobs WHERE url = ?",
        ("https://example.com/jobs/1",),
    ).fetchone()
    assert row["fit_score"] is None
    assert row["score_reasoning"] is None
    assert row["scored_at"] is None

    result = runner.invoke(app, ["jobs", "remove", "https://example.com/jobs/1", "--yes"])
    assert result.exit_code == 0
    count = get_connection().execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    assert count == 0


def test_targeted_run_force_resets_only_selected_job(monkeypatch, tmp_path):
    app_dir = _configure_app_dir(monkeypatch, tmp_path)

    from applypilot.cli import app
    from applypilot.database import get_connection, init_db
    from applypilot import pipeline

    conn = init_db(app_dir / "applypilot.db")
    _insert_job(conn, url="https://example.com/jobs/1", score_reasoning="old")
    _insert_job(conn, url="https://example.com/jobs/2", score_reasoning="keep")

    monkeypatch.setitem(pipeline._STAGE_RUNNERS, "score", lambda **_: {"status": "ok"})

    runner = CliRunner()
    result = runner.invoke(app, ["run", "score", "--job-id", "https://example.com/jobs/1", "--force"])
    assert result.exit_code == 0

    rows = {
        row["url"]: row
        for row in get_connection().execute(
            "SELECT url, fit_score, score_reasoning FROM jobs ORDER BY url"
        ).fetchall()
    }
    assert rows["https://example.com/jobs/1"]["fit_score"] is None
    assert rows["https://example.com/jobs/1"]["score_reasoning"] is None
    assert rows["https://example.com/jobs/2"]["fit_score"] == 8
    assert rows["https://example.com/jobs/2"]["score_reasoning"] == "keep"


def test_manual_packet_mark_reset_and_legacy_apply_mark(monkeypatch, tmp_path):
    app_dir = _configure_app_dir(monkeypatch, tmp_path)

    from applypilot.cli import app
    from applypilot.database import get_connection, init_db

    conn = init_db(app_dir / "applypilot.db")
    _insert_job(conn)

    runner = CliRunner()
    result = runner.invoke(app, ["manual", "packet", "https://example.com/jobs/1"])
    assert result.exit_code == 0
    assert "https://apply.example.com/1" in result.output
    assert "/tmp/resume.txt" in result.output

    result = runner.invoke(
        app,
        ["manual", "mark", "https://example.com/jobs/1", "--status", "failed", "--reason", "manual-test"],
    )
    assert result.exit_code == 0
    row = get_connection().execute("SELECT apply_status, apply_error FROM jobs").fetchone()
    assert row["apply_status"] == "failed"
    assert row["apply_error"] == "manual-test"

    result = runner.invoke(app, ["manual", "reset", "--failed"])
    assert result.exit_code == 0
    row = get_connection().execute("SELECT apply_status, apply_error, apply_attempts FROM jobs").fetchone()
    assert row["apply_status"] is None
    assert row["apply_error"] is None
    assert row["apply_attempts"] == 0

    result = runner.invoke(app, ["apply", "--mark-applied", "apply.example.com/1"])
    assert result.exit_code == 0
    row = get_connection().execute("SELECT apply_status, applied_at FROM jobs").fetchone()
    assert row["apply_status"] == "applied"
    assert row["applied_at"] is not None


def test_apply_positional_ids_force_resets_selected_state(monkeypatch, tmp_path):
    app_dir = _configure_app_dir(monkeypatch, tmp_path)

    from applypilot.cli import app
    from applypilot.database import get_connection, init_db
    from applypilot.config import PROFILE_PATH
    from applypilot.apply import launcher

    conn = init_db(app_dir / "applypilot.db")
    _insert_job(conn, apply_status="failed", apply_error="old", apply_attempts=2)
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_PATH.write_text("{}", encoding="utf-8")

    monkeypatch.setattr("applypilot.config.check_tier", lambda *_: None)
    calls = {}

    def fake_apply_main(**kwargs):
        calls.update(kwargs)

    monkeypatch.setattr(launcher, "main", fake_apply_main)

    runner = CliRunner()
    result = runner.invoke(app, ["apply", "https://example.com/jobs/1", "--force"])
    assert result.exit_code == 0
    assert calls["target_urls"] == ["https://example.com/jobs/1"]

    row = get_connection().execute("SELECT apply_status, apply_error, apply_attempts FROM jobs").fetchone()
    assert row["apply_status"] is None
    assert row["apply_error"] is None
    assert row["apply_attempts"] == 0
