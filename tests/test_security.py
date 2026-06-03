import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _profile(password: str = "job-site-secret") -> dict:
    return {
        "personal": {
            "full_name": "Test Candidate",
            "preferred_name": "Test",
            "email": "test@example.com",
            "password": password,
            "phone": "555-1212",
            "address": "1 Main St",
            "city": "San Francisco",
            "province_state": "CA",
            "country": "USA",
            "postal_code": "94105",
            "linkedin_url": "",
            "github_url": "",
            "portfolio_url": "",
            "website_url": "",
        },
        "work_authorization": {
            "legally_authorized_to_work": "Yes",
            "require_sponsorship": "No",
            "work_permit_type": "",
        },
        "availability": {"earliest_start_date": "Immediately"},
        "compensation": {
            "salary_expectation": "120000",
            "salary_currency": "USD",
            "salary_range_min": "110000",
            "salary_range_max": "130000",
        },
        "experience": {
            "years_of_experience_total": "5",
            "education_level": "BS",
            "target_role": "Software Engineer",
        },
        "skills_boundary": {"languages": ["Python"]},
        "resume_facts": {},
        "eeo_voluntary": {},
    }


def test_cloud_llm_fails_closed_without_explicit_opt_in(monkeypatch):
    from applypilot import llm

    monkeypatch.setenv("APPLYPILOT_PRIVACY_MODE", "strict")
    monkeypatch.delenv("APPLYPILOT_ALLOW_CLOUD_LLM", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret-key")
    monkeypatch.delenv("LLM_URL", raising=False)
    llm._instance = None

    with pytest.raises(RuntimeError, match="APPLYPILOT_ALLOW_CLOUD_LLM"):
        llm.get_client()


def test_mcp_config_is_pinned_and_gmail_absent_by_default(monkeypatch):
    from applypilot.apply.launcher import _make_mcp_config

    monkeypatch.delenv("APPLYPILOT_ENABLE_GMAIL_MCP", raising=False)
    mcp = _make_mcp_config(9222)

    assert "gmail" not in mcp["mcpServers"]
    args = mcp["mcpServers"]["playwright"]["args"]
    assert any(arg.startswith("@playwright/mcp@") for arg in args)
    assert all("@latest" not in arg for arg in args)


def test_gmail_mcp_is_opt_in_and_pinned(monkeypatch):
    from applypilot.apply.launcher import _make_mcp_config

    monkeypatch.setenv("APPLYPILOT_ENABLE_GMAIL_MCP", "1")
    monkeypatch.setenv("APPLYPILOT_ALLOW_VULNERABLE_GMAIL_MCP", "1")
    mcp = _make_mcp_config(9222)

    gmail_args = mcp["mcpServers"]["gmail"]["args"]
    assert any(arg.startswith("@gongrzhe/server-gmail-autoauth-mcp@") for arg in gmail_args)
    assert all("@latest" not in arg for arg in gmail_args)


def test_gmail_mcp_request_is_blocked_without_vulnerability_ack(monkeypatch):
    from applypilot.apply.launcher import _make_mcp_config

    monkeypatch.setenv("APPLYPILOT_ENABLE_GMAIL_MCP", "1")
    monkeypatch.delenv("APPLYPILOT_ALLOW_VULNERABLE_GMAIL_MCP", raising=False)

    assert "gmail" not in _make_mcp_config(9222)["mcpServers"]


def test_prompt_does_not_include_password_or_api_keys(tmp_path, monkeypatch):
    from applypilot import config
    from applypilot.apply import prompt

    app_dir = tmp_path / "app"
    tailored = app_dir / "tailored" / "resume.txt"
    tailored.parent.mkdir(parents=True)
    tailored.write_text("resume text", encoding="utf-8")
    tailored.with_suffix(".pdf").write_bytes(b"%PDF-1.4\n")

    profile = _profile(password="job-site-secret")
    profile_path = app_dir / "profile.json"
    search_path = app_dir / "searches.yaml"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    search_path.write_text("location: {}\n", encoding="utf-8")

    monkeypatch.setattr(config, "APP_DIR", app_dir)
    monkeypatch.setattr(config, "PROFILE_PATH", profile_path)
    monkeypatch.setattr(config, "SEARCH_CONFIG_PATH", search_path)
    monkeypatch.setattr(config, "APPLY_WORKER_DIR", app_dir / "apply-workers")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret-key")
    monkeypatch.setenv("CAPSOLVER_API_KEY", "capsolver-secret-key")
    monkeypatch.delenv("APPLYPILOT_ENABLE_CAPSOLVER", raising=False)

    built = prompt.build_prompt(
        {
            "url": "https://example.com/job",
            "title": "Software Engineer",
            "site": "Example",
            "fit_score": 8,
            "tailored_resume_path": str(tailored),
            "application_url": "https://example.com/apply",
            "cover_letter_path": "",
        },
        tailored_resume="Tailored resume text",
        dry_run=True,
    )

    assert "job-site-secret" not in built
    assert "gemini-secret-key" not in built
    assert "openai-secret-key" not in built
    assert "capsolver-secret-key" not in built
    assert "CapSolver is disabled" in built


def test_clean_chrome_profile_is_default(tmp_path, monkeypatch):
    from applypilot import config
    from applypilot.apply import chrome

    monkeypatch.setattr(config, "CHROME_WORKER_DIR", tmp_path / "chrome-workers")
    monkeypatch.delenv("APPLYPILOT_CLONE_CHROME_PROFILE", raising=False)

    def fail_if_called():
        raise AssertionError("real Chrome profile should not be read by default")

    monkeypatch.setattr(config, "get_chrome_user_data", fail_if_called)

    profile_dir = chrome.setup_worker_profile(0)
    assert profile_dir == tmp_path / "chrome-workers" / "worker-0"
    assert (profile_dir / "Default").is_dir()
