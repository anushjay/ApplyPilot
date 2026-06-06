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


def test_cloud_validation_client_uses_explicit_cloud_provider(monkeypatch):
    from applypilot import llm

    monkeypatch.setenv("APPLYPILOT_ALLOW_CLOUD_LLM", "1")
    monkeypatch.setenv("CLOUD_LLM_PROVIDER", "openai")
    monkeypatch.setenv("CLOUD_LLM_MODEL", "gpt-test")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret-key")
    monkeypatch.setenv("LLM_URL", "http://localhost:11434/v1")
    llm._cloud_instance = None

    client = llm.get_cloud_client()
    try:
        assert client.base_url == "https://api.openai.com/v1"
        assert client.model == "gpt-test"
        assert client.api_key == "openai-secret-key"
    finally:
        client.close()
        llm._cloud_instance = None


def test_resume_validator_accepts_preserved_company_in_subtitle():
    from applypilot.scoring.validator import validate_json_fields

    profile = {
        "resume_facts": {
            "preserved_companies": ["New Relic", "Splunk"],
            "preserved_school": "New Jersey Institute of Technology",
        }
    }
    data = {
        "title": "Senior Partner Manager, AWS",
        "summary": "Partnerships leader with cloud GTM experience.",
        "skills": {"Tools": "AWS, Azure"},
        "experience": [
            {
                "header": "Director, ISV and Cloud Partnerships",
                "subtitle": "New Relic | 2025-Present",
                "bullets": ["Led cloud partnerships."],
            },
            {
                "header": "Director, Technology Partnerships",
                "subtitle": "Splunk | 2021-2025",
                "bullets": ["Scaled partner programs."],
            },
        ],
        "projects": [
            {
                "header": "Partner Enablement Programs",
                "subtitle": "AWS | 2021-2025",
                "bullets": ["Built partner enablement programs."],
            }
        ],
        "education": "New Jersey Institute of Technology | Master's Degree",
    }

    result = validate_json_fields(data, profile)

    assert result["passed"] is True


def test_resume_assembler_uses_profile_title_not_llm_title():
    from applypilot.scoring.tailor import assemble_resume_text

    profile = {
        "personal": {"full_name": "Test Candidate", "email": "test@example.com"},
        "experience": {"current_job_title": "Director, Technology Partnerships"},
    }
    data = {
        "title": "Senior Partner Manager",
        "summary": "Partnerships leader.",
        "skills": {"Tools": "AWS"},
        "experience": [
            {
                "header": "Director, Technology Partnerships",
                "subtitle": "Splunk | 2021-2025",
                "bullets": ["Led partnerships."],
            }
        ],
        "projects": [
            {
                "header": "Partner Program",
                "subtitle": "AWS | 2021-2025",
                "bullets": ["Built enablement."],
            }
        ],
        "education": "School | Degree",
    }

    rendered = assemble_resume_text(data, profile).splitlines()

    assert rendered[1] == "Director, Technology Partnerships"
    assert "Senior Partner Manager" not in rendered[:3]


def test_resume_assembler_places_each_education_item_on_own_line():
    from applypilot.scoring.tailor import assemble_resume_text

    profile = {
        "personal": {"full_name": "Test Candidate", "email": "test@example.com"},
        "experience": {"current_job_title": "Director, Technology Partnerships"},
    }
    data = {
        "title": "Senior Partner Manager",
        "summary": "Partnerships leader.",
        "skills": {"Tools": "AWS"},
        "experience": [],
        "projects": [],
        "education": [
            "New Jersey Institute of Technology | Master's Degree",
            "Anna University | Bachelor's Degree",
        ],
    }

    rendered = assemble_resume_text(data, profile).splitlines()
    education_index = rendered.index("EDUCATION")

    assert rendered[education_index + 1] == "New Jersey Institute of Technology | Master's Degree"
    assert rendered[education_index + 2] == "Anna University | Bachelor's Degree"


def test_resume_pdf_renders_education_lines_separately():
    from applypilot.scoring.pdf import build_html, parse_resume

    resume = parse_resume(
        """Test Candidate
Director, Technology Partnerships
test@example.com

SUMMARY
Partnerships leader.

EDUCATION
New Jersey Institute of Technology | Master's Degree
Anna University | Bachelor's Degree
"""
    )

    html = build_html(resume)

    assert html.count('class="edu-line"') == 2
    assert "New Jersey Institute of Technology | Master&#x27;s Degree" in html
    assert "Anna University | Bachelor&#x27;s Degree" in html


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


def test_prompt_can_upload_default_resume_pdf(tmp_path, monkeypatch):
    from applypilot import config
    from applypilot.apply import prompt

    app_dir = tmp_path / "app"
    app_dir.mkdir(parents=True)
    default_pdf = app_dir / "resume.pdf"
    default_pdf.write_bytes(b"%PDF-1.4 default\n")
    profile_path = app_dir / "profile.json"
    search_path = app_dir / "searches.yaml"
    profile_path.write_text(json.dumps(_profile()), encoding="utf-8")
    search_path.write_text("location: {}\n", encoding="utf-8")

    monkeypatch.setattr(config, "APP_DIR", app_dir)
    monkeypatch.setattr(config, "PROFILE_PATH", profile_path)
    monkeypatch.setattr(config, "SEARCH_CONFIG_PATH", search_path)
    monkeypatch.setattr(config, "RESUME_PDF_PATH", default_pdf)
    monkeypatch.setattr(config, "APPLY_WORKER_DIR", app_dir / "apply-workers")

    built = prompt.build_prompt(
        {
            "url": "https://example.com/job",
            "title": "Software Engineer",
            "site": "Example",
            "fit_score": 8,
            "application_url": "https://example.com/apply",
        },
        tailored_resume="Default resume text",
        dry_run=True,
        resume_mode="default",
    )

    upload_pdf = app_dir / "apply-workers" / "current" / "Test_Candidate_Resume.pdf"
    assert upload_pdf.exists()
    assert upload_pdf.read_bytes() == b"%PDF-1.4 default\n"
    assert f"Resume PDF (upload this): {upload_pdf}" in built
    assert "Default resume text" in built


def test_apply_default_resume_mode_does_not_require_tailored_resume(tmp_path, monkeypatch):
    from applypilot import config
    from applypilot import database
    from applypilot.apply.launcher import acquire_job
    from applypilot.database import init_db

    db_path = tmp_path / "applypilot.db"
    monkeypatch.setattr(config, "DB_PATH", db_path)
    monkeypatch.setattr(database, "DB_PATH", db_path)
    conn = init_db(db_path)
    conn.execute(
        """
        INSERT INTO jobs (
            url, title, site, fit_score, full_description
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (
            "https://example.com/job",
            "Partner Manager",
            "Example",
            8,
            "Partnerships role",
        ),
    )
    conn.commit()

    tailored_job = acquire_job(target_url="https://example.com/job", resume_mode="tailored")
    default_job = acquire_job(target_url="https://example.com/job", resume_mode="default")

    assert tailored_job is None
    assert default_job is not None
    assert default_job["title"] == "Partner Manager"


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
