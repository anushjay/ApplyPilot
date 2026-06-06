import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _base_evidence(**overrides):
    evidence = {
        "role_family": "AI product partnerships",
        "dimension_scores": {
            "target_role_alignment": 9,
            "partnerships_alignment": 9,
            "product_platform_alignment": 9,
            "industry_domain_alignment": 8,
            "sales_motion_alignment": 7,
            "technical_domain_alignment": 8,
            "seniority_alignment": 9,
            "location_timezone_alignment": 8,
        },
        "positive_evidence": ["Led AI/cloud partnerships at New Relic and Splunk"],
        "hard_requirement_gaps": [],
        "dealbreaker_gaps": [],
        "matched_keywords": ["AI partnerships", "platform partnerships"],
        "recommended_score": 9,
        "confidence": "high",
    }
    evidence.update(overrides)
    return evidence


def test_professional_services_financial_sales_evidence_caps_score():
    from applypilot.scoring.scorer import _compute_score

    evidence = _base_evidence(
        role_family="professional services sales",
        hard_requirement_gaps=[
            {
                "requirement": "No evidence of selling professional services",
                "job_evidence_quote": "Experience selling professional services",
                "resume_gap": "Resume does not show professional-services sales",
            },
            {
                "requirement": "No financial services / banking domain experience",
                "job_evidence_quote": "Enterprise Financial Services",
                "resume_gap": "Resume does not show banking domain ownership",
            },
            {
                "requirement": "No full quota-carrying sales lifecycle evidence",
                "job_evidence_quote": "full sales lifecycle",
                "resume_gap": "Resume shows partner revenue but not full sales lifecycle",
            },
            {
                "requirement": "Role prefers EST/CST while candidate is Bay Area",
                "job_evidence_quote": "Eastern or Central timezone preferred",
                "resume_gap": "Candidate location is Bay Area",
            },
        ],
        matched_keywords=["partnerships", "SaaS", "GTM"],
        recommended_score=9,
    )

    score, keywords, reasoning = _compute_score(evidence)

    assert score == 5
    assert keywords == "partnerships, SaaS, GTM"
    assert "Final score capped at 5" in reasoning


def test_dealbreaker_gap_caps_score_below_five():
    from applypilot.scoring.scorer import _compute_score

    evidence = _base_evidence(
        role_family="semiconductor foundry business development",
        dealbreaker_gaps=[
            {
                "requirement": "Role depends on semiconductor foundry expertise absent from resume",
                "job_evidence_quote": "semiconductor foundry services",
                "resume_gap": "Resume does not show foundry services experience",
            }
        ],
        recommended_score=8,
    )

    score, _, reasoning = _compute_score(evidence)

    assert score == 4
    assert "Final score capped at 4" in reasoning


def test_strong_ai_partnership_evidence_scores_high():
    from applypilot.scoring.scorer import _compute_score

    score, keywords, reasoning = _compute_score(_base_evidence())

    assert score == 9
    assert keywords == "AI partnerships, platform partnerships"
    assert "Final score capped" not in reasoning


def test_unsupported_string_gaps_do_not_cap_score():
    from applypilot.scoring.scorer import _compute_score

    evidence = _base_evidence(
        hard_requirement_gaps=[
            "Financial services domain required",
            "Full sales lifecycle required",
            "Owning a book of business required",
        ],
        recommended_score=9,
    )

    score, _, reasoning = _compute_score(evidence)

    assert score == 9
    assert "Final score capped" not in reasoning


def test_gap_with_unrelated_quote_does_not_cap_score():
    from applypilot.scoring.scorer import _compute_score

    evidence = _base_evidence(
        hard_requirement_gaps=[
            {
                "requirement": "Financial services domain required",
                "job_evidence_quote": "Strong understanding of MSP and GSI business models",
                "resume_gap": "Resume does not show financial services domain experience",
            }
        ],
        recommended_score=9,
    )

    score, _, reasoning = _compute_score(evidence)

    assert score == 9
    assert "Final score capped" not in reasoning


def test_emea_region_gap_caps_otherwise_strong_role_at_seven():
    from applypilot.scoring.scorer import _compute_score

    evidence = _base_evidence(
        hard_requirement_gaps=[
            {
                "requirement": "EMEA or Europe timezone required",
                "job_evidence_quote": "We are open to hiring for this position in wider EMEA region.",
                "resume_gap": "Candidate is based in the San Francisco Bay Area.",
            }
        ],
        recommended_score=9,
    )

    score, _, reasoning = _compute_score(evidence)

    assert score == 7
    assert "Final score capped at 7" in reasoning


def test_low_location_timezone_dimension_caps_score_at_seven():
    from applypilot.scoring.scorer import _compute_score

    evidence = _base_evidence(
        dimension_scores={
            "target_role_alignment": 9,
            "partnerships_alignment": 9,
            "product_platform_alignment": 8,
            "industry_domain_alignment": 9,
            "sales_motion_alignment": 7,
            "technical_domain_alignment": 9,
            "seniority_alignment": 9,
            "location_timezone_alignment": 3,
        },
        recommended_score=9,
    )

    score, _, reasoning = _compute_score(evidence)

    assert score == 7
    assert "location/timezone mismatch" in reasoning


def test_extract_json_allows_markdown_preface():
    from applypilot.scoring.scorer import _extract_json

    parsed = _extract_json('```json\n{"role_family": "partnerships"}\n```')

    assert parsed == {"role_family": "partnerships"}


def test_extract_json_allows_raw_control_characters():
    from applypilot.scoring.scorer import _extract_json

    parsed = _extract_json('{"role_family": "partner\nmarketing"}')

    assert parsed == {"role_family": "partner\nmarketing"}


def test_unparseable_score_response_raises_instead_of_zero():
    from applypilot.scoring.scorer import ScoreParseError, _score_from_response

    try:
        _score_from_response("Okay", {"title": "Partnerships Manager"})
    except ScoreParseError:
        pass
    else:
        raise AssertionError("Expected ScoreParseError")


def test_compact_preferences_remove_verbose_score_caps():
    from applypilot.scoring.scorer import _compact_scoring_preferences

    compact = _compact_scoring_preferences(
        {
            "target_roles": ["technology partnerships"],
            "deprioritize": ["product design"],
            "watch_for_hard_gaps": ["JD required"],
            "dealbreakers": ["base pay below floor"],
            "score_caps": [{"reason": "primary product-design function", "contains": ["product designer"]}],
            "salary_caps": [{"floor": 200000, "ceiling": 4}],
        }
    )

    assert compact["target_roles"] == ["technology partnerships"]
    assert compact["minimum_base_salary"] == 200000
    assert compact["configured_cap_themes"] == ["primary product-design function"]
    assert "score_caps" not in compact
    assert "salary_caps" not in compact


def test_score_job_retries_with_compact_prompt_after_parse_failure(monkeypatch):
    from applypilot.scoring import scorer

    class FakeClient:
        def __init__(self):
            self.calls = 0

        def chat(self, messages, max_tokens=1200, temperature=0.1):
            self.calls += 1
            if self.calls == 1:
                return "Okay"
            assert "COMPACT SCORING PREFERENCES" in messages[1]["content"]
            return """
            {
              "role_family": "technology partnerships",
              "dimension_scores": {
                "target_role_alignment": 8,
                "partnerships_alignment": 8,
                "product_platform_alignment": 7,
                "industry_domain_alignment": 7,
                "sales_motion_alignment": 6,
                "technical_domain_alignment": 8,
                "seniority_alignment": 8,
                "location_timezone_alignment": 8
              },
              "positive_evidence": ["partnership evidence"],
              "hard_requirement_gaps": [],
              "dealbreaker_gaps": [],
              "matched_keywords": ["partnerships"],
              "recommended_score": 8,
              "confidence": "medium"
            }
            """

    fake_client = FakeClient()
    monkeypatch.setattr(scorer, "get_client", lambda: fake_client)
    monkeypatch.setattr(
        scorer.config,
        "load_search_config",
        lambda: {"scoring_preferences": {"target_roles": ["technology partnerships"]}},
    )

    result = scorer.score_job(
        "Resume with partnerships experience",
        {
            "title": "Senior Director, Business Development (GSI)",
            "site": "linkedin",
            "location": "Remote",
            "full_description": "GSI partnership role",
        },
    )

    assert result["score"] == 8
    assert fake_client.calls == 2
    assert "compact retry" in result["reasoning"]


def test_score_job_returns_error_when_full_and_compact_parse_fail(monkeypatch):
    from applypilot.scoring import scorer

    class FakeClient:
        def chat(self, messages, max_tokens=1200, temperature=0.1):
            return "Okay"

    monkeypatch.setattr(scorer, "get_client", lambda: FakeClient())
    monkeypatch.setattr(scorer.config, "load_search_config", lambda: {})

    result = scorer.score_job(
        "Resume",
        {
            "title": "Bad model output",
            "site": "indeed",
            "location": "Remote",
            "full_description": "Description",
        },
    )

    assert result["score"] is None
    assert result["error"] is True


def test_configured_score_cap_matches_title(monkeypatch):
    from applypilot.scoring import scorer

    monkeypatch.setattr(
        scorer.config,
        "load_search_config",
        lambda: {
            "scoring_preferences": {
                "score_caps": [
                    {
                        "field": "title",
                        "contains": ["partner marketing"],
                        "ceiling": 5,
                        "reason": "primary partner marketing function",
                    }
                ]
            }
        },
    )

    cap = scorer._configured_score_cap({"title": "Director, Enterprise Partner Marketing"})

    assert cap == (5, "primary partner marketing function")


def test_configured_score_cap_matches_product_design_title(monkeypatch):
    from applypilot.scoring import scorer

    monkeypatch.setattr(
        scorer.config,
        "load_search_config",
        lambda: {
            "scoring_preferences": {
                "score_caps": [
                    {
                        "field": "title",
                        "contains": ["product designer"],
                        "ceiling": 5,
                        "reason": "primary product-design function",
                    }
                ]
            }
        },
    )

    cap = scorer._configured_score_cap({"title": "Product Designer, Marketplace"})

    assert cap == (5, "primary product-design function")


def test_configured_salary_cap_when_advertised_range_below_floor(monkeypatch):
    from applypilot.scoring import scorer

    monkeypatch.setattr(
        scorer.config,
        "load_search_config",
        lambda: {
            "scoring_preferences": {
                "salary_caps": [
                    {
                        "floor": 200000,
                        "ceiling": 4,
                        "reason": "advertised base pay below $200,000 floor",
                    }
                ]
            }
        },
    )

    cap = scorer._configured_score_cap({"salary": "USD123,372-USD161,925/yearly"})

    assert cap == (4, "advertised base pay below $200,000 floor ($123,372, $161,925)")


def test_configured_salary_cap_does_not_match_when_range_reaches_floor(monkeypatch):
    from applypilot.scoring import scorer

    monkeypatch.setattr(
        scorer.config,
        "load_search_config",
        lambda: {
            "scoring_preferences": {
                "salary_caps": [
                    {
                        "floor": 200000,
                        "ceiling": 4,
                        "reason": "advertised base pay below $200,000 floor",
                    }
                ]
            }
        },
    )

    cap = scorer._configured_score_cap({"salary": "USD203,000-USD274,500/yearly"})

    assert cap is None


def test_configured_salary_cap_ignores_unrelated_description_dollar_amount(monkeypatch):
    from applypilot.scoring import scorer

    monkeypatch.setattr(
        scorer.config,
        "load_search_config",
        lambda: {
            "scoring_preferences": {
                "salary_caps": [
                    {
                        "floor": 200000,
                        "ceiling": 4,
                        "reason": "advertised base pay below $200,000 floor",
                    }
                ]
            }
        },
    )

    cap = scorer._configured_score_cap({"full_description": "Managed a $5,000 pilot budget for experiments."})

    assert cap is None


def test_configured_salary_cap_reads_compensation_context_in_description(monkeypatch):
    from applypilot.scoring import scorer

    monkeypatch.setattr(
        scorer.config,
        "load_search_config",
        lambda: {
            "scoring_preferences": {
                "salary_caps": [
                    {
                        "floor": 200000,
                        "ceiling": 4,
                        "reason": "advertised base pay below $200,000 floor",
                    }
                ]
            }
        },
    )

    cap = scorer._configured_score_cap({"full_description": "The base salary range is $130,000 - $140,000 per year."})

    assert cap == (4, "advertised base pay below $200,000 floor ($130,000, $140,000)")


def test_apply_configured_cap_lowers_score_and_explains_reason(monkeypatch):
    from applypilot.scoring import scorer

    monkeypatch.setattr(
        scorer.config,
        "load_search_config",
        lambda: {
            "scoring_preferences": {
                "score_caps": [
                    {
                        "field": "title",
                        "contains": ["product designer"],
                        "ceiling": 5,
                        "reason": "primary product-design function",
                    }
                ],
                "salary_caps": [
                    {
                        "floor": 200000,
                        "ceiling": 4,
                        "reason": "advertised base pay below $200,000 floor",
                    }
                ],
            }
        },
    )

    score, reasoning = scorer._apply_configured_cap(
        8,
        "Role family: Product Partnerships.",
        {"title": "Product Designer, Marketplace", "salary": "USD123,372-USD161,925/yearly"},
    )

    assert score == 4
    assert "advertised base pay below $200,000 floor" in reasoning


def test_merge_preferences_maps_legacy_hard_gaps_key():
    from applypilot.scoring.scorer import _default_scoring_preferences, _merge_preferences

    prefs = _merge_preferences(_default_scoring_preferences(), {"hard_gaps": ["timezone required"]})

    assert prefs["watch_for_hard_gaps"] == ["timezone required"]


def test_scoring_config_provides_prompt_preferences_weights_and_ceiling():
    from applypilot.scoring.scorer import (
        _default_scoring_preferences,
        _dimension_weights,
        _score_ceiling_config,
        _scoring_prompt,
    )

    assert "job fit evidence evaluator" in _scoring_prompt()
    assert "AI partnerships" in _default_scoring_preferences()["target_roles"]
    assert _default_scoring_preferences()["dealbreakers"] == []
    assert _dimension_weights()["partnerships_alignment"] == 0.20
    assert _score_ceiling_config()["location_timezone"]["mismatch_ceiling"] == 7


def _insert_job(conn, url="https://example.test/job", title="Director, AI Partnerships", **overrides):
    values = {
        "url": url,
        "title": title,
        "site": "example",
        "location": "Remote",
        "full_description": "Lead AI and technology partnerships.",
        "discovered_at": "2026-06-05T00:00:00+00:00",
    }
    values.update(overrides)
    conn.execute(
        """
        INSERT INTO jobs (
            url, title, site, location, full_description, discovered_at,
            fit_score, score_reasoning, local_fit_score, local_score_reasoning,
            cloud_validation_status
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            values.get("url"),
            values.get("title"),
            values.get("site"),
            values.get("location"),
            values.get("full_description"),
            values.get("discovered_at"),
            values.get("fit_score"),
            values.get("score_reasoning"),
            values.get("local_fit_score"),
            values.get("local_score_reasoning"),
            values.get("cloud_validation_status"),
        ),
    )
    conn.commit()
    return values


def test_schema_migration_adds_hybrid_score_columns(tmp_path):
    import sqlite3

    from applypilot.database import ensure_columns

    conn = sqlite3.connect(tmp_path / "legacy.sqlite")
    conn.execute("CREATE TABLE jobs (url TEXT PRIMARY KEY, title TEXT)")

    added = ensure_columns(conn)

    columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()}
    assert "local_fit_score" in columns
    assert "cloud_validation_status" in columns
    assert "local_fit_score" in added


def test_run_local_scoring_writes_local_and_final_score(monkeypatch, tmp_path):
    from applypilot.database import init_db
    from applypilot.scoring import scorer

    conn = init_db(tmp_path / "jobs.sqlite")
    _insert_job(conn)
    resume = tmp_path / "resume.txt"
    resume.write_text("Partnerships resume", encoding="utf-8")

    monkeypatch.setattr(scorer, "RESUME_PATH", resume)
    monkeypatch.setattr(scorer, "get_connection", lambda: conn)
    monkeypatch.setattr(scorer, "get_client", lambda: object())
    monkeypatch.setattr(
        scorer,
        "score_job",
        lambda resume_text, job, **kwargs: {"score": 8, "keywords": "AI partnerships", "reasoning": "Strong fit."},
    )

    result = scorer.run_local_scoring()

    row = conn.execute("SELECT * FROM jobs").fetchone()
    assert result["scored"] == 1
    assert row["local_fit_score"] == 8
    assert row["fit_score"] == 8
    assert row["local_score_error"] is None
    assert "Strong fit." in row["score_reasoning"]


def test_run_local_scoring_parse_failure_keeps_final_score_null(monkeypatch, tmp_path):
    from applypilot.database import init_db
    from applypilot.scoring import scorer

    conn = init_db(tmp_path / "jobs.sqlite")
    _insert_job(conn)
    resume = tmp_path / "resume.txt"
    resume.write_text("Partnerships resume", encoding="utf-8")

    monkeypatch.setattr(scorer, "RESUME_PATH", resume)
    monkeypatch.setattr(scorer, "get_connection", lambda: conn)
    monkeypatch.setattr(scorer, "get_client", lambda: object())
    monkeypatch.setattr(
        scorer,
        "score_job",
        lambda resume_text, job, **kwargs: {"score": None, "reasoning": "Score parse error: Okay", "error": True},
    )

    result = scorer.run_local_scoring()

    row = conn.execute("SELECT * FROM jobs").fetchone()
    assert result["errors"] == 1
    assert row["local_score_error"] == "Score parse error: Okay"
    assert row["local_fit_score"] is None
    assert row["fit_score"] is None


def test_run_cloud_validation_lowers_final_score_when_cloud_is_stricter(monkeypatch, tmp_path):
    from applypilot.database import init_db
    from applypilot.scoring import scorer

    conn = init_db(tmp_path / "jobs.sqlite")
    _insert_job(
        conn,
        fit_score=9,
        score_reasoning="Local keywords\nLocal reasoning.",
        local_fit_score=9,
        local_score_reasoning="Local keywords\nLocal reasoning.",
    )
    resume = tmp_path / "resume.txt"
    resume.write_text("Partnerships resume", encoding="utf-8")

    monkeypatch.setattr(scorer, "RESUME_PATH", resume)
    monkeypatch.setattr(scorer, "get_connection", lambda: conn)
    monkeypatch.setattr(scorer, "get_cloud_client", lambda: object())
    monkeypatch.setattr(
        scorer,
        "score_job",
        lambda resume_text, job, **kwargs: {"score": 6, "keywords": "GSI", "reasoning": "Cloud found a timezone gap."},
    )

    result = scorer.run_cloud_validation(min_local_score=8)

    row = conn.execute("SELECT * FROM jobs").fetchone()
    assert result["validated"] == 1
    assert result["lowered"] == 1
    assert row["cloud_fit_score"] == 6
    assert row["fit_score"] == 6
    assert row["cloud_validation_status"] == "validated"
    assert "Cloud validation lowered final score from 9 to 6" in row["score_reasoning"]


def test_cloud_validation_threshold_reads_env(monkeypatch):
    from applypilot.scoring.scorer import _cloud_validation_min_local_score

    monkeypatch.setenv("APPLYPILOT_CLOUD_VALIDATION_MIN_LOCAL_SCORE", "9")

    assert _cloud_validation_min_local_score() == 9


def test_cloud_validation_threshold_rejects_invalid_env(monkeypatch):
    import pytest

    from applypilot.scoring.scorer import _cloud_validation_min_local_score

    monkeypatch.setenv("APPLYPILOT_CLOUD_VALIDATION_MIN_LOCAL_SCORE", "11")

    with pytest.raises(RuntimeError, match="APPLYPILOT_CLOUD_VALIDATION_MIN_LOCAL_SCORE"):
        _cloud_validation_min_local_score()


def test_run_cloud_validation_keeps_local_score_when_cloud_is_higher(monkeypatch, tmp_path):
    from applypilot.database import init_db
    from applypilot.scoring import scorer

    conn = init_db(tmp_path / "jobs.sqlite")
    _insert_job(conn, fit_score=8, score_reasoning="Local.", local_fit_score=8, local_score_reasoning="Local.")
    resume = tmp_path / "resume.txt"
    resume.write_text("Partnerships resume", encoding="utf-8")

    monkeypatch.setattr(scorer, "RESUME_PATH", resume)
    monkeypatch.setattr(scorer, "get_connection", lambda: conn)
    monkeypatch.setattr(scorer, "get_cloud_client", lambda: object())
    monkeypatch.setattr(
        scorer,
        "score_job",
        lambda resume_text, job, **kwargs: {"score": 9, "keywords": "AI", "reasoning": "Cloud agrees."},
    )

    scorer.run_cloud_validation(min_local_score=8)

    row = conn.execute("SELECT * FROM jobs").fetchone()
    assert row["cloud_fit_score"] == 9
    assert row["fit_score"] == 8
    assert "Cloud validation kept final score at 8" in row["score_reasoning"]


def test_run_cloud_validation_failure_preserves_local_final_score(monkeypatch, tmp_path):
    from applypilot.database import init_db
    from applypilot.scoring import scorer

    conn = init_db(tmp_path / "jobs.sqlite")
    _insert_job(conn, fit_score=8, score_reasoning="Local.", local_fit_score=8, local_score_reasoning="Local.")
    resume = tmp_path / "resume.txt"
    resume.write_text("Partnerships resume", encoding="utf-8")

    monkeypatch.setattr(scorer, "RESUME_PATH", resume)
    monkeypatch.setattr(scorer, "get_connection", lambda: conn)
    monkeypatch.setattr(scorer, "get_cloud_client", lambda: object())
    monkeypatch.setattr(
        scorer,
        "score_job",
        lambda resume_text, job, **kwargs: {"score": None, "reasoning": "Cloud parse failed", "error": True},
    )

    result = scorer.run_cloud_validation(min_local_score=8)

    row = conn.execute("SELECT * FROM jobs").fetchone()
    assert result["failed"] == 1
    assert row["fit_score"] == 8
    assert row["score_reasoning"] == "Local."
    assert row["cloud_validation_status"] == "failed"
    assert row["cloud_validation_error"] == "Cloud parse failed"
