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
