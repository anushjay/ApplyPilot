"""Job fit scoring: LLM-powered evaluation of candidate-job match quality.

Scores jobs on a 1-10 scale by comparing the user's resume against each
job description. All personal data is loaded at runtime from the user's
profile and resume file.
"""

import logging
import json
import re
import time
from datetime import datetime, timezone
from typing import Any

from applypilot import config
from applypilot.config import RESUME_PATH
from applypilot.database import get_connection, get_jobs_by_stage
from applypilot.llm import get_client

log = logging.getLogger(__name__)


def _parse_score_response(response: str) -> dict:
    """Parse the LLM's score response into structured data.

    Args:
        response: Raw LLM response text.

    Returns:
        {"score": int, "keywords": str, "reasoning": str}
    """
    score = 0
    keywords = ""
    reasoning = response

    for line in response.split("\n"):
        line = line.strip()
        if line.startswith("SCORE:"):
            try:
                score = int(re.search(r"\d+", line).group())
                score = max(1, min(10, score))
            except (AttributeError, ValueError):
                score = 0
        elif line.startswith("KEYWORDS:"):
            keywords = line.replace("KEYWORDS:", "").strip()
        elif line.startswith("REASONING:"):
            reasoning = line.replace("REASONING:", "").strip()

    return {"score": score, "keywords": keywords, "reasoning": reasoning}


def _scoring_config_section(name: str, expected_type: type) -> Any:
    """Load and validate one section from packaged scoring.yaml."""
    value = config.load_scoring_config().get(name)
    if not isinstance(value, expected_type):
        raise RuntimeError(f"Invalid scoring config: missing or invalid '{name}'")
    return value


def _scoring_prompt() -> str:
    """Load the system prompt from packaged scoring.yaml."""
    return _scoring_config_section("prompt", str)


def _default_scoring_preferences() -> dict[str, Any]:
    """Load default scoring preferences from packaged scoring.yaml."""
    return _scoring_config_section("default_preferences", dict)


def _dimension_weights() -> dict[str, float]:
    """Load score dimension weights from packaged scoring.yaml."""
    raw = _scoring_config_section("dimension_weights", dict)
    weights: dict[str, float] = {}
    for name, weight in raw.items():
        try:
            weights[str(name)] = float(weight)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Invalid scoring config: weight for '{name}' must be numeric") from exc
    if not weights:
        raise RuntimeError("Invalid scoring config: dimension_weights cannot be empty")
    return weights


def _score_ceiling_config() -> dict[str, Any]:
    """Load score ceiling policy from packaged scoring.yaml."""
    return _scoring_config_section("score_ceiling", dict)


def _config_int(value: Any, default: int) -> int:
    """Read an integer config value with a fallback."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_dict(value: Any) -> dict[str, Any]:
    """Return a dict section or an empty dict for malformed optional config."""
    return value if isinstance(value, dict) else {}


def _text_for_cap_field(job: dict[str, Any], field: str) -> str:
    """Return the job text field used by a configured score cap rule."""
    if field == "title":
        return str(job.get("title") or "")
    if field == "description":
        return str(job.get("full_description") or job.get("description") or "")
    if field == "location":
        return str(job.get("location") or "")
    if field == "salary":
        return str(job.get("salary") or "")
    return " ".join(
        str(job.get(name) or "")
        for name in ("title", "site", "location", "salary", "description", "full_description")
    )


def _salary_amounts(text: str, require_compensation_context: bool = False) -> list[int]:
    """Extract plausible annual USD salary amounts from job text.

    Supports forms like ``USD123,000``, ``$123,000``, ``$123k``, and
    hourly rates such as ``$25 per hour`` converted to annualized base pay.
    """
    amounts: list[int] = []
    pattern = re.compile(
        r"(?i)(?:(usd\s*|\$)\s*([0-9][0-9,]*(?:\.\d+)?)\s*(k)?|([0-9]+(?:\.\d+)?)\s*k\b)"
    )
    for match in pattern.finditer(text or ""):
        raw = match.group(2) or match.group(4)
        if not raw:
            continue
        amount = float(raw.replace(",", ""))
        has_k_suffix = bool(match.group(3) or match.group(4))
        context = text[max(0, match.start() - 40):match.end() + 80].lower()
        if require_compensation_context and not any(
            term in context
            for term in (
                "salary",
                "base pay",
                "base salary",
                "compensation",
                "pay range",
                "annual",
                "yearly",
                "per year",
                "per hour",
                "hourly",
                "/hr",
            )
        ):
            continue
        if has_k_suffix:
            amount *= 1000
        elif amount < 1000 and any(term in context for term in ("hour", "hourly", "/hr", "per hr")):
            amount *= 2080
        elif amount < 1000:
            continue
        amounts.append(int(round(amount)))
    return amounts


def _configured_salary_cap(job: dict[str, Any]) -> tuple[int, str] | None:
    """Return a configured salary cap when advertised base pay is below floor."""
    prefs = _scoring_preferences()
    rules = prefs.get("salary_caps", [])
    if not isinstance(rules, list):
        return None

    salary_field = str(job.get("salary") or "").strip()
    if salary_field:
        amounts = _salary_amounts(salary_field)
    else:
        description = str(job.get("full_description") or job.get("description") or "")
        amounts = _salary_amounts(description, require_compensation_context=True)
    if not amounts:
        return None

    matches: list[tuple[int, str]] = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        floor = _config_int(rule.get("floor") or rule.get("min_base"), 0)
        if floor <= 0:
            continue
        mode = str(rule.get("mode") or "max_below_floor").strip().lower()
        if mode == "any_below_floor":
            below_floor = any(amount < floor for amount in amounts)
        else:
            below_floor = max(amounts) < floor
        if not below_floor:
            continue

        ceiling = max(1, min(10, _config_int(rule.get("ceiling"), 4)))
        amount_summary = ", ".join(f"${amount:,}" for amount in sorted(set(amounts)))
        reason = str(rule.get("reason") or f"advertised base pay below ${floor:,}").strip()
        matches.append((ceiling, f"{reason} ({amount_summary})"))

    if not matches:
        return None
    return min(matches, key=lambda item: item[0])


def _configured_score_cap(job: dict[str, Any]) -> tuple[int, str] | None:
    """Return the lowest matching user-configured score cap for a job.

    Example searches.yaml shape:

      scoring_preferences:
        score_caps:
          - field: title
            contains: ["partner marketing"]
            ceiling: 5
            reason: primary partner marketing function
    """
    prefs = _scoring_preferences()
    rules = prefs.get("score_caps", [])
    if not isinstance(rules, list):
        return None

    matches: list[tuple[int, str]] = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        patterns = rule.get("contains") or rule.get("patterns") or []
        if isinstance(patterns, str):
            patterns = [patterns]
        patterns = [str(pattern).strip().lower() for pattern in patterns if str(pattern).strip()]
        if not patterns:
            continue

        field = str(rule.get("field") or "all").strip().lower()
        text = _text_for_cap_field(job, field).lower()
        require_all = bool(rule.get("require_all", False))
        matched = all(pattern in text for pattern in patterns) if require_all else any(pattern in text for pattern in patterns)
        if not matched:
            continue

        ceiling = _config_int(rule.get("ceiling"), 10)
        ceiling = max(1, min(10, ceiling))
        reason = str(rule.get("reason") or f"configured score cap matched {field}").strip()
        matches.append((ceiling, reason))

    if not matches:
        salary_cap = _configured_salary_cap(job)
        return salary_cap
    salary_cap = _configured_salary_cap(job)
    if salary_cap:
        matches.append(salary_cap)
    return min(matches, key=lambda item: item[0])


def _apply_configured_cap(score: int, reasoning: str, job: dict[str, Any]) -> tuple[int, str]:
    """Apply the lowest matching configured cap to a computed score."""
    configured_cap = _configured_score_cap(job)
    if not configured_cap:
        return score, reasoning
    ceiling, cap_reason = configured_cap
    if score <= ceiling:
        return score, reasoning
    score = ceiling
    return score, f"{reasoning} Final score capped at {score} due to {cap_reason}."


def _merge_preferences(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge scoring preference lists without mutating defaults."""
    merged = {key: list(value) if isinstance(value, list) else value for key, value in base.items()}
    override = dict(override or {})
    if "hard_gaps" in override and "watch_for_hard_gaps" not in override:
        override["watch_for_hard_gaps"] = override.pop("hard_gaps")
    for key, value in (override or {}).items():
        if isinstance(value, list) and isinstance(merged.get(key), list):
            merged[key] = value
        else:
            merged[key] = value
    return merged


def _scoring_preferences() -> dict[str, Any]:
    """Load user-editable scoring preferences from searches.yaml."""
    search_cfg = config.load_search_config()
    return _merge_preferences(_default_scoring_preferences(), search_cfg.get("scoring_preferences", {}))


def _extract_json(response: str) -> dict[str, Any]:
    """Parse a JSON object from an LLM response."""
    text = response.strip()
    try:
        parsed = json.loads(text, strict=False)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(text[start:end + 1], strict=False)
    if not isinstance(parsed, dict):
        raise ValueError("LLM response JSON must be an object")
    return parsed


def _coerce_score(value: Any, default: int = 1) -> int:
    """Clamp a numeric score into the 1-10 range."""
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        score = default
    return max(1, min(10, score))


def _strings(value: Any) -> list[str]:
    """Return a clean list of strings from unknown JSON values."""
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _content_tokens(text: str) -> set[str]:
    """Tokenize text for lightweight requirement/evidence consistency checks."""
    stopwords = {
        "required", "requirement", "requires", "experience", "expertise", "domain",
        "candidate", "resume", "role", "preferred", "needed", "clearly",
    }
    return {
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9/&+-]*", text.lower())
        if len(token) >= 4 and token not in stopwords
    }


def _gap_has_supporting_quote(requirement: str, evidence: str) -> bool:
    """Check that a gap's job quote supports the stated requirement."""
    requirement_tokens = _content_tokens(requirement)
    evidence_tokens = _content_tokens(evidence)
    return bool(requirement_tokens and evidence_tokens and requirement_tokens & evidence_tokens)


def _gap_strings(value: Any) -> list[str]:
    """Return gap descriptions only when the LLM provides job evidence.

    This prevents user preference watchlists from being copied into gaps unless
    the job posting itself contains evidence that the requirement matters.
    """
    if not isinstance(value, list):
        return []

    gaps: list[str] = []
    for item in value:
        if isinstance(item, dict):
            requirement = str(item.get("requirement") or "").strip()
            evidence = str(item.get("job_evidence_quote") or "").strip()
            resume_gap = str(item.get("resume_gap") or "").strip()
            if requirement and evidence and _gap_has_supporting_quote(requirement, evidence):
                detail = requirement
                if resume_gap:
                    detail = f"{detail} ({resume_gap})"
                gaps.append(detail)
        elif isinstance(item, str) and item.strip():
            # Backward-compatible fallback for models that do not follow the
            # object schema. These are treated as soft gaps and handled by
            # dimension scores, not score ceilings.
            continue
    return gaps


def _compute_score(evidence: dict[str, Any]) -> tuple[int, str, str]:
    """Compute final score from structured LLM evidence.

    The LLM judges the role, dimensions, and hard gaps. Code applies stable
    scoring math so keyword-overlap enthusiasm cannot bypass missing evidence.
    """
    dimensions_raw = evidence.get("dimension_scores", {})
    dimensions = dimensions_raw if isinstance(dimensions_raw, dict) else {}
    weights = _dimension_weights()
    weighted_total = 0.0
    for name, weight in weights.items():
        weighted_total += _coerce_score(dimensions.get(name), default=1) * weight

    recommended = _coerce_score(evidence.get("recommended_score"), default=round(weighted_total))
    base_score = round((weighted_total * 0.75) + (recommended * 0.25))

    hard_gaps = _gap_strings(evidence.get("hard_requirement_gaps"))
    dealbreakers = _gap_strings(evidence.get("dealbreaker_gaps"))
    ceiling_cfg = _score_ceiling_config()

    ceiling = 10
    ceiling_reasons: list[str] = []
    multiple_gap_cfg = _as_dict(ceiling_cfg.get("multiple_hard_gaps"))
    if dealbreakers:
        ceiling = _config_int(ceiling_cfg.get("dealbreaker"), 4)
        ceiling_reasons.append("dealbreaker gaps")
    elif len(hard_gaps) >= _config_int(multiple_gap_cfg.get("count"), 3):
        ceiling = _config_int(multiple_gap_cfg.get("ceiling"), 5)
        ceiling_reasons.append("multiple hard requirement gaps")
    elif len(hard_gaps) >= 1:
        ceiling = _config_int(ceiling_cfg.get("any_hard_gap"), 7)
        ceiling_reasons.append("hard requirement gaps")

    location_cfg = _as_dict(ceiling_cfg.get("location_timezone"))
    location_timezone_score = _coerce_score(dimensions.get("location_timezone_alignment"), default=10)
    severe_threshold = _config_int(location_cfg.get("severe_threshold"), 2)
    severe_ceiling = _config_int(location_cfg.get("severe_ceiling"), 6)
    mismatch_threshold = _config_int(location_cfg.get("mismatch_threshold"), 3)
    mismatch_ceiling = _config_int(location_cfg.get("mismatch_ceiling"), 7)
    if location_timezone_score <= severe_threshold and ceiling > severe_ceiling:
        ceiling = severe_ceiling
        ceiling_reasons.append("severe location/timezone mismatch")
    elif location_timezone_score <= mismatch_threshold and ceiling > mismatch_ceiling:
        ceiling = mismatch_ceiling
        ceiling_reasons.append("location/timezone mismatch")

    final_score = max(1, min(10, base_score, ceiling))
    keywords = ", ".join(_strings(evidence.get("matched_keywords")))

    positives = "; ".join(_strings(evidence.get("positive_evidence"))[:3]) or "No strong resume-backed evidence identified"
    gaps = "; ".join((dealbreakers + hard_gaps)[:4]) or "No major hard requirement gaps identified"
    role_family = str(evidence.get("role_family") or "unknown role family")
    confidence = str(evidence.get("confidence") or "unknown")
    dimension_summary = ", ".join(
        f"{name}={_coerce_score(dimensions.get(name), default=1)}"
        for name in weights
    )

    reasoning = (
        f"Role family: {role_family}. Positives: {positives}. Gaps: {gaps}. "
        f"Dimensions: {dimension_summary}. Confidence: {confidence}."
    )
    if final_score < base_score:
        reason = ", ".join(ceiling_reasons) or "score ceiling"
        reasoning += f" Final score capped at {final_score} due to {reason}."

    return final_score, keywords, reasoning


def score_job(resume_text: str, job: dict) -> dict:
    """Score a single job against the resume.

    Args:
        resume_text: The candidate's full resume text.
        job: Job dict with keys: title, site, location, full_description.

    Returns:
        {"score": int, "keywords": str, "reasoning": str}
    """
    job_text = (
        f"TITLE: {job['title']}\n"
        f"COMPANY: {job['site']}\n"
        f"LOCATION: {job.get('location', 'N/A')}\n\n"
        f"DESCRIPTION:\n{(job.get('full_description') or '')[:6000]}"
    )
    preferences = json.dumps(_scoring_preferences(), indent=2)

    messages = [
        {"role": "system", "content": _scoring_prompt()},
        {
            "role": "user",
            "content": (
                f"SCORING PREFERENCES (target preferences and watchlists, not job requirements):\n{preferences}\n\n"
                f"RESUME:\n{resume_text}\n\n---\n\nJOB POSTING:\n{job_text}"
            ),
        },
    ]

    try:
        client = get_client()
        response = client.chat(messages, max_tokens=1200, temperature=0.1)
        try:
            evidence = _extract_json(response)
            score, keywords, reasoning = _compute_score(evidence)
            score, reasoning = _apply_configured_cap(score, reasoning, job)
            return {"score": score, "keywords": keywords, "reasoning": reasoning}
        except Exception as parse_error:
            log.warning("Structured score parse failed for '%s': %s", job.get("title", "?"), parse_error)
            return _parse_score_response(response)
    except Exception as e:
        log.error("LLM error scoring job '%s': %s", job.get("title", "?"), e)
        return {"score": 0, "keywords": "", "reasoning": f"LLM error: {e}"}


def run_scoring(limit: int = 0, rescore: bool = False) -> dict:
    """Score unscored jobs that have full descriptions.

    Args:
        limit: Maximum number of jobs to score in this run.
        rescore: If True, re-score all jobs (not just unscored ones).

    Returns:
        {"scored": int, "errors": int, "elapsed": float, "distribution": list}
    """
    resume_text = RESUME_PATH.read_text(encoding="utf-8")
    conn = get_connection()

    if rescore:
        query = "SELECT * FROM jobs WHERE full_description IS NOT NULL"
        if limit > 0:
            query += f" LIMIT {limit}"
        jobs = conn.execute(query).fetchall()
    else:
        jobs = get_jobs_by_stage(conn=conn, stage="pending_score", limit=limit)

    if not jobs:
        log.info("No unscored jobs with descriptions found.")
        return {"scored": 0, "errors": 0, "elapsed": 0.0, "distribution": []}

    # Convert sqlite3.Row to dicts if needed
    if jobs and not isinstance(jobs[0], dict):
        columns = jobs[0].keys()
        jobs = [dict(zip(columns, row)) for row in jobs]

    log.info("Scoring %d jobs sequentially...", len(jobs))
    t0 = time.time()
    completed = 0
    errors = 0
    results: list[dict] = []

    for job in jobs:
        result = score_job(resume_text, job)
        result["url"] = job["url"]
        completed += 1

        if result["score"] == 0:
            errors += 1

        results.append(result)

        log.info(
            "[%d/%d] score=%d  %s",
            completed, len(jobs), result["score"], job.get("title", "?")[:60],
        )

    # Write scores to DB
    now = datetime.now(timezone.utc).isoformat()
    for r in results:
        conn.execute(
            "UPDATE jobs SET fit_score = ?, score_reasoning = ?, scored_at = ? WHERE url = ?",
            (r["score"], f"{r['keywords']}\n{r['reasoning']}", now, r["url"]),
        )
    conn.commit()

    elapsed = time.time() - t0
    log.info("Done: %d scored in %.1fs (%.1f jobs/sec)", len(results), elapsed, len(results) / elapsed if elapsed > 0 else 0)

    # Score distribution
    dist = conn.execute("""
        SELECT fit_score, COUNT(*) FROM jobs
        WHERE fit_score IS NOT NULL
        GROUP BY fit_score ORDER BY fit_score DESC
    """).fetchall()
    distribution = [(row[0], row[1]) for row in dist]

    return {
        "scored": len(results),
        "errors": errors,
        "elapsed": elapsed,
        "distribution": distribution,
    }
