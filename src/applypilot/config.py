"""ApplyPilot configuration: paths, platform detection, user data."""

import os
import platform
import shutil
from pathlib import Path

# User data directory — all user-specific files live here
APP_DIR = Path(os.environ.get("APPLYPILOT_DIR", Path.home() / ".applypilot"))

# Core paths
DB_PATH = APP_DIR / "applypilot.db"
PROFILE_PATH = APP_DIR / "profile.json"
RESUME_PATH = APP_DIR / "resume.txt"
RESUME_PDF_PATH = APP_DIR / "resume.pdf"
SEARCH_CONFIG_PATH = APP_DIR / "searches.yaml"
ENV_PATH = APP_DIR / ".env"

# Generated output
TAILORED_DIR = APP_DIR / "tailored_resumes"
COVER_LETTER_DIR = APP_DIR / "cover_letters"
LOG_DIR = APP_DIR / "logs"

# Chrome worker isolation
CHROME_WORKER_DIR = APP_DIR / "chrome-workers"
APPLY_WORKER_DIR = APP_DIR / "apply-workers"

# Package-shipped config (YAML registries)
PACKAGE_DIR = Path(__file__).parent
CONFIG_DIR = PACKAGE_DIR / "config"


def get_chrome_path() -> str:
    """Auto-detect Chrome/Chromium executable path, cross-platform.

    Override with CHROME_PATH environment variable.
    """
    env_path = os.environ.get("CHROME_PATH")
    if env_path and Path(env_path).exists():
        return env_path

    system = platform.system()

    if system == "Windows":
        candidates = [
            Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")) / "Google/Chrome/Application/chrome.exe",
            Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")) / "Google/Chrome/Application/chrome.exe",
            Path(os.environ.get("LOCALAPPDATA", "")) / "Google/Chrome/Application/chrome.exe",
        ]
    elif system == "Darwin":
        candidates = [
            Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
            Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
        ]
    else:  # Linux
        candidates = []
        for name in ("google-chrome", "google-chrome-stable", "chromium-browser", "chromium"):
            found = shutil.which(name)
            if found:
                candidates.append(Path(found))

    for c in candidates:
        if c and c.exists():
            return str(c)

    # Fall back to PATH search
    for name in ("google-chrome", "google-chrome-stable", "chromium-browser", "chromium", "chrome"):
        found = shutil.which(name)
        if found:
            return found

    raise FileNotFoundError(
        "Chrome/Chromium not found. Install Chrome or set CHROME_PATH environment variable."
    )


def get_chrome_user_data() -> Path:
    """Default Chrome user data directory, cross-platform."""
    system = platform.system()
    if system == "Windows":
        return Path(os.environ.get("LOCALAPPDATA", "")) / "Google" / "Chrome" / "User Data"
    elif system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "Google" / "Chrome"
    else:
        return Path.home() / ".config" / "google-chrome"


def ensure_dirs():
    """Create all required directories."""
    for d in [APP_DIR, TAILORED_DIR, COVER_LETTER_DIR, LOG_DIR, CHROME_WORKER_DIR, APPLY_WORKER_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def load_profile() -> dict:
    """Load user profile from ~/.applypilot/profile.json."""
    import json
    if not PROFILE_PATH.exists():
        raise FileNotFoundError(
            f"Profile not found at {PROFILE_PATH}. Run `applypilot init` first."
        )
    return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))


def load_search_config() -> dict:
    """Load search configuration from ~/.applypilot/searches.yaml."""
    import yaml
    if not SEARCH_CONFIG_PATH.exists():
        return {}
    return yaml.safe_load(SEARCH_CONFIG_PATH.read_text(encoding="utf-8"))


def load_sites_config() -> dict:
    """Load sites.yaml configuration (sites list, manual_ats, blocked, etc.)."""
    import yaml
    path = CONFIG_DIR / "sites.yaml"
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def job_board_sites(search_cfg: dict) -> list[str] | None:
    """Return JobSpy board names from search config.

    Accept both ``boards`` and ``sites`` so existing user configs keep working.
    """
    sites = search_cfg.get("sites")
    if sites is not None:
        return sites
    return search_cfg.get("boards")


def location_filters(search_cfg: dict) -> tuple[list[str], list[str]]:
    """Return accepted and rejected location patterns from search config.

    Accept both nested ``location`` config and older flat keys.
    """
    location_cfg = search_cfg.get("location", {}) or {}
    accept = search_cfg.get("location_accept")
    reject = search_cfg.get("location_reject_non_remote")
    if accept is None:
        accept = location_cfg.get("accept_patterns", [])
    if reject is None:
        reject = location_cfg.get("reject_patterns", [])
    return accept, reject


def discovery_sources(search_cfg: dict) -> dict[str, bool]:
    """Return enabled discovery source flags.

    Defaults preserve the historical behavior: all discovery sources run
    unless explicitly disabled in ``searches.yaml``.
    """
    raw = search_cfg.get("discovery_sources", {}) or {}
    return {
        "jobspy": bool(raw.get("jobspy", True)),
        "workday": bool(raw.get("workday", True)),
        "smart_extract": bool(raw.get("smart_extract", raw.get("smartextract", True))),
    }


def is_manual_ats(url: str | None) -> bool:
    """Check if a URL routes through an ATS that requires manual application."""
    if not url:
        return False
    sites_cfg = load_sites_config()
    domains = sites_cfg.get("manual_ats", [])
    url_lower = url.lower()
    return any(domain in url_lower for domain in domains)


def load_blocked_sites() -> tuple[set[str], list[str]]:
    """Load blocked sites and URL patterns from sites.yaml.

    Returns:
        (blocked_site_names, blocked_url_patterns)
    """
    cfg = load_sites_config()
    blocked = cfg.get("blocked", {})
    sites = set(blocked.get("sites", []))
    patterns = blocked.get("url_patterns", [])
    return sites, patterns


def load_blocked_sso() -> list[str]:
    """Load blocked SSO domains from sites.yaml."""
    cfg = load_sites_config()
    return cfg.get("blocked_sso", [])


def load_base_urls() -> dict[str, str | None]:
    """Load site base URLs for URL resolution from sites.yaml."""
    cfg = load_sites_config()
    return cfg.get("base_urls", {})


# ---------------------------------------------------------------------------
# Default values — referenced across modules instead of magic numbers
# ---------------------------------------------------------------------------

DEFAULTS = {
    "min_score": 7,
    "max_apply_attempts": 3,
    "max_tailor_attempts": 5,
    "poll_interval": 60,
    "apply_timeout": 300,
    "viewport": "1280x900",
    "playwright_mcp_version": "0.0.75",
    "gmail_mcp_version": "1.1.11",
}

SENSITIVE_ENV_KEYS = (
    "GEMINI_API_KEY",
    "OPENAI_API_KEY",
    "LLM_API_KEY",
    "CAPSOLVER_API_KEY",
    "ANTHROPIC_API_KEY",
    "CLAUDE_API_KEY",
)

TRUTHY = {"1", "true", "yes", "on"}


def env_flag(name: str, default: bool = False) -> bool:
    """Return a boolean environment flag using strict, explicit truthy values."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in TRUTHY


def privacy_mode() -> str:
    """Current privacy mode. Defaults to strict for new installs."""
    return os.environ.get("APPLYPILOT_PRIVACY_MODE", "strict").strip().lower() or "strict"


def is_strict_privacy() -> bool:
    """Whether privacy-sensitive integrations should fail closed."""
    return privacy_mode() == "strict"


def cloud_llm_allowed() -> bool:
    """Whether cloud LLM providers may receive resume/profile/application data."""
    return env_flag("APPLYPILOT_ALLOW_CLOUD_LLM", default=False)


def gmail_mcp_enabled() -> bool:
    """Whether the Gmail MCP server may be exposed to Claude Code."""
    return env_flag("APPLYPILOT_ENABLE_GMAIL_MCP", default=False) and env_flag(
        "APPLYPILOT_ALLOW_VULNERABLE_GMAIL_MCP",
        default=False,
    )


def gmail_mcp_requested() -> bool:
    """Whether Gmail MCP was requested but may still be blocked by safety gates."""
    return env_flag("APPLYPILOT_ENABLE_GMAIL_MCP", default=False)


def capsolver_enabled() -> bool:
    """Whether CapSolver automation may be included in browser-agent prompts."""
    return env_flag("APPLYPILOT_ENABLE_CAPSOLVER", default=False)


def clone_chrome_profile_enabled() -> bool:
    """Whether worker Chrome profiles may be seeded from the user's real profile."""
    return env_flag("APPLYPILOT_CLONE_CHROME_PROFILE", default=False)


def require_cloud_llm_allowed(provider: str) -> None:
    """Fail closed before sending personal data to a cloud LLM/agent."""
    if is_strict_privacy() and not cloud_llm_allowed():
        raise RuntimeError(
            f"{provider} would receive resume/profile/application data, but "
            "APPLYPILOT_PRIVACY_MODE=strict and APPLYPILOT_ALLOW_CLOUD_LLM is not set. "
            "Set APPLYPILOT_ALLOW_CLOUD_LLM=1 only after you accept that data-sharing risk, "
            "or configure LLM_URL for a local OpenAI-compatible model."
        )


def secure_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Write a sensitive local artifact and mark it owner-only where possible."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding=encoding)
    try:
        path.chmod(0o600)
    except OSError:
        pass


def secure_file_mode(path: Path) -> str:
    """Return an octal file mode string or a useful unavailable marker."""
    try:
        return oct(path.stat().st_mode & 0o777)
    except FileNotFoundError:
        return "missing"
    except OSError:
        return "unknown"


def sensitive_values(profile: dict | None = None) -> list[str]:
    """Collect known secret values for log/prompt redaction."""
    values: list[str] = []
    for key in SENSITIVE_ENV_KEYS:
        value = os.environ.get(key, "")
        if value:
            values.append(value)

    if profile:
        password = profile.get("personal", {}).get("password", "")
        if password:
            values.append(password)

    return [v for v in values if len(v) >= 4]


def redact_sensitive(text: str, profile: dict | None = None) -> str:
    """Remove known local secrets from text before persisting it."""
    redacted = text
    for value in sensitive_values(profile):
        redacted = redacted.replace(value, "[REDACTED]")
    return redacted


def load_env():
    """Load environment variables from ~/.applypilot/.env if it exists."""
    from dotenv import load_dotenv
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)
    # Also try CWD .env as fallback
    load_dotenv()


# ---------------------------------------------------------------------------
# Tier system — feature gating by installed dependencies
# ---------------------------------------------------------------------------

TIER_LABELS = {
    1: "Discovery",
    2: "AI Scoring & Tailoring",
    3: "Full Auto-Apply",
}

TIER_COMMANDS: dict[int, list[str]] = {
    1: ["init", "run discover", "run enrich", "status", "dashboard"],
    2: ["run score", "run tailor", "run cover", "run pdf", "run"],
    3: ["apply"],
}


def get_tier() -> int:
    """Detect the current tier based on available dependencies.

    Tier 1 (Discovery):            Python + pip
    Tier 2 (AI Scoring & Tailoring): + LLM API key
    Tier 3 (Full Auto-Apply):       + Claude Code CLI + Chrome
    """
    load_env()

    has_local_llm = bool(os.environ.get("LLM_URL"))
    has_cloud_llm = any(os.environ.get(k) for k in ("GEMINI_API_KEY", "OPENAI_API_KEY"))
    has_llm = has_local_llm or (has_cloud_llm and (not is_strict_privacy() or cloud_llm_allowed()))
    if not has_llm:
        return 1

    has_claude = shutil.which("claude") is not None
    try:
        get_chrome_path()
        has_chrome = True
    except FileNotFoundError:
        has_chrome = False

    if has_claude and has_chrome and (not is_strict_privacy() or cloud_llm_allowed()):
        return 3

    return 2


def check_tier(required: int, feature: str) -> None:
    """Raise SystemExit with a clear message if the current tier is too low.

    Args:
        required: Minimum tier needed (1, 2, or 3).
        feature: Human-readable description of the feature being gated.
    """
    current = get_tier()
    if current >= required:
        return

    from rich.console import Console
    _console = Console(stderr=True)

    missing: list[str] = []
    has_local_llm = bool(os.environ.get("LLM_URL"))
    has_cloud_llm = any(os.environ.get(k) for k in ("GEMINI_API_KEY", "OPENAI_API_KEY"))
    has_usable_llm = has_local_llm or (has_cloud_llm and (not is_strict_privacy() or cloud_llm_allowed()))
    if required >= 2 and not has_usable_llm:
        if has_cloud_llm and is_strict_privacy() and not cloud_llm_allowed():
            missing.append("Cloud LLM opt-in — set APPLYPILOT_ALLOW_CLOUD_LLM=1 or configure LLM_URL")
        else:
            missing.append("LLM API key — run [bold]applypilot init[/bold] or set GEMINI_API_KEY")
    if required >= 3:
        if is_strict_privacy() and not cloud_llm_allowed():
            missing.append("Claude Code cloud opt-in — set APPLYPILOT_ALLOW_CLOUD_LLM=1 for auto-apply")
        if not shutil.which("claude"):
            missing.append("Claude Code CLI — install from [bold]https://claude.ai/code[/bold]")
        try:
            get_chrome_path()
        except FileNotFoundError:
            missing.append("Chrome/Chromium — install or set CHROME_PATH")

    _console.print(
        f"\n[red]'{feature}' requires {TIER_LABELS.get(required, f'Tier {required}')} (Tier {required}).[/red]\n"
        f"Current tier: {TIER_LABELS.get(current, f'Tier {current}')} (Tier {current})."
    )
    if missing:
        _console.print("\n[yellow]Missing:[/yellow]")
        for m in missing:
            _console.print(f"  - {m}")
    _console.print()
    raise SystemExit(1)
