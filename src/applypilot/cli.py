"""ApplyPilot CLI — the main entry point."""

from __future__ import annotations

import logging
import os
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from applypilot import __version__

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)

app = typer.Typer(
    name="applypilot",
    help="AI-powered end-to-end job application pipeline.",
    no_args_is_help=True,
)
jobs_app = typer.Typer(help="Inspect and manage jobs without direct database access.")
manual_app = typer.Typer(help="Manual application workflow helpers.")
console = Console()
log = logging.getLogger(__name__)

# Valid pipeline stages (in execution order)
VALID_STAGES = ("discover", "enrich", "score", "tailor", "cover", "pdf")
JOB_STAGES = (
    "all", "pending-detail", "enriched", "pending-score", "scored",
    "pending-tailor", "tailored", "pending-apply", "manual", "failed", "applied",
)
JOB_OUTPUT_FIELDS = (
    "url", "application_url", "title", "site", "location", "fit_score",
    "apply_status", "apply_error", "tailored_resume_path", "cover_letter_path",
    "applied_at",
)

app.add_typer(jobs_app, name="jobs")
app.add_typer(manual_app, name="manual")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bootstrap() -> None:
    """Common setup: load env, create dirs, init DB."""
    from applypilot.config import load_env, ensure_dirs
    from applypilot.database import init_db

    load_env()
    ensure_dirs()
    init_db()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"[bold]applypilot[/bold] {__version__}")
        raise typer.Exit()


def _stage_key(stage: str) -> str:
    return (stage or "all").strip().lower().replace("_", "-")


def _db_stage(stage: str) -> str:
    return _stage_key(stage).replace("-", "_")


def _read_cli_ids(job_ids: Optional[list[str]] = None, ids_file: Optional[Path] = None) -> list[str]:
    from applypilot.database import read_ids_file

    values = list(job_ids or [])
    if ids_file:
        values.extend(read_ids_file(ids_file))
    return values


def _resolve_cli_ids(job_ids: Optional[list[str]] = None, ids_file: Optional[Path] = None) -> list[str]:
    ids = _read_cli_ids(job_ids, ids_file)
    if not ids:
        return []
    from applypilot.database import get_connection, resolve_job_ids

    try:
        return resolve_job_ids(get_connection(), ids)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc


def _resolve_one_job_url(job_id: str) -> str:
    from applypilot.database import get_connection, resolve_job_id

    job = resolve_job_id(get_connection(), job_id)
    if not job:
        console.print(f"[red]No matching job found:[/red] {job_id}")
        raise typer.Exit(code=1)
    return job["url"]


def _public_job(job: dict) -> dict:
    return {field: job.get(field) for field in JOB_OUTPUT_FIELDS}


def _print_jobs(jobs: list[dict], output_format: str) -> None:
    if output_format == "json":
        console.print(json.dumps([_public_job(job) for job in jobs], indent=2))
        return
    if output_format != "table":
        console.print("[red]Invalid --format.[/red] Choose table or json.")
        raise typer.Exit(code=1)

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Score", justify="right")
    table.add_column("Status")
    table.add_column("Title")
    table.add_column("Site")
    table.add_column("URL")
    for job in jobs:
        table.add_row(
            "" if job.get("fit_score") is None else str(job.get("fit_score")),
            job.get("apply_status") or "",
            job.get("title") or "",
            job.get("site") or "",
            job.get("url") or "",
        )
    console.print(table)


def _print_job(job: dict, output_format: str) -> None:
    if output_format == "json":
        console.print(json.dumps(_public_job(job), indent=2))
        return
    if output_format != "table":
        console.print("[red]Invalid --format.[/red] Choose table or json.")
        raise typer.Exit(code=1)
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Field", style="bold")
    table.add_column("Value")
    for field, value in _public_job(job).items():
        table.add_row(field, "" if value is None else str(value))
    console.print(table)


def _run_stage_alias(
    stage: str,
    job_ids: Optional[list[str]] = None,
    min_score: int = 7,
    workers: int = 1,
    validation: str = "normal",
    force: bool = False,
) -> None:
    _bootstrap()
    valid_modes = ("strict", "normal", "lenient")
    if validation not in valid_modes:
        console.print(
            f"[red]Invalid --validation value:[/red] '{validation}'. "
            f"Choose from: {', '.join(valid_modes)}"
        )
        raise typer.Exit(code=1)
    resolved = _resolve_cli_ids(job_ids)
    from applypilot.pipeline import run_pipeline

    result = run_pipeline(
        stages=[stage],
        min_score=min_score,
        workers=workers,
        validation_mode=validation,
        job_ids=resolved or None,
        force=force,
    )
    if result.get("errors"):
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-V",
        help="Show version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """ApplyPilot — AI-powered end-to-end job application pipeline."""


@app.command()
def init() -> None:
    """Run the first-time setup wizard (profile, resume, search config)."""
    from applypilot.wizard.init import run_wizard

    run_wizard()


@app.command()
def run(
    stages: Optional[list[str]] = typer.Argument(
        None,
        help=(
            "Pipeline stages to run. "
            f"Valid: {', '.join(VALID_STAGES)}, all. "
            "Defaults to 'all' if omitted."
        ),
    ),
    min_score: int = typer.Option(7, "--min-score", help="Minimum fit score for tailor/cover stages."),
    workers: int = typer.Option(1, "--workers", "-w", help="Parallel threads for discovery/enrichment stages."),
    stream: bool = typer.Option(False, "--stream", help="Run stages concurrently (streaming mode)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview stages without executing."),
    job_id: Optional[list[str]] = typer.Option(None, "--job-id", help="Target a specific job URL/application URL. Repeatable."),
    ids_file: Optional[Path] = typer.Option(None, "--ids-file", help="Newline-delimited file of job URLs/application URLs."),
    force: bool = typer.Option(False, "--force", help="Clear selected stage output before rerunning targeted jobs."),
    validation: str = typer.Option(
        "normal",
        "--validation",
        help=(
            "Validation strictness for tailor/cover stages. "
            "strict: banned words = errors, judge must pass. "
            "normal: banned words = warnings only (default, recommended for Gemini free tier). "
            "lenient: banned words ignored, LLM judge skipped (fastest, fewest API calls)."
        ),
    ),
) -> None:
    """Run pipeline stages: discover, enrich, score, tailor, cover, pdf."""
    _bootstrap()

    from applypilot.pipeline import run_pipeline

    stage_list = stages if stages else ["all"]

    # Validate stage names
    for s in stage_list:
        if s != "all" and s not in VALID_STAGES:
            console.print(
                f"[red]Unknown stage:[/red] '{s}'. "
                f"Valid stages: {', '.join(VALID_STAGES)}, all"
            )
            raise typer.Exit(code=1)

    # Gate AI stages behind Tier 2
    llm_stages = {"score", "tailor", "cover"}
    if any(s in stage_list for s in llm_stages) or "all" in stage_list:
        from applypilot.config import check_tier
        check_tier(2, "AI scoring/tailoring")

    # Validate the --validation flag value
    valid_modes = ("strict", "normal", "lenient")
    if validation not in valid_modes:
        console.print(
            f"[red]Invalid --validation value:[/red] '{validation}'. "
            f"Choose from: {', '.join(valid_modes)}"
        )
        raise typer.Exit(code=1)

    resolved_job_ids = _resolve_cli_ids(job_id, ids_file)

    result = run_pipeline(
        stages=stage_list,
        min_score=min_score,
        dry_run=dry_run,
        stream=stream,
        workers=workers,
        validation_mode=validation,
        job_ids=resolved_job_ids or None,
        force=force,
    )

    if result.get("errors"):
        raise typer.Exit(code=1)


@jobs_app.command("list")
def jobs_list(
    stage: str = typer.Argument("all", help=f"Stage filter. Valid: {', '.join(JOB_STAGES)}."),
    min_score: int = typer.Option(7, "--min-score", help="Minimum score for score-aware stages."),
    limit: int = typer.Option(50, "--limit", "-l", help="Maximum rows to show. Use 0 for all."),
    output_format: str = typer.Option("table", "--format", help="Output format: table or json."),
) -> None:
    """List jobs by workflow stage."""
    _bootstrap()
    normalized = _stage_key(stage)
    if normalized not in JOB_STAGES:
        console.print(f"[red]Unknown stage:[/red] {stage}. Valid: {', '.join(JOB_STAGES)}")
        raise typer.Exit(code=1)

    from applypilot.database import get_connection, get_jobs_by_stage

    jobs = get_jobs_by_stage(
        conn=get_connection(),
        stage=_db_stage(normalized),
        min_score=min_score,
        limit=limit,
    )
    _print_jobs(jobs, output_format)


@jobs_app.command("show")
def jobs_show(
    job_id: str = typer.Argument(..., help="Job URL/application URL."),
    output_format: str = typer.Option("table", "--format", help="Output format: table or json."),
) -> None:
    """Show one job by URL/application URL."""
    _bootstrap()
    from applypilot.database import get_connection, resolve_job_id

    job = resolve_job_id(get_connection(), job_id)
    if not job:
        console.print(f"[red]No matching job found:[/red] {job_id}")
        raise typer.Exit(code=1)
    _print_job(job, output_format)


@jobs_app.command("add")
def jobs_add(
    url: str = typer.Argument(..., help="Job URL to import."),
    title: Optional[str] = typer.Option(None, "--title", help="Job title."),
    site: Optional[str] = typer.Option(None, "--site", help="Source/company/site."),
    location: Optional[str] = typer.Option(None, "--location", help="Job location."),
) -> None:
    """Import one job URL for direct manual workflow runs."""
    _bootstrap()
    from applypilot.database import add_job, get_connection

    inserted = add_job(get_connection(), url=url, title=title, site=site, location=location)
    if inserted:
        console.print(f"[green]Added job:[/green] {url}")
    else:
        console.print(f"[yellow]Job already exists:[/yellow] {url}")


@jobs_app.command("ids")
def jobs_ids(
    stage: str = typer.Argument("all", help=f"Stage filter. Valid: {', '.join(JOB_STAGES)}."),
    min_score: int = typer.Option(7, "--min-score", help="Minimum score for score-aware stages."),
    limit: int = typer.Option(50, "--limit", "-l", help="Maximum IDs to print. Use 0 for all."),
) -> None:
    """Print canonical job IDs, one URL per line."""
    _bootstrap()
    normalized = _stage_key(stage)
    if normalized not in JOB_STAGES:
        console.print(f"[red]Unknown stage:[/red] {stage}. Valid: {', '.join(JOB_STAGES)}")
        raise typer.Exit(code=1)

    from applypilot.database import get_connection, get_jobs_by_stage

    jobs = get_jobs_by_stage(
        conn=get_connection(),
        stage=_db_stage(normalized),
        min_score=min_score,
        limit=limit,
    )
    for job in jobs:
        console.print(job["url"])


@jobs_app.command("remove")
def jobs_remove(
    job_ids: Optional[list[str]] = typer.Argument(None, help="Job URLs/application URLs to delete."),
    ids_file: Optional[Path] = typer.Option(None, "--ids-file", help="Newline-delimited file of job URLs/application URLs."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Confirm deletion."),
) -> None:
    """Remove jobs from the discovery database."""
    _bootstrap()
    resolved = _resolve_cli_ids(job_ids, ids_file)
    if not resolved:
        console.print("[red]Provide at least one JOB_ID or --ids-file.[/red]")
        raise typer.Exit(code=1)
    if not yes:
        console.print("[red]Refusing to delete without --yes.[/red]")
        raise typer.Exit(code=1)

    from applypilot.database import delete_jobs, get_connection

    count = delete_jobs(get_connection(), resolved)
    console.print(f"[green]Removed {count} job(s).[/green]")


@jobs_app.command("reset")
def jobs_reset(
    stage: str = typer.Argument(..., help="Stage output to clear: enrich, score, tailor, cover, or apply."),
    job_ids: Optional[list[str]] = typer.Argument(None, help="Job URLs/application URLs to reset."),
    ids_file: Optional[Path] = typer.Option(None, "--ids-file", help="Newline-delimited file of job URLs/application URLs."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Confirm reset."),
) -> None:
    """Clear selected stage output without rerunning the stage."""
    _bootstrap()
    normalized = _db_stage(stage)
    if normalized not in {"enrich", "score", "tailor", "cover", "apply"}:
        console.print("[red]Stage must be one of: enrich, score, tailor, cover, apply.[/red]")
        raise typer.Exit(code=1)
    resolved = _resolve_cli_ids(job_ids, ids_file)
    if not resolved:
        console.print("[red]Provide at least one JOB_ID or --ids-file.[/red]")
        raise typer.Exit(code=1)
    if not yes:
        console.print("[red]Refusing to reset without --yes.[/red]")
        raise typer.Exit(code=1)

    from applypilot.database import get_connection, reset_jobs_for_stage

    count = reset_jobs_for_stage(get_connection(), normalized, resolved, include_applied=True)
    console.print(f"[green]Reset {normalized} output for {count} job(s).[/green]")


@app.command()
def apply(
    job_ids: Optional[list[str]] = typer.Argument(None, help="Specific job URLs/application URLs to apply."),
    limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Max applications to submit."),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of parallel browser workers."),
    min_score: int = typer.Option(7, "--min-score", help="Minimum fit score for job selection."),
    model: str = typer.Option("haiku", "--model", "-m", help="Claude model name."),
    continuous: bool = typer.Option(False, "--continuous", "-c", help="Run forever, polling for new jobs."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview actions without submitting."),
    headless: bool = typer.Option(False, "--headless", help="Run browsers in headless mode."),
    url: Optional[str] = typer.Option(None, "--url", help="Apply to a specific job URL."),
    ids_file: Optional[Path] = typer.Option(None, "--ids-file", help="Newline-delimited file of job URLs/application URLs."),
    force: bool = typer.Option(False, "--force", help="Clear failed/manual/in-progress state before applying selected jobs."),
    resume_mode: Optional[str] = typer.Option(None, "--resume-mode", help="Resume source: tailored, default, or path to .pdf/.txt. Env: APPLYPILOT_APPLY_RESUME_MODE."),
    gen: bool = typer.Option(False, "--gen", help="Generate prompt file for manual debugging instead of running."),
    mark_applied: Optional[str] = typer.Option(None, "--mark-applied", help="Manually mark a job URL as applied."),
    mark_failed: Optional[str] = typer.Option(None, "--mark-failed", help="Manually mark a job URL as failed (provide URL)."),
    fail_reason: Optional[str] = typer.Option(None, "--fail-reason", help="Reason for --mark-failed."),
    reset_failed: bool = typer.Option(False, "--reset-failed", help="Reset all failed jobs for retry."),
) -> None:
    """Launch auto-apply to submit job applications."""
    _bootstrap()

    from applypilot.config import (
        check_tier,
        PROFILE_PATH as _profile_path,
        RESUME_PDF_PATH as _resume_pdf_path,
    )
    from applypilot.database import get_connection

    # --- Utility modes (no Chrome/Claude needed) ---

    if mark_applied:
        from applypilot.apply.launcher import mark_job
        mark_job(_resolve_one_job_url(mark_applied), "applied")
        console.print(f"[green]Marked as applied:[/green] {mark_applied}")
        return

    if mark_failed:
        from applypilot.apply.launcher import mark_job
        mark_job(_resolve_one_job_url(mark_failed), "failed", reason=fail_reason)
        console.print(f"[yellow]Marked as failed:[/yellow] {mark_failed} ({fail_reason or 'manual'})")
        return

    if reset_failed:
        from applypilot.apply.launcher import reset_failed as do_reset
        count = do_reset()
        console.print(f"[green]Reset {count} failed job(s) for retry.[/green]")
        return

    resolved_job_ids = _resolve_cli_ids(job_ids, ids_file)
    if url and resolved_job_ids:
        console.print("[red]Use either --url or positional JOB_ID/--ids-file, not both.[/red]")
        raise typer.Exit(code=1)
    if force and resolved_job_ids:
        from applypilot.database import reset_jobs_for_stage
        reset_count = reset_jobs_for_stage(get_connection(), "apply", resolved_job_ids)
        console.print(f"[yellow]Reset apply state for {reset_count} job(s).[/yellow]")
    elif force and url:
        from applypilot.database import reset_jobs_for_stage
        reset_url = _resolve_one_job_url(url)
        reset_count = reset_jobs_for_stage(get_connection(), "apply", [reset_url])
        console.print(f"[yellow]Reset apply state for {reset_count} job(s).[/yellow]")

    # --- Full apply mode ---
    effective_resume_mode = (resume_mode or os.environ.get("APPLYPILOT_APPLY_RESUME_MODE") or "tailored").strip()
    resume_mode_key = effective_resume_mode.lower()
    if resume_mode_key in {"tailored", "default"}:
        effective_resume_mode = resume_mode_key
    else:
        resume_path = Path(effective_resume_mode).expanduser()
        if resume_path.suffix.lower() not in {".pdf", ".txt"}:
            console.print("[red]Invalid --resume-mode.[/red] Use 'tailored', 'default', or a .pdf/.txt path.")
            raise typer.Exit(code=1)
        if not resume_path.exists():
            console.print(f"[red]Resume file not found:[/red] {resume_path}")
            raise typer.Exit(code=1)
        if resume_path.suffix.lower() == ".txt" and not resume_path.with_suffix(".pdf").exists():
            console.print(
                f"[red]Resume PDF not found for TXT source.[/red]\n"
                f"Expected: [bold]{resume_path.with_suffix('.pdf')}[/bold]"
            )
            raise typer.Exit(code=1)
        effective_resume_mode = str(resume_path)

    if not effective_resume_mode:
        console.print("[red]Invalid --resume-mode.[/red] Use 'tailored', 'default', or a .pdf/.txt path.")
        raise typer.Exit(code=1)

    # Check 1: Tier 3 required (Claude Code CLI + Chrome)
    check_tier(3, "auto-apply")

    # Check 2: Profile exists
    if not _profile_path.exists():
        console.print(
            "[red]Profile not found.[/red]\n"
            "Run [bold]applypilot init[/bold] to create your profile first."
        )
        raise typer.Exit(code=1)

    if effective_resume_mode == "default" and not _resume_pdf_path.exists():
        console.print(
            f"[red]Default resume PDF not found.[/red]\n"
            f"Expected: [bold]{_resume_pdf_path}[/bold]"
        )
        raise typer.Exit(code=1)

    # Check 3: Tailored resumes exist (skip for --gen with --url or default resume mode)
    if effective_resume_mode == "tailored" and not (gen and url):
        conn = get_connection()
        ready = conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE tailored_resume_path IS NOT NULL AND applied_at IS NULL"
        ).fetchone()[0]
        if ready == 0:
            console.print(
                "[red]No tailored resumes ready.[/red]\n"
                "Run [bold]applypilot run score tailor[/bold] first to prepare applications."
            )
            raise typer.Exit(code=1)

    if gen:
        from applypilot.apply.launcher import gen_prompt
        target = url or ""
        if not target:
            console.print("[red]--gen requires --url to specify which job.[/red]")
            raise typer.Exit(code=1)
        prompt_file = gen_prompt(target, min_score=min_score, model=model, resume_mode=effective_resume_mode)
        if not prompt_file:
            console.print("[red]No matching job found for that URL.[/red]")
            raise typer.Exit(code=1)
        mcp_path = _profile_path.parent / ".mcp-apply-0.json"
        console.print(f"[green]Wrote prompt to:[/green] {prompt_file}")
        console.print("\n[bold]Run manually:[/bold]")
        console.print(
            f"  claude --model {model} -p "
            f"--mcp-config {mcp_path} "
            f"--permission-mode bypassPermissions < {prompt_file}"
        )
        return

    from applypilot.apply.launcher import main as apply_main

    effective_limit = limit if limit is not None else (0 if continuous else (len(resolved_job_ids) if resolved_job_ids else 1))

    console.print("\n[bold blue]Launching Auto-Apply[/bold blue]")
    console.print(f"  Limit:    {'unlimited' if continuous else effective_limit}")
    console.print(f"  Workers:  {workers}")
    console.print(f"  Model:    {model}")
    console.print(f"  Headless: {headless}")
    console.print(f"  Dry run:  {dry_run}")
    console.print(f"  Resume:   {effective_resume_mode}")
    if url:
        console.print(f"  Target:   {url}")
    if resolved_job_ids:
        console.print(f"  Targets:  {len(resolved_job_ids)}")
    console.print()

    apply_main(
        limit=effective_limit,
        target_url=url,
        target_urls=resolved_job_ids or None,
        min_score=min_score,
        headless=headless,
        model=model,
        dry_run=dry_run,
        continuous=continuous,
        workers=workers,
        resume_mode=effective_resume_mode,
    )


@manual_app.command("packet")
def manual_packet(
    job_id: str = typer.Argument(..., help="Job URL/application URL."),
    resume_mode: Optional[str] = typer.Option("tailored", "--resume-mode", help="Resume source: tailored, default, or path to .pdf/.txt."),
    gen_prompt: bool = typer.Option(False, "--gen-prompt", help="Generate a Claude prompt file for this job."),
) -> None:
    """Print the materials needed to apply manually."""
    _bootstrap()
    from applypilot import config
    from applypilot.database import get_connection, resolve_job_id

    job = resolve_job_id(get_connection(), job_id)
    if not job:
        console.print(f"[red]No matching job found:[/red] {job_id}")
        raise typer.Exit(code=1)

    effective_resume_mode = (resume_mode or "tailored").strip()
    resume_path = ""
    if effective_resume_mode == "tailored":
        resume_path = job.get("tailored_resume_path") or ""
    elif effective_resume_mode == "default":
        resume_path = str(config.RESUME_PDF_PATH if config.RESUME_PDF_PATH.exists() else config.RESUME_PATH)
    else:
        resume_path = str(Path(effective_resume_mode).expanduser())

    cover_path = job.get("cover_letter_path") or ""
    cover_pdf = str(Path(cover_path).with_suffix(".pdf")) if cover_path and Path(cover_path).with_suffix(".pdf").exists() else ""
    prompt_path = ""
    if gen_prompt:
        from applypilot.apply.launcher import gen_prompt as do_gen_prompt
        prompt = do_gen_prompt(job["url"], resume_mode=effective_resume_mode)
        if not prompt:
            console.print("[red]Could not generate prompt for this job.[/red]")
            raise typer.Exit(code=1)
        prompt_path = str(prompt)

    table = Table(title="Manual Application Packet", show_header=True, header_style="bold cyan")
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row("title", job.get("title") or "")
    table.add_row("site", job.get("site") or "")
    table.add_row("score", "" if job.get("fit_score") is None else str(job.get("fit_score")))
    table.add_row("job_url", job.get("url") or "")
    table.add_row("application_url", job.get("application_url") or job.get("url") or "")
    table.add_row("resume_path", resume_path)
    table.add_row("cover_letter_path", cover_path)
    table.add_row("cover_letter_pdf", cover_pdf)
    table.add_row("prompt_path", prompt_path)
    console.print(table)


@manual_app.command("mark")
def manual_mark(
    job_id: str = typer.Argument(..., help="Job URL/application URL."),
    status: str = typer.Option(..., "--status", help="Status: applied, failed, or manual."),
    reason: Optional[str] = typer.Option(None, "--reason", help="Reason for failed/manual status."),
) -> None:
    """Mark a manual application outcome."""
    _bootstrap()
    from applypilot.database import get_connection, mark_job_status

    url = _resolve_one_job_url(job_id)
    try:
        count = mark_job_status(get_connection(), url, status, reason=reason)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    if count == 0:
        console.print(f"[red]No matching job found:[/red] {job_id}")
        raise typer.Exit(code=1)
    console.print(f"[green]Marked {url} as {status}.[/green]")


@manual_app.command("reset")
def manual_reset(
    failed: bool = typer.Option(False, "--failed", help="Reset failed jobs."),
    manual: bool = typer.Option(False, "--manual", help="Reset manual jobs."),
    in_progress: bool = typer.Option(False, "--in-progress", help="Reset in-progress locks."),
) -> None:
    """Reset selected application statuses for retry."""
    _bootstrap()
    if not (failed or manual or in_progress):
        console.print("[red]Choose at least one reset flag: --failed, --manual, or --in-progress.[/red]")
        raise typer.Exit(code=1)
    from applypilot.database import get_connection, reset_manual_statuses

    count = reset_manual_statuses(get_connection(), failed=failed, manual=manual, in_progress=in_progress)
    console.print(f"[green]Reset {count} job(s).[/green]")


@app.command()
def discover(
    force: bool = typer.Option(False, "--force", help="Accepted for symmetry; discovery uses search config."),
    workers: int = typer.Option(1, "--workers", "-w", help="Parallel threads for discovery sub-sources."),
) -> None:
    """Run job discovery."""
    _run_stage_alias("discover", workers=workers, force=force)


@app.command()
def enrich(
    job_ids: Optional[list[str]] = typer.Argument(None, help="Specific job URLs/application URLs to enrich."),
    force: bool = typer.Option(False, "--force", help="Clear enrichment output before rerunning selected jobs."),
    workers: int = typer.Option(1, "--workers", "-w", help="Parallel threads for enrichment."),
) -> None:
    """Enrich pending or selected jobs."""
    _run_stage_alias("enrich", job_ids=job_ids, workers=workers, force=force)


@app.command()
def score(
    job_ids: Optional[list[str]] = typer.Argument(None, help="Specific job URLs/application URLs to score."),
    force: bool = typer.Option(False, "--force", help="Clear score output before rerunning selected jobs."),
) -> None:
    """Score pending or selected jobs."""
    _run_stage_alias("score", job_ids=job_ids, force=force)


@app.command()
def tailor(
    job_ids: Optional[list[str]] = typer.Argument(None, help="Specific job URLs/application URLs to tailor."),
    force: bool = typer.Option(False, "--force", help="Clear tailored resume output before rerunning selected jobs."),
    min_score: int = typer.Option(7, "--min-score", help="Minimum fit score."),
    validation: str = typer.Option("normal", "--validation", help="Validation mode: strict, normal, or lenient."),
) -> None:
    """Tailor resumes for pending or selected jobs."""
    _run_stage_alias("tailor", job_ids=job_ids, min_score=min_score, validation=validation, force=force)


@app.command()
def cover(
    job_ids: Optional[list[str]] = typer.Argument(None, help="Specific job URLs/application URLs for cover letters."),
    force: bool = typer.Option(False, "--force", help="Clear cover letter output before rerunning selected jobs."),
    min_score: int = typer.Option(7, "--min-score", help="Minimum fit score."),
    validation: str = typer.Option("normal", "--validation", help="Validation mode: strict, normal, or lenient."),
) -> None:
    """Generate cover letters for pending or selected jobs."""
    _run_stage_alias("cover", job_ids=job_ids, min_score=min_score, validation=validation, force=force)


@app.command()
def status() -> None:
    """Show pipeline statistics from the database."""
    _bootstrap()

    from applypilot.database import get_stats

    stats = get_stats()

    console.print("\n[bold]ApplyPilot Pipeline Status[/bold]\n")

    # Summary table
    summary = Table(title="Pipeline Overview", show_header=True, header_style="bold cyan")
    summary.add_column("Metric", style="bold")
    summary.add_column("Count", justify="right")

    summary.add_row("Total jobs discovered", str(stats["total"]))
    summary.add_row("With full description", str(stats["with_description"]))
    summary.add_row("Pending enrichment", str(stats["pending_detail"]))
    summary.add_row("Enrichment errors", str(stats["detail_errors"]))
    summary.add_row("Scored by LLM", str(stats["scored"]))
    summary.add_row("Pending scoring", str(stats["unscored"]))
    summary.add_row("Tailored resumes", str(stats["tailored"]))
    summary.add_row("Pending tailoring (7+)", str(stats["untailored_eligible"]))
    summary.add_row("Cover letters", str(stats["with_cover_letter"]))
    summary.add_row("Ready to apply", str(stats["ready_to_apply"]))
    summary.add_row("Applied", str(stats["applied"]))
    summary.add_row("Apply errors", str(stats["apply_errors"]))

    console.print(summary)

    # Score distribution
    if stats["score_distribution"]:
        dist_table = Table(title="\nScore Distribution", show_header=True, header_style="bold yellow")
        dist_table.add_column("Score", justify="center")
        dist_table.add_column("Count", justify="right")
        dist_table.add_column("Bar")

        max_count = max(count for _, count in stats["score_distribution"]) or 1
        for score, count in stats["score_distribution"]:
            bar_len = int(count / max_count * 30)
            if score >= 7:
                color = "green"
            elif score >= 5:
                color = "yellow"
            else:
                color = "red"
            bar = f"[{color}]{'=' * bar_len}[/{color}]"
            dist_table.add_row(str(score), str(count), bar)

        console.print(dist_table)

    # By site
    if stats["by_site"]:
        site_table = Table(title="\nJobs by Source", show_header=True, header_style="bold magenta")
        site_table.add_column("Site")
        site_table.add_column("Count", justify="right")

        for site, count in stats["by_site"]:
            site_table.add_row(site or "Unknown", str(count))

        console.print(site_table)

    console.print()


@app.command()
def dashboard() -> None:
    """Generate and open the HTML dashboard in your browser."""
    _bootstrap()

    from applypilot.view import open_dashboard

    open_dashboard()


@app.command()
def doctor(
    security: bool = typer.Option(False, "--security", help="Include privacy and security hardening checks."),
) -> None:
    """Check your setup and diagnose missing requirements."""
    import shutil
    import sys
    from pathlib import Path
    from applypilot.config import (
        load_env, PROFILE_PATH, RESUME_PATH, RESUME_PDF_PATH,
        SEARCH_CONFIG_PATH, ENV_PATH, get_chrome_path,
    )

    load_env()

    ok_mark = "[green]OK[/green]"
    fail_mark = "[red]MISSING[/red]"
    warn_mark = "[yellow]WARN[/yellow]"

    results: list[tuple[str, str, str]] = []  # (check, status, note)

    # --- Tier 1 checks ---
    # Profile
    if PROFILE_PATH.exists():
        results.append(("profile.json", ok_mark, str(PROFILE_PATH)))
    else:
        results.append(("profile.json", fail_mark, "Run 'applypilot init' to create"))

    # Resume
    if RESUME_PATH.exists():
        results.append(("resume.txt", ok_mark, str(RESUME_PATH)))
    elif RESUME_PDF_PATH.exists():
        results.append(("resume.txt", warn_mark, "Only PDF found — plain-text needed for AI stages"))
    else:
        results.append(("resume.txt", fail_mark, "Run 'applypilot init' to add your resume"))

    # Search config
    if SEARCH_CONFIG_PATH.exists():
        results.append(("searches.yaml", ok_mark, str(SEARCH_CONFIG_PATH)))
    else:
        results.append(("searches.yaml", fail_mark, "Run 'applypilot init' to create"))

    # jobspy (discovery dep installed separately)
    try:
        import jobspy  # noqa: F401
        results.append(("python-jobspy", ok_mark, "Job board scraping available"))
    except ImportError:
        results.append(("python-jobspy", warn_mark,
                        "pip install --no-deps python-jobspy && pip install pydantic tls-client requests markdownify regex"))

    # --- Tier 2 checks ---
    import os
    has_gemini = bool(os.environ.get("GEMINI_API_KEY"))
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    has_local = bool(os.environ.get("LLM_URL"))
    if has_gemini:
        model = os.environ.get("LLM_MODEL", "gemini-2.0-flash")
        results.append(("LLM API key", ok_mark, f"Gemini ({model})"))
    elif has_openai:
        model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
        results.append(("LLM API key", ok_mark, f"OpenAI ({model})"))
    elif has_local:
        results.append(("LLM API key", ok_mark, f"Local: {os.environ.get('LLM_URL')}"))
    else:
        results.append(("LLM API key", fail_mark,
                        "Set GEMINI_API_KEY in ~/.applypilot/.env (run 'applypilot init')"))

    # --- Tier 3 checks ---
    # Claude Code CLI
    claude_bin = shutil.which("claude")
    if claude_bin:
        results.append(("Claude Code CLI", ok_mark, claude_bin))
    else:
        results.append(("Claude Code CLI", fail_mark,
                        "Install from https://claude.ai/code (needed for auto-apply)"))

    # Chrome
    try:
        chrome_path = get_chrome_path()
        results.append(("Chrome/Chromium", ok_mark, chrome_path))
    except FileNotFoundError:
        results.append(("Chrome/Chromium", fail_mark,
                        "Install Chrome or set CHROME_PATH env var (needed for auto-apply)"))

    # Node.js / npx (for Playwright MCP)
    npx_bin = shutil.which("npx")
    if npx_bin:
        results.append(("Node.js (npx)", ok_mark, npx_bin))
    else:
        results.append(("Node.js (npx)", fail_mark,
                        "Install Node.js 18+ from nodejs.org (needed for auto-apply)"))

    # CapSolver (optional)
    capsolver = os.environ.get("CAPSOLVER_API_KEY")
    if capsolver:
        results.append(("CapSolver API key", ok_mark, "CAPTCHA solving enabled"))
    else:
        results.append(("CapSolver API key", "[dim]optional[/dim]",
                        "Set CAPSOLVER_API_KEY in .env for CAPTCHA solving"))

    # --- Render results ---
    console.print()
    console.print("[bold]ApplyPilot Doctor[/bold]\n")

    col_w = max(len(r[0]) for r in results) + 2
    for check, status, note in results:
        pad = " " * (col_w - len(check))
        console.print(f"  {check}{pad}{status}  [dim]{note}[/dim]")

    console.print()

    # Tier summary
    from applypilot.config import get_tier, TIER_LABELS
    tier = get_tier()
    console.print(f"[bold]Current tier: Tier {tier} — {TIER_LABELS[tier]}[/bold]")

    if tier == 1:
        console.print("[dim]  → Tier 2 unlocks: scoring, tailoring, cover letters (needs LLM API key)[/dim]")
        console.print("[dim]  → Tier 3 unlocks: auto-apply (needs Claude Code CLI + Chrome + Node.js)[/dim]")
    elif tier == 2:
        console.print("[dim]  → Tier 3 unlocks: auto-apply (needs Claude Code CLI + Chrome + Node.js)[/dim]")

    if security:
        from applypilot import config as cfg
        from applypilot.apply.launcher import _make_mcp_config

        sec_rows: list[tuple[str, str, str]] = []
        sec_rows.append(("Privacy mode", ok_mark if cfg.is_strict_privacy() else warn_mark, cfg.privacy_mode()))
        sec_rows.append((
            "Cloud LLM opt-in",
            ok_mark if cfg.cloud_llm_allowed() else warn_mark,
            "enabled" if cfg.cloud_llm_allowed() else "disabled; Gemini/OpenAI/Claude fail closed in strict mode",
        ))
        sec_rows.append((
            "Gmail MCP",
            warn_mark if cfg.gmail_mcp_enabled() else ok_mark,
            (
                "enabled despite vulnerable transitive deps"
                if cfg.gmail_mcp_enabled()
                else (
                    "requested but blocked; set APPLYPILOT_ALLOW_VULNERABLE_GMAIL_MCP=1 to override"
                    if cfg.gmail_mcp_requested()
                    else "disabled"
                )
            ),
        ))
        sec_rows.append((
            "CapSolver",
            warn_mark if cfg.capsolver_enabled() else ok_mark,
            "enabled" if cfg.capsolver_enabled() else "disabled",
        ))
        sec_rows.append((
            "Chrome profile cloning",
            warn_mark if cfg.clone_chrome_profile_enabled() else ok_mark,
            "enabled" if cfg.clone_chrome_profile_enabled() else "disabled; clean worker profiles",
        ))

        for label, path in (
            ("profile.json mode", PROFILE_PATH),
            ("resume.txt mode", RESUME_PATH),
            (".env mode", ENV_PATH),
            ("searches.yaml mode", SEARCH_CONFIG_PATH),
        ):
            mode = cfg.secure_file_mode(path)
            status = ok_mark if mode in ("0o600", "missing") else warn_mark
            sec_rows.append((label, status, mode))

        mcp = _make_mcp_config(9222)
        pw_args = mcp["mcpServers"]["playwright"]["args"]
        pinned = any("@latest" not in arg and "@playwright/mcp@" in arg for arg in pw_args)
        sec_rows.append(("Playwright MCP pin", ok_mark if pinned else warn_mark, " ".join(pw_args)))
        sec_rows.append((
            "Gmail MCP config",
            ok_mark if "gmail" not in mcp["mcpServers"] else warn_mark,
            "absent by default" if "gmail" not in mcp["mcpServers"] else "present",
        ))

        scanner_bins = ["bandit", "semgrep", "pip-audit", "detect-secrets", "ruff", "npm"]
        for name in scanner_bins:
            venv_candidate = Path(sys.executable).with_name(name)
            found = shutil.which(name) or (str(venv_candidate) if venv_candidate.exists() else None)
            sec_rows.append((f"scanner: {name}", ok_mark if found else warn_mark, found or "not installed"))

        console.print("\n[bold]Security Checks[/bold]\n")
        sec_col_w = max(len(r[0]) for r in sec_rows) + 2
        for check, status, note in sec_rows:
            pad = " " * (sec_col_w - len(check))
            console.print(f"  {check}{pad}{status}  [dim]{note}[/dim]")

    console.print()


if __name__ == "__main__":
    app()
