"""Typer CLI for the capstone: `ops <subcommand>`.

Usage:
    python 05-capstone/cli.py init
    python 05-capstone/cli.py triage
    python 05-capstone/cli.py propose-replies
    python 05-capstone/cli.py schedule "30min coffee with Sara next week"
    python 05-capstone/cli.py send <draft_id>     # respects DRY_RUN
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import typer
from dotenv import load_dotenv

from capstone_agent import run
from integrations import gcal, gmail
from safety import dry_run_enabled

load_dotenv()

app = typer.Typer(help="Personal Ops Assistant CLI")


@app.command()
def init() -> None:
    """Run the OAuth flow for both Gmail and Calendar (browser opens)."""
    typer.echo("Authenticating Gmail...")
    _ = gmail.list_unread(limit=1)
    typer.echo("Authenticating Google Calendar...")
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    _ = gcal.list_events(now.isoformat(), (now + timedelta(days=1)).isoformat())
    typer.echo("OAuth tokens cached. You're ready.")
    _dry_run_banner()


@app.command()
def triage() -> None:
    """Summarize unread inbox without drafting or sending anything."""
    _dry_run_banner()
    output = run("Triage my unread inbox. Summarize which emails need a reply and why. Do NOT draft or send anything.")
    typer.echo(output)


@app.command("propose-replies")
def propose_replies() -> None:
    """Draft replies (saved to Gmail Drafts) for each email that deserves one."""
    _dry_run_banner()
    output = run(
        "Process my unread inbox. For each email that deserves a reply, use CRM "
        "context and propose time slots when relevant, then create a Gmail DRAFT. "
        "Do not send anything."
    )
    typer.echo(output)


@app.command()
def schedule(request: str) -> None:
    """Schedule a meeting based on a natural-language request."""
    _dry_run_banner()
    output = run(f"Scheduling request: {request}")
    typer.echo(output)


@app.command()
def send(draft_id: str) -> None:
    """Promote a draft to a send. Requires DRY_RUN=false AND explicit approval."""
    if dry_run_enabled():
        typer.echo("DRY_RUN is enabled - set CAPSTONE_DRY_RUN=false to actually send.")
        raise typer.Exit(code=0)
    answer = typer.confirm(f"Really send draft {draft_id}?")
    if not answer:
        typer.echo("Aborted.")
        raise typer.Exit(code=1)
    result = gmail.send_draft(draft_id)
    typer.echo(f"Sent: {result}")


def _dry_run_banner() -> None:
    if dry_run_enabled():
        typer.secho(
            "[DRY_RUN] No real emails will be sent and no events will be created.",
            fg=typer.colors.YELLOW,
        )


if __name__ == "__main__":
    app()
