#!/usr/bin/env python3
"""
sprint_risk_analyzer.py

Pulls active sprint data from Jira, computes velocity/burndown risk metrics
deterministically (no LLM math), then sends the computed metrics + ticket
context to Claude for a structured risk assessment.

SETUP
-----
1. pip install requests anthropic python-dotenv

2. Create a .env file (or export these as env vars):
       JIRA_BASE_URL=https://yourcompany.atlassian.net
       JIRA_EMAIL=you@yourcompany.com
       JIRA_API_TOKEN=xxxxx              # https://id.atlassian.com/manage-profile/security/api-tokens
       JIRA_BOARD_ID=123                 # find via /rest/agile/1.0/board
       ANTHROPIC_API_KEY=sk-ant-xxxxx

3. Run:
       python sprint_risk_analyzer.py

   Optional flags:
       --sprint-id 456          # analyze a specific sprint instead of the active one
       --history 4              # number of past sprints to use for velocity trend (default 4)
       --output report.md       # write the report to a file instead of stdout
       --dry-run                # print the computed metrics + prompt, skip the Claude call

NOTES
-----
- Uses a READ-ONLY Jira API token. Do not use a token with write scopes for this.
- All velocity/burndown arithmetic happens in Python (compute_risk_signals),
  never in the LLM call — the model only interprets numbers it's given.
- Designed to be run on a schedule (cron / Step Functions / Airflow) every
  1-2 days mid-sprint. Running it constantly is noise; running it once at
  sprint end is too late to act on.
"""

import argparse
import os
import sys
import json
from datetime import datetime, timezone
from typing import Any

import requests
from anthropic import Anthropic

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # fine if python-dotenv isn't installed; env vars can be set directly


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

JIRA_BASE_URL = os.environ.get("JIRA_BASE_URL", "").rstrip("/")
JIRA_EMAIL = os.environ.get("JIRA_EMAIL", "")
JIRA_API_TOKEN = os.environ.get("JIRA_API_TOKEN", "")
JIRA_BOARD_ID = os.environ.get("JIRA_BOARD_ID", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

REQUIRED_VARS = {
    "JIRA_BASE_URL": JIRA_BASE_URL,
    "JIRA_EMAIL": JIRA_EMAIL,
    "JIRA_API_TOKEN": JIRA_API_TOKEN,
    "JIRA_BOARD_ID": JIRA_BOARD_ID,
}


def check_config(require_anthropic: bool = True) -> None:
    missing = [k for k, v in REQUIRED_VARS.items() if not v]
    if require_anthropic and not ANTHROPIC_API_KEY:
        missing.append("ANTHROPIC_API_KEY")
    if missing:
        print(f"Missing required environment variables: {', '.join(missing)}", file=sys.stderr)
        print("Set them in a .env file or export them before running.", file=sys.stderr)
        sys.exit(1)


# --------------------------------------------------------------------------
# Jira data pull
# --------------------------------------------------------------------------

def jira_session() -> requests.Session:
    s = requests.Session()
    s.auth = (JIRA_EMAIL, JIRA_API_TOKEN)
    s.headers.update({"Accept": "application/json"})
    return s


def get_active_sprint(session: requests.Session, board_id: str) -> dict:
    """Return the currently active sprint for the given board."""
    url = f"{JIRA_BASE_URL}/rest/agile/1.0/board/{board_id}/sprint"
    resp = session.get(url, params={"state": "active"})
    resp.raise_for_status()
    sprints = resp.json().get("values", [])
    if not sprints:
        raise RuntimeError("No active sprint found on this board.")
    return sprints[0]


def get_sprint_by_id(session: requests.Session, sprint_id: str) -> dict:
    url = f"{JIRA_BASE_URL}/rest/agile/1.0/sprint/{sprint_id}"
    resp = session.get(url)
    resp.raise_for_status()
    return resp.json()


def get_sprint_issues(session: requests.Session, sprint_id: str) -> list[dict]:
    """Pull all issues in a sprint with the fields needed for risk analysis."""
    url = f"{JIRA_BASE_URL}/rest/agile/1.0/sprint/{sprint_id}/issue"
    fields = "summary,status,assignee,customfield_10016,issuelinks,updated,created"
    # NOTE: customfield_10016 is the default Jira Cloud "Story Points" field ID.
    # This varies per instance — check yours via:
    #   GET /rest/api/3/field  and search for "Story Points"
    issues = []
    start_at = 0
    while True:
        resp = session.get(url, params={
            "fields": fields,
            "startAt": start_at,
            "maxResults": 100,
        })
        resp.raise_for_status()
        data = resp.json()
        issues.extend(data.get("issues", []))
        if start_at + 100 >= data.get("total", 0):
            break
        start_at += 100
    return issues


def get_past_sprints(session: requests.Session, board_id: str, count: int) -> list[dict]:
    """Return the N most recently closed sprints for velocity history."""
    url = f"{JIRA_BASE_URL}/rest/agile/1.0/board/{board_id}/sprint"
    resp = session.get(url, params={"state": "closed"})
    resp.raise_for_status()
    sprints = resp.json().get("values", [])
    sprints.sort(key=lambda s: s.get("endDate", ""), reverse=True)
    return sprints[:count]


def get_sprint_velocity(session: requests.Session, sprint: dict) -> tuple[float, float]:
    """Return (committed_points, completed_points) for a closed sprint."""
    issues = get_sprint_issues(session, sprint["id"])
    committed = sum(_story_points(i) for i in issues)
    completed = sum(
        _story_points(i) for i in issues
        if i["fields"]["status"]["statusCategory"]["key"] == "done"
    )
    return committed, completed


def _story_points(issue: dict) -> float:
    pts = issue["fields"].get("customfield_10016")
    return float(pts) if pts is not None else 0.0


def _is_blocked(issue: dict) -> tuple[bool, str]:
    """Detect blocked status via issue links or a 'Blocked' status name."""
    status_name = issue["fields"]["status"]["name"].lower()
    if "block" in status_name:
        return True, f"Status: {issue['fields']['status']['name']}"
    for link in issue["fields"].get("issuelinks", []):
        link_type = link.get("type", {}).get("inward", "") or link.get("type", {}).get("outward", "")
        if "block" in link_type.lower() and "inwardIssue" in link:
            blocker_key = link["inwardIssue"]["key"]
            return True, f"Blocked by {blocker_key}"
    return False, ""


# --------------------------------------------------------------------------
# Deterministic risk math (no LLM involved)
# --------------------------------------------------------------------------

def compute_risk_signals(
    committed_points: float,
    completed_points: float,
    days_elapsed: int,
    days_total: int,
    historical_velocity: list[float],
) -> dict[str, Any]:
    days_elapsed = max(days_elapsed, 1)  # avoid div/0 on day 0
    days_remaining = max(days_total - days_elapsed, 0)

    ideal_burn_rate = committed_points / days_total if days_total else 0
    actual_burn_rate = completed_points / days_elapsed
    projected_completion_points = actual_burn_rate * days_total
    projected_completion_pct = (
        round((projected_completion_points / committed_points) * 100, 1)
        if committed_points else 0.0
    )

    avg_historical_velocity = (
        round(sum(historical_velocity) / len(historical_velocity), 1)
        if historical_velocity else None
    )
    latest_velocity = historical_velocity[0] if historical_velocity else None
    velocity_trend = (
        round(latest_velocity - avg_historical_velocity, 1)
        if avg_historical_velocity is not None and latest_velocity is not None
        else None
    )

    # "Mathematically cannot recover" check: even at best-observed historical
    # daily rate, can remaining points be closed in remaining days?
    best_daily_rate = (
        max(historical_velocity) / days_total if historical_velocity and days_total else actual_burn_rate
    )
    max_recoverable_points = best_daily_rate * days_remaining
    remaining_points = max(committed_points - completed_points, 0)
    mathematically_recoverable = max_recoverable_points >= remaining_points

    return {
        "days_elapsed": days_elapsed,
        "days_total": days_total,
        "days_remaining": days_remaining,
        "committed_points": committed_points,
        "completed_points": completed_points,
        "remaining_points": remaining_points,
        "ideal_burn_rate_per_day": round(ideal_burn_rate, 2),
        "actual_burn_rate_per_day": round(actual_burn_rate, 2),
        "burn_rate_variance": round(actual_burn_rate - ideal_burn_rate, 2),
        "projected_completion_pct": projected_completion_pct,
        "avg_historical_velocity": avg_historical_velocity,
        "velocity_trend_vs_avg": velocity_trend,
        "mathematically_recoverable": mathematically_recoverable,
    }


# --------------------------------------------------------------------------
# Claude call
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a sprint risk analyst for a software delivery program. You will be given:
1. Computed velocity/burndown metrics (already calculated — do not recalculate)
2. Raw ticket-level data (status, blockers, assignees)
3. Historical sprint performance

Your job is to produce a risk assessment, not a status summary. Follow this structure exactly:

1. RISK LEVEL: One of [Low / Medium / High / Critical], with one sentence justifying the level.
2. KEY DRIVERS: The 2-4 specific factors most responsible for the risk level (cite the actual numbers/tickets provided — never invent data not given).
3. WHAT'S DIFFERENT FROM NORMAL: Compare current sprint to historical velocity pattern. Only flag genuine anomalies, not normal sprint-to-sprint variance.
4. BLOCKED/AT-RISK TICKETS: List specific tickets blocking completion, with the blocker reason if available. If none, say so.
5. RECOMMENDED ACTION: 1-3 concrete actions a PM could take today. If risk is Low, it is fine to say "no action needed."

Rules:
- Never state a numeric claim that wasn't in the provided data. If something is missing, say so explicitly rather than estimating.
- The data includes a "mathematically_recoverable" flag — respect it. If false, do not describe the sprint as merely "at risk"; state plainly that the current commitment will not be met without descoping, and frame the recommendation around descope/re-plan rather than "push harder."
- Do not pad the analysis with generic Agile advice. Every sentence should be specific to this sprint's actual data.
- If risk is Low, say so briefly — don't manufacture concern to seem thorough.
"""

USER_PROMPT_TEMPLATE = """Analyze sprint risk for: {sprint_name}

COMPUTED METRICS:
- Days elapsed: {days_elapsed} of {days_total} ({days_remaining} remaining)
- Committed points: {committed_points}
- Completed points: {completed_points}
- Remaining points: {remaining_points}
- Ideal burn rate: {ideal_burn_rate_per_day} pts/day
- Actual burn rate: {actual_burn_rate_per_day} pts/day
- Burn rate variance: {burn_rate_variance} (negative = behind pace)
- Projected completion: {projected_completion_pct}% of commitment
- Mathematically recoverable given remaining days at best historical rate: {mathematically_recoverable}
- Average historical velocity (last {history_count} sprints): {avg_historical_velocity}
- Velocity trend vs. average: {velocity_trend_vs_avg}

BLOCKED / AT-RISK TICKETS:
{blocked_tickets_text}

FULL TICKET LIST ({ticket_count} tickets):
{ticket_list_text}

HISTORICAL VELOCITY (most recent first, last {history_count} sprints):
{historical_velocity_list}

Produce the risk assessment per your instructions.
"""


def build_user_prompt(sprint_name: str, metrics: dict, issues: list[dict], historical_velocity: list[float]) -> str:
    blocked = []
    ticket_lines = []
    for issue in issues:
        key = issue["key"]
        summary = issue["fields"]["summary"]
        status = issue["fields"]["status"]["name"]
        assignee = (issue["fields"].get("assignee") or {}).get("displayName", "Unassigned")
        pts = _story_points(issue)
        ticket_lines.append(f"- {key} [{status}, {pts}pts, {assignee}]: {summary}")

        is_blocked, reason = _is_blocked(issue)
        if is_blocked:
            blocked.append(f"- {key}: {summary} — {reason}")

    return USER_PROMPT_TEMPLATE.format(
        sprint_name=sprint_name,
        history_count=len(historical_velocity),
        blocked_tickets_text="\n".join(blocked) if blocked else "None flagged.",
        ticket_list_text="\n".join(ticket_lines) if ticket_lines else "No tickets found.",
        ticket_count=len(issues),
        historical_velocity_list=", ".join(str(v) for v in historical_velocity) or "No history available.",
        **metrics,
    )


def call_claude(user_prompt: str) -> str:
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=800,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Jira sprint velocity/burndown risk analyzer")
    parser.add_argument("--sprint-id", help="Analyze a specific sprint ID instead of the active sprint")
    parser.add_argument("--history", type=int, default=4, help="Number of past sprints for velocity trend (default 4)")
    parser.add_argument("--output", help="Write report to this file instead of stdout")
    parser.add_argument("--dry-run", action="store_true", help="Print computed metrics + prompt, skip the Claude call")
    args = parser.parse_args()

    check_config(require_anthropic=not args.dry_run)
    session = jira_session()

    # 1. Get target sprint
    sprint = get_sprint_by_id(session, args.sprint_id) if args.sprint_id else get_active_sprint(session, JIRA_BOARD_ID)
    sprint_name = sprint.get("name", "Unknown Sprint")
    start = datetime.fromisoformat(sprint["startDate"].replace("Z", "+00:00"))
    end = datetime.fromisoformat(sprint["endDate"].replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)
    days_total = max((end - start).days, 1)
    days_elapsed = min(max((now - start).days, 0), days_total)

    # 2. Pull current sprint issues
    issues = get_sprint_issues(session, sprint["id"])
    committed_points = sum(_story_points(i) for i in issues)
    completed_points = sum(
        _story_points(i) for i in issues
        if i["fields"]["status"]["statusCategory"]["key"] == "done"
    )

    # 3. Pull historical velocity
    past_sprints = get_past_sprints(session, JIRA_BOARD_ID, args.history)
    historical_velocity = []
    for s in past_sprints:
        _, completed = get_sprint_velocity(session, s)
        historical_velocity.append(completed)

    # 4. Compute risk signals (deterministic)
    metrics = compute_risk_signals(
        committed_points=committed_points,
        completed_points=completed_points,
        days_elapsed=days_elapsed,
        days_total=days_total,
        historical_velocity=historical_velocity,
    )

    # 5. Build prompt
    user_prompt = build_user_prompt(sprint_name, metrics, issues, historical_velocity)

    if args.dry_run:
        print("=== COMPUTED METRICS ===")
        print(json.dumps(metrics, indent=2))
        print("\n=== USER PROMPT ===")
        print(user_prompt)
        return

    # 6. Call Claude
    report = call_claude(user_prompt)

    output_text = f"# Sprint Risk Report: {sprint_name}\n\n_Generated {now.strftime('%Y-%m-%d %H:%M UTC')}_\n\n{report}\n"

    if args.output:
        with open(args.output, "w") as f:
            f.write(output_text)
        print(f"Report written to {args.output}")
    else:
        print(output_text)


if __name__ == "__main__":
    main()
