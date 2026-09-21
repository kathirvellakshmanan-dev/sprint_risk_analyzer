# sprint_risk_analyzer
Pulls active sprint data from Jira, computes velocity/burndown risk metrics deterministically (no LLM math), then sends the computed metrics + ticket context to Claude for a structured risk assessment.
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
