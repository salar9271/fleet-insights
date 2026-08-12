"""Generate the LLM-authored fleet driving-safety brief from the current
KMeans cluster profile and write reports/fleet_brief.json + fleet_brief.md.

Usage:
    .venv/bin/python scripts/generate_report.py           # calls the API
    .venv/bin/python scripts/generate_report.py --mock    # uses the stored example

Falls back to the stored example at reports/fleet_brief_example.json
automatically if ANTHROPIC_API_KEY is not set, so the repo stays runnable
for anyone who clones it without a key.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.clustering import run_clustering
from src.genai_report import GenAIReportError, generate_report, load_api_key

REPORTS_DIR = Path("reports")
MOCK_EXAMPLE_PATH = REPORTS_DIR / "fleet_brief_example.json"
JSON_OUT_PATH = REPORTS_DIR / "fleet_brief.json"
MD_OUT_PATH = REPORTS_DIR / "fleet_brief.md"


def render_markdown(report):
    lines = [
        "# Fleet Driving-Safety Brief",
        "",
        report["summary"],
        "",
        f"**Higher-risk cluster:** {report['risk_cluster']}",
        "",
        "## Evidence",
    ]
    lines += [f"- {item}" for item in report["evidence"]]
    lines += [
        "",
        "## Recommendation",
        report["recommendation"],
        "",
        "## Caveats",
    ]
    lines += [f"- {item}" for item in report["caveats"]]
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mock", action="store_true",
        help="skip the API call and use the stored example report",
    )
    args = parser.parse_args()

    api_key = load_api_key()
    use_mock = args.mock or not api_key

    if use_mock:
        if args.mock:
            print("--mock passed: skipping the Anthropic API call.")
        else:
            print("NOTICE: ANTHROPIC_API_KEY not found (checked .env and the "
                  "environment) -- skipping the Anthropic API call.")
        print(f"Using stored example report: {MOCK_EXAMPLE_PATH}")
        with open(MOCK_EXAMPLE_PATH) as f:
            report = json.load(f)
    else:
        print(f"Calling the Anthropic API (model claude-sonnet-4-6)...")
        results = run_clustering()
        cluster_profile = results["profile"].drop(columns=["size"])
        cluster_sizes = results["profile"]["size"]
        test_ari = results["test_ari"]
        try:
            report = generate_report(cluster_profile, cluster_sizes, test_ari, api_key)
        except GenAIReportError as e:
            print(f"ERROR generating report: {e}", file=sys.stderr)
            sys.exit(1)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(JSON_OUT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    MD_OUT_PATH.write_text(render_markdown(report))

    print(f"\nSaved {JSON_OUT_PATH}")
    print(f"Saved {MD_OUT_PATH}")


if __name__ == "__main__":
    main()
