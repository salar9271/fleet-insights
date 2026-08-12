"""Turns the KMeans cluster profile into an LLM-generated driving-safety
brief via the Anthropic API (model claude-sonnet-4-6, as requested).

The model only ever sees already-computed cluster statistics -- feature
z-scores per cluster, cluster sizes, and the held-out test ARI -- never the
raw sensor data or the true class labels, so the clustering itself stays
unsupervised. The dataset's real limitations (see docs/data_selection.md and
reports/clustering_findings.md) are passed in as facts so the model must
ground its caveats in them rather than inventing or omitting its own.
"""

import json
import os

from dotenv import load_dotenv

MODEL = "claude-sonnet-4-6"

REQUIRED_KEYS = ("summary", "risk_cluster", "evidence", "recommendation", "caveats")

KNOWN_LIMITATIONS = (
    "No vehicle or driver identifier exists in this dataset -- clusters cannot "
    "be attributed to a specific car or driver, and no multi-vehicle or "
    "cross-driver generalization claim can be made.",
    "Each behavior class (SLOW / NORMAL / AGGRESSIVE) was recorded as a single "
    "continuous session, not a pool of independent trips, so these results "
    "describe the sessions actually recorded, not a broader population.",
    "Inertial sensors (accelerometer/gyroscope) cannot separate NORMAL from "
    "SLOW driving, because that distinction lives in vehicle speed, which "
    "these sensors do not measure -- not in acceleration or rotation.",
)


class GenAIReportError(Exception):
    """Raised on an Anthropic API failure or unparseable model output."""


def load_api_key(env_path=".env"):
    """Loads .env (if present) and returns ANTHROPIC_API_KEY, or None."""
    load_dotenv(dotenv_path=env_path)
    return os.environ.get("ANTHROPIC_API_KEY") or None


def build_prompt(cluster_profile, cluster_sizes, test_ari):
    table = cluster_profile.copy()
    table.insert(0, "size", cluster_sizes)
    table_text = table.to_string(float_format=lambda v: f"{v:.3f}")

    facts = "\n".join(f"- {fact}" for fact in KNOWN_LIMITATIONS)

    return f"""You are writing a short internal driving-safety brief for a fleet
analytics team. The input below is the output of an UNSUPERVISED KMeans
clustering run over window-level driving-sensor features (accelerometer and
gyroscope statistics). Each row is one cluster; each column (other than
"size") is that cluster's MEAN Z-SCORE for one feature, i.e. how far above
or below the overall average that cluster sits on that feature, in standard
deviations. "size" is the number of training windows assigned to the cluster.

Cluster profile (mean z-score per feature, by cluster):
{table_text}

Units: AccX/Y/Z (and every Acc-prefixed feature above, including
accY_rate_below_neg3 and accY_rate_above_pos3) are in m/s^2, NOT g -- these
are raw accelerometer readings, not g-force multiples. So "accY_rate_below_neg3"
is the fraction of samples with AccY < -3 m/s^2, and "accY_rate_above_pos3" is
the fraction with AccY > +3 m/s^2. When describing these in the brief, say
"m/s^2", never "g" or "g-units".

External validation: on held-out test windows never used for clustering,
comparing cluster assignment to the (withheld-during-fitting) true driving-
behavior label gives an Adjusted Rand Index (ARI) of {test_ari:.4f}.

Known, confirmed limitations of the underlying dataset (do not invent others,
and do not omit these -- restate them, in your own words, as the "caveats"):
{facts}

Task: write a short driving-safety brief. Identify which cluster represents
higher-intensity / higher-risk driving, cite the specific z-score features
that support that call (name the feature and its value), and give one
concrete operational recommendation for a fleet team. Ground every claim in
the numbers above -- do not speculate beyond what the cluster profile and
ARI actually show.

Respond with ONLY a single JSON object (no markdown fences, no commentary
before or after it) with exactly these keys:
- "summary": string, 2-3 sentences, plain-language summary of what the
  clusters represent.
- "risk_cluster": string, which cluster (by its label/id above) represents
  higher-intensity driving.
- "evidence": array of strings, each one a specific z-score feature/value
  from the table above that supports the risk_cluster call.
- "recommendation": string, one concrete operational recommendation.
- "caveats": array of strings, the limitations above, restated -- do not add
  or remove any.
"""


def _extract_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    return json.loads(text)


def generate_report(cluster_profile, cluster_sizes, test_ari, api_key, model=MODEL):
    """Calls the Anthropic API and returns the parsed report dict.

    Raises GenAIReportError on any API failure or unparseable/invalid output.
    """
    import anthropic

    prompt = build_prompt(cluster_profile, cluster_sizes, test_ari)
    client = anthropic.Anthropic(api_key=api_key)

    try:
        response = client.messages.create(
            model=model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.AuthenticationError as e:
        raise GenAIReportError(f"Anthropic authentication failed: {e}") from e
    except anthropic.RateLimitError as e:
        raise GenAIReportError(f"Anthropic rate limit hit: {e}") from e
    except anthropic.APIStatusError as e:
        raise GenAIReportError(f"Anthropic API error ({e.status_code}): {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise GenAIReportError(f"Could not reach the Anthropic API: {e}") from e

    if response.stop_reason == "refusal":
        raise GenAIReportError("Model declined to generate the report (stop_reason=refusal).")

    text = next((b.text for b in response.content if b.type == "text"), "")
    if not text:
        raise GenAIReportError("Model response contained no text content.")

    try:
        report = _extract_json(text)
    except json.JSONDecodeError as e:
        raise GenAIReportError(
            f"Model output was not valid JSON: {e}\nRaw output:\n{text}"
        ) from e

    missing = [k for k in REQUIRED_KEYS if k not in report]
    if missing:
        raise GenAIReportError(f"Model output missing required keys: {missing}")

    return report
