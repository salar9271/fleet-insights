"""FastAPI service for scoring driving-sensor CSV uploads against the
trained KMeans clustering model and returning a driving-safety brief.

Reuses the existing pipelines unchanged: window features come from
src/features.py (same code used to build the training data), scoring uses
the saved models/scaler.joblib + models/kmeans.joblib (see
src/clustering.py), and the narrative comes from src/genai_report.py.

Endpoints
---------
GET /health
    Liveness check. Returns {"status": "ok"}.

POST /analyse?use_llm=true
    Multipart file upload, field name "file": a CSV with columns
    AccX, AccY, AccZ, GyroX, GyroY, GyroZ, Timestamp (Class optional).
    The upload is segmented into sessions and windowed exactly as at
    training time, each window is scaled and assigned to a cluster with
    the saved scaler + KMeans, and the response is:

        {
          "n_windows": int,
          "window_length": int,
          "class_provided": bool,
          "windows": [
            {"window_index": int, "session_id": str, "cluster": int,
             "true_class": str | null},
            ...
          ],
          "cluster_summary": {"<cluster id>": {"count": int, "share": float}, ...},
          "llm_used": bool,
          "notice": str | null,
          "brief": {...} | null
        }

    Query param `use_llm` (default true): when false, or when
    ANTHROPIC_API_KEY is not configured, the Anthropic call is skipped and
    "brief" is null with "notice" explaining why -- the numeric results are
    still returned. An Anthropic API failure degrades the same way rather
    than failing the request.

    Malformed input (unparseable CSV, missing required columns, non-numeric
    sensor data, or too few contiguous samples to form even one window)
    returns 422 with a JSON body describing the problem.

Run with: scripts/run_api.sh
"""

import io
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from src import features
from src.clustering import get_feature_columns, run_clustering
from src.genai_report import GenAIReportError, generate_report, load_api_key

MODELS_DIR = Path("models")
REQUIRED_COLUMNS = ["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ", "Timestamp"]
NUMERIC_COLUMNS = REQUIRED_COLUMNS

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["scaler"] = joblib.load(MODELS_DIR / "scaler.joblib")
    state["kmeans"] = joblib.load(MODELS_DIR / "kmeans.joblib")

    # Static context for the LLM brief: how each cluster is characterized
    # (feature z-scores) and how well clustering agrees with true labels on
    # held-out data. Refit deterministically (fixed random_state) from the
    # same training data used to produce the saved model -- not derived
    # from any one /analyse request.
    results = run_clustering()
    state["cluster_profile"] = results["profile"].drop(columns=["size"])
    state["test_ari"] = float(results["test_ari"])

    yield
    state.clear()


app = FastAPI(title="Fleet Insights API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


def _load_and_validate_csv(raw_bytes):
    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse file as CSV: {e}")

    if df.empty:
        raise HTTPException(status_code=422, detail="Uploaded CSV has no rows.")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Missing required column(s): {missing}. "
                f"Required: {REQUIRED_COLUMNS} (Class optional)."
            ),
        )

    for col in NUMERIC_COLUMNS:
        coerced = pd.to_numeric(df[col], errors="coerce")
        n_bad = int(coerced.isna().sum())
        if n_bad:
            raise HTTPException(
                status_code=422,
                detail=f"Column '{col}' has {n_bad} non-numeric value(s).",
            )
        df[col] = coerced

    has_class = "Class" in df.columns
    if not has_class:
        # segment_sessions() cuts at class changes; a constant placeholder
        # means it only cuts at timestamp gaps, which is correct when no
        # label was supplied.
        df["Class"] = "UNKNOWN"

    return df, has_class


@app.post("/analyse")
async def analyse(file: UploadFile = File(...), use_llm: bool = Query(True)):
    raw = await file.read()
    df, has_class = _load_and_validate_csv(raw)

    windows = features.make_windows(df, split_name="upload")
    if len(windows) == 0:
        raise HTTPException(
            status_code=422,
            detail=(
                f"No complete windows could be formed -- each session needs "
                f"at least {features.DEFAULT_WINDOW_LENGTH} contiguous rows."
            ),
        )

    feature_cols = get_feature_columns(windows)
    X = state["scaler"].transform(windows[feature_cols])
    cluster_labels = state["kmeans"].predict(X)

    per_window = [
        {
            "window_index": i,
            "session_id": str(windows.iloc[i]["session_id"]),
            "cluster": int(cluster_labels[i]),
            "true_class": str(windows.iloc[i]["class"]) if has_class else None,
        }
        for i in range(len(windows))
    ]

    total = len(cluster_labels)
    upload_cluster_sizes = pd.Series(0, index=state["cluster_profile"].index)
    cluster_summary = {}
    for cluster_id in state["cluster_profile"].index:
        count = int((cluster_labels == cluster_id).sum())
        upload_cluster_sizes[cluster_id] = count
        cluster_summary[str(cluster_id)] = {
            "count": count,
            "share": round(count / total, 4) if total else 0.0,
        }

    brief = None
    llm_used = False
    notice = None
    api_key = load_api_key()

    if not use_llm:
        notice = "use_llm=false -- returning numeric results only."
    elif not api_key:
        notice = "ANTHROPIC_API_KEY not configured -- returning numeric results only."
    else:
        try:
            brief = generate_report(
                state["cluster_profile"], upload_cluster_sizes, state["test_ari"], api_key
            )
            llm_used = True
        except GenAIReportError as e:
            notice = f"LLM brief unavailable: {e}"

    return {
        "n_windows": len(windows),
        "window_length": features.DEFAULT_WINDOW_LENGTH,
        "class_provided": has_class,
        "windows": per_window,
        "cluster_summary": cluster_summary,
        "llm_used": llm_used,
        "notice": notice,
        "brief": brief,
    }
