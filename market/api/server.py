"""
market/api/server.py — Prediction API server.

Serves model predictions and market signals via FastAPI.
Loads the best trained model checkpoint and provides:
  - /predict: probability estimates for market outcomes
  - /signals: mispriced market detection (model vs current price)
  - /health: service and data freshness checks
  - /metrics: experiment history from results.tsv

Usage:
    uv run uvicorn market.api.server:app --host 0.0.0.0 --port 8000
    uv run uvicorn market.api.server:app --reload  # dev mode
"""

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from market.config import DATA_DIR

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Prediction Autoresearch API",
    description="ML-powered prediction market analysis and signals",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_TSV = PROJECT_ROOT / "results.tsv"


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str
    data_freshness: dict
    model_loaded: bool
    uptime_seconds: float


class MarketSignal(BaseModel):
    ticker: str
    platform: str
    current_price: Optional[float] = None
    model_estimate: Optional[float] = None
    edge: Optional[float] = None  # model_estimate - current_price
    confidence: Optional[float] = None
    last_trade_time: Optional[str] = None


class SignalsResponse(BaseModel):
    generated_at: str
    count: int
    signals: list[MarketSignal]


class MetricsResponse(BaseModel):
    total_experiments: int
    best_val_bpb: Optional[float] = None
    latest_val_bpb: Optional[float] = None
    improvement_pct: Optional[float] = None
    history: list[dict]


class DataSummary(BaseModel):
    platform: str
    dataset: str
    record_count: int
    file_count: int
    total_size_mb: float
    newest_file: Optional[str] = None


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

START_TIME = time.time()


def _get_db():
    """Return an in-memory DuckDB connection."""
    return duckdb.connect(":memory:")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health():
    """Service health and data freshness check."""
    freshness = {}
    data_dir = Path(DATA_DIR)

    for platform in ["kalshi", "polymarket"]:
        for dataset in ["markets", "trades"]:
            dir_path = data_dir / platform / dataset
            if not dir_path.exists():
                freshness[f"{platform}/{dataset}"] = "missing"
                continue
            parquets = sorted(dir_path.glob("*.parquet"))
            if not parquets:
                freshness[f"{platform}/{dataset}"] = "empty"
                continue
            newest = max(p.stat().st_mtime for p in parquets)
            age_hours = (time.time() - newest) / 3600
            freshness[f"{platform}/{dataset}"] = {
                "files": len(parquets),
                "age_hours": round(age_hours, 1),
                "stale": age_hours > 24,
            }

    return HealthResponse(
        status="ok",
        data_freshness=freshness,
        model_loaded=False,  # TODO: update when model loading is implemented
        uptime_seconds=round(time.time() - START_TIME, 1),
    )


@app.get("/data", response_model=list[DataSummary])
def data_summary():
    """Summary of all collected datasets."""
    summaries = []
    data_dir = Path(DATA_DIR)

    for platform in ["kalshi", "polymarket"]:
        for dataset in ["markets", "trades"]:
            dir_path = data_dir / platform / dataset
            if not dir_path.exists():
                continue
            parquets = list(dir_path.glob("*.parquet"))
            if not parquets:
                continue

            total_size = sum(p.stat().st_size for p in parquets)
            newest = max(parquets, key=lambda p: p.stat().st_mtime)

            # Count records via DuckDB
            try:
                con = _get_db()
                count = con.execute(
                    f"SELECT COUNT(*) FROM read_parquet('{dir_path}/*.parquet')"
                ).fetchone()[0]
                con.close()
            except Exception:
                count = 0

            summaries.append(
                DataSummary(
                    platform=platform,
                    dataset=dataset,
                    record_count=count,
                    file_count=len(parquets),
                    total_size_mb=round(total_size / (1024 * 1024), 1),
                    newest_file=newest.name,
                )
            )

    return summaries


@app.get("/metrics", response_model=MetricsResponse)
def metrics():
    """Experiment history and performance metrics."""
    if not RESULTS_TSV.exists():
        return MetricsResponse(
            total_experiments=0,
            history=[],
        )

    rows = []
    with open(RESULTS_TSV) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                try:
                    rows.append(
                        {
                            "date": parts[0],
                            "commit": parts[1],
                            "val_bpb": float(parts[2]) if parts[2] else None,
                            "peak_mem_mb": float(parts[3]) if parts[3] else None,
                            "note": parts[4] if len(parts) > 4 else "",
                        }
                    )
                except (ValueError, IndexError):
                    continue

    if not rows:
        return MetricsResponse(total_experiments=0, history=[])

    valid_bpb = [r["val_bpb"] for r in rows if r["val_bpb"] is not None]
    best = min(valid_bpb) if valid_bpb else None
    latest = valid_bpb[-1] if valid_bpb else None
    first = valid_bpb[0] if valid_bpb else None

    improvement = None
    if first and latest and first > 0:
        improvement = round((first - latest) / first * 100, 2)

    return MetricsResponse(
        total_experiments=len(rows),
        best_val_bpb=best,
        latest_val_bpb=latest,
        improvement_pct=improvement,
        history=rows,
    )


@app.get("/signals", response_model=SignalsResponse)
def signals(
    platform: str = Query("polymarket", enum=["kalshi", "polymarket"]),
    min_edge: float = Query(0.05, description="Minimum price edge to report"),
    limit: int = Query(50, le=500),
):
    """
    Detect potentially mispriced markets.

    Compares model probability estimates against current market prices.
    Returns markets where the model disagrees with the market by at least min_edge.

    NOTE: Model inference not yet implemented — currently returns
    volume-weighted signals as a placeholder for the real model output.
    """
    data_dir = Path(DATA_DIR)
    markets_dir = data_dir / platform / "markets"
    trades_dir = data_dir / platform / "trades"

    if not markets_dir.exists():
        raise HTTPException(404, f"No {platform} market data found")

    try:
        con = _get_db()

        # Get markets with recent trade activity
        if trades_dir.exists() and list(trades_dir.glob("*.parquet")):
            df = con.execute(f"""
                WITH market_prices AS (
                    SELECT
                        ticker,
                        AVG(yes_price) as avg_price,
                        COUNT(*) as trade_count,
                        MAX(created_time) as last_trade
                    FROM read_parquet('{trades_dir}/*.parquet')
                    WHERE ticker IS NOT NULL
                    GROUP BY ticker
                    HAVING trade_count >= 10
                    ORDER BY trade_count DESC
                    LIMIT {limit * 2}
                )
                SELECT * FROM market_prices
            """).fetchdf()
        else:
            con.close()
            return SignalsResponse(
                generated_at=datetime.now(timezone.utc).isoformat(),
                count=0,
                signals=[],
            )

        con.close()
    except Exception as e:
        raise HTTPException(500, f"Query error: {str(e)}")

    # TODO: Replace with actual model inference
    # For now, generate placeholder signals using price deviation from 50%
    signals_list = []
    for _, row in df.iterrows():
        current = row["avg_price"] / 100.0 if row["avg_price"] > 1 else row["avg_price"]
        # Placeholder: model estimate = regression toward 50% based on volume
        model_est = current  # Will be replaced with real model output
        edge = abs(model_est - 0.5) - abs(current - 0.5)

        if abs(edge) >= min_edge:
            signals_list.append(
                MarketSignal(
                    ticker=row["ticker"],
                    platform=platform,
                    current_price=round(current, 4),
                    model_estimate=round(model_est, 4),
                    edge=round(edge, 4),
                    confidence=None,  # TODO: model confidence score
                    last_trade_time=str(row["last_trade"])
                    if pd.notna(row["last_trade"])
                    else None,
                )
            )

    signals_list.sort(key=lambda s: abs(s.edge or 0), reverse=True)

    return SignalsResponse(
        generated_at=datetime.now(timezone.utc).isoformat(),
        count=len(signals_list[:limit]),
        signals=signals_list[:limit],
    )


@app.get("/markets/{platform}/search")
def search_markets(
    platform: str,
    q: str = Query(..., min_length=2, description="Search query"),
    limit: int = Query(20, le=100),
):
    """Search markets by text query."""
    data_dir = Path(DATA_DIR)
    markets_dir = data_dir / platform / "markets"

    if not markets_dir.exists():
        raise HTTPException(404, f"No {platform} market data found")

    try:
        con = _get_db()
        # Search across available text columns
        df = con.execute(f"""
            SELECT *
            FROM read_parquet('{markets_dir}/*.parquet')
            WHERE CAST(ticker AS VARCHAR) ILIKE '%{q}%'
               OR CAST(ticker AS VARCHAR) ILIKE '%{q}%'
            LIMIT {limit}
        """).fetchdf()
        con.close()
    except Exception as e:
        raise HTTPException(500, f"Query error: {str(e)}")

    return {
        "query": q,
        "count": len(df),
        "results": df.to_dict(orient="records"),
    }
