from __future__ import annotations

import argparse
import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from stress_analytics.repository import ArtifactRepository


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "_stress_test_data"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "_outputs" / "stress_analytics"

templates = Jinja2Templates(directory=PACKAGE_ROOT / "templates")
repository: ArtifactRepository | None = None
refresh_result: dict[str, int] = {"indexed": 0, "reused": 0, "failed": 0}


def _configured_path(environment_name: str, default: Path) -> Path:
    return Path(os.environ.get(environment_name, str(default))).expanduser()


def _repository() -> ArtifactRepository:
    if repository is None:
        raise HTTPException(status_code=503, detail="Analytics index is not ready")
    return repository


@asynccontextmanager
async def lifespan(_: FastAPI):
    global repository, refresh_result
    repository = ArtifactRepository(
        _configured_path("STRESS_ANALYTICS_DATA_DIR", DEFAULT_DATA_DIR),
        _configured_path("STRESS_ANALYTICS_CACHE_DIR", DEFAULT_CACHE_DIR),
    )
    refresh_result = await asyncio.to_thread(repository.refresh)
    yield


app = FastAPI(title="RLHF Signal Room", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=PACKAGE_ROOT / "static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {"data_dir": str(_configured_path("STRESS_ANALYTICS_DATA_DIR", DEFAULT_DATA_DIR))},
    )


@app.get("/api/status")
async def status():
    return {
        "ready": repository is not None,
        "data_dir": str(_configured_path("STRESS_ANALYTICS_DATA_DIR", DEFAULT_DATA_DIR)),
        "refresh": refresh_result,
    }


@app.post("/api/refresh")
async def refresh():
    global refresh_result
    refresh_result = await asyncio.to_thread(_repository().refresh)
    return refresh_result


@app.get("/api/runs")
async def runs():
    return await asyncio.to_thread(_repository().list_runs)


@app.get("/api/runs/{artifact_id}/overview")
async def overview(artifact_id: int):
    try:
        return await asyncio.to_thread(_repository().overview, artifact_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Artifact not found") from None


@app.get("/api/runs/{artifact_id}/timeline")
async def timeline(
    artifact_id: int,
    start: float | None = None,
    end: float | None = None,
    points: int = Query(default=900, ge=100, le=2000),
):
    try:
        return await asyncio.to_thread(
            _repository().timeline,
            artifact_id,
            start,
            end,
            points,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Artifact not found") from None


@app.get("/api/runs/{artifact_id}/routes")
async def routes(
    artifact_id: int,
    start: float | None = None,
    end: float | None = None,
):
    try:
        return await asyncio.to_thread(
            _repository().route_summary,
            artifact_id,
            start,
            end,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Artifact not found") from None


@app.get("/api/runs/{artifact_id}/issues")
async def issues(artifact_id: int):
    try:
        return await asyncio.to_thread(_repository().issue_events, artifact_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Artifact not found") from None


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore downloaded RLHF diagnostics")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    os.environ["STRESS_ANALYTICS_DATA_DIR"] = str(args.data_dir.resolve())
    os.environ["STRESS_ANALYTICS_CACHE_DIR"] = str(args.cache_dir.resolve())
    print(f"RLHF Signal Room: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()