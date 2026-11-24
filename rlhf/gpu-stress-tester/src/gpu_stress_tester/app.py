from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict, Field

from .config import profile_summaries
from .supervisor import RunManager, Settings

PACKAGE_ROOT = Path(__file__).resolve().parent


class StartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: str = "standard"
    rounds: int | None = Field(default=None, ge=1, le=100_000)
    samples: int | None = Field(default=None, ge=1, le=10_000)
    sft_epochs: int | None = Field(default=None, ge=1, le=20)
    dpo_epochs: int | None = Field(default=None, ge=1, le=20)


def create_app(settings: Settings | None = None) -> FastAPI:
    application = FastAPI(title="VoiceMed ASR RLHF Stress Testing Panel")
    manager = RunManager(settings or Settings.from_environment())
    templates = Jinja2Templates(directory=PACKAGE_ROOT / "templates")
    application.mount(
        "/static",
        StaticFiles(directory=PACKAGE_ROOT / "static"),
        name="static",
    )

    @application.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request):
        return templates.TemplateResponse(request, "index.html")

    @application.get("/health")
    async def health():
        return {"ok": True}

    @application.get("/api/status")
    async def status():
        return manager.status()

    @application.get("/api/profiles")
    async def profiles():
        return {"profiles": profile_summaries()}

    @application.get("/api/logs")
    async def logs(lines: int = Query(default=200, ge=10, le=2_000)):
        return {"lines": manager.log_tail(lines)}

    @application.post("/api/start", status_code=202)
    async def start(payload: StartRequest):
        overrides = payload.model_dump(exclude={"profile"}, exclude_none=True)
        try:
            return manager.start(payload.profile, overrides)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @application.post("/api/stop", status_code=202)
    async def stop():
        try:
            return manager.request_stop()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @application.post("/api/force-stop", status_code=202)
    async def force_stop():
        try:
            return manager.force_stop()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    return application


app = create_app()