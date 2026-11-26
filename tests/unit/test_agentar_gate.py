"""AGENTAR_ENABLED decides whether the Agentar routes exist and whether the lifecycle tick runs.

The route table is fixed when `main` is imported, so each case imports it in a fresh interpreter
with the flag set in the environment.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from apscheduler.schedulers.asyncio import AsyncIOScheduler

import main
from config import settings

ROOT = Path(__file__).resolve().parents[2]
TICK_JOB = "agentar_lifecycle_tick"

_ROUTE_TABLE = "import json, main; print(json.dumps(sorted(main.app.openapi()['paths'])))"


def _paths(flag: str) -> set[str]:
    env = {k: v for k, v in os.environ.items() if k != "AGENTAR_ENABLED"}
    env["AGENTAR_ENABLED"] = flag
    out = subprocess.run([sys.executable, "-c", _ROUTE_TABLE], cwd=ROOT, env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    return set(json.loads(out.stdout.strip().splitlines()[-1]))


def _is_agentar(path: str) -> bool:
    return path == "/api/me" or path.startswith(("/api/me/", "/api/agentar"))


@pytest.fixture(scope="module")
def paths_off() -> set[str]:
    return _paths("false")


@pytest.fixture(scope="module")
def paths_on() -> set[str]:
    return _paths("true")


def test_flag_off_mounts_no_agentar_route(paths_off):
    assert sorted(p for p in paths_off if _is_agentar(p)) == []


def test_flag_off_keeps_the_routes_main_already_serves(paths_off):
    for path in (
        "/api/chat",
        "/api/virtual-participants",
        "/api/virtual-participants/{bot_id}",
        "/api/auth/login",
        "/api/auth/jitsi/issue",
        "/api/auth/config",
        "/health",
    ):
        assert path in paths_off, path


def test_flag_on_mounts_the_agentar_routes(paths_on, paths_off):
    agentar = {p for p in paths_on if _is_agentar(p)}
    assert {"/api/me", "/api/agentar/meetings", "/api/agentar/files", "/api/agentar/events"} <= agentar
    assert paths_on - agentar == paths_off


@pytest.mark.parametrize("enabled", [False, True])
def test_lifecycle_tick_is_scheduled_only_with_the_flag(monkeypatch, enabled):
    monkeypatch.setattr(settings, "agentar_enabled", enabled)
    scheduler = AsyncIOScheduler()
    main.schedule_agentar_tick(scheduler)
    assert (scheduler.get_job(TICK_JOB) is not None) is enabled
