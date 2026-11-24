from pathlib import Path

from fastapi.testclient import TestClient


def test_health_and_missing_dataset_status(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("STRESS_OUTPUT_DIR", str(tmp_path / "default-outputs"))
    monkeypatch.setenv("ADMEDVOICE_ARCHIVE", str(tmp_path / "default-missing.zip"))
    from gpu_stress_tester.app import create_app
    from gpu_stress_tester.supervisor import Settings

    app = create_app(Settings(tmp_path / "missing.zip", tmp_path / "outputs"))
    client = TestClient(app)

    assert client.get("/health").json() == {"ok": True}
    status = client.get("/api/status").json()
    assert status["status"] == "idle"
    assert status["dataset_ready"] is False
    assert "Start unavailable" in status["dataset_error"]
    profiles = client.get("/api/profiles").json()["profiles"]
    assert profiles["smoke"]["samples"] == 8
    assert profiles["max-load"]["rounds"] == 100_000


def test_start_rejects_missing_archive(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("STRESS_OUTPUT_DIR", str(tmp_path / "default-outputs"))
    monkeypatch.setenv("ADMEDVOICE_ARCHIVE", str(tmp_path / "default-missing.zip"))
    from gpu_stress_tester.app import create_app
    from gpu_stress_tester.supervisor import Settings

    app = create_app(Settings(tmp_path / "missing.zip", tmp_path / "outputs"))
    response = TestClient(app).post("/api/start", json={"profile": "smoke"})

    assert response.status_code == 409
    assert "ADMEDVOICE archive not found" in response.json()["detail"]