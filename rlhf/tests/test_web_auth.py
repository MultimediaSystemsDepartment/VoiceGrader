from __future__ import annotations

import os
import io
import wave

import pytest
from fastapi.testclient import TestClient

import web.app as web_app
from tests.conftest import make_audio, make_pair
from utils.admin_password import load_password_config, save_password_config, verify_password


@pytest.fixture()
def client(monkeypatch, store):
    monkeypatch.setattr(web_app, "_get_store", lambda: store)
    monkeypatch.setattr(web_app, "_ADMIN_CFG_PATH", str(store.db_path.parent / "admin_password.json"))
    return TestClient(web_app.app)


def _upload_audio():
    data = io.BytesIO()
    with wave.open(data, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\x01\x00" * 3200)
    return data.getvalue()


@pytest.mark.parametrize("purpose", ["training", "evaluation"])
def test_gui_upload_preview_accept_and_explicit_activation(client, store, monkeypatch, tmp_path, purpose):
    monkeypatch.setattr(web_app, "_CORPUS_UPLOAD_ROOT", tmp_path / "uploads")
    client.post("/admin/login", data={"password": "change-me"})
    response = client.post("/admin/corpora/import", data={"name": "uploaded", "purpose": purpose},
                           files=[("manifest", ("list.csv", b"file_path;transcript\nclip.wav;Test words.\n")),
                                  ("audio", ("clip.wav", _upload_audio()))])
    assert response.status_code == 200
    assert "Accept import" in response.text
    assert store.get_corpora() == []
    preview_url = response.url.path
    assert client.get(preview_url + "/audio/0").status_code == 200
    accepted = client.post(preview_url + "/accept", follow_redirects=False)
    assert accepted.status_code == 303
    corpus = store.get_corpora()[0]
    assert not corpus["active"]
    assert corpus["purpose"] == purpose
    assert client.post(preview_url + "/accept").status_code == 400
    assert client.post(preview_url + "/cancel").status_code == 400
    if purpose == "evaluation":
        assert "/admin/canary?" in accepted.headers["location"]
        assert store.get_canary_sets()[0]["verified_count"] == 0
        assert client.post("/admin/corpora/activation",
                           data={"corpus_id": corpus["id"], "active": "1"}).status_code == 400
    else:
        assert not store.get_random_audio_files(5, 1)
        assert client.post("/admin/corpora/activation",
                           data={"corpus_id": corpus["id"], "active": "1"}).status_code == 200
        assert len(store.get_random_audio_files(5, 1)) == 1


@pytest.mark.parametrize("path", ["../outside.wav", "C:\\outside.wav", "missing.wav"])
def test_gui_upload_rejects_unsafe_or_missing_manifest_audio(client, store, monkeypatch, tmp_path, path):
    root = tmp_path / "uploads"
    monkeypatch.setattr(web_app, "_CORPUS_UPLOAD_ROOT", root)
    client.post("/admin/login", data={"password": "change-me"})
    response = client.post("/admin/corpora/import", data={"name": "bad", "purpose": "evaluation"},
                           files=[("manifest", ("list.csv", f"file_path;transcript\n{path};Words.\n".encode())),
                                  ("audio", ("clip.wav", _upload_audio()))])
    assert response.status_code == 400
    assert store.get_corpora() == []
    assert list(root.iterdir()) == []


def test_gui_upload_authentication_and_cancellation(client, store, monkeypatch, tmp_path):
    root = tmp_path / "uploads"
    monkeypatch.setattr(web_app, "_CORPUS_UPLOAD_ROOT", root)
    assert client.post("/admin/corpora/import", data={}).status_code == 403
    client.post("/admin/login", data={"password": "change-me"})
    response = client.post("/admin/corpora/import", data={"name": "cancel", "purpose": "training"},
                           files=[("manifest", ("list.csv", b"file_path\nclip.wav\n")),
                                  ("audio", ("clip.wav", _upload_audio()))])
    assert response.status_code == 200
    assert client.post(response.url.path + "/cancel", follow_redirects=False).status_code == 303
    assert list(root.iterdir()) == []
    assert store.get_corpora() == []


@pytest.mark.parametrize("limit", ["MAX_UPLOAD_BYTES", "MAX_MANIFEST_BYTES", "MAX_AUDIO_FILES"])
def test_gui_upload_enforces_bounds_without_partial_import(client, store, monkeypatch, tmp_path, limit):
    root = tmp_path / "uploads"
    monkeypatch.setattr(web_app, "_CORPUS_UPLOAD_ROOT", root)
    monkeypatch.setattr(f"web.corpus_import.{limit}", 1)
    client.post("/admin/login", data={"password": "change-me"})
    uploads = [("manifest", ("list.csv", b"file_path\nclip.wav\n")),
               ("audio", ("clip.wav", _upload_audio()))]
    if limit == "MAX_AUDIO_FILES":
        uploads.append(("audio", ("second.wav", _upload_audio())))
    response = client.post("/admin/corpora/import", data={"name": "bounded", "purpose": "training"},
                           files=uploads)
    assert response.status_code == 400
    assert store.get_corpora() == []
    assert not root.exists() or list(root.iterdir()) == []


def test_gui_cancel_preserves_committed_audio_even_with_stale_stage_marker(
    client, store, monkeypatch, tmp_path,
):
    import json

    root = tmp_path / "uploads"
    monkeypatch.setattr(web_app, "_CORPUS_UPLOAD_ROOT", root)
    client.post("/admin/login", data={"password": "change-me"})
    response = client.post("/admin/corpora/import", data={"name": "retained", "purpose": "training"},
                           files=[("manifest", ("list.csv", b"file_path\nclip.wav\n")),
                                  ("audio", ("clip.wav", _upload_audio()))])
    assert client.post(response.url.path + "/accept", follow_redirects=False).status_code == 303
    stage = root / response.url.path.rsplit("/", 1)[-1]
    marker = json.loads((stage / "stage.json").read_text())
    marker["status"] = "validated"
    (stage / "stage.json").write_text(json.dumps(marker))
    assert client.post(response.url.path + "/cancel").status_code == 400
    assert (stage / "clip.wav").is_file()
    assert store.count_audio_files() == 1


def test_setup_probes_are_read_only_and_annotations_keep_participant_identity(client, store):
    for _ in range(12):
        response = client.get("/setup", follow_redirects=False)
        assert response.status_code == 200
        assert "set-cookie" not in response.headers
    assert store.count_active_sessions(timeout_s=120) == 0
    assert client.post("/setup", data={"participant_id": "  "}).url.path == "/setup"
    pair = make_pair(store, make_audio(store).id)
    assert client.post("/setup", data={"participant_id": "Alice"}).url.path == "/annotate"
    assert client.cookies["participant_id"] == "Alice"
    assert client.post("/api/submit", data={"pair_id": pair.id, "preference": "a"}).status_code == 200
    assert store.count_human_annotations() == 1
    assert client.get("/admin/corpora", follow_redirects=False).status_code == 307


def test_bootstrap_login_is_required_and_wrong_passwords_do_not_create_sessions(client, store):
    assert client.get("/").url.path == "/setup"
    assert client.get("/annotate").url.path == "/setup"
    login_page = client.get("/admin/login")
    assert login_page.status_code == 200
    assert "change-me" not in login_page.text
    assert "New-installation password" not in login_page.text
    assert client.get("/admin/corpora", follow_redirects=False).headers["location"] == "/admin/login"
    assert client.post("/admin/corpora/activation", data={}).status_code == 403
    for password in ("wrong", "\u0142adne has\u0142o"):
        response = client.post("/admin/login", data={"password": password})
        assert "Invalid password." in response.text
        assert not client.cookies.get("admin_session")
    assert client.post("/admin/login", data={"password": "change-me"}).url.path == "/admin"
    assert client.get("/admin/password").status_code == 200
    assert client.get("/admin/corpora").status_code == 200
    assert store.get_setting("admin_session_id")


def test_password_change_persists_and_revokes_sessions_including_after_reload(client, store):
    client.post("/admin/login", data={"password": "change-me"})
    old_cookie = client.cookies["admin_session"]
    password = "  different passphrase  "
    response = client.post("/admin/password", data={
        "current_password": "change-me", "new_password": password, "confirm_password": password,
    }, follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/admin/login"
    assert not client.cookies.get("admin_session")
    config_path = store.db_path.parent / "admin_password.json"
    assert config_path.is_file()
    assert password not in config_path.read_text(encoding="utf-8")
    assert verify_password(password, load_password_config(config_path))
    assert not verify_password("change-me", load_password_config(config_path))
    client.cookies.set("admin_session", old_cookie)
    assert client.get("/admin/corpora", follow_redirects=False).status_code == 307
    client.cookies.clear()
    assert client.post("/admin/login", data={"password": "change-me"}).url.path == "/admin/login"
    reloaded = TestClient(web_app.app)
    assert reloaded.post("/admin/login", data={"password": password}).url.path == "/admin"
    assert reloaded.get("/admin/corpora").status_code == 200


def test_external_password_reset_revokes_existing_cookie(client, store):
    client.post("/admin/login", data={"password": "change-me"})
    save_password_config(store.db_path.parent / "admin_password.json", "replacement passphrase")
    assert client.get("/admin/corpora", follow_redirects=False).status_code == 307
    assert client.post("/admin/login", data={"password": "replacement passphrase"}).url.path == "/admin"


def test_logout_requires_admin_session_and_preserves_other_sessions(client, store):
    assert client.post("/admin/logout").status_code == 403
    client.post("/admin/login", data={"password": "change-me"})
    session = store.get_setting("admin_session_id")
    anonymous = TestClient(web_app.app)
    assert anonymous.post("/admin/logout").status_code == 403
    assert store.get_setting("admin_session_id") == session
    assert client.post("/admin/logout").url.path == "/admin/login"
    assert not client.cookies.get("admin_session")
    assert store.get_setting("admin_session_id") == ""


def test_annotation_capacity_gate_still_reports_error_without_creating_session(client, store):
    from tests.conftest import make_session

    store.set_setting("max_concurrent_sessions", "1")
    existing = make_session(store, "Existing")
    store.upsert_session(existing.id, "Existing")
    response = client.post("/setup", data={"participant_id": "Another"})
    assert response.status_code == 200
    assert "Server is at capacity (1/1 active annotators)." in response.text
    assert not client.cookies.get("session_id")


@pytest.mark.parametrize("current,new,confirmation", [
    ("incorrect", "replacement", "replacement"),
    ("change-me", "short", "short"),
    ("change-me", "replacement", "different"),
])
def test_invalid_password_change_leaves_bootstrap_and_session_unchanged(
    client, store, current, new, confirmation,
):
    client.post("/admin/login", data={"password": "change-me"})
    cookie = client.cookies["admin_session"]
    response = client.post("/admin/password", data={
        "current_password": current, "new_password": new, "confirm_password": confirmation,
    })
    assert response.status_code == 400
    assert client.cookies["admin_session"] == cookie
    assert not (store.db_path.parent / "admin_password.json").exists()
    assert client.get("/admin/corpora").status_code == 200


@pytest.mark.parametrize("content", ["{", "{}", "[]"])
def test_corrupt_password_configuration_never_falls_back_to_bootstrap(
    client, store, content, caplog,
):
    (store.db_path.parent / "admin_password.json").write_text(content, encoding="utf-8")
    response = client.post("/admin/login", data={"password": "change-me"})
    assert response.status_code == 503
    assert not client.cookies.get("admin_session")
    assert "could not be loaded" in caplog.text


def test_password_write_failure_keeps_existing_config_and_login(
    client, store, monkeypatch,
):
    config_path = store.db_path.parent / "admin_password.json"
    save_password_config(config_path, "existing passphrase")
    before = config_path.read_bytes()
    client.post("/admin/login", data={"password": "existing passphrase"})

    def fail_replace(source, destination):
        raise PermissionError("test write failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    response = client.post("/admin/password", data={
        "current_password": "existing passphrase",
        "new_password": "replacement", "confirm_password": "replacement",
    })
    assert response.status_code == 500
    assert config_path.read_bytes() == before
    assert client.get("/admin/corpora").status_code == 200
    assert list(config_path.parent.glob("tmp*")) == []


def test_headless_password_tool_writes_the_same_persistent_format(monkeypatch, tmp_path):
    import sys
    import utils.admin_password as passwords

    path = tmp_path / "admin_password.json"
    monkeypatch.setattr(sys, "argv", ["admin_password", "--config", str(path)])
    monkeypatch.setattr(passwords.getpass, "getpass", lambda prompt: "terminal passphrase")
    assert passwords.main() == 0
    assert verify_password("terminal passphrase", load_password_config(path))
    assert not verify_password("change-me", load_password_config(path))


def test_browser_refill_conflict_stays_on_admin_page_without_requeueing(client, store):
    client.post("/admin/login", data={"password": "change-me"})
    store.set_setting("queue_refill_status", "queued")
    store.set_setting("queue_refill_log", "Original request")
    response = client.post("/admin/trigger_refill", headers={"accept": "text/html"})
    assert response.status_code == 200
    assert response.url.path == "/admin"
    assert "Queue refill is already pending or running." in response.text
    assert "GPU worker" in response.text
    assert store.get_setting("queue_refill_log") == "Original request"
    response = client.post("/admin/trigger_refill", headers={"accept": "application/json"})
    assert response.status_code == 409
    assert "already pending" in response.json()["detail"]
