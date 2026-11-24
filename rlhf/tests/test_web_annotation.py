from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_empty_pair_response_exposes_refill_status() -> None:
    app_source = (ROOT / "web" / "app.py").read_text(encoding="utf-8")

    assert '"done": True' in app_source
    assert '"queue_refill_status": store.get_setting(' in app_source


def test_annotation_audio_rendering_runs_off_event_loop() -> None:
    app_source = (ROOT / "web" / "app.py").read_text(encoding="utf-8")

    assert "def _cached_audio_wav(file_path: str) -> Path:" in app_source
    assert "await asyncio.to_thread(_cached_audio_wav, af.file_path)" in app_source
    assert 'Path(DEFAULT_PATHS.output_dir) / "audio_cache"' in app_source
    assert "return FileResponse(cached_path, media_type=\"audio/wav\")" in app_source


def test_training_gate_rejection_has_non_error_admin_status() -> None:
    app_source = (ROOT / "web" / "app.py").read_text(encoding="utf-8")
    template = (ROOT / "web" / "templates" / "admin.html").read_text(
        encoding="utf-8"
    )

    assert "is_policy_gate_rejection(description, proc.returncode)" in app_source
    assert 'set_setting("training_status", "guarded")' in app_source
    assert "gate activated" in template
    assert "settings.training_status == 'guarded'" in template


def test_admin_shows_scheduler_aligned_training_progress() -> None:
    app_source = (ROOT / "web" / "app.py").read_text(encoding="utf-8")
    template = (ROOT / "web" / "templates" / "admin.html").read_text(
        encoding="utf-8"
    )

    assert "new_actionable, training_threshold = training_trigger_progress(store)" in app_source
    assert "training_remaining" in app_source
    assert "{{ new_actionable }} / {{ settings.training_threshold }}" in template
    assert "{{ training_remaining }} remaining" in template


def test_annotator_template_retries_while_refill_is_active() -> None:
    template = (ROOT / "web" / "templates" / "annotate.html").read_text(
        encoding="utf-8"
    )

    assert 'id="state-waiting"' in template
    assert '["queued", "running"].includes(data.queue_refill_status)' in template
    assert "setTimeout(() => loadNextPair(false), 10000)" in template


def test_bulk_gt_copy_is_admin_only_and_exposed_in_monitor_ui() -> None:
    app_source = (ROOT / "web" / "app.py").read_text(encoding="utf-8")
    template = (ROOT / "web" / "templates" / "canary.html").read_text(encoding="utf-8")

    route_start = app_source.index(
        '@app.post("/admin/canary/copy-all/{canary_set_id}")'
    )
    route_source = app_source[route_start : route_start + 700]
    assert "if not _is_admin(request):" in route_source
    assert "bulk_copy_canary_ground_truth(canary_set_id)" in route_source
    assert 'action="/admin/canary/copy-all/{{ active_set.id }}"' in template
    assert "Copy GT to all" in template


def test_frozen_monitor_can_be_inspected_and_unfrozen() -> None:
    app_source = (ROOT / "web" / "app.py").read_text(encoding="utf-8")
    template = (ROOT / "web" / "templates" / "canary.html").read_text(encoding="utf-8")

    assert '@app.post("/admin/canary/unfreeze/{canary_set_id}")' in app_source
    assert "unfreeze_canary_set(canary_set_id)" in app_source
    assert "Unfreeze transcripts" in template
    assert "Evaluate now / retry" in template
    assert 'class="canary-transcript-table"' in template
    assert "{{ item.transcript or" in template
    assert "Automatic evaluation" in template


def test_model_health_comparison_exposes_monitor_audio() -> None:
    template = (ROOT / "web" / "templates" / "model_health.html").read_text(
        encoding="utf-8"
    )

    assert 'src="/admin/canary/audio/{{ prediction.canary_item_id }}"' in template
    assert "{{ prediction.canary_item_id[:8] }}" in template
    assert "{{ prediction.audio_id[:8] }}" in template
    assert "evaluation.output_diff.changed" in template
    assert "prediction.previous_hypothesis" in template


def test_monitor_ui_supports_selection_reroll_and_locked_wording() -> None:
    app_source = (ROOT / "web" / "app.py").read_text(encoding="utf-8")
    monitor = (ROOT / "web" / "templates" / "canary.html").read_text(encoding="utf-8")
    health = (ROOT / "web" / "templates" / "model_health.html").read_text(
        encoding="utf-8"
    )

    assert '@app.post("/admin/canary/reroll/{item_id}")' in app_source
    assert 'name="canary_set_id"' in monitor
    assert 'action="/admin/canary/reroll/{{ item.id }}"' in monitor
    assert "Zbiór oceny jest zablokowany" in monitor
    assert 'name="canary_set_id"' in health
    assert "Formatting Signals" not in health
    assert "fragmentation_points" not in health
    assert "completed.sort(key=checkpoint_time)" in app_source
    assert "chart_ticks" in app_source
    assert 'class="chart-axis-label"' in health
    assert 'class="chart-value chart-value--wer"' in health
    assert 'class="chart-value chart-value--cer"' in health
    assert "Δ corpus WER vs base (95% CI)" in health
    assert "paired_corpus_wer_from_predictions" in app_source


def test_monitor_ui_can_delete_drafts_and_archive_locked_history() -> None:
    app_source = (ROOT / "web" / "app.py").read_text(encoding="utf-8")
    template = (ROOT / "web" / "templates" / "canary.html").read_text(encoding="utf-8")

    assert '@app.post("/admin/canary/remove/{canary_set_id}")' in app_source
    assert "remove_canary_set(canary_set_id)" in app_source
    assert 'action="/admin/canary/remove/{{ active_set.id }}"' in template
    assert "Delete draft" in template
    assert "Archive evaluation set" in template
    assert "Show archived" in template


def test_monitor_items_show_item_and_audio_identifiers() -> None:
    template = (ROOT / "web" / "templates" / "canary.html").read_text(encoding="utf-8")

    assert "{{ item.id[:8] }}" in template
    assert "{{ item.audio_id[:8] }}" in template


def test_reward_audit_is_admin_only_and_exposes_diagnostics() -> None:
    app_source = (ROOT / "web" / "app.py").read_text(encoding="utf-8")
    template = (ROOT / "web" / "templates" / "reward_audit.html").read_text(
        encoding="utf-8"
    )
    admin_nav = (ROOT / "web" / "templates" / "_admin_nav.html").read_text(
        encoding="utf-8"
    )

    route_start = app_source.index(
        '@app.get("/admin/reward-audit", response_class=HTMLResponse)'
    )
    assert "if not _is_admin(request):" in app_source[route_start : route_start + 500]
    assert '@app.post("/admin/reward-audit/create")' in app_source
    assert '@app.get("/admin/reward-audit/audio/{pair_id}")' in app_source
    assert 'action="/admin/reward-audit/create"' in template
    assert "item.reward_a" in template
    assert "item.margin" in template
    assert "item.agreement" in template
    assert 'href="/admin/reward-audit"' in admin_nav


def test_authenticated_admin_pages_share_navigation() -> None:
    template_dir = ROOT / "web" / "templates"
    for name in (
        "admin.html",
        "annotation_audit.html",
        "canary.html",
        "model_health.html",
        "reward_audit.html",
        "diagnostics.html",
        "corpora.html",
    ):
        source = (template_dir / name).read_text(encoding="utf-8")
        assert '{% include "_admin_nav.html" %}' in source

    navigation = (template_dir / "_admin_nav.html").read_text(encoding="utf-8")
    for href in (
        "/annotate",
        "/admin/canary",
        "/admin/model-health",
        "/admin/annotation-audit",
        "/admin/reward-audit",
        "/admin/diagnostics",
        "/admin/corpora",
        "/admin",
    ):
        assert f'href="{href}"' in navigation
    assert 'action="/admin/logout"' in navigation


def _corpus_admin_client(monkeypatch, store, authenticated=True):
    import web.app as web_app
    from fastapi.testclient import TestClient

    monkeypatch.setattr(web_app, "_get_store", lambda: store)
    monkeypatch.setattr(web_app, "_ADMIN_CFG_PATH", str(store.db_path.parent / "admin_password.json"))
    client = TestClient(web_app.app)
    if authenticated:
        client.post("/admin/login", data={"password": "change-me"})
    return client


def test_corpus_page_stages_data_and_activation_preserves_holdout_exclusions(
    monkeypatch, store,
) -> None:
    from storage.schema import AudioFile, CanarySet, Corpus

    held = AudioFile.new("held.wav", "monitor", "Held text.")
    store.insert_audio_file(held)
    store.create_canary_set(CanarySet.new("monitor", 1, "explicit", target_size=1), [held.id])
    corpus = Corpus.new("<script>corpus</script>")
    corpus.active = False
    recordings = [
        AudioFile.new("sibling.wav", corpus.name, "Held  text."),
        AudioFile.new("other.wav", corpus.name, "<img src=x onerror=alert(1)>"),
    ]
    store.import_corpus_audio(corpus, recordings)
    client = _corpus_admin_client(monkeypatch, store)
    url = f"/admin/corpora?corpus_id={corpus.id}"
    response = client.get(url)
    assert response.status_code == 200
    assert response.context["selected"]["audio_count"] == 2
    assert response.context["selected"]["utterance_count"] == 2
    assert response.context["selected"]["eligible_count"] == 0
    assert "&lt;script&gt;corpus&lt;/script&gt;" in response.text
    assert "<img src=x onerror=alert(1)>" not in response.text
    assert "corpora.activateConfirm" in response.text
    assert store.get_corpora()[0]["active"] == 0
    activated = client.post("/admin/corpora/activation",
                            data={"corpus_id": corpus.id, "active": "1"}, follow_redirects=False)
    assert activated.status_code == 303
    assert activated.headers["location"] == url
    response = client.get(url)
    assert response.context["selected"]["eligible_count"] == 1
    eligibility = {audio["id"]: audio["learning_eligible"] for audio in response.context["audio_files"]}
    assert eligibility == {recordings[0].id: 0, recordings[1].id: 1}
    client.post("/admin/corpora/activation", data={"corpus_id": corpus.id, "active": "0"})
    assert client.get(url).context["selected"]["eligible_count"] == 0
    assert store.count_audio_files(corpus.name) == 2


def test_corpus_page_requires_admin_session_and_rejects_invalid_requests(monkeypatch, store) -> None:
    from storage.schema import Corpus

    corpus = store.register_corpus(Corpus.new("empty"))
    client = _corpus_admin_client(monkeypatch, store, authenticated=False)
    response = client.get("/admin/corpora", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/admin/login"
    assert client.post("/admin/corpora/activation",
                       data={"corpus_id": corpus["id"], "active": "0"}).status_code == 403
    assert client.get("/admin/corpora/audio/unknown").status_code == 403
    client.post("/admin/login", data={"password": "change-me"})
    assert client.get("/admin/corpora").status_code == 200
    for page, status in (("0", 400), ("abc", 400), ("2", 404)):
        assert client.get(f"/admin/corpora?page={page}").status_code == status
    assert client.get("/admin/corpora?corpus_id=unknown").status_code == 404
    assert client.post("/admin/corpora/activation",
                       data={"corpus_id": corpus["id"], "active": "true"}).status_code == 400
    assert client.post("/admin/corpora/activation",
                       data={"corpus_id": "unknown", "active": "1"}).status_code == 404
    assert client.get("/admin/corpora/audio/unknown").status_code == 404
    assert store.get_corpora()[0]["active"] == 1


def test_corpus_preview_is_paginated_without_duplicate_or_missing_rows(monkeypatch, store) -> None:
    from storage.schema import AudioFile, Corpus

    corpus = Corpus.new("paged")
    corpus.active = False
    recordings = [AudioFile.new(f"{index}.wav", corpus.name) for index in range(26)]
    store.import_corpus_audio(corpus, recordings)
    client = _corpus_admin_client(monkeypatch, store)
    first = client.get(f"/admin/corpora?corpus_id={corpus.id}")
    second = client.get(f"/admin/corpora?corpus_id={corpus.id}&page=2")
    assert len(first.context["audio_files"]) == 25
    assert len(second.context["audio_files"]) == 1
    assert first.context["pages"] == second.context["pages"] == 2
    ids = [row["id"] for row in first.context["audio_files"] + second.context["audio_files"]]
    assert set(ids) == {audio.id for audio in recordings}
    assert len(ids) == len(set(ids))
    assert "page=2" in first.text
    assert "page=3" not in second.text


def test_corpus_audio_preserves_nested_archive_paths_and_distinct_caches(
    monkeypatch, store, tmp_path, caplog,
) -> None:
    import io
    import wave
    import zipfile
    from dataclasses import replace

    import soundfile as sf
    import web.app as web_app
    from storage.schema import AudioFile

    client = _corpus_admin_client(monkeypatch, store)
    monkeypatch.setattr(web_app, "_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(web_app, "DEFAULT_PATHS",
                        replace(web_app.DEFAULT_PATHS, output_dir=str(tmp_path / "outputs")))
    for corpus_name, samples in (("first", b"\x00\x10"), ("second", b"\x00\x20")):
        source = io.BytesIO()
        with wave.open(source, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16_000)
            wav.writeframes(samples * 1_600)
        directory = tmp_path / "_data" / corpus_name
        directory.mkdir(parents=True)
        archive_path = directory / "clips.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("sample.wav", source.getvalue())
        audio = AudioFile.new(f"zip:_data/{corpus_name}/clips.zip!sample.wav", corpus_name)
        store.insert_audio_file(audio)
        response = client.get(f"/admin/corpora/audio/{audio.id}")
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        waveform, sample_rate = sf.read(io.BytesIO(response.content))
        assert sample_rate == 16_000
        assert len(waveform) == 1_600
        assert waveform.mean() == pytest.approx(0.125 if corpus_name == "first" else 0.25)
    monkeypatch.setattr(web_app, "_PROJECT_ROOT", str(tmp_path / "other-project"))
    monkeypatch.setattr(web_app, "_CONTAINER_DATA_DIR", str(tmp_path / "_data"))
    assert client.get(f"/admin/corpora/audio/{audio.id}").status_code == 200
    archive_path.unlink()
    response = client.get(f"/admin/corpora/audio/{audio.id}")
    assert response.status_code == 500
    assert response.json()["detail"] == "Could not load corpus audio"
    assert "Corpus audio load failed" in caplog.text
    assert web_app._container_path("zip:C:/old-machine/clips.zip!sample.wav") == (
        f"zip:{web_app._CONTAINER_DATA_DIR}/clips.zip!sample.wav"
    )


def test_diagnostics_download_is_admin_only_and_allowlisted() -> None:
    app_source = (ROOT / "web" / "app.py").read_text(encoding="utf-8")

    admin_start = app_source.index('@app.get("/admin", response_class=HTMLResponse)')
    route_start = app_source.index(
        '@app.get("/admin/diagnostics/{run_id}/download/{kind}")'
    )
    canary_start = app_source.index(
        '@app.get("/admin/canary", response_class=HTMLResponse)'
    )
    assert "return templates.TemplateResponse(" in app_source[admin_start:route_start]
    assert admin_start < route_start < canary_start
    route = app_source[route_start : route_start + 900]
    assert "if not _is_admin(request):" in route
    assert "bundle_path(DEFAULT_PATHS, run_id, kind)" in route
    assert 'headers={"Cache-Control": "no-store"}' in route
    assert "path.is_symlink()" in route
    assert "Sensitive download confirmation required" in route


def test_diagnostics_page_has_language_and_pending_feedback() -> None:
    app_source = (ROOT / "web" / "app.py").read_text(encoding="utf-8")
    template = (ROOT / "web" / "templates" / "diagnostics.html").read_text(
        encoding="utf-8"
    )

    assert 'id="lang-pl-link"' in template
    assert 'id="lang-en-link"' in template
    assert "Waiting for collector acknowledgement" in template
    assert "diagnostics-command-form" in template
    assert "last_request_id" in app_source
    assert "_diagnostics_pending_action" in app_source


def test_annotation_audit_is_mute_only_and_uses_stored_rewards() -> None:
    app_source = (ROOT / "web" / "app.py").read_text(encoding="utf-8")
    template = (ROOT / "web" / "templates" / "annotation_audit.html").read_text(
        encoding="utf-8"
    )

    assert '@app.get("/admin/annotation-audit"' in app_source
    assert 'action="/admin/annotation-audit/annotation"' in template
    assert 'action="/admin/annotation-audit/annotator"' in template
    assert "item.reward_a" in template
    assert "item.reward_b" in template
    assert "item.reward_source" in template
    assert 'value="mute"' not in template
    assert "/delete" not in template.lower()
    assert "/ban" not in template.lower()


def test_annotation_audit_exposes_excluded_recordings_view() -> None:
    app_source = (ROOT / "web" / "app.py").read_text(encoding="utf-8")
    template = (ROOT / "web" / "templates" / "annotation_audit.html").read_text(
        encoding="utf-8"
    )

    assert 'view not in {"interactions", "excluded"}' in app_source
    assert "get_excluded_audio_audit(" in app_source
    assert '@app.get("/admin/annotation-audit/excluded-audio/{audio_id}")' in app_source
    assert "Excluded recordings" in template
    assert 'name="dataset"' in template
    assert 'name="query"' in template
    assert "item.exclusion_reason" in template
    assert "item.reference_transcript" in template
    assert "item.excluded_by" in template
    assert 'action="/admin/annotation-audit/audio-curation"' in template
    assert "item.latest_actor" in template
    assert "Re-exclude" in template
    assert 'option value="restored"' in template
    assert '@app.post("/admin/annotation-audit/audio-curation")' in app_source


def test_active_policy_is_prominent_in_admin_and_model_health() -> None:
    app_source = (ROOT / "web" / "app.py").read_text(encoding="utf-8")
    admin = (ROOT / "web" / "templates" / "admin.html").read_text(encoding="utf-8")
    health = (ROOT / "web" / "templates" / "model_health.html").read_text(
        encoding="utf-8"
    )

    assert "def _active_policy_view" in app_source
    assert '"active_policy": _active_policy_view(store)' in app_source
    assert "Currently deployed policy" in health
    assert "evaluation.is_active" in health
    assert "Deployed" in health
    assert "Active policy" in admin


def test_models_page_publishes_only_active_healthy_policy() -> None:
    app_source = (ROOT / "web" / "app.py").read_text(encoding="utf-8")
    health = (ROOT / "web" / "templates" / "model_health.html").read_text(
        encoding="utf-8"
    )
    navigation = (ROOT / "web" / "templates" / "_admin_nav.html").read_text(
        encoding="utf-8"
    )

    route_start = app_source.index('@app.post("/admin/model-health/publish-latest")')
    route = app_source[route_start : route_start + 3000]
    assert "resolve_active_policy_checkpoint(" in route
    assert 'row["health_status"] == "healthy"' in route
    assert "publish_latest_model(" in route
    assert 'action="/admin/model-health/publish-latest"' in health
    assert "publication_allowed" in health
    assert "Publish as latest" in health
    assert "Models &amp; Publishing" in navigation


def test_promotion_gate_is_explicit_in_evaluation_interfaces() -> None:
    app_source = (ROOT / "web" / "app.py").read_text(encoding="utf-8")
    canary = (ROOT / "web" / "templates" / "canary.html").read_text(encoding="utf-8")
    health = (ROOT / "web" / "templates" / "model_health.html").read_text(
        encoding="utf-8"
    )

    assert "get_promotion_gate_canary_set()" in app_source
    assert 'action="/admin/canary/promotion-gate/{{ active_set.id }}"' in canary
    assert "Use for promotion gate" in canary
    assert "Choose another promotion gate before archiving" in canary
    assert "PROMOTION GATE" in canary
    assert "PROMOTION GATE" in health


def test_admin_renders_training_error_without_a_current_step() -> None:
    admin = (ROOT / "web" / "templates" / "admin.html").read_text(encoding="utf-8")

    error_condition = "settings.training_status == 'failed' and settings.training_error"
    progress_end = admin.index("{% endif %}", admin.index("pipeline-progress"))
    error_start = admin.index(error_condition)

    assert error_start > progress_end
    assert "{{ settings.training_error }}" in admin[error_start:]


def test_admin_exposes_guarded_convergence_tuning() -> None:
    app_source = (ROOT / "web" / "app.py").read_text(encoding="utf-8")
    admin = (ROOT / "web" / "templates" / "admin.html").read_text(encoding="utf-8")

    for name in (
        "training_sft_epochs",
        "training_sft_learning_rate",
        "training_sft_warmup_steps",
        "training_dpo_epochs",
        "training_dpo_learning_rate",
        "training_dpo_beta",
        "training_max_grad_norm",
        "training_early_stopping_patience",
        "training_synthetic_ratio",
        "training_synthetic_pool_multiplier",
        "training_synthetic_reward_margin",
        "training_synthetic_min_token_edits",
    ):
        assert f'name="{name}"' in admin
    assert "TrainingTuning.from_mapping(form)" in app_source
    assert "training_tuning.as_settings().items()" in app_source
    assert "Latest training diagnostics" in admin
    assert "diagnostic.optimizer_updates" in admin
    assert "diagnostic.mean_gradient_norm" in admin
    assert "diagnostic.mean_synthetic_reward_margin" in admin
