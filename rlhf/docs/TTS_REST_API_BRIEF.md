# Minimal TTS REST API Brief

## Goal

The RLHF service sends a batch of Polish texts with selected TTS engines and
voices. The TTS service synthesizes the whole batch asynchronously, packages
the results into one ZIP archive, and exposes it through a temporary HTTP
download URL.

The services initially run on one trusted machine or network. No authentication
is required for the first version.

## Responsibilities

### RLHF service

- selects texts, engines, and voices;
- splits large campaigns into practical batches;
- submits synthesis jobs;
- polls job status;
- downloads one ZIP archive per completed job;
- unpacks the archive and matches files using its manifest;
- stores audio permanently on its side.

### TTS service

- advertises available engines and compatible voices;
- synthesizes submitted batches;
- reports job progress;
- packages successful files and per-item results into a ZIP archive;
- retains completed archives for a known period;
- serves completed archives from HTTP URLs until expiry.

RLHF does not upload recordings or voice-cloning reference audio. Cloned voices
are prepared and managed by the TTS service.

---

## Required API

### 1. Discover capabilities

```http
GET /v1/capabilities
```

Example response:

```json
{
  "api_version": "1",
  "languages": ["pl"],
  "audio_formats": ["wav"],
  "archive_formats": ["zip"],
  "max_batch_size": 1000,
  "result_retention_seconds": 86400,
  "engines": [
    {
      "id": "xtts",
      "version": "2.1",
      "voices": [
        {"id": "clone-anna-v2", "name": "Anna"},
        {"id": "clone-piotr-v1", "name": "Piotr"}
      ]
    },
    {
      "id": "other-engine",
      "version": "1.0",
      "voices": [
        {"id": "clone-marek-v1", "name": "Marek"}
      ]
    }
  ]
}
```

Voice lists are engine-specific. The RLHF service will submit only combinations
advertised here.

### 2. Submit a synthesis job

```http
POST /v1/jobs
Content-Type: application/json
```

Example request:

```json
{
  "request_id": "a1b2c3d4",
  "language": "pl",
  "audio_format": "wav",
  "archive_format": "zip",
  "items": [
    {
      "item_id": "item-001",
      "text": "Text to synthesize.",
      "engine": "xtts",
      "voice_id": "clone-anna-v2"
    },
    {
      "item_id": "item-002",
      "text": "Another text to synthesize.",
      "engine": "other-engine",
      "voice_id": "clone-marek-v1"
    }
  ]
}
```

Example response:

```http
HTTP/1.1 202 Accepted
```

```json
{
  "job_id": "job-987",
  "status": "queued"
}
```

Requirements:

- `request_id` identifies the submission. Repeating the same `request_id`
  should return the same job instead of creating duplicate work.
- `item_id` is supplied by RLHF and must be returned unchanged.
- Each item may use a different advertised engine and voice.
- One invalid item should be recorded as failed without failing successful
  items in the same archive.

### 3. Poll job status

```http
GET /v1/jobs/{job_id}
```

Example response:

```json
{
  "job_id": "job-987",
  "status": "completed_with_failures",
  "total_items": 1000,
  "successful_items": 998,
  "failed_items": 2,
  "archive_url": "http://tts-service:8090/v1/results/job-987/archive",
  "archive_format": "zip",
  "byte_size": 184322000,
  "sha256": "optional archive checksum",
  "ready_at": "2026-11-13T15:20:00Z",
  "expires_at": "2026-11-14T15:20:00Z"
}
```

Job statuses:

```text
queued
running
completed
completed_with_failures
failed
```

While a job is running, the response may contain simple progress counts:

```json
{
  "job_id": "job-987",
  "status": "running",
  "total_items": 1000,
  "processed_items": 420
}
```

### 4. Download the result archive

The `archive_url` may point to the TTS service or another HTTP storage service.
It must be reachable from the RLHF service.

```http
GET {archive_url}
```

Expected behavior:

- return one ZIP archive with HTTP 200;
- use `Content-Type: application/zip`;
- include `Content-Length` when known;
- return the same archive while it is retained;
- return HTTP 410 when the archive has expired.

The retention period starts when the archive becomes ready.

The ZIP contains:

```text
manifest.json
audio/
  file-000001.wav
  file-000002.wav
  ...
```

`manifest.json` maps provider-generated safe file names to RLHF item IDs and
records failures:

```json
{
  "job_id": "job-987",
  "items": [
    {
      "item_id": "item-001",
      "status": "success",
      "file": "audio/file-000001.wav",
      "engine": "xtts",
      "engine_version": "2.1",
      "voice_id": "clone-anna-v2"
    },
    {
      "item_id": "item-002",
      "status": "failed",
      "error": {
        "code": "synthesis_failed",
        "message": "Engine could not synthesize this item.",
        "retryable": false
      }
    }
  ]
}
```

Create an archive even when some items fail. It contains all successful audio
files and the status of every requested item. A job-level `failed` status is
reserved for cases where no usable result archive can be produced.

Batch size should be bounded. One archive containing hundreds or approximately
one thousand files is reasonable; one archive containing an entire very large
campaign is not. Bounded batches avoid thousands of HTTP downloads without
making a single retry require downloading the whole corpus again.

---

## Error Format

For API-level errors, return a non-2xx HTTP status and a small JSON body:

```json
{
  "error": {
    "code": "invalid_voice",
    "message": "Voice clone-x is not available for engine xtts.",
    "retryable": false
  }
}
```

Use `retryable: true` only for temporary failures.

---

## General Conventions

- JSON uses UTF-8.
- Timestamps use ISO 8601 in UTC.
- Polish is represented as `"pl"`.
- WAV inside ZIP is sufficient for the first version.
- Native sample rate and channel count may vary; RLHF will normalize audio.
- Result URLs must not use `localhost` when RLHF runs in another container.
- Large campaigns are split by RLHF according to `max_batch_size`.

---

## Not Required in the First Version

- authentication;
- callbacks or webhooks;
- successful-download acknowledgement;
- audio upload from RLHF;
- shared filesystem paths;
- permanent result storage on the TTS side;
- deterministic generation;
- pronunciation or audio-quality validation;
- separate download URLs for every synthesized file;
- campaign, training, or annotation settings;
- cancellation endpoint;
- advanced synthesis controls.

The complete internal architecture and rationale are documented in
`docs/TTS_AUGMENTATION_DESIGN.md`.
