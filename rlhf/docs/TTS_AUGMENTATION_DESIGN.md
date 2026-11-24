# TTS Augmentation Integration Design

## Status and Purpose

This document records the complete architectural discussion about integrating an
external text-to-speech (TTS) service with the RLHF-for-ASR system. It is an
information-dense design record for:

- onboarding an implementation agent without repeating the original discussion;
- reminding maintainers why the proposed boundaries and data model exist;
- guiding the next task: defining the detailed REST communication contract;
- separating settled decisions from deferred work and still-open implementation
  details.

The catalogue and migration 008 campaign implementation now exist. The active
operator workflow is in the root README; this document retains design rationale,
not a verified provider wire specification. The temporary provider adapter uses
the observed `/api/v1/tts/jobs` interface, not the supplied batch/ZIP contract.
Accepted jobs resume; ambiguous submissions require review. Independent SFT/refill
caps default to 0%, and SFT permits one TTS rendition per utterance per epoch.
Full live-campaign and approved converted-model interoperability remain unverified.

---

## 1. Objective

Add TTS-generated Polish speech to the existing system as training-only
augmentation. The primary uses are:

1. **SFT augmentation**: synthesize additional audio renditions of known
   transcripts and use the requested text as the supervised ASR target.
2. **RLHF augmentation**: generate Whisper hypotheses for synthesized audio,
   send those pairs through the existing human annotation flow, and use the
   resulting preferences in reward-model and DPO training.
3. **Coverage repair, deferred**: allow operators to add standalone texts for
   rare words, terminology, or other missing linguistic coverage, synthesize
   them, and add the resulting examples to the catalogue.

The initial source of synthesis text is the transcripts already available in
the database. The future coverage-repair source should fit the same model, but
its authoring and auditing UI is explicitly deferred.

TTS augmentation belongs only to learning paths. Frozen real-speech evaluation
must remain the policy-promotion authority.

---

## 2. Original-System Context

Before the catalogue migration, the system was coupled to named datasets,
especially ADMEDVOICE:

- `storage/schema.py::AudioFile` combines physical file path, dataset name,
  transcript, and audio properties in one record.
- `paths.py::PathsConfig` contains dedicated ADMEDVOICE and Common Voice path
  fields.
- `inference/hypothesis_generator.py` maps hard-coded dataset names to manifest
  fields.
- `run_generate_pairs.py` restricts dataset choices at the CLI.
- `prepare_data.py` understands the internal archive layouts of ADMEDVOICE and
  Common Voice.

Adding `"tts"` as one more special-cased dataset would preserve the wrong
abstraction. The integration should instead generalize the catalogue so that
text, corpus membership, and individual audio renditions are distinct.

The current learning loop must remain recognizable:

1. Audio enters the catalogue.
2. The active Whisper policy generates beam and sampled hypotheses.
3. Annotators compare hypothesis pairs.
4. Human preferences train the reward model and contribute authoritative DPO
   labels.
5. Optional SFT consumes human corrections and, after this integration,
   controlled synthetic-audio examples.
6. Candidate policies are evaluated and promoted through the frozen real-audio
   monitor.

Existing invariants continue to apply:

- all audio decoding goes through `utils/audio.py::load_audio()`;
- Whisper generation and teacher-forced label encoding remain locked to Polish
  transcription;
- database access belongs in `storage/feedback_store.py`;
- schema changes require a migration;
- persisted records use UUID4 factories;
- frozen WER Monitor audio is excluded from all learning paths;
- human A/B labels remain authoritative;
- human annotation pairs and temporary reward-model-ranked synthetic DPO pairs
  remain different concepts.

---

## 3. Core Terminology

Use unambiguous terminology because the project will contain two different
kinds of "synthetic" data:

- **Synthetic audio**: a waveform produced by a TTS engine.
- **Synthetic preference**: a chosen/rejected Whisper hypothesis pair selected
  by the reward model for temporary, current-round DPO data.
- **Recorded audio**: a human recording imported from ADMEDVOICE, Common Voice,
  or another real-speech corpus.
- **Utterance**: linguistic content independent of any recording.
- **Audio rendition / audio asset**: one recorded or synthesized realization of
  an utterance.
- **Synthesis profile**: an RLHF-side selection of TTS provider, engine, voice,
  and optional supported parameters.
- **Synthesis campaign**: a persisted operator request that applies one or more
  synthesis profiles to a frozen selection of utterances.

Do not call TTS audio "synthetic DPO data." That phrase already refers to the
ephemeral preference artifact generated during a DPO round.

---

## 4. Architectural Boundary

Treat TTS as a separate, governed data-augmentation subsystem, not as another
step inside the RLHF training pipeline.

```text
Existing transcripts or future custom texts
    |
    v
Synthesis campaign in RLHF system
    |
    | REST: text + language + engine/voice profile
    v
External TTS service
    |
    | one time-limited ZIP archive per bounded provider job
    v
RLHF download/import worker
    |
    v
Durable RLHF-managed audio catalogue
    |                       |
    v                       v
Capped SFT sampling    Capped annotation-queue sampling
                            |
                            v
                    Human A/B preferences
                            |
                            v
                    Reward model and DPO
```

The TTS implementation is a black box. The colleague responsible for TTS must
be free to use different models, engines, cloning implementations, schedulers,
and storage mechanisms as long as the service satisfies the agreed REST
contract.

### RLHF/ASR system owns

- selecting source utterances;
- freezing campaign membership;
- choosing discovered synthesis profiles;
- enforcing corpus and training/evaluation separation;
- submitting and tracking asynchronous synthesis work;
- promptly downloading completed batch archives;
- unpacking each archive using its result manifest;
- durable local storage after download;
- minimal file validation;
- retry and expiry handling;
- SFT and annotation-queue sampling ratios;
- training snapshot provenance;
- annotation, reward-model, DPO, and evaluation integration.

### TTS service owns

- synthesis implementation;
- engine and voice lifecycle;
- cloned-voice enrollment and reference recordings;
- legal/licensing concerns for voices and generated audio;
- concurrency and internal job scheduling;
- validation of TTS-specific parameters;
- generation result archives and manifests;
- result retention and deletion;
- truthful engine, version, and voice metadata;
- explicit job/item failures.

### Important cloning boundary

Cloned voices are pre-provisioned on the TTS side. RLHF discovers and refers to
opaque voice IDs such as `clone-anna-v2`. RLHF does not upload existing
recordings, speaker enrollment material, or cloning references.

---

## 5. Target Data Model

The exact migration is not yet designed, but the logical model should contain
the following concepts.

### 5.1 Corpora

A corpus is a logical collection, not a hard-coded path field.

```text
corpora
  id
  name
  origin             recorded | synthesized | mixed
  provider
  language
  description
  active
  created_at
```

Examples:

```text
admedvoice-original
commonvoice-pl-25
tts-admedvoice-clones-2026-11
tts-custom-coverage-2026-12
```

ADMEDVOICE and Common Voice import logic should become source adapters that
populate this generic model rather than defining the model.

### 5.2 Utterances

An utterance represents text independently of audio.

```text
utterances
  id
  text
  normalized_text
  language
  source_kind        imported_transcript | custom_coverage
  source_reference
  created_at
```

The first implementation needs a complete workflow only for
`imported_transcript`. The `custom_coverage` source kind reserves the loose-text
database concept without requiring its UI or auditing workflow immediately.

A separate loose-text table is not inherently necessary. A custom utterance
with no real recording is simply an utterance with no rendition yet, or later
with only TTS renditions.

### 5.3 Audio assets

An audio asset is one physical realization of an utterance.

```text
audio_assets
  id
  utterance_id
  corpus_id
  storage_uri
  sha256             optional if not used by the provider
  content_type
  duration_s
  sample_rate
  channels
  origin             recorded | tts
  voice_id           nullable for recorded audio
  synthesis_item_id  nullable for recorded audio
  technical_status   valid | invalid
  excluded
  exclusion_reason
  created_at
```

The existing reversible audio-curation behavior must carry forward. Old assets
must not be overwritten merely because a newer engine version or voice rendition
is generated.

One utterance may have:

- one or more recorded renditions;
- several cloned voices;
- the same voice synthesized by different engines;
- future noise- or speed-augmented renditions.

Preserving `utterance_id` is essential for preventing repeated renditions of the
same text from dominating SFT.

### 5.4 Synthesis profiles

A synthesis profile is an RLHF-side saved engine/voice selection.

```text
synthesis_profiles
  id
  provider
  engine
  engine_version_constraint
  voice_id
  parameters_json
  active
  created_at
```

Example:

```text
provider:    colleague-tts
engine:      xtts
voice_id:    clone-anna-v2
parameters:  provider-supported defaults
```

Profiles are preferable to a generic "five variants" integer because the
intended variation currently comes from different voices and different engines.

### 5.5 Synthesis campaigns

```text
synthesis_campaigns
  id
  name
  provider
  source_corpus_id
  selection_mode       fixed_count | percentage | all
  selection_value
  selection_seed
  status
  intended_for_sft
  intended_for_annotation
  selection_config_json
  created_by
  created_at
  started_at
  completed_at
```

The campaign's exact utterance membership must be materialized before or when
work begins. Later corpus changes must not silently alter a running campaign.

### 5.6 Synthesis items

One item represents one requested utterance/profile combination.

```text
synthesis_items
  id
  campaign_id
  utterance_id
  synthesis_profile_id
  client_request_id
  provider_job_id
  provider_item_id
  status
  request_metadata_json
  result_metadata_json
  ready_at
  download_expires_at
  audio_asset_id
  attempt_count
  error_code
  error_message
  created_at
  updated_at
```

Item-level persistence is mandatory for resumability at the expected scale. A
small number of bad items must not fail or repeat an otherwise successful large
campaign.

---

## 6. Campaign Selection Semantics

The admin UI must support:

```text
Fixed count   synthesize N eligible utterances
Percentage    synthesize a target percentage of eligible utterances
All           synthesize every eligible utterance
```

### 6.1 Percentage means cumulative coverage

Generated samples are retained. Increasing a target from 20% to 40% should
select only the missing 20 percentage points.

Example:

```text
Eligible utterances:          10,000
Already covered:               2,000 (20%)
New cumulative target:           40%
New utterances selected:       2,000
Final covered utterances:      4,000 (40%)
```

It must not select a fresh 40% and accidentally produce roughly 60% total
coverage.

Coverage is profile-sensitive. The system needs to know whether a combination
has already been produced:

```text
utterance + provider + engine + voice + relevant profile configuration
```

Normal campaigns fill missing coverage. Explicit future modes may include:

- add newly selected profiles to previously covered utterances;
- retry failed items;
- regenerate after engine/version changes;
- force an additional rendition.

Forced regeneration creates a new immutable asset; it does not overwrite the
old asset.

### 6.2 Frozen selection

The selected utterance IDs are persisted in synthesis items. A selection seed
should be stored for random or percentage selection so maintainers can explain
how campaign membership was chosen. This seed controls RLHF-side selection; it
does not require deterministic TTS output.

### 6.3 Profile matrix

Expected work is:

```text
selected utterances x selected compatible synthesis profiles
```

For 10,000 utterances and five profiles, the campaign contains 50,000 synthesis
items.

The provider may require smaller submission batches. Batch size is determined
from capability discovery or configuration; it is not the campaign boundary.

---

## 7. Capability Discovery

The RLHF UI must not assume which engines, cloned voices, or synthesis controls
exist. The TTS service should expose capabilities including:

- REST API version;
- supported languages;
- available engines and versions;
- available voices;
- voice type, such as cloned or stock;
- engine/voice compatibility;
- supported output formats;
- maximum submission batch size;
- normal result-retention duration;
- optional controls per engine, for example speaking rate, pitch, style,
  emotion, seed, or pronunciation overrides.

The UI displays only advertised controls and prevents invalid engine/voice
combinations.

The minimal useful provider may support only:

```text
text + language + engine + voice_id
```

Optional capabilities must not become mandatory merely because another engine
supports them.

An option such as "provider chooses a compatible Polish cloned voice" may be
supported later, but explicitly selected engine/voice profiles are the primary
model because different voices and engines are intentional augmentation axes.

---

## 8. Data Delivery Decision

### 8.1 Selected approach

The TTS service returns one time-limited HTTP(S) download URL for each completed
provider job. The result is a ZIP archive containing the successful audio files
and a manifest that maps files and failures back to RLHF item IDs. RLHF promptly
streams the archive, unpacks it, and moves valid audio into RLHF-managed durable
storage.

A large campaign is divided into bounded provider jobs. A job may reasonably
contain hundreds or approximately one thousand files, subject to the batch size
advertised by the provider. Do not package an entire very large campaign into
one archive: a failed transfer would require downloading the whole campaign
again.

This is preferred over:

- returning host filesystem paths;
- requiring a shared volume;
- embedding Base64 audio in JSON;
- treating the TTS service as permanent storage.

### 8.2 Why URL-based delivery

It works on the current single-machine deployment and remains topology-agnostic:

- on one Docker host, a URL can use a resolvable service name;
- on multiple machines, it can address the remote TTS service;
- it can later point to MinIO, S3-compatible storage, or a signed object URL;
- RLHF import logic does not need to understand the TTS filesystem.

One archive per provider job avoids thousands of small HTTP downloads while
retaining bounded retries and partial campaign progress. ZIP is primarily a
packaging format; WAV data may not compress substantially.

Inside Docker, a result URL must not use `localhost` unless the download server
actually runs in the RLHF container. The TTS service should use a hostname
resolvable from the RLHF worker, such as the Docker service name.

### 8.3 Ownership after delivery

The boundary is:

> TTS owns temporary generation and time-limited delivery. RLHF owns durable
> storage after a successful download.

RLHF does not send its existing audio to the provider as part of this protocol.

---

## 9. Retention and Acknowledgement

There is no acknowledgement endpoint in the first protocol.

An earlier idea was to let RLHF tell TTS that a result had been imported. This
was rejected to keep the protocol simple. RLHF never sends the downloaded audio
back to TTS, and it does not notify TTS after successful import.

### TTS responsibilities

- retain each completed job archive for a documented duration;
- serve the same archive URL during that interval;
- report when the archive became available;
- preferably report its exact expiration time;
- delete the archive according to its own retention policy after expiration.

### RLHF responsibilities

- poll provider jobs;
- download each archive promptly when that job completes;
- retry transient archive-download failures while it remains available;
- unpack the manifest and import successful files item by item;
- store imported audio durably;
- never rely on TTS for permanent storage.

### Per-job archive retention

Retention begins when the provider finishes and packages that bounded job, not
when the larger RLHF campaign starts. Large campaigns may run for days, so each
provider-job archive has its own retention window.

Capability discovery may advertise:

```json
{
  "result_retention_seconds": 86400
}
```

Each completed provider job should preferably include `ready_at` and
`expires_at`. Explicit archive expiry is safer than deriving it because provider
configuration or operational conditions may vary.

---

## 10. Minimal Batch Download Result

The later REST specification should refine the shape, but a completed provider
job needs approximately:

```json
{
  "job_id": "provider job ID",
  "status": "completed_with_failures",
  "total_items": 1000,
  "successful_items": 998,
  "failed_items": 2,
  "archive_url": "http://tts-service:8090/results/job-123/archive",
  "archive_format": "zip",
  "ready_at": "2026-11-13T15:20:00Z",
  "expires_at": "2026-11-14T15:20:00Z",
  "content_type": "application/zip",
  "byte_size": 184322000,
  "sha256": "optional archive checksum"
}
```

The archive contains a `manifest.json` plus an `audio/` directory. The manifest
records every requested RLHF item ID. Successful entries map to provider-created
safe relative file names and record actual engine, version, and voice metadata.
Failed entries contain an error code, message, and retryable flag.

An archive is still produced when some items fail. A job-level failure means no
usable archive could be produced.

`content_type` and `byte_size` are useful but do not imply an elaborate quality
pipeline. An archive checksum may be supplied and verified but is not mandatory
for the first version.

The client-provided item ID is required for reliable correlation and idempotent
or retry-safe orchestration.

---

## 11. Minimal Import Validation

The first implementation deliberately does not build a TTS quality-auditing
system. It performs only checks needed to establish that a file was received and
can enter the existing audio pipeline:

1. The HTTP request completed successfully.
2. The response body is a non-empty ZIP archive.
3. `Content-Length` or declared byte size matches when supplied.
4. A supplied archive checksum matches, if checksum support is used.
5. The archive contains a valid manifest.
6. Manifest paths resolve to files inside the archive.
7. Each successful audio file can be opened through the canonical audio loader.

The decode check protects against HTML error pages returned with HTTP 200,
truncated files, and other non-audio payloads.

The first version does not perform:

- ASR-versus-requested-text comparison;
- silence detection;
- clipping or loudness analysis;
- language identification;
- pronunciation checks;
- semantic rejection;
- human review or approval.

The provider-job and item states can remain simple:

```text
archive_pending
archive_downloaded
unpacking
imported
partially_imported
failed
expired
```

An item becomes imported after the file is durably stored and decodable.

### Safe local import

The worker should:

1. stream the ZIP to a temporary file;
2. perform archive-level checks;
3. unpack only manifest-declared safe relative paths;
4. validate each successful audio file;
5. atomically move valid files into permanent RLHF storage;
6. create or connect durable audio-asset records;
7. remove the temporary archive and extracted temporary files.

Partial archive downloads and partially extracted files must never appear as
catalogue assets.

---

## 12. Retry and Expiry Behavior

While the archive is not expired, RLHF may retry:

- connection failures;
- timeouts;
- HTTP 429;
- HTTP 500, 502, 503, and 504;
- incomplete transfers.

Examples of permanent or terminal outcomes:

- the provider marks the job permanently failed;
- the archive is corrupt after successful re-download;
- HTTP 404 or 410 is returned after expiry;
- the archive reaches `expires_at` without successful import;
- the provider rejects the requested profile or input.

Per-item synthesis failures are reported in the archive manifest and do not
invalidate successful files. The later REST contract must distinguish retryable
and permanent provider failures explicitly. It must not return success-shaped
results for failed jobs or items.

---

## 13. Campaign and Item Lifecycles

Possible campaign states:

```text
draft
queued
running
downloading
completed
completed_with_failures
cancelled
failed
```

Possible item states:

```text
planned
submitted
provider_queued
provider_running
generated
generation_failed
packaged
imported
retryable_failure
permanent_failure
expired
cancelled
```

The exact state machine belongs in the REST/implementation specification.
Required properties are:

- item-level resumability;
- partial campaign success;
- retry without repeating successful items;
- cancellation of outstanding work without deleting imported assets;
- durable status across coordinator restarts.

---

## 14. SFT Integration

An imported TTS asset provides:

```text
audio asset waveform -> linked utterance text
```

This makes it directly eligible for supervised ASR training when the campaign
permits SFT use.

### 14.1 Independent SFT controls

The admin console must provide tunable settings whose final defaults will be
chosen experimentally:

1. **Maximum TTS share of SFT samples or batches.**
2. **Maximum TTS renditions of one utterance per epoch.**

The first prevents a fivefold stored-data expansion from making most SFT
optimization synthetic. The second prevents five voices reading the same text
from overwhelming textual diversity.

Generating more data does not automatically increase its training weight.
Campaign coverage and SFT sampling ratio are separate concepts.

### 14.2 Training provenance

Each SFT round should record enough provenance to explain its composition:

- recorded versus TTS asset counts;
- unique utterance counts;
- corpus IDs;
- synthesis providers;
- engines and versions;
- voice IDs or profiles;
- effective TTS sampling ratio.

This supports later diagnosis if one engine, voice, or campaign correlates with
a regression.

---

## 15. RLHF Integration

An imported TTS asset enters the existing human-feedback path:

```text
TTS audio
    |
    v
Active Whisper policy generates candidates
    |
    v
Pending hypothesis pair
    |
    v
Human A/B annotation
    |
    v
Reward-model and current-policy DPO datasets
```

The known requested transcript does not itself produce a DPO preference.
Human annotation remains authoritative. Existing guarded, reward-model-ranked
synthetic preference generation remains a separate temporary DPO stage.

### 15.1 Independent annotation-queue control

The admin console must provide a tunable target or maximum TTS share for queue
refill. This is separate from the SFT ratio.

Without explicit sampling, a catalogue expanded fivefold by TTS might cause the
annotation queue to contain almost entirely cloned speech. Refill should be able
to schedule a deliberate mix, for example:

```text
30 recorded assets
20 TTS assets
```

The exact default will be chosen after implementation and testing.

---

## 16. Evaluation Isolation

TTS augmentation is training-only.

The frozen policy-promotion monitor must continue to use real recorded speech
and remain excluded from:

- synthesis source selection where that would leak monitor transcripts;
- SFT;
- human annotation acquisition;
- reward-model training;
- human DPO;
- temporary synthetic DPO sampling.

Changing the speaker or engine does not prevent text leakage. Selection must
respect the underlying utterance identity and the existing monitor exclusions.

TTS-specific evaluation may later be added as a diagnostic dashboard, but it
must not replace real-speech policy-promotion gating.

---

## 17. Admin Console

### 17.1 Campaign creation

The initial form should contain:

- campaign name;
- source corpus;
- selection mode: fixed count, percentage, or all;
- count or target cumulative percentage;
- synthesis profiles chosen from discovered compatible engine/voice pairs;
- intended use: SFT, annotation queue, or both;
- estimated new utterance count;
- expected synthesis-item count;
- current and projected coverage;
- create campaign action.

Example:

```text
Campaign:                 November ADMEDVOICE augmentation
Source:                   ADMEDVOICE training corpus
Selection:                cumulative 40%
Already covered:          20%
New utterances:           2,000
Profiles selected:        5
Expected result files:    10,000
Eligible for SFT:         yes
Eligible for annotation:  yes
```

### 17.2 Global tuning controls

Provide editable settings for:

- TTS provider base URL;
- download timeout;
- maximum parallel downloads;
- maximum TTS share in SFT;
- maximum renditions per utterance per SFT epoch;
- target or maximum TTS share in annotation refill.

The ratio values are intentionally undecided. The requirement is to make them
operator-tunable after implementation.

### 17.3 Campaign status

Display:

- total planned items;
- submitted, generating, packaged, imported, failed, and expired
  counts;
- result-expiry pressure;
- selected profiles;
- retry failed items;
- cancel outstanding items.

### 17.4 Deferred auditing UI

An interface for listening to and approving/rejecting synthetic examples would
be useful, but it is not part of the first iteration. The first implementation
should not build review workflow, semantic quality scoring, or quality
heuristics merely to anticipate it.

The data model should preserve enough provenance that a later audit interface
can identify campaign, utterance, provider, engine, voice, and stored asset.

---

## 18. Loose-Text and Rare-Word Coverage Module

The system should eventually support texts entered specifically to repair
coverage, for example:

- rare medical words;
- drug names;
- dosages and numbers;
- abbreviations;
- error-derived phrases;
- manually curated terminology.

This thread is frozen for now. The initial architecture only needs to ensure it
is possible later.

The likely future workflow is:

1. operator enters or imports custom texts;
2. normalization and duplicate detection run;
3. texts become `custom_coverage` utterances;
4. an ordinary synthesis campaign selects them;
5. imported audio uses the same SFT and annotation controls as transcript-based
   augmentation.

Potential future metadata includes tags such as `drug-name`, `dosage`, and
`abbreviation`, but no such schema or UI is required now.

---

## 19. Scale and Performance Assumptions

Design for approximately five synthesis profiles across the ADMEDVOICE corpus,
producing roughly five times as many audio assets as the original recorded
corpus.

Consequences:

- campaign work must be itemized and persisted;
- provider submissions must be batched;
- status APIs need pagination;
- downloads must stream to disk rather than buffer full audio in memory;
- import must be restart-safe;
- successful items must not be repeated after partial failure;
- database queries and indexes must support campaign/status filtering;
- training samplers must avoid loading all assets into memory unnecessarily;
- generation coverage and training sampling weight must remain independent.

The web process should not perform synthesis or bulk downloads synchronously.
A coordinator or dedicated background integration worker should own network and
import work, preserving the existing CPU-only web-container boundary.

---

## 20. Reproducibility and Provenance

The RLHF side does not require the same TTS request to reproduce an identical
waveform, and it does not validate TTS determinism.

It should still store:

- provider;
- engine and reported version;
- voice ID;
- requested parameters;
- provider-reported seed if one exists;
- result identifiers;
- downloaded asset identity and optional checksum.

This records what produced the imported waveform without restricting the TTS
implementation.

Campaign selection reproducibility is different: RLHF should record selected
utterance IDs and the selection seed so the source set is explainable.

---

## 21. Settled Decisions

The following decisions should be treated as the current design baseline:

1. TTS is an external REST service and implementation details remain its
   owner's responsibility.
2. Services initially run on one trusted machine or safe subnetwork.
3. URL-based ZIP archive delivery is used so the same protocol can later work
   across machines without thousands of per-file downloads.
4. Large campaigns are split into bounded provider jobs; each completed job
   produces one archive containing audio plus a per-item manifest.
5. RLHF downloads archives promptly, unpacks them, and owns durable storage.
6. TTS archives are temporary and expire under a provider-owned retention
   policy.
7. The first protocol has no successful-download acknowledgement endpoint.
8. RLHF sends text and discovered synthesis selections, not existing recordings
   or cloning references.
9. Cloned voices are pre-provisioned and represented by provider voice IDs.
10. Campaign selection supports fixed count, percentage, and all.
11. Percentage is a cumulative coverage target that preserves old samples.
12. Campaign membership is frozen and persisted.
13. Variation is represented by engine/voice synthesis profiles, not merely an
    integer variant count.
14. Design for roughly five profiles and fivefold audio growth.
15. Import validation is intentionally minimal: successful non-empty archive,
    valid manifest, and decodable audio, with optional size/checksum verification.
16. Quality heuristics and human synthetic-audio audit are deferred.
17. SFT and annotation-queue TTS ratios are independent, admin-tunable settings.
18. Final ratio defaults will be selected after implementation.
19. Known transcripts are the first synthesis text source.
20. Custom loose texts for rare-word coverage are planned but deferred.
21. TTS audio may enter SFT and the human RLHF queue.
22. A known requested transcript does not automatically become a DPO
    chosen/rejected label.
23. Policy promotion remains based on frozen real recorded speech only.

---

## 22. Explicit Non-Goals for the First Iteration

- Building the detailed rare-word/custom-text authoring module.
- Building a synthetic-audio listening and approval interface.
- Automatically judging whether TTS pronounced the requested text correctly.
- Language identification, silence, clipping, loudness, or pronunciation
  scoring.
- Uploading RLHF audio to enroll cloned voices.
- Sharing filesystem paths between services.
- Requiring deterministic waveform generation.
- Making TTS storage permanent.
- Adding a result acknowledgement callback.
- Replacing the real-audio WER Monitor.
- Automatically creating DPO labels from requested transcripts.
- Finalizing training-ratio defaults before empirical tuning.

---

## 23. Migration Direction

Implementation should be staged to preserve current behavior.

### Phase 1: generic catalogue foundation

- introduce corpora, utterances, and audio assets through migrations;
- migrate existing audio rows and transcripts without changing their meaning;
- preserve IDs or provide reliable mappings for existing pairs and annotations;
- keep ADMEDVOICE and Common Voice importers working as adapters;
- preserve evaluation-set and audio-exclusion invariants;
- update queries to use generic eligibility rather than hard-coded dataset
  names.

### Phase 2: provider and campaign orchestration

- add provider configuration and capability discovery;
- add synthesis profiles, campaigns, and items;
- implement background submission, polling, expiry, retry, and download;
- implement durable local storage and minimal validation;
- expose campaign creation and status in the admin console.

### Phase 3: training and annotation integration

- add origin-aware SFT sampling and rendition-density controls;
- add origin-aware queue-refill sampling;
- record training snapshot provenance;
- test that monitor exclusions cover new catalogue queries;
- verify current-policy pair and DPO invariants for TTS assets.

### Deferred phases

- custom coverage text authoring;
- synthetic-audio audit and review;
- richer quality diagnostics;
- automated curriculum based on ASR errors or terminology coverage.

---

## 24. Risks and Required Safeguards

### Text leakage

The same underlying sentence must not enter training through TTS when its
utterance belongs to the frozen promotion monitor. Audio-origin checks alone are
insufficient.

### Synthetic dominance

Fivefold storage growth can silently dominate SFT and annotation sampling.
Independent, enforced sampling controls are mandatory.

### Repeated-text dominance

Multiple voices of one sentence increase acoustic variety but not linguistic
variety. Preserve utterance identity and cap renditions per epoch.

### Expired results

Large campaigns require prompt per-job archive download. Jobs must be bounded so
an expired or interrupted transfer does not require repeating or downloading the
whole campaign.

### Container addressing

Provider result URLs must be resolvable from the worker process. `localhost`
often points to the wrong container.

### Partial import

Download to temporary files and atomically publish only complete, decodable
audio.

### Provider drift

Store engine and voice provenance so later changes can be traced and problematic
profiles can be disabled.

### Schema migration safety

Existing annotations, pairs, audio exclusions, monitor membership, and training
history must remain valid. Do not replace the current table without a migration
and compatibility plan.

---

## 25. Next Required Design Task: REST Specification

The next discussion should convert this architecture into a precise provider
contract covering:

1. API versioning.
2. Capability-discovery endpoints and schemas.
3. Engine and voice compatibility representation.
4. Submission endpoint and batch limits.
5. Client correlation IDs and idempotency semantics.
6. Asynchronous provider job and item statuses.
7. Polling endpoints, pagination, and recommended polling behavior.
8. Partial success.
9. Completed-job archive metadata.
10. ZIP manifest schema and safe file mapping.
11. Archive download behavior while retained.
12. `ready_at`, `expires_at`, and retention guarantees.
13. Retryable versus permanent job and item errors.
14. Cancellation semantics.
15. Text encoding, language values, and maximum input lengths.
16. Audio and archive format requirements.
17. Compatibility with current trusted-network deployment while leaving room
    for future authentication headers.

That specification should remain engine-agnostic and should not require features
that capability discovery marks optional.

---

## 26. Implementation-Agent Onboarding Checklist

Before implementing this design, an agent should:

1. Read `CONTRACT.md`, especially audio, current-policy DPO, language-lock, and
   frozen-monitor invariants.
2. Inspect `storage/schema.py`, all migrations, and the relevant
   `FeedbackStore` catalogue and learning-data queries.
3. Trace catalogue ingestion through `prepare_data.py`,
   `inference/hypothesis_generator.py`, and `run_generate_pairs.py`.
4. Trace SFT, human DPO, reward-model, temporary synthetic-DPO, and monitor
   dataset queries separately.
5. Identify every location that assumes ADMEDVOICE/Common Voice names or stores
   transcript text directly on an audio row.
6. Design a backwards-compatible migration before changing runtime behavior.
7. Add tests proving that recorded-audio behavior is unchanged.
8. Add tests proving that monitor utterances and assets cannot enter any TTS
   learning path.
9. Add tests for cumulative campaign coverage, frozen selection, profile
   matrices, restart safety, partial success, expiry, and atomic download.
10. Add tests for independent SFT and annotation TTS sampling controls.
11. Run the narrowest relevant suites throughout implementation.
12. Run `py -3.12 -m pytest -q` before completion.

---

## 27. Concise Reference Flow

```text
1. Existing transcripts become reusable utterances.
2. Existing recordings become recorded assets linked to utterances.
3. Admin chooses fixed/count/all cumulative coverage.
4. Admin chooses discovered compatible engine/voice profiles.
5. RLHF freezes the selected utterance/profile item matrix.
6. A background worker submits bounded asynchronous provider jobs.
7. TTS synthesizes each job and packages audio plus a manifest into one ZIP.
8. TTS retains each completed archive for a known period.
9. RLHF promptly downloads and safely unpacks each completed archive.
10. RLHF verifies decodable files and stores them atomically.
11. Imported TTS assets enter capped SFT and/or capped annotation refill.
12. Human annotation remains the authority for RLHF preferences.
13. Real frozen monitor audio remains the authority for policy promotion.
```
