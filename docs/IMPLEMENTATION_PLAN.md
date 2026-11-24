# VoiceGrader implementation plan

Updated: 2026-11-23.

Working roadmap, not a completed-feature specification. It may be removed or
replaced once implementation and release documentation cover these decisions.
Update milestone status and validation evidence as work progresses.

## Current state and next scope

Local TTS source integration now supplies a separate authenticated REST v0.7
service and CPU Piper. The local adapter verifies archive/item identity,
checksums and actual voice, while retaining existing external campaigns on
their original adapter. Reference assets live under
`_data/tts/reference_voices/Arek.wav` and matching `Arek.txt`; no private voice
has been supplied. GPU profiles are staged only: no GPU weights were downloaded,
no XTTS terms accepted and no GPU TTS engine launched. Source redistribution
licensing remains unresolved. CPU live validation is recorded in the root README;
staging GPU profiles is not GPU synthesis/voice-quality certification.

Repository/service skeleton is complete:

- Three components: RLHF web, training worker and ASR.
- Shared RLHF source for the web and training images.
- Root Compose configuration and joint README.
- Original source folders unchanged; development takes place in VoiceGrader.
- Imported ASR code has minimal publication/default-selection/status wiring.
- RLHF web/training images built and deployed for approved local testing.
- Isolated Large-v3 short-clip GPU proof passed; ASR restored after GPU work.
- Six fresh pairs available in the regular UI from an explicitly approved corpus.

**Milestones 2 and 3 have CPU foundations and the first approved GPU/import
proof. Milestones 4-7 are implemented with local regression coverage.**
Augmented learning and approved converted-model interoperability remain
unverified; this document does not itself authorize those operations.

The CPU configuration/compatibility slice of 2 and the database foundation of 3
are implemented. Bounded short-clip GPU execution is now verified; long-clip
capacity and the full coordinator campaign remain unverified. The first generic-ingestion
workflow was agreed and implemented: local manifests, recorded audio with optional
transcripts, CLI reporting and explicit corpus activation. The first browser
corpus-management page now lists corpora, previews recordings and controls
activation; standalone TTS text collections remain out of this scope.

Foundation validation at that stage: all 309 RLHF tests and both packaging tests passed; all 10 Jinja
templates compiled. At that foundation stage all 67 imported ASR files were
identical to the source snapshot. RLHF tests used process-local `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`
because an unrelated global pytest plugin could not load its native dependency.
No model downloads, actual Large-v3 processing or GPU training were tested
at that foundation stage.

The subsequent generic CLI import slice passed all 325 RLHF tests and both
packaging tests. Duration acceptance was verified at 0.1 and 30 seconds and
rejection at one sample beyond either boundary; no GPU or live services ran.

The browser corpus-management slice passed all 330 RLHF tests; all 11 Jinja
templates and changed Python modules compiled. JavaScript syntax passed using
VS Code's bundled runtime. HTTP tests verify admin access, activation,
holdout-aware counts, pagination and correct ZIP playback. A temporary standalone
browser fixture could not load local assets, so it is not live visual validation.
No application services, containers or API requests were started.

Admin authentication is now enforced and showcase mode is removed. The public
initial password must be changed before access beyond localhost.
Subsequently the user authorized local builds/restarts, missing model downloads,
read-only selection of eight original recordings and temporary ASR pauses for
the isolated GPU proof. The user subsequently approved one small TTS API sanity
check; its outcome and contract mismatch are recorded below.

### Approved operational proof and browser test dataset

The refill initially remained queued because no worker was running and the live
database was empty. Duplicate browser refill requests now redirect to an escaped
admin notice; JSON clients still receive HTTP 409.

The isolated proof used six recordings for learning and two distinct held-out
recordings. Human A/B choices and approved dataset-reference SFT targets were
recorded through UUID4 factories/storage helpers. Actual generation, reward
training, SFT and DPO saved finite weights; SFT made six updates and final DPO
made five, with one explicitly logged AMP-overflow skip. The final DPO started
from stock Large-v3 with an identical frozen FP16 reference (initial offset zero).
No candidate was activated or published. WER was 0% before/after on two clips:
execution evidence only, not an improvement claim or a passing promotion gate.

| Stage | Wall time | Sampled peak GPU memory |
| --- | ---: | ---: |
| Generation | 58.66 s | 7,340 MiB |
| Reward training | 70.26 s | 3,536 MiB |
| SFT | 11.23 s | 8,948 MiB |
| DPO/evaluation, final attempt | 18.55 s | 14,030 MiB |

The proof fixed DPO's counting of AMP-skipped updates and implicit adapter
recovery despite an initialized stock-model deployment. Earlier DPO attempts
are retained as superseded validation attempts, not accepted final proof.
Weights are cached persistently. Current regression evidence: 340 RLHF tests,
two packaging tests and all 11 templates compiled.

With separate explicit approval, the six training recordings were imported and
activated as `large-v3-functional-test` in the regular UI. One bounded invocation
of the real coordinator refill generated six fresh pairs, cleared the request
and exited. No proof annotations/checkpoints or held-out clips were transferred.
ASR was restored; no automatic coordinator/training remained running at that stage.
The web's automatic refill signal initially requeued work because six pairs are
below its default minimum of 20. For this one-batch test, local automatic refill
is disabled (`queue_min_size=0`, previously 20); the normal batch setting remains
50.

Subsequently the user completed six official-interface annotations and approved
the normal coordinator. The original six-clip pool was exhausted for the current
policy; neither a stopped worker nor an auto-refill minimum of 0 could replenish it.
Imported 100 additional unique 1.92-7.00 s recordings using the recorded-manifest
validator, excluding every prior proof recording/text and preserving the archive.
Enabled minimum 20/batch 50 and paused ASR for the continuous GPU session.
The web automatically queued work; the coordinator produced 30 distinct pairs
from 50 inputs, omitting 20 identical-candidate cases. All 30 audio responses
were playable mono 16-kHz WAVs. Six human annotations remained intact, stock
policy unchanged, training threshold 50, and TTS shares 0%. Full regression:
363 RLHF tests passed. Genuine model promotion still needs a frozen monitor.

## Original campaign implementation agreement (2026-11-23)

At that stage the provider-source-dependent batch API work was postponed, but functional
campaigns should use the currently working UI API in the meantime. Implement
only the necessary temporary provider adapter; keep its endpoint/payload, voice
mapping and individual-result download contract isolated and explicitly marked
for review when the provider source arrives. Do not claim v0.7 batch/ZIP
compatibility.

- Implement persistent campaign controls and current-API execution, not mocked
  progress or permanently disabled synthesis controls.
- Persist accepted provider job IDs and resume polling after restart. An
  ambiguous submission timeout requires operator review; never automatically
  resubmit an unknown-outcome POST.
- Ship independent SFT/refill TTS shares at 0%; operators enable participation
  explicitly. Shares are hard caps, not guaranteed targets. Initially allow at
  most one TTS rendition per utterance per SFT epoch. Known requested text is an
  SFT target when permitted, never automatic human DPO feedback.
- Implement the HTTP publication/artifact bridge for separate-host deployment.
  Use a separate service credential rather than the human admin password.
- Both file and live ASR default to the same installed approved publication;
  retain existing explicit stock-model choices.
- Check/update at ASR startup. The later approved extension adds drained worker
  hot reload for later updates, without interrupting existing files/live sessions.
  Failed updates restore the previous installed pointer and runtime;
  no approved publication means an explicit stock-model bootstrap state.
- Normal publication requires monitor approval; isolated proof adapters must
  not be silently promoted.
- Defer full ecosystem/release validation until these pieces exist, but run
  focused regression and bounded component checks during implementation.

These decisions are now implemented. Local validation: 363 RLHF tests passed;
ASR/packaging tests cover staged updates, rollback, explicit stock selection and
failed GPU load preservation. A real approved-model conversion/transfer remains
unverified. A later user-authorized 10-item Piper campaign completed through the
CPU web dispatcher with GPU training stopped.

## Architecture and constraints

Authentication validation: all 336 RLHF tests passed; all 11 templates and
changed Python modules compiled. The headless password tool entry point ran
successfully. Builds and service starts were not authorized for that earlier
authentication slice; later operational testing is recorded above.

| Component | Responsibility |
| --- | --- |
| RLHF web | Annotation, admin/campaign controls, status, authenticated publication delivery, CPU campaign submission/collection and HTTP dispatch for requested quality reviews |
| Training worker | Generation, reward training, SFT/DPO, evaluation and model conversion |
| ASR | File/live transcription, reusable SNR/WhiSQA quality endpoint, startup update and drained model-worker reload |

TTSCompare is external, with provisional base URL
`https://ttscompare.rmp.caise.pl`. Keep it configurable. An approved standalone
sanity check completed one short generic Polish sentence using only Piper
(`pl_PL-gosia-medium`): one submission, one status poll, one result request and
one WAV download. Samples were finite/non-silent: 22,050 Hz, 1.950476 s, 86,060
bytes. Provider metadata reports real TTS and 0.155266 s synthesis time. No
corpus data, additional engines, retries or learning imports were used.

The observed working interface is the older UI API:
`POST /api/v1/tts/jobs`, `GET /api/v1/tts/jobs/{job_id}`, and `/result`.
Its single-text/engine-array payload, HTTP 200 acceptance and individual audio
URLs differ from the supplied RLHF `request_id/items`, HTTP 202 and ZIP contract.
`GET /v1/capabilities` returns HTML HTTP 200; `/api/v1/capabilities` returns JSON
HTTP 404. Therefore the documented batch contract is not verified at this URL.
Confirm its deployment/location with the provider before implementing the
batch client; JSON content/shape must be verified, not just HTTP success.

Other engines, batching, request idempotence, retention and ZIP manifests remain
untested. Missing provider source still does not prevent planning, but a fully
self-hosted release requires that source/image and redistribution rights.

Preserve these boundaries throughout:

- Make changes only in VoiceGrader; leave original source folders untouched.
- Alter ASR as little as possible; no UI redesign or inference-backend rewrite.
- Keep web CPU-only, with no ML subprocesses or Docker socket.
- Preserve existing annotations, IDs, exclusions and training history.
- Use migrations and the existing storage layer for database changes.
- Preserve Polish/transcribe decoding and label-prefix invariants.
- Keep policy/reference checkpoint and precision consistency for DPO.
- Decode RLHF audio through the shared audio loader.
- Keep TTS audio distinct from temporary synthetic preference artifacts.
- Human preferences remain authoritative.
- Frozen real-speech evaluation remains the promotion authority.
- Treat evaluation text and all linked renditions as excluded from learning.
- Runtime data, credentials and model weights stay outside the source archive.

See the [RLHF contract](../rlhf/CONTRACT.md),
[existing TTS design](../rlhf/docs/TTS_AUGMENTATION_DESIGN.md) and
[provider documentation](../_dev_documentation/TTSCompare_REST_v0.7_opis_dla_RLHF_v3.md).
Resolve disagreements using verified current code/tests and observed provider
behavior, then update active documentation. Provider examples are not evidence
of live responses.

## Milestones

### 1. General-use baseline

Status: authenticated baseline implemented and locally deployed; public initial
password must be changed before broader access.

- Remove showcase functionality entirely; always require admin login.
- Ship the public initial password `change-me`; allow optional password changes
  through the admin UI and shared desktop/headless tools.
- Persist salted hashes outside Git in the shared output volume, independent of
  model lineage; fail closed on invalid configuration and revoke sessions on reset.
- Establish shared configuration conventions and document responsibilities.
- Preserve current annotation, training and ASR behavior.

Acceptance: existing tests pass; participant identities are preserved, admin
authentication is enforced, and changed passwords survive restarts. Change the
public initial password before exposing the system beyond localhost.

### 2. Large-v3 readiness

Status: CPU configuration/compatibility slice and bounded short-clip GPU smoke
test passed. Long-clip sizing and full coordinator campaign remain unverified.

First implementation slice, without GPU execution:

- Trace model configuration through generation, reward encoding, SFT, DPO,
  evaluation and publication; distinguish policy and reward-encoder choices.
- Align the intended Large-v3 policy configuration and model-specific processor.
- Verify feature handling supports the model's 128 mel bins instead of assuming
  Medium's 80.
- Validate adapter/base-model compatibility at relevant load/resume boundaries.
- Create an explicit fresh Large-v3 lineage; never reuse Medium adapters or
  silently select them through checkpoint recovery.
- Preserve Medium checkpoints, feedback and history. Record the current policy
  identity so existing feedback is not treated as newly generated Large-v3 data.
- Add focused regression tests for changed behavior.

The approved small GPU slice is completed (evidence above). Further capacity
work, with execution approval:

- Extend generation/SFT/DPO/evaluation checks to representative longer clips.
- Measure peak VRAM on representative-length clips.
- Choose microbatch/accumulation settings while preserving DPO reference
  consistency and effective batch intent.
- Keep ASR GPU workloads separate during the initial test.

Acceptance: configuration and compatibility tests pass; a fresh lineage cannot
load incompatible adapters. Hardware readiness is not declared complete until
the approved small GPU round succeeds.

Dependency: does not require a working TTS service or catalogue migration.

### 3. Catalogue foundation for augmentation

Status: database foundation, CLI recorded-audio importer and validated browser
Training/Evaluation ingestion implemented; broader archive/text ingestion remains pending.

Text identity uses Unicode NFC and collapsed whitespace, preserving case,
punctuation and numbers. Legacy asset IDs and relationships are retained.
The shared learning selection excludes monitor-linked utterances and matching
verified monitor text. This is not fuzzy/near-duplicate detection. Synthesized
assets can be catalogued but remain ineligible for learning until milestone 6.

Original CLI ingestion slice (retained):

- Shared local folder/ZIP plus standard CSV; no browser uploader.
- Recorded audio with optional reference transcripts; no standalone text corpus.
- Header probing and shared-loader decoding before database registration.
- Reject clips outside 0.1-30 seconds; no implicit truncation in this importer.
- Reject the whole import if any row is invalid and report errors explicitly.
- Atomic corpus/asset registration, inactive by default; explicit activation.
- Path-based idempotence without replacing existing transcripts or IDs.
- Deactivate a corpus before appending recordings; preserve existing importers.
- CLI list/import/activate/deactivate first; simple browser controls afterward.

The browser slice adds admin-gated corpus summaries, 25-recording previews,
reference text/provenance, playback and explicit activation/deactivation.
It shares learning eligibility with the existing storage view and never runs
GPU work. Nested ZIP source paths and
same-named archives from different corpora retain distinct audio/cache identity.
Admin authentication is now enforced. The public initial password must still
be changed before broader deployment.

The subsequent self-service slice adds Training/Evaluation tabs and bounded
audio-plus-CSV uploads using the shared validator, staged preview, explicit
acceptance/cancellation and atomic imports. Evaluation requires transcripts
and learning-overlap checks; migration 009 permanently excludes evaluation-purpose
corpora from learning. Imported monitor transcripts remain unverified until reviewed,
and freeze/baseline/promotion-gate selection reuse the existing monitor workflow.
DPO now consumes the designated frozen monitor directly for before/after WER,
without operator CSV export. Existing annotations and monitor revisions are retained.
Validation: 379 RLHF tests passed and 13 templates compiled. The rebuilt web/training
services preserve all 55 live annotations, the 40-clip frozen monitor and threshold
100. A browser evaluation upload passed validation and playable-audio preview,
then was cancelled without adding a corpus or monitor.

- Separate utterance identity from individual audio renditions.
- Introduce generic corpus membership, recorded/TTS origin and provenance.
- Plan incremental migrations and compatibility queries before changing tables.
- Preserve existing audio IDs or explicit mappings, pair relationships,
  annotations, curation history, monitor membership and training history.
- Keep existing ADMEDVOICE/Common Voice import behavior working.
- Make learning eligibility generic rather than adding more hard-coded dataset
  branches.
- Prevent held-out text from re-entering learning through another recording or
  a TTS rendition. Define and test how duplicate text identities are handled.
- Ensure future campaign selection uses eligible learning text only.

Acceptance: multiple renditions can share one utterance; legacy records retain
their meaning and relationships; migrations preserve existing state; evaluation
exclusions cover related renditions and synthesis source selection.

Dependency: foundational for milestones 4-6; no live TTS access required.
This is the most consequential database change and must be validated separately.
The emergency ZIP snapshot is a fallback, not a replacement for safe migrations.

### 4. TTS client and durable import

Status: implemented against the temporary current jobs API; fixture-tested and
live-verified with a user-authorized 10-item Piper campaign. Submission, collection
and validated import completed with GPU training stopped. The supplied batch API
is deferred.

- Configure provider URL and explicit engine/voice profiles; do not infer
  capabilities from the incompatible documentation endpoint.
- Persist campaign, provider-job and item state.
- Freeze campaign membership and engine/voice profiles.
- Submit one utterance/profile per current-API job, with bounded worker batches.
- Persist accepted IDs, poll with bounded backoff and download promptly.
  Unknown-outcome POSTs require operator review, not automatic retries.
- Match results to frozen campaign items/job IDs.
- Validate individual result/audio responses and decoded samples; batch ZIP
  and provider-supplied manifest checksums are not implemented for this API.
- Safely stage imports and publish only complete valid assets.
- Handle partial failures, expiry, restart/resume and retries without duplicates.
- Treat returned URLs as untrusted; keep downloads within approved origins.
- Reconcile documented all-items-failed semantics before live integration.

Acceptance: a small contract-fixture campaign survives worker interruption;
successful items import once; failed/expired items remain explicit. Confirm
actual interoperability later using approved non-sensitive input.

Dependency: milestone 3. Live validation requires separate approval and access.

### 5. Functional campaign UI

Status: implemented and HTTP/template-tested. The deployed admin page and
read-only work estimate were verified in the browser; no campaign was submitted.

- Source corpus and intended use: SFT, annotation queue or both.
- Fixed count, cumulative percentage or all eligible utterances.
- Compatible engine/voice profiles and estimated work/coverage.
- Campaign progress, explicit errors and retry controls.
- Cancellation only within provider-supported/local semantics; do not promise
  cancellation of remote synthesis without a provider cancellation API.

Acceptance: operators manage campaigns without scripts; increasing cumulative
coverage selects missing combinations rather than duplicating existing work.

Dependency: milestone 4. Preserve current UI style; no visual redesign or
custom terminology authoring.

### 6. Controlled learning integration

Status: implemented and sampling/eligibility-tested; real augmented training pending.

- Independent TTS participation limits for SFT and annotation refill.
- Cap renditions of the same utterance per epoch.
- Record corpus, origin, utterance and synthesis provenance in training snapshots.
- Let TTS assets enter existing hypothesis generation and human feedback.
- Do not turn requested TTS text into automatic human DPO preferences.
- Preserve current-policy eligibility, synthetic-preference caps and promotion
  gates.
- Ship independent SFT/refill limits at 0%; enable only explicitly.
  Permit at most one TTS rendition per utterance per SFT epoch.

Acceptance: measured training/refill composition respects configured limits;
augmentation cannot silently dominate or contaminate frozen evaluation.

Dependency: milestones 3-4 and validated policy configuration from milestone 2.

### 7. Model publication and minimal ASR bridge

Status: implemented and fixture-tested; stock Large-v3 drained reload and real
file inference verified locally. Approved-model transfer and live audio
transcription remain to be validated end-to-end.

- Merge only an approved adapter into its exact base model.
- Convert to CTranslate2 in the training worker, not the CPU web process.
- Publish immutable artifacts with publication identity, provenance and checksums.
- Test converted-model behavior; adapter evaluation alone does not verify the
  final inference artifact.
- Expose publication metadata and artifact delivery through a governed endpoint.
- Configure one installed model path for ASR file and live transcription.
- Compare publication identity rather than relying only on timestamps; deliberate
  rollback must propagate too.
- Stage and verify downloads before switching versions.
- Use a controlled update/restart mechanism outside transcription logic; retain
  the previous artifact and explicitly recover from failed activation.
- Report the loaded publication identity.

Acceptance: a verified trained model reaches both ASR paths; failed download,
verification or activation does not replace the usable installed version.

Dependency: milestone 2 and publication controls. May be developed separately
from TTS augmentation once a suitable approved checkpoint exists.
ASR changes cover configuration, identity/update wiring, drained runtime control
and reuse of existing quality scorers through an authenticated endpoint.

### 8. End-to-end validation and release preparation

Status: pending.

- Exercise augmentation, annotation/SFT, DPO, evaluation, publication and ASR.
- Test failure recovery, backups and restoration.
- Establish reproducible dependency/model versions and supported environments.
- Confirm application, dependency, model, voice and generated-audio
  redistribution/use rights.
- Document clean installation, configuration, training and update operations.
- Provide an approved non-sensitive reproducible example for SoftwareX.
- Distinguish a source archive from image archives and runtime backups.

Acceptance: a clean installation reproduces the documented workflow; release
scope, external dependencies and licensing are explicit.

Dependency: milestones 1-7 for the complete ecosystem.

## Validation approach

- Run the smallest relevant existing regression tests for each change.
- Add tests for introduced behavior and meaningful failure cases.
- Run the complete RLHF suite before finishing RLHF implementation milestones.
- Validate Compose without starting containers when only packaging changes.
- Do not infer GPU/API functionality from configuration tests.
- Record commands, outcomes and remaining unverified assumptions per milestone.
- Do not introduce destructive runtime migrations or rely on an emergency revert
  as normal error handling.

## Deferred scope

UI redesign, custom rare-word/terminology authoring, automatic quality rejection,
automatic GPU scheduling, zero-downtime parallel model swapping, Kubernetes-specific
orchestration and TTS provider implementation are outside the first integration.

The later approved extension implements synthesized-audio quality scoring/review
(report-only) with the existing ASR scorers and drained file/live worker reload.
Reload pauses admissions, preserves existing operations, has a drain deadline and
restores the previous pointer/runtime on failure. Quality requests/results/errors
and human notes persist; neither low scores nor offline ASR change learning eligibility.

## Completion record

| Milestone | Evidence |
| --- | --- |
| Repository skeleton | Two packaging tests and 286 RLHF tests passed; imported sources checksum-verified; no runtime execution |
| 2, CPU slice | Large-v3 defaults and isolated state, adapter/config/activation guards, 80/128-bin collator tests; no model downloads or GPU execution |
| 3, database foundation | Generic corpora/utterances, preserved asset IDs, atomic migration with rollback/concurrency tests, holdout text exclusions, legacy/new export merging |
| 1 | Admin authentication/password persistence implemented and locally deployed |
| 2, GPU proof | Six short training clips, two holdout clips; real reward/SFT/DPO execution, no activation or quality-improvement claim |
| 3, recorded import/UI | Generic manifest import, duration validation, activation and playable regular-UI pairs |
| 4-6, implementation | Migration 008, current jobs adapter/worker, durable import, campaign/review UI and independent default-zero sampling caps; 363 RLHF tests passed |
| 7, implementation | Authenticated immutable publication, converted WER check, staged ASR update, CUDA load validation, rollback identities and file/live defaults; no approved-model live transfer yet |
| Reload/quality extension | 385 RLHF tests, 19 ASR/packaging tests and 13 compiled templates; real-process shutdown regression; local GPU quality assessment, live-session drain/admission rejection, stock Large-v3 reload and post-reload file transcription passed; campaign persistence/review fixture-tested, no approved-model transfer |
| Independent CPU campaign dispatch | 386 RLHF tests passed, including web-only campaign lifecycle regression; removed coordinator campaign loop; deployed web/training images; all 10 previously pending user-authorized Piper items completed with job IDs, imported audio and no errors while training stayed stopped and ASR stayed running; 55 annotations, frozen 40-clip monitor, threshold 100 and default-zero TTS caps unchanged |
| TTS Augmentation review UI | Renamed navigation/main headings; responsive recording cards, readable quality metrics and collapsed technical data; existing-job collection retry without synthesis; explicit new-synthesis action with hover/focus risk warning instead of a checkbox, backend acknowledgement retained. All 387 RLHF tests, 19 root tests and 13 template compilations passed. Browser fixture verified 1440/1024/768/390-pixel layouts without page/card overflow, null versus zero scores and mouse/keyboard warnings; web image deployed without triggering retries |
| 8 | Deferred full ecosystem/release validation |
| Normal auto-refill session | 100 additional recorded assets; web-triggered refill completed with 30 playable pairs; minimum20/batch50, normal worker running, ASR paused; original six annotations/holdout/archive preserved |
