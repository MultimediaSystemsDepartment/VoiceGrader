# Source provenance

Imported on 2026-11-23 from local source snapshots. Original folders were not
modified. No remote repository or release was fetched for this import.

## RLHF

- Source: `C:\Users\akkur\Documents\Git\rlhf-for-asr`.
- Git HEAD: `f4a45b6829fc2a1dbe7471327eadea278c0a7e13`.
- Imported working-tree content of tracked files, except
  `config/admin_password.json`.
- Also imported the existing untracked design documents
  `docs/TTS_AUGMENTATION_DESIGN.md` and `docs/TTS_REST_API_BRIEF.md`.
- Editor configuration, ignored runtime state, data and model weights excluded.
- GPU `Dockerfile` relocated to `training/Dockerfile`, without content changes.
- Web/training application logic and existing tests were unchanged at import
  time. Subsequent VoiceGrader changes are tracked in the implementation plan.

The snapshot therefore includes two design documents beyond the listed commit;
it is not represented as an exact checkout of that commit.

## ASR

- Source: `C:\Users\akkur\Documents\Git\asr_workflow-main`.
- The supplied folder has no Git metadata; no upstream commit is asserted.
- Application, deployment references, utility scripts and third-party notices
  imported without modifying their content.
- Excluded caches, environments, model/checkpoint directories, local `data/`
  contents and the two transcript fixtures in `util/caise-asr-test/`.
- Every copied source file was SHA-256 compared with its local source during
  import. This checks copy integrity, not upstream authenticity.

## Publication boundary

Credentials, recordings, transcripts from local datasets, trained model
weights and runtime state are not distributable source fixtures.
Confirm application licenses, model licenses and bundled dependency notices
before publishing the archive. No new license is assigned by this import.
