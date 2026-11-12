# Release provenance

This release identifies `h3_gta555_decfg_20261013T190132Z`, **step 4000**. All four tensor/state payloads retain their original bytes and [checksums](../provenance/checkpoint.json). The published trainer metadata replaces three private filesystem paths with portable paths; its original file hash is recorded separately.

Core training files were recovered from historical snapshots and compared with the original run's source hashes. [source_files.json](../provenance/source_files.json) records the source hashes, exact historical matches, release edits, and final hashes. The training entrypoint and core model/data/checkpoint/DeCFG/sampler/native-cache files match before portable-path changes. Shared helpers without recorded historical hashes are marked accordingly. The vendored Diffusers runtime is separately hashed; no unavailable upstream Git revision is claimed.

Public packaging adds dataset extraction, model download, launchers, and evaluation preparation. The long-inference contract accepts the selected case count and checks published checkpoint fingerprints directly, replacing dependencies on a private 100-case suite and training-manifest file. Its denoising, latent-prefix continuation, and decoder algorithms are preserved. Later actor-attention, pose-attention, and DMD models are not used.

The public 5000-clip dataset is a subset of the original 8552 clips. Exact retraining to the released weights is not claimed. Saved optimizer state does not replace the original sample stream and settings.

The eight cases preserve recorded input references, Proxy files, seeds, and window prompts. Result previews are the project page's compressed encodes, not lossless outputs for metric computation. Each case records the original full-resolution output hash.

Checks cover CPU conversion/training/checkpoint/sampling behavior, public-data extraction, complete 17-window input preparation, prompt/seed/hash identity, and full decoding of all released videos. See [release_validation.json](../provenance/release_validation.json) for the measured checks and their scope. Full-model GPU training and inference were not rerun during packaging.
