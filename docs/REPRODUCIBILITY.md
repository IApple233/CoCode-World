# Checkpoint and validation

The released checkpoint is **CoCode-World step 4000**. Inference uses its EMA LoRA with the official MiniMax-H3 base. Tensor and optimizer payload checksums are listed in [checkpoint.json](../provenance/checkpoint.json).

The original training used 8,552 clips; the public dataset contains 5,000 selected clips. The released recipe supports further training, but the public subset does not reproduce the original sample stream. Optimizer resume requires matching data and training settings.

The eight evaluation cases retain their input images, proxy videos, prompts, and seeds. Result previews preserve all 1,440 frames at a compressed resolution of 1600 × 450. They are qualitative examples rather than lossless metric references. Hardware, kernels, and video encoding can affect exact output equality.

Source fingerprints are recorded in [source_files.json](../provenance/source_files.json). The package separates training, conditioning, and inference without changing the released model architecture or tensor names. The bundled Diffusers runtime has its own [file manifest](../provenance/diffusers_files.json).

Validation covers CPU training/checkpoint behavior, data preparation, input checksums, and video decoding. The results and scope are recorded in [release_validation.json](../provenance/release_validation.json). Full-model GPU training and inference were not rerun during repository preparation.
