# Eight step-4000 evaluation examples

These cases are selected from the original CoCode-World step-4000 EMA results shown on the project page. Each directory contains the **actual edited first-frame input** (`first_frame.png`), **original 60-second Proxy555 input** (`proxy.mp4`), first-window prompt and system prompt, all **17 window prompts** in `prompts.json`, the inference seed, and a compressed preview of the recorded result.

The first-frame input is an appearance reference; generated frame 0 is not hard-clamped to it. The first-window prompt alone does not describe the complete 60 seconds: use the full window schedule to reproduce the long result. A small context-parallel padding suffix may be appended by the original text-presentation code when caching.

| Case | Title | Category | Seed | Inputs and result |
|---|---|---|---|---|
| [case_01](case_01/case.json) | Ride through the city | riding | 469162610 | [First frame](case_01/first_frame.png) · [Prompt](case_01/prompt.txt) · [Result](case_01/result_preview.mp4) |
| [case_02](case_02/case.json) | Across the desert | riding | 900877685 | [First frame](case_02/first_frame.png) · [Prompt](case_02/prompt.txt) · [Result](case_02/result_preview.mp4) |
| [case_03](case_03/case.json) | A quiet rest | interaction | 2045812713 | [First frame](case_03/first_frame.png) · [Prompt](case_03/prompt.txt) · [Result](case_03/result_preview.mp4) |
| [case_04](case_04/case.json) | Village stop | exploration | 255886286 | [First frame](case_04/first_frame.png) · [Prompt](case_04/prompt.txt) · [Result](case_04/result_preview.mp4) |
| [case_05](case_05/case.json) | Moonlit meeting | interaction | 1841378506 | [First frame](case_05/first_frame.png) · [Prompt](case_05/prompt.txt) · [Result](case_05/result_preview.mp4) |
| [case_06](case_06/case.json) | Blue water tanks | interaction | 123464386 | [First frame](case_06/first_frame.png) · [Prompt](case_06/prompt.txt) · [Result](case_06/result_preview.mp4) |
| [case_07](case_07/case.json) | Along the water | riding | 2040715247 | [First frame](case_07/first_frame.png) · [Prompt](case_07/prompt.txt) · [Result](case_07/result_preview.mp4) |
| [case_08](case_08/case.json) | From beach to boulevard | exploration | 1503099102 | [First frame](case_08/first_frame.png) · [Prompt](case_08/prompt.txt) · [Result](case_08/result_preview.mp4) |

Run `python scripts/verify_release.py --decode-videos` from the repository root to verify inputs and decode all videos. See [the evaluation guide](../../docs/EVALUATION.md) for inference commands.

Input PNGs and Proxy videos are unchanged from the recorded inference inputs; SHA256 checksums are supplied. Result previews are the 1600 × 450 project-page encodes of the original 2560 × 720 outputs, with all 1,440 frames preserved. These eight selected examples are qualitative illustrations, not a randomly sampled test set or a quantitative benchmark. No aggregate metric is claimed.
