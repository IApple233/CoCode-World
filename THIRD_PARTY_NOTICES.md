# Third-party notices

- **Hugging Face Diffusers**: `diffusers/` and its native MiniMax-H3 converter use [Apache-2.0](diffusers/LICENSE). Copyright headers are retained. This runtime snapshot includes local changes. Upstream: https://github.com/huggingface/diffusers.
- **Code World Model**: presentation wrapping in `cocode_world/prompting.py` is adapted from `src/cwm_h3_inference/presentation.py` at revision `17fa82ba2fb8168aa0d30fc51a1d1d0ebbc9180b`, under [Apache-2.0](third_party_licenses/code-world-model-LICENSE). The adaptation adds synchronized two-view task/system instructions around native reference tokens. Upstream: https://github.com/buaacyw/code-world-model.
- **MiniMax-H3**: base weights are downloaded separately and retain the [MiniMax-H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE).
- **DiffSynth-Studio Ref2VA DeCFG**: the training preset is downloaded separately from [ModelScope](https://www.modelscope.cn/models/DiffSynth-Studio/MiniMax-H3-TrainingAdapter) and retains its upstream terms.

- **Twemoji**: the four README icons in `assets/icons/` use artwork from [Twemoji 15.1.0](https://github.com/jdecked/twemoji/tree/v15.1.0), copyright Twitter, Inc. and contributors, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The root MIT license covers original CoCode-World code. It does not replace third-party software licenses or the terms of model weights, datasets, and example media.
