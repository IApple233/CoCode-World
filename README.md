<h1 align="center">CoCode-World</h1>
<p align="center"><strong>Shared 3D Proxy Representation for Multi-Player World Models</strong></p>

<p align="center">
  <img src="assets/icons/page.svg" width="18" height="18" alt=""> <a href="https://iapple233.github.io/CCW/">Project Page</a> &nbsp;·&nbsp;
  <img src="assets/icons/checkpoint.svg" width="18" height="18" alt=""> <a href="https://huggingface.co/iapple1/CoCode-World-Model">Checkpoint</a> &nbsp;·&nbsp;
  <img src="assets/icons/data.svg" width="18" height="18" alt=""> <a href="https://huggingface.co/datasets/iapple1/CoCode-World-Data">Dataset</a> &nbsp;·&nbsp;
  <img src="assets/icons/evaluation.svg" width="18" height="18" alt=""> <a href="examples/evaluation">Evaluation Cases</a>
</p>

Official implementation of **CoCode-World**. A shared 3D proxy guides synchronized video generation from two player perspectives, while a reference image and text prompt define the appearance of the world.

## Gallery

Six examples from our [project page](https://iapple233.github.io/CCW/#long-events). Each preview shows both player views; click to watch the full video.

<table>
  <tr>
    <td width="50%" align="center"><strong>Ride through the city</strong><br><a href="examples/evaluation/case_01/result_preview.mp4"><img src="assets/gallery/city.gif" width="100%" alt="Two synchronized views of a motorcycle ride through the city"></a></td>
    <td width="50%" align="center"><strong>Across the desert</strong><br><a href="examples/evaluation/case_02/result_preview.mp4"><img src="assets/gallery/desert.gif" width="100%" alt="Two synchronized views of a car crossing the desert at sunset"></a></td>
  </tr>
  <tr>
    <td width="50%" align="center"><strong>A quiet rest</strong><br><a href="examples/evaluation/case_03/result_preview.mp4"><img src="assets/gallery/rest.gif" width="100%" alt="Two characters meet on a sunlit path, seen from both perspectives"></a></td>
    <td width="50%" align="center"><strong>Village stop</strong><br><a href="examples/evaluation/case_04/result_preview.mp4"><img src="assets/gallery/village.gif" width="100%" alt="Two characters explore a village in synchronized views"></a></td>
  </tr>
  <tr>
    <td width="50%" align="center"><strong>Blue water tanks</strong><br><a href="examples/evaluation/case_06/result_preview.mp4"><img src="assets/gallery/tanks.gif" width="100%" alt="Two characters walk beside blue water tanks, seen from both perspectives"></a></td>
    <td width="50%" align="center"><strong>Along the water</strong><br><a href="examples/evaluation/case_07/result_preview.mp4"><img src="assets/gallery/water.gif" width="100%" alt="Two synchronized views of a watercraft ride at sunset"></a></td>
  </tr>
</table>

[Explore all eight evaluation cases →](examples/evaluation)

## Getting started

For training, first download the official **[MiniMax-H3 weights](https://huggingface.co/MiniMaxAI/MiniMax-H3)**, then continue task training from **[DiffSynth-Studio's Ref2VA DeCFG LoRA](https://www.modelscope.cn/models/DiffSynth-Studio/MiniMax-H3-TrainingAdapter)**. The DeCFG adapter is a frozen training preset; inference uses MiniMax-H3 and our EMA LoRA.

<details>
<summary><strong>Installation and model downloads</strong></summary>

Linux, Python 3.11, FFmpeg, and CUDA-compatible PyTorch are required. The launchers use eight GPUs per node.

```bash
git clone https://github.com/IApple233/CoCode-World.git
cd CoCode-World
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install -e ./diffusers

# Base model and CoCode-World checkpoint for inference.
python scripts/download_models.py base
python scripts/download_models.py cocode

# Training additionally requires the Ref2VA DeCFG preset.
python scripts/download_models.py decfg
```

Install `ffmpeg` with your system package manager. Use the included Diffusers runtime for MiniMax-H3 support. [Environment and model details →](docs/SETUP.md)

</details>

<details>
<summary><strong>Training</strong></summary>

```bash
python scripts/download_models.py training
hf download iapple1/CoCode-World-Data --repo-type dataset \
  --include 'train/*' 'manifests/train.jsonl' --local-dir datasets/CoCode-World-Data
```

Follow the [training guide](docs/TRAINING.md) to prepare and cache the data, then launch:

```bash
bash scripts/train.sh cache/train/manifest.jsonl outputs/cocode-world
```

The guide covers multiple nodes, resume behavior, and the released configuration. The public dataset is a selected subset of the original training data; exact checkpoint reproduction is not claimed.

</details>

<details>
<summary><strong>Evaluation</strong></summary>

The [eight cases](examples/evaluation) include the first-frame reference, prompts, proxy video, seed, and recorded result.

```bash
python scripts/verify_release.py
python scripts/prepare_evaluation.py --case-id case_01 --output-dir cache/eval_inputs
```

Follow the [evaluation guide](docs/EVALUATION.md) to cache the inputs, then generate:

```bash
bash scripts/evaluate.sh cache/eval/manifest.jsonl cache/eval_inputs/plan.json outputs/evaluation
```

</details>

<details>
<summary><strong>Repository structure and validation</strong></summary>

| Directory | Contents |
|---|---|
| `cocode_world/` | Model, training, data preparation, and inference |
| `scripts/` | Download, preparation, training, and evaluation entrypoints |
| `configs/` | Released training configuration |
| `examples/evaluation/` | Eight complete evaluation cases |
| `tests/` | CPU regression tests |
| `diffusers/` | MiniMax-H3 runtime with its upstream license |

```bash
python -m pip install pytest
python -m pytest -q
python scripts/verify_release.py --decode-videos
```

[Checkpoint and validation notes →](docs/REPRODUCIBILITY.md)

</details>

## Acknowledgments and license

Built on [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3), [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio), [Diffusers](https://github.com/huggingface/diffusers), and [Code World Model](https://github.com/buaacyw/code-world-model).

Original code is released under [MIT](LICENSE). Third-party software, model weights, datasets, and example media retain their respective terms. See [third-party notices](THIRD_PARTY_NOTICES.md).
