# Simple-WAM

![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)
[![Project Page](https://img.shields.io/badge/Project_Page-Simple--WAM-2ea44f.svg)](https://zrporz.github.io/Simple-WAM-Web/)
[![Hugging Face Model](https://img.shields.io/badge/Hugging_Face-Model-f7c843.svg)](https://huggingface.co/rpzhou/Simple-WAM)

Codebase for **What Makes World Action Models Generalize?
An Empirical Study of Test-Time Future Modeling**.

## Environment Setup

### 1. Create the Environment

```bash
conda create -n simple-wam python=3.10 -y
conda activate simple-wam
```

### 2. Install Dependencies

Core model and data dependencies are pinned in
[requirements.txt](./requirements.txt); benchmark-specific environments are not included.

```bash
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e .
```

## Dataset Download

These trajectory datasets are needed for training, not for
simulation evaluation with the released checkpoints.

### LIBERO

Download the dataset from
[yuanty/LIBERO-fastwam](https://huggingface.co/datasets/yuanty/LIBERO-fastwam)
into `data/libero_mujoco3.3.2/`, then extract each archive:

```bash
mkdir -p data/libero_mujoco3.3.2
# Place all four .tar.gz files in this directory first.
for archive in data/libero_mujoco3.3.2/*.tar.gz; do
  tar -xzf "$archive" -C data/libero_mujoco3.3.2
done
```

The existing data configs expect this layout:

```text
data/libero_mujoco3.3.2/
  libero_10_no_noops_lerobot/
  libero_goal_no_noops_lerobot/
  libero_object_no_noops_lerobot/
  libero_spatial_no_noops_lerobot/
```

### RoboTwin

Download all `robotwin2.0.tar.gz.part-*` files from
[yuanty/robotwin2.0-fastwam](https://huggingface.co/datasets/yuanty/robotwin2.0-fastwam)
into `data/robotwin2.0/`. Concatenate the parts in filename order and extract:

```bash
mkdir -p data/robotwin2.0
# Place every archive part in this directory first.
cat data/robotwin2.0/robotwin2.0.tar.gz.part-* | tar -xzf - -C data/robotwin2.0
```

Expected layout:

```text
data/robotwin2.0/
  robotwin2.0/
    data/
    meta/
    videos/
```


## Evaluation

### 1. Download Checkpoints and Simulator Assets

Download the released checkpoints and simulator assets from
[rpzhou/Simple-WAM](https://huggingface.co/rpzhou/Simple-WAM):

```bash
huggingface-cli download rpzhou/Simple-WAM \
  --repo-type model \
  --include "simplewam_checkpoints/*" "benchmark_assets/*" \
  --local-dir checkpoints

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
```

If the files are already downloaded, skip the download command. Expected layout:

```text
checkpoints/
  simplewam_checkpoints/
    libero_joint_2cam224_1e-4.pt
    libero_joint_2cam224_1e-4_fewshot10.pt
    ...
    robotwin_joint_3cam_384_1e-4_fewshot10.pt
    robotwin_joint_3cam_384_1e-4_taskgen.pt
    robotwin_joint_3cam_384_1e-4_taskgen_video.pt
    libero_dataset_stats.json
    robotwin_dataset_stats.json
  benchmark_assets/
    libero_standard.tar.gz
    libero_plus.tar.gz
    robotwin_assets.tar.gz
```

Wan VAE, text encoder, and tokenizer files are downloaded automatically on first
use if missing. For offline evaluation, prepare those files in the same
`checkpoints/` model-cache directory beforehand. The trained VideoDiT and
ActionDiT weights come from the released `.pt` file.

### 2. LIBERO

Install the simulator dependencies into the existing environment.

```bash
pip install -c requirements.txt \
  mujoco==3.3.2 robosuite==1.4.0 bddl==1.0.1 gym==0.25.2 \
  easydict==1.9 cloudpickle==2.1.0 future==0.18.2 \
  opencv-python==4.6.0.66 matplotlib==3.5.3 \
  h5py==3.16.0 scipy==1.13.1 PyOpenGL==3.1.10

tar -xzf checkpoints/benchmark_assets/libero_standard.tar.gz -C third_party/LIBERO
pip install -e third_party/LIBERO

export PYTHONPATH="$(pwd)/third_party/LIBERO:${PYTHONPATH:-}"
export LIBERO_CONFIG_PATH="$(pwd)/third_party/LIBERO/.libero"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

The archive restores the omitted `third_party/LIBERO/libero/` directory, including
assets, BDDL files, and initial states. Its paths are resolved through the bundled
`third_party/LIBERO/.libero/config.yaml`.

Evaluate the full-data checkpoint on all four suites, with 50 trials per task:

```bash
config_name=libero_joint_2cam224_1e-4
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python experiments/libero/run_libero_manager.py \
  "task=${config_name}" \
  "ckpt=checkpoints/simplewam_checkpoints/${config_name}.pt" \
  EVALUATION.dataset_stats_path=checkpoints/simplewam_checkpoints/libero_dataset_stats.json \
  'MULTIRUN.task_suite_names=[libero_spatial,libero_object,libero_goal,libero_10]' \
  MULTIRUN.num_gpus=8 \
  MULTIRUN.max_tasks_per_gpu=1 \
  MULTIRUN.persistent_workers=true \
  EVALUATION.num_trials=50 \
  EVALUATION.num_inference_steps=10 \
  EVALUATION.freeze_future_video_noise=true \
  EVALUATION.torch_compile_action=true \
  EVALUATION.offload_text_encoder=true
```

Use the same command with a different `config_name` to evaluate another released
checkpoint. For task generalization, replace `MULTIRUN.task_suite_names` with the
held-out suite:

| Checkpoint/config suffix | Evaluation suites |
| --- | --- |
| `_fewshot5`, `_fewshot10` | All four suites |
| `_taskgen_spatial`, `_taskgen_video_spatial` | `[libero_spatial]` |
| `_taskgen_object`, `_taskgen_video_object` | `[libero_object]` |
| `_taskgen_goal`, `_taskgen_video_goal` | `[libero_goal]` |
| `_taskgen_long`, `_taskgen_video_long` | `[libero_10]` |

All suffixes above follow `libero_joint_2cam224_1e-4`. Results are saved under
`evaluate_results/libero/<config_name>/`.

### 3. LIBERO-plus

Install the simulator dependencies listed in **LIBERO** above, then add the
image-corruption dependencies and extract the LIBERO-plus archive:

```bash
conda install -c conda-forge imagemagick -y
pip install -c requirements.txt Wand==0.7.2 scikit-image==0.25.2

tar -xzf checkpoints/benchmark_assets/libero_plus.tar.gz -C third_party/LIBERO-plus
pip install -e third_party/LIBERO-plus

export PYTHONPATH="$(pwd)/third_party/LIBERO-plus:$(pwd)/src:${PYTHONPATH:-}"
export LIBERO_CONFIG_PATH="$(pwd)/third_party/LIBERO-plus/.libero"
export NUMBA_CACHE_DIR="$(pwd)/.numba_cache"
export MPLCONFIGDIR="$(pwd)/.matplotlib_cache"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

python -c "import libero.libero; print(libero.libero.__file__)"
```

The printed path must be under `third_party/LIBERO-plus`. Both benchmarks use
the same Python package name, `libero`; installing LIBERO-plus replaces the
editable LIBERO installation. The explicit `PYTHONPATH` and `LIBERO_CONFIG_PATH`
above select the plus benchmark in this shell. To switch back to standard
LIBERO, rerun its editable install and environment exports from the previous
section.

Use the same released LIBERO checkpoint and normalization statistics. This
command evaluates all four robustness suites with one rollout per task variant:

```bash
config_name=libero_joint_2cam224_1e-4
run_id=$(date +%Y%m%d_%H%M%S)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python experiments/libero/run_libero_manager.py \
  "task=${config_name}" \
  "ckpt=checkpoints/simplewam_checkpoints/${config_name}.pt" \
  EVALUATION.dataset_stats_path=checkpoints/simplewam_checkpoints/libero_dataset_stats.json \
  "EVALUATION.output_dir=./evaluate_results/libero_plus/${config_name}/${run_id}" \
  'MULTIRUN.task_suite_names=[libero_spatial,libero_object,libero_goal,libero_10]' \
  MULTIRUN.num_gpus=8 \
  MULTIRUN.max_tasks_per_gpu=1 \
  MULTIRUN.persistent_workers=true \
  EVALUATION.num_trials=1 \
  EVALUATION.num_inference_steps=10 \
  EVALUATION.freeze_future_video_noise=true \
  EVALUATION.torch_compile_action=true \
  EVALUATION.offload_text_encoder=true
```

After evaluation, summarize Camera, Robot, Language, and the other components:

```bash
python experiments/libero/summarize_libero_plus_components.py \
  --run-dir "evaluate_results/libero_plus/${config_name}/${run_id}"
```

The summary CSV and JSON files are written into that run directory.

### 4. RoboTwin

Our reference RoboTwin environment uses CUDA Toolkit **12.8.1** and PyTorch
**2.7.1+cu128**. The key simulator packages are SAPIEN **3.0.0b1**, mplib
**0.2.1**, cuRobo **0.7.5**, Warp **1.14.0**, Open3D **0.19.0**, and toppra
**0.6.8**. RoboTwin rendering also requires a working Vulkan setup and `ffmpeg`.

Machine and driver setups vary. Follow the
[official RoboTwin repository](https://github.com/RoboTwin-Platform/RoboTwin)
for installation instructions and environment-specific troubleshooting, while
using the versions above as the reference configuration for reproducing our
evaluation.

After configuring RoboTwin, extract the released assets and expose the
Simple-WAM policy and checkpoints:

```bash
tar -xzf checkpoints/benchmark_assets/robotwin_assets.tar.gz -C third_party/RoboTwin

ln -sfn ../../../experiments/robotwin/simplewam_policy \
  third_party/RoboTwin/policy/simplewam_policy
ln -sfn ../../checkpoints third_party/RoboTwin/checkpoints
```

Evaluate the few-shot checkpoint on all tasks, in both clean and randomized phases:

```bash
config_name=robotwin_joint_3cam_384_1e-4_fewshot10
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python experiments/robotwin/run_robotwin_manager.py \
  "task=${config_name}" \
  "ckpt=checkpoints/simplewam_checkpoints/${config_name}.pt" \
  EVALUATION.dataset_stats_path=checkpoints/simplewam_checkpoints/robotwin_dataset_stats.json \
  EVALUATION.eval_phase=both \
  EVALUATION.eval_num_episodes=100 \
  EVALUATION.num_inference_steps=10 \
  EVALUATION.freeze_future_video_noise=true \
  EVALUATION.torch_compile_action=true \
  EVALUATION.offload_text_encoder=true \
  MULTIRUN.num_gpus=8 \
  MULTIRUN.max_tasks_per_gpu=1
```

For the generalization checkpoints, change `config_name` and append the matching
override to evaluate only the held-out tasks:

| `config_name` | Additional override |
| --- | --- |
| `robotwin_joint_3cam_384_1e-4_taskgen` | `'EVALUATION.task_names=${data.train.robotwin_drop_tasks}'` |
| `robotwin_joint_3cam_384_1e-4_taskgen_video` | `'EVALUATION.task_names=${data.train.robotwin_action_loss_drop_tasks}'` |


## Training

### 1. Preprocessing

Run the following commands from the repository root after installing Simple-WAM
and extracting the datasets. Use the full-data configs below, not a few-shot or
task-generalization config, so the caches cover all training subsets.
For RoboTwin, place the normalization statistics at
`data/robotwin2.0/dataset_stats.json` before computing video or metadata caches.

#### Prepare the ActionDiT Initialization

The preprocessing scripts download missing Wan weights and tokenizer files on
first use. Set their local cache directory:

```bash
mkdir -p checkpoints
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
```

Interpolate the pretrained Wan2.2 video backbone to the 1024-dimensional
ActionDiT backbone, with alpha scaling enabled:

```bash
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/simplewam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda \
  --dtype bfloat16
```

#### Precompute Text Cache

Encode the task instructions from each dataset's `meta/tasks.jsonl`:

```bash
# LIBERO
CUDA_VISIBLE_DEVICES=0 python scripts/precompute_text_embeds.py \
  task=libero_joint_2cam224_1e-4 \
  +overwrite=false

# RoboTwin
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 scripts/precompute_text_embeds.py \
  task=robotwin_joint_3cam_384_1e-4 \
  +overwrite=false
```

Outputs are written to `data/text_embeds_cache/libero/` and
`data/text_embeds_cache/robotwin/`, as configured by
`data.train.text_embedding_cache_dir`. Existing entries are skipped; use
`+overwrite=true` to regenerate them. The script loads the text encoder even
though training uses `model.load_text_encoder=false`.

#### Precompute Video Latents

Precomputed video latents matching the released configs are available from
[rpzhou/simplewam-precompute-cache](https://huggingface.co/datasets/rpzhou/simplewam-precompute-cache).
Download and install them from the repository root:

```bash
huggingface-cli download rpzhou/simplewam-precompute-cache \
  --repo-type dataset \
  --include "video_latents/libero.tar.gz" "video_latents/robotwin/*" \
  --local-dir ./cache_downloads

mkdir -p ./data/video_latents/libero ./data/video_latents/robotwin
tar -xzf ./cache_downloads/video_latents/libero.tar.gz \
  -C ./data/video_latents/libero --strip-components=1

set -o pipefail
cat ./cache_downloads/video_latents/robotwin/robotwin_part_{aa,ab,ac,ad,ae,af} | \
  tar -xzf - -C ./data/video_latents/robotwin --strip-components=1
```

Alternatively, compute the caches locally. The commands below use eight GPUs,
with one process per GPU. Adjust
`CUDA_VISIBLE_DEVICES` and `--nproc_per_node` together for your machine.

```bash
# LIBERO
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 scripts/precompute_video_latents.py \
  task=libero_joint_2cam224_1e-4 \
  data.train.use_precomputed_metadata_cache=false \
  data.train.metadata_cache_dir=null \
  data.train.video_latent_cache_dir=./data/video_latents/libero \
  data.train.video_latent_storage_format=npz_compressed \
  overwrite_video_latents=false \
  video_latent_save_dtype=fp16 \
  video_latent_batch_size=8 \
  video_latent_num_workers=8

# RoboTwin
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 scripts/precompute_video_latents.py \
  task=robotwin_joint_3cam_384_1e-4 \
  data.train.robotwin_clean_only=false \
  data.train.use_precomputed_metadata_cache=false \
  data.train.metadata_cache_dir=null \
  data.train.video_latent_cache_dir=./data/video_latents/robotwin \
  data.train.video_latent_storage_format=npz_compressed \
  overwrite_video_latents=false \
  video_latent_save_dtype=fp16 \
  video_latent_batch_size=8 \
  video_latent_num_workers=8
```

The output paths match the current training configs. `./data/video_latents/libero`
and `./data/video_latents/robotwin` are relative to the current working directory.
Run these commands from the repository root to store caches under its `data/`.
If you change the VAE, frame sampling, resolution, or camera layout, generate a
new cache and update `data.train.video_latent_cache_dir` accordingly.

#### Precompute RoboTwin Metadata Cache

Ensure `data/robotwin2.0/dataset_stats.json` is present first. This CPU-only step
caches normalized actions, proprioception, padding masks, and instructions; it
does not run the VAE or text encoder.

A compatible precomputed cache is also available from
[rpzhou/simplewam-precompute-cache](https://huggingface.co/datasets/rpzhou/simplewam-precompute-cache):

```bash
huggingface-cli download rpzhou/simplewam-precompute-cache \
  --repo-type dataset \
  --include "metadata_cache/robotwin/*" \
  --local-dir ./cache_downloads

mkdir -p ./data/metadata_cache/robotwin
cp -a ./cache_downloads/metadata_cache/robotwin/. \
  ./data/metadata_cache/robotwin/
```

To recompute it locally, note that the metadata script expects a **resolved
configuration**, not a task YAML with Hydra defaults. Export one without
starting preprocessing:

```bash
mkdir -p ./data/metadata_cache
python scripts/precompute_video_latents.py \
  task=robotwin_joint_3cam_384_1e-4 \
  data.train.robotwin_clean_only=false \
  data.train.use_precomputed_video_latents=false \
  data.train.use_precomputed_metadata_cache=false \
  data.train.metadata_cache_dir=null \
  --cfg job --resolve > ./data/metadata_cache/robotwin_precompute.yaml

python scripts/precompute_metadata_cache.py \
  --config ./data/metadata_cache/robotwin_precompute.yaml \
  --output-cache-dir ./data/metadata_cache/robotwin \
  --shard-size 10000 \
  --batch-size 64 \
  --num-workers 8
```

The relative output path `./data/metadata_cache/robotwin` matches
`data.train.metadata_cache_dir` in the RoboTwin training config. Run from the
repository root. Existing arrays are skipped unless
`--overwrite` is passed.
Regenerate the cache if normalization statistics, action processing, or sample
indexing changes.

### 2. Training Commands

Run from the repository root after preparing the datasets, ActionDiT
initialization, text cache, and video latents above. RoboTwin also uses the
metadata cache. Training uses DeepSpeed ZeRO-1 and BF16 through
`scripts/train_zero1.sh`; no simulator environment is needed for training.

#### LIBERO

Train on all four suites with eight GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/train_zero1.sh 8 \
  task=libero_joint_2cam224_1e-4 \
  gradient_accumulation_steps=2 \
  wandb.mode=offline
```

Replace `task=` with the corresponding config below; keep the other arguments
unchanged:

| Task config | Training setting |
| --- | --- |
| `libero_joint_2cam224_1e-4` | Full data, all four suites |
| `libero_joint_2cam224_1e-4_fewshot5` | 5 demonstrations per task |
| `libero_joint_2cam224_1e-4_fewshot10` | 10 demonstrations per task |
| `libero_joint_2cam224_1e-4_taskgen_spatial` | Exclude the spatial suite |
| `libero_joint_2cam224_1e-4_taskgen_object` | Exclude the object suite |
| `libero_joint_2cam224_1e-4_taskgen_goal` | Exclude the goal suite |
| `libero_joint_2cam224_1e-4_taskgen_long` | Exclude the long-horizon suite (`libero_10`) |
| `libero_joint_2cam224_1e-4_taskgen_video_spatial` | Spatial suite contributes video supervision only |
| `libero_joint_2cam224_1e-4_taskgen_video_object` | Object suite contributes video supervision only |
| `libero_joint_2cam224_1e-4_taskgen_video_goal` | Goal suite contributes video supervision only |
| `libero_joint_2cam224_1e-4_taskgen_video_long` | Long-horizon suite contributes video supervision only |

LIBERO-plus evaluates the same trained LIBERO models; it does not require a
separate training command.

#### RoboTwin

Train the few-shot setting with eight GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/train_zero1.sh 8 \
  task=robotwin_joint_3cam_384_1e-4_fewshot10 \
  gradient_accumulation_steps=16 \
  wandb.mode=offline
```

Replace `task=` to select another training setting:

| Task config | Training setting |
| --- | --- |
| `robotwin_joint_3cam_384_1e-4` | Full clean and randomized data |
| `robotwin_joint_3cam_384_1e-4_fewshot10` | 10 episodes per task sampled from the combined clean and randomized data |
| `robotwin_joint_3cam_384_1e-4_taskgen` | Exclude tasks listed in `data.train.robotwin_drop_tasks` |
| `robotwin_joint_3cam_384_1e-4_taskgen_video` | Tasks in `data.train.robotwin_action_loss_drop_tasks` contribute video supervision only |

#### Batch Size and Outputs

The examples use global batch sizes of 128 (LIBERO) and 1024 (RoboTwin), giving
a per-GPU microbatch of 8 in both commands:

```text
per_gpu_batch_size = global_batch_size / (num_gpus * gradient_accumulation_steps)
```

When changing GPU count, update both `CUDA_VISIBLE_DEVICES` and the positional
argument to `train_zero1.sh`. Increase `gradient_accumulation_steps` to reduce
per-GPU memory while keeping `global_batch_size` fixed; the division above must
remain exact. Epoch counts, checkpoint intervals, and any step limits come from
the selected task config.

Each launch writes its resolved `config.yaml`, `checkpoints/weights/step_XXXXXX.pt`,
and `checkpoints/state/` under `runs/<task>/<timestamp>/`. `wandb.mode=offline` keeps W&B
logging local; use `wandb.enabled=false` to disable it entirely.

## Acknowledgements

We thank the authors and maintainers of the following open-source projects:

- [FastWAM](https://github.com/yuantianyuan01/FastWAM) for the codebase on which Simple-WAM is built and the preprocessed datasets used in this repository.
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) for the benchmark and simulation environment.
- [LIBERO-plus](https://github.com/sylvestf/LIBERO-plus) for the robustness benchmark.
- [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) for the simulation environment, tasks, and assets.
