#!/usr/bin/env bash
# Reproduce DGS-Net main results (train 2 stages on AIGIBench ProGAN/SDv1.4 -> test on UFD / GenImage / AIGIBench).
#
# Usage:
#   bash run_reproduce.sh setup      # conda env + python deps
#   bash run_reproduce.sh data       # download train/test data (where automatable)
#   bash run_reproduce.sh weights    # download released checkpoints (for test-only reproduction)
#   bash run_reproduce.sh test       # evaluate released checkpoint  -> log_test.log
#   bash run_reproduce.sh train      # stage1 + stage2 training from scratch
#   bash run_reproduce.sh test_own   # evaluate your own trained stage-2 model
#   bash run_reproduce.sh all        # setup, data, weights, test
#
# Override via env vars: DATA_ROOT, GPU, ENV_NAME
set -euo pipefail
cd "$(dirname "$0")"

ENV_NAME=${ENV_NAME:-DGS-Net}
GPU=${GPU:-0}
DATA_ROOT=${DATA_ROOT:-$PWD/datasets}
AIGIBENCH_ROOT=${AIGIBENCH_ROOT:-$DATA_ROOT/AIGIBench}
export UFD_ROOT=${UFD_ROOT:-$DATA_ROOT/UniversalFakeDetect}
export GENIMAGE_ROOT=${GENIMAGE_ROOT:-$DATA_ROOT/Genimage}
export AIGIBENCH_TEST_ROOT=${AIGIBENCH_TEST_ROOT:-$AIGIBENCH_ROOT/test}
CKPT_DIR=$PWD/checkpoints
TRAIN_NAME=5class-car-cat-chair-horse-sdv1.4
TRAIN_CLASSES=car,cat,chair,horse,sdv1.4

# Google Drive folders from README
WEIGHTS_GDRIVE=https://drive.google.com/drive/folders/1_MznDWLcZDNCeWj6AeP8MrJBLoMomxe7
UFD_GDRIVE=https://drive.google.com/drive/folders/1nkCXClC7kFM01_fqmLrVNtnOYEFPtWO-

activate() {
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME"
}

setup() {
  conda env list | grep -q "^$ENV_NAME " || conda create -n "$ENV_NAME" -y python=3.10
  activate
  pip install torch==2.9.1 torchvision==0.24.1   # pick the CUDA wheel index matching your driver if needed
  pip install -r requirements.txt
  pip install gdown "huggingface_hub[cli]"
  python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
}

data() {
  activate
  mkdir -p "$DATA_ROOT"
  # AIGIBench (train: ProGAN & SDv1.4; test: all subsets)
  if [ ! -d "$AIGIBENCH_ROOT/train" ]; then
    huggingface-cli download HorizonTEL/AIGIBench --repo-type dataset --local-dir "$AIGIBENCH_ROOT"
    # unpack any archives shipped by the HF repo
    find "$AIGIBENCH_ROOT" -maxdepth 2 \( -name '*.zip' -o -name '*.tar*' \) | while read -r f; do
      case "$f" in *.zip) unzip -qn "$f" -d "$(dirname "$f")";; *) tar -xf "$f" -C "$(dirname "$f")";; esac
    done
  fi
  # UniversalFakeDetect test set
  if [ ! -d "$UFD_ROOT" ]; then
    gdown --folder "$UFD_GDRIVE" -O "$UFD_ROOT"
  fi
  cat <<MSG
[data] Done where automatable. Manual steps:
  * GenImage is not linked in the README: place it at $GENIMAGE_ROOT
    (layout: <GENIMAGE_ROOT>/<generator>/{0_real,1_fake}).
  * Unzip any remaining archives so UFD / AIGIBench test dirs look like <root>/<subset>/{0_real,1_fake}.
  * Training layout expected: $AIGIBENCH_ROOT/train/{train,val}/<class>/{0_real,1_fake}
    (train.py reads <dataroot>/train and <dataroot>/val). Check with: ls $AIGIBENCH_ROOT/train
  * Training also needs the caption file $AIGIBENCH_ROOT/train_text/train.json (datasets.py loads it).
MSG
}

weights() {
  activate
  mkdir -p "$CKPT_DIR"
  gdown --folder "$WEIGHTS_GDRIVE" -O "$CKPT_DIR" --remaining-ok
  # gdown may nest files in a subfolder; flatten
  find "$CKPT_DIR" -mindepth 2 -name '*.pth' -exec mv -n {} "$CKPT_DIR"/ \;
  ls -lh "$CKPT_DIR"
}

test_released() {
  activate
  # clip_models.py loads ./checkpoints/model_epoch_step1.pth (frozen branch) -> must exist too
  for f in model_epoch_step1.pth model_epoch_step2.pth; do
    [ -f "$CKPT_DIR/$f" ] || { echo "Missing $CKPT_DIR/$f (run: bash $0 weights)"; exit 1; }
  done
  CUDA_VISIBLE_DEVICES=$GPU python test.py --model_path "$CKPT_DIR/model_epoch_step2.pth" --gpu_ids 0
  echo "Results in log_test.log (compare with the committed reference; ~0.1 deviation expected)."
}

train() {
  activate
  mkdir -p "$CKPT_DIR"
  local root="$AIGIBENCH_ROOT/train"
  [ -d "$root/train" ] || { echo "Missing $root/train (run: bash $0 data)"; exit 1; }
  [ -f "$AIGIBENCH_ROOT/train_text/train.json" ] || echo "WARNING: $AIGIBENCH_ROOT/train_text/train.json not found; datasets.py will fail without it"
  # Stage 1: train linear head on frozen CLIP (teacher)
  CUDA_VISIBLE_DEVICES=$GPU python train.py --name "$TRAIN_NAME" --dataroot "$root" \
      --classes "$TRAIN_CLASSES" --train_stage 0 --niter 3 --gpu_ids 0
  local s1; s1=$(ls -dt "$CKPT_DIR/${TRAIN_NAME}"*/ | head -1)
  cp "${s1}model_epoch_last.pth" "$CKPT_DIR/model_epoch_step1.pth"   # path hardcoded in models/clip_models.py
  echo "Stage 1 checkpoint -> $CKPT_DIR/model_epoch_step1.pth"

  # Stage 2: LoRA fine-tuning with distillation-guided gradient surgery
  CUDA_VISIBLE_DEVICES=$GPU python train.py --name "$TRAIN_NAME" --dataroot "$root" \
      --classes "$TRAIN_CLASSES" --train_stage 1 --niter 1 --gpu_ids 0
  local s2; s2=$(ls -dt "$CKPT_DIR/${TRAIN_NAME}"*/ | head -1)
  cp "${s2}model_epoch_last.pth" "$CKPT_DIR/model_epoch_step2_own.pth"
  echo "Stage 2 checkpoint -> $CKPT_DIR/model_epoch_step2_own.pth"
}

test_own() {
  activate
  CUDA_VISIBLE_DEVICES=$GPU python test.py --model_path "$CKPT_DIR/model_epoch_step2_own.pth" --gpu_ids 0
}

case "${1:-}" in
  setup)    setup ;;
  data)     data ;;
  weights)  weights ;;
  test)     test_released ;;
  train)    train ;;
  test_own) test_own ;;
  all)      setup; data; weights; test_released ;;
  *) sed -n '2,14p' "$0"; exit 1 ;;
esac
