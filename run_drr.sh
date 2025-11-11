#!/bin/bash
set -e

# --- 設定 ---
USER_ID=$(id -u)
GROUP_ID=$(id -g)
NAS_PARENT_DIR="/mnt/nas/eto/LIDC"
DOCKER_IMAGE_NAME="drr-generator"

# --- 関数定義 ---
function print_usage() {
    echo "Usage: $0 [MODE]"
    echo "A simple wrapper for running the DRR generator in a Docker container."
    echo ""
    echo "Modes:"
    echo "  (no argument) - Process a single sample file (--single)."
    echo "  [N]           - Process N random files (--random N)."
    echo "  all           - Process all available files (no extra args)."
    echo "  --[ARG]       - Pass any other arguments directly to generate_drr.py."
}

# --- メインロジック ---
MODE=$1
PYTHON_ARGS=""

if [ -z "$MODE" ]; then
    # 引数なし -> シングルテストモード
    PYTHON_ARGS="--single"
elif [[ "$MODE" =~ ^[0-9]+$ ]]; then
    # 引数が数字 -> ランダムモード
    PYTHON_ARGS="--random $MODE"
elif [ "$MODE" == "all" ]; then
    # 'all' -> 全ファイルモード（引数なし）
    PYTHON_ARGS=""
elif [[ "$MODE" == --* ]]; then
    # '--'で始まる引数 -> そのまま渡す
    PYTHON_ARGS="$@"
else
    echo "Error: Invalid mode '$MODE'."
    print_usage
    exit 1
fi

echo "DRR Generation Script Wrapper"
echo "Host User ID: $USER_ID, Group ID: $GROUP_ID"
echo "Target Mode: '${MODE:-single}' -> Python args: '${PYTHON_ARGS}'"
echo "====================================================="

# Docker実行の共通オプション
DOCKER_OPTS="--gpus all --rm -u ${USER_ID}:${GROUP_ID} -v $(pwd):/workspace -v ${NAS_PARENT_DIR}:/data -e MPLCONFIGDIR=/workspace/.config/matplotlib"

# Dockerコンテナを実行
# --generateフラグは常に追加する
docker run $DOCKER_OPTS ${DOCKER_IMAGE_NAME} python3 generate_drr.py --generate $PYTHON_ARGS

if [ $? -ne 0 ]; then
    echo "▲▲▲ An error occurred during the script execution. ▲▲▲"
fi

echo "====================================================="
echo "Script finished."
echo "====================================================="
