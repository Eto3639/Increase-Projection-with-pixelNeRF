#!/bin/bash

# --- 設定 ---
# 権限の問題を解決するため、ホストの現在のユーザーID/グループIDでコンテナを実行
USER_ID=$(id -u)
GROUP_ID=$(id -g)

# マウントするディレクトリの定義
# 注意: これらのパスはご自身の環境に合わせて調整が必要な場合があります
NAS_PARENT_DIR="/mnt/nas/eto/LIDC" # run_drr_all.shから推測
DRR_DATASET_DIR="$(pwd)/drr_dataset" # config.ymlの設定と合わせる

# ★根本対策: コンテナ内のホームディレクトリを作成・指定
mkdir -p .home

echo "学習スクリプトを開始します。"
echo "実行ユーザーID: $USER_ID, グループID: $GROUP_ID でコンテナを実行します。"
echo "====================================================="

# docker run コマンド
docker run --gpus all -i --rm \
  -u "${USER_ID}:${GROUP_ID}" \
  -v "$(pwd)":/workspace \
  -v "${NAS_PARENT_DIR}":/data \
  -v "${DRR_DATASET_DIR}":"/workspace/drr_dataset" \
  -e HOME=/workspace/.home \
  -e WANDB_API_KEY=$(cat .wandb_api_key) \
  nerf_multiview \
  python3 train.py

# エラーコードをチェック
if [ $? -ne 0 ]; then
    echo "▲▲▲ Dockerコンテナの実行中にエラーが発生しました ▲▲▲"
fi

echo "====================================================="
echo "学習スクリプトが完了しました。"
echo "====================================================="
