#!/bin/bash

# --- 設定 ---

# 権限の問題を解決するため、ホストの現在のユーザーID/グループIDでコンテナを実行
USER_ID=$(id -u)
GROUP_ID=$(id -g)

# NAS上のLIDC親ディレクトリ（コンテナ内の /data にマウントする対象）
NAS_PARENT_DIR="/mnt/nas/eto/LIDC"

echo "DRR一括生成スクリプト（コンテナ内ループ版）を開始します。"
echo "実行ユーザーID: $USER_ID, グループID: $GROUP_ID でコンテナを実行します。"
echo "NASマウント: ${NAS_PARENT_DIR} -> /data"
echo "====================================================="
echo "コンテナを1回だけ起動し、Pythonスクリプト内で全ファイルを処理します。"
echo "（ライブラリのインポートに数秒かかります...）"

# docker run コマンドを1回だけ実行し、generate_drr.py に処理を任せる
# 各行末の '\' の直後にスペースがないことを確認
docker run --gpus all --rm \
  -u "${USER_ID}:${GROUP_ID}" \
  -v "$(pwd)":/workspace \
  -v "${NAS_PARENT_DIR}":/data \
  -e MPLCONFIGDIR=/workspace/.config/matplotlib \
  drr-generator \
  python3 generate_drr.py
  
# エラーコードをチェック (オプション)
if [ $? -ne 0 ]; then
    echo "▲▲▲ Dockerコンテナの実行中にエラーが発生しました ▲▲▲"
fi

echo "====================================================="
echo "すべてのファイルの処理が完了しました。"
echo "====================================================="