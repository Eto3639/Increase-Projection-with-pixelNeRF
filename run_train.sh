# 権限の問題を解決するため、ホストの現在のユーザーID/グループIDでコンテナを実行
USER_ID=$(id -u)
GROUP_ID=$(id -g)

    # NAS上のLIDC親ディレクトリ（コンテナ内の /data にマウントする対象）
    # generate_drr.py 内の CT_NIFTI_DIR と整合性を取る
NAS_PARENT_DIR="/mnt/nas/eto/LIDC" 
    
    # DRRデータセットの親ディレクトリ (ホスト側)
    # train.py が /workspace/drr_dataset を参照するため、
    # ホストの ./drr_dataset を /workspace/drr_dataset にマウントする
DRR_DATASET_DIR="./drr_dataset"

echo "NeRF学習スクリプト (train.py) をコンテナで実行します。"
echo "実行ユーザーID: $USER_ID, グループID: $GROUP_ID"
echo "NASマウント: ${NAS_PARENT_DIR} -> /data (主にデータ生成用)"
echo "プロジェクトマウント: $(pwd) -> /workspace"
echo "DRRデータセットマウント: ${DRR_DATASET_DIR} -> /workspace/drr_dataset" # データセット読み込み用
echo "====================================================="

docker run --gpus all --rm \
    -u "${USER_ID}:${GROUP_ID}" \
    -v "$(pwd)":/workspace \
    -v "${NAS_PARENT_DIR}":/data \
    -v "${DRR_DATASET_DIR}":"/workspace/drr_dataset" \
    -e MPLCONFIGDIR=/workspace/.config/matplotlib \
    -e TORCH_HOME=/workspace/.cache/torch \
    nerf-trainer \
    python3 train.py --debug
      
    # エラーコードをチェック
if [ $? -ne 0 ]; then
        echo "▲▲▲ Dockerコンテナの実行中にエラーが発生しました ▲▲▲"
fi

echo "====================================================="
echo "学習スクリプトの実行が完了しました。"
echo "====================================================="