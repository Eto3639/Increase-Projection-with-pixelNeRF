#!/bin/bash

# スクリプトの第1引数を、CTファイルへの「相対パス」として使用
# 例: ./run_drr.sh CT_Nifti/0153_3000669.000000-NA-59527_ct.nii.gz
CT_RELATIVE_PATH=$1

# 引数が指定されていない場合はエラー
if [ -z "$CT_RELATIVE_PATH" ]; then
    echo "エラー: CTファイルへの相対パスを引数として指定してください。"
    echo "例: $0 CT_Nifti/0153_....nii.gz"
    exit 1
fi

# NASのLIDC親ディレクトリをコンテナ内の/dataにマウント
NAS_PARENT_DIR="/mnt/nas/eto/LIDC"

echo "コンテナを起動します..."
echo "NASマウント: ${NAS_PARENT_DIR} -> /data"
echo "対象ファイル: /data/${CT_RELATIVE_PATH}"

docker run --gpus all --rm \
  -v ./:/workspace \
  -v "${NAS_PARENT_DIR}":/data \
  drr-generator \
  python3 generate_drr_dataset.py "$CT_RELATIVE_PATH"