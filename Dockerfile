# ベースイメージとしてPyTorch公式イメージを使用
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# 作業ディレクトリを設定
WORKDIR /workspace

# 必要なシステムパッケージをインストール
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# 必要なPythonパッケージをインストール
# matplotlib, torchio, PyYAML, scikit-image, pandas, lpips, wandb, optuna
RUN pip install --no-cache-dir \
    matplotlib \
    torchio \
    PyYAML \
    scikit-image \
    pandas \
    lpips \
    wandb \
    optuna

# DiffDRR をpipで直接インストール
RUN pip install --no-cache-dir diffdrr

# ソースコードをコンテナにコピー
COPY . /workspace

# Matplotlibのキャッシュディレクトリを作成し、適切な権限を付与
RUN mkdir -p /workspace/.config/matplotlib && \
    chmod -R 777 /workspace/.config

# コンテナ起動時のデフォルトコマンド
CMD ["/bin/bash"]
