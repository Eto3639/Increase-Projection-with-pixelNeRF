# 1. ベースイメージ
# DiffDRRが依存するPyTorchとCUDA環境を指定します。
# PyTorch 2.0.1 / CUDA 11.8 (DiffDRRがテストされている環境)
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# 2. 必要なPythonパッケージのインストール
# DiffDRRと、スクリプトで使うライブラリ
RUN pip install \
    pyyaml \
    tqdm \
    diffdrr \
    torchio \
    matplotlib \
    imageio \
    imageio-ffmpeg \
    nibabel \
    timm \
    lpips \
    optuna \
    wandb

# 3. アプリケーションコード用のディレクトリ作成
WORKDIR /workspace