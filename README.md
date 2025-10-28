# LIDC-DRR-NeRF: Conditional NeRFによる胸部CTからのDRR新規ビュー合成

## 1. 概要

このプロジェクトは、胸部CTデータセット（LIDC-IDRI）から生成したDRR（デジタル再構成X線画像）を用いて、Conditional NeRFモデル（pixelNeRFベース）を学習させるためのフレームワークです。

**目的:** 2つの直交する角度（0度: 正面、90度: 側面）のDRRを入力として受け取り、任意の新しい角度からのDRR画像を生成（Novel View Synthesis）するモデルを構築します。

## 2. 特徴

- **pixelNeRFベース:** 画像特徴量を条件として利用することで、少数の入力ビューからでもシーンを再構成します。
- **DiffDRRライブラリ:** 微分可能なDRR生成ライブラリ `diffdrr` を使用して、CTデータから高品質なDRRとカメラパラメータを生成します。
- **Docker環境:** 依存関係を隔離し、再現性を高めるために、データ生成と学習プロセスをDockerコンテナ内で実行します。
- **モジュラーな構成:** データセット生成、データローダー、モデル、学習ループがそれぞれ独立したスクリプトで管理されており、拡張性に優れています。

## 3. 必要な環境

- **OS:** Linux (Ubuntu推奨)
- **GPU:** NVIDIA GPU (CUDA対応)
- **ソフトウェア:**
    - Docker
    - NVIDIA Container Toolkit (GPUをDockerコンテナ内で使用するため)

## 4. セットアップ

### 4.1. Dockerイメージのビルド

本プロジェクトでは、学習用の `nerf-trainer` というDockerイメージを使用します。まず、リポジトリのルートで以下のコマンドを実行してイメージをビルドしてください。

```bash
sudo docker build -t nerf-trainer .
```

### 4.2. データセットの準備

- **CTデータ:** LIDC-IDRIデータセットなど、NIfTI形式 (`.nii.gz`) のCTデータを準備します。
- **ディレクトリ構成:** CTデータと生成されるDRRデータセットは、`run_drr_all.sh` や `run_train.sh` で指定されたディレクトリに配置する必要があります。スクリプト内のパスを環境に合わせて適宜変更してください。

## 5. 使い方

### 5.1. DRRデータセットの生成

`run_drr_all.sh` スクリプトは、指定されたディレクトリ内のすべてのNIfTIファイルを探索し、`generate_drr.py` を使ってDRRデータセットを生成します。

1.  **設定:** `run_drr_all.sh` を開き、以下の変数を環境に合わせて設定します。
    - `CT_NIFTI_DIR`: 入力となるNIfTIファイルが格納されているディレクトリ。
    - `OUTPUT_DIR`: 生成されたDRRデータセットを保存するディレクトリ。

2.  **実行:** 以下のコマンドでスクリプトを実行します。

    ```bash
    bash run_drr_all.sh
    ```

    これにより、`OUTPUT_DIR` 以下に各患者IDごとのディレクトリが作成され、その中に `.pt` ファイル（DRRテンソルとカメラパラメータ）と `.png` ファイル（可視化用画像）が保存されます。

### 5.2. モデルの学習

`run_train.sh` スクリプトは、`nerf-trainer` コンテナ内で `train.py` を実行し、モデルの学習を開始します。

1.  **設定:**
    - `config.yml` を開き、データセットのパスや学習パラメータを設定します。（詳細は後述）
    - `run_train.sh` を開き、`DRR_DATASET_DIR` が `config.yml` で指定した `root_dir` と一致するように設定します。

2.  **実行:** 以下のコマンドで学習を開始します。`sudo` が必要になる場合があります。

    ```bash
    sudo bash run_train.sh
    ```

    - **デバッグモード:** スクリプトはデフォルトで `--debug` フラグ付きで実行され、1エポックあたり1バッチのみ処理します。これにより、パイプラインが正常に動作するかを迅速に確認できます。本格的な学習を行う場合は、`run_train.sh` 内の `--debug` を削除してください。
    - **学習の再開:** `--resume_from` オプションでチェックポイントファイルを指定することで、学習を途中から再開できます。
      ```bash
      python3 train.py --resume_from checkpoints/pixelnerf_epoch_0020.pth
      ```

## 6. 設定ファイル (`config.yml`)

学習の挙動は `config.yml` で制御します。

- **`dataset`**:
    - `root_dir`: DRRデータセットのルートディレクトリパス。
    - `target_angles`: ターゲット（正解データ）として使用するDRRの角度リスト。
    - `img_norm_type`: 画像の正規化手法 (`max_val` または `min_max`)。
- **`training`**:
    - `device`: 使用するデバイス (`cuda` または `cpu`)。
    - `learning_rate`: 学習率。
    - `num_epochs`: 総エポック数。
    - `batch_size`: バッチサイズ。
    - `num_rays_per_batch`: 1バッチあたりでサンプリングするレイの数。
    - `checkpoint_dir`: チェックポイントの保存先ディレクトリ。
- **`model`**:
    - `n_pos_freqs`: 位置エンコーディングの周波数。
    - `d_feature`: 画像特徴量の次元数。
    - `n_samples_coarse`, `n_samples_fine`: Coarse / Fineサンプリングでのサンプル点数。
    - `use_fine_model`: Fineモデルを使用するかどうか。
- **`geometry`**:
    - `invert_extrinsics`: `diffdrr` が出力するWorld-to-Camera行列を、NeRFで必要なCamera-to-World行列に変換するために逆行列をとるかどうかの設定。通常は `True` です。

## 7. 主要なファイル構成

```
.
├── Dockerfile              # 学習用Dockerイメージの定義
├── config.yml              # 学習設定ファイル
├── dataset.py              # PyTorch Datasetクラス
├── generate_drr.py         # 単一CTファイルからDRRを生成
├── model.py                # PixelNeRFモデルのアーキテクチャ
├── train.py                # 学習ループ本体
├── run_drr_all.sh          # 全CTデータに対してDRR生成を実行
└── run_train.sh            # Dockerコンテナで学習を実行
```
