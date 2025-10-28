# ファイル名: generate_drr.py
# ★★★
# このスクリプトは `run_drr.sh` から1回だけ呼び出されます。
# スクリプト内部で /data/CT_Nifti ディレクトリをスキャンし、
# 見つかった全ての .nii.gz / .nii ファイルを順番に処理します。
# ★★★

import torch
import numpy as np
import os
from pathlib import Path
import sys
import matplotlib.pyplot as plt

# --- DiffDRR/Torchio Imports ---
try:
    import torchio as tio
    from diffdrr.drr import DRR
    from diffdrr.data import read
    from diffdrr.pose import RigidTransform, make_matrix, euler_angles_to_matrix
    print("PyTorch, DiffDRR, TorchIO のインポートに成功しました。")

except ImportError as e:
    print(f"エラー: 必要なモジュールが見つかりません: {e}")
    print("diffdrr, torchio, matplotlib が正しくインストールされているか確認してください。")
    print("pip install diffdrr torchio matplotlib")
    sys.exit(1)

# --- ハードコード設定 ---

# 1. 検索対象ディレクトリ (コンテナ内のパス)
CT_NIFTI_DIR = Path("/data/CT_Nifti")

# 2. ベース出力ディレクトリ (コンテナ内のパス)
BASE_OUTPUT_DIR = Path("drr_dataset") # (/workspace/drr_dataset になります)

# 3. レンダリング設定 (ハードコード)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SDD = 1500.0
HEIGHT = 512
WIDTH = 512
DELX = 1.0
VISUALIZE = True # PNGでの可視化を有効にする
ANGLES_TO_GENERATE = [0, 30, 60, 90, 120, 150, 180]

# --- 関数定義 ---

def process_single_file(nifti_file_path: Path, drr_instance: DRR, common_translations: torch.Tensor):
    """
    単一のNiftiファイル（のSubject）を受け取り、全角度のDRRを生成する。
    DRRインスタンスは再利用する。
    """

    # --- 1. 出力先の決定 ---
    filename_stem = nifti_file_path.stem
    try:
        patient_id = filename_stem.split('_')[0]
    except IndexError:
        patient_id = filename_stem

    output_dir_pt = BASE_OUTPUT_DIR / patient_id / "pt"
    output_dir_png = BASE_OUTPUT_DIR / patient_id / "PNG"
    output_dir_pt.mkdir(parents=True, exist_ok=True)
    output_dir_png.mkdir(parents=True, exist_ok=True)

    print(f"  -> .pt 出力先: {output_dir_pt}")
    print(f"  -> .png 出力先: {output_dir_png}")

    # --- 2. カメラ内部パラメータを一度だけ取得 ---
    try:
        intrinsics = drr_instance.detector.intrinsic
    except Exception as e:
        print(f"  ❌ カメラ内部パラメータの取得に失敗: {e}")
        return # このファイルの処理をスキップ

    # --- 3. 角度リストをループしてDRRとパラメータを生成 ---
    for angle in ANGLES_TO_GENERATE:
        print(f"    --- {angle}度のDRRを生成中... ---")

        try:
            # 1. ポーズ (Extrinsics) の定義 (X軸回転)
            rotations_euler = torch.tensor([[np.deg2rad(angle), 0.0, 0.0]], device=DEVICE, dtype=torch.float32)

            # 2. レンダリング実行
            img_tensor = drr_instance(
                rotations_euler,
                common_translations,
                parameterization="euler_angles",
                convention="ZXY"
            )

            output_tensor_raw = img_tensor.squeeze().cpu().detach()
            print(f"      レンダリング結果 (Min/Max/Mean): {output_tensor_raw.min():.4f}, {output_tensor_raw.max():.4f}, {output_tensor_raw.mean():.4f}")


            # 3. DRRテンソル (.pt) の保存
            output_pt_path = output_dir_pt / f"angle_{angle:03d}.pt"
            torch.save(output_tensor_raw, output_pt_path)

            # 4. カメラパラメータ (.pt) の保存
            R = euler_angles_to_matrix(rotations_euler, convention="ZXY")
            matrix = make_matrix(R, common_translations)
            extrinsics = matrix

            cam_params = {
                "intrinsics": intrinsics.squeeze().cpu().detach(),
                "extrinsics": extrinsics.squeeze().cpu().detach()
            }
            output_params_path = output_dir_pt / f"params_{angle:03d}.pt"
            torch.save(cam_params, output_params_path)
            print(f"      ✅ .pt ファイルを保存しました (DRR, Params)")


            # 5. 可視化PNGの保存
            if VISUALIZE:
                save_visualization(output_tensor_raw.numpy(), output_dir_png, angle)

        except Exception as e:
            print(f"    ❌ {angle}度の生成中にエラーが発生しました: {e}")
            continue # この角度をスキップして次へ

def save_visualization(img_np, output_directory, angle):
    """
    DRRテンソル（numpy）をPNG画像として保存する (99.9パーセンタイルでクリップ)
    """
    try:
        img_np[img_np < 0] = 0
        p_99_9 = np.percentile(img_np, 99.9)

        if p_99_9 == 0:
            p_99_9 = img_np.max()

        if p_99_9 == 0:
            img_scaled = np.zeros(img_np.shape, dtype="uint8")
        else:
            img_np[img_np > p_99_9] = p_99_9
            img_scaled = (img_np / p_99_9) * 255
            img_scaled = img_scaled.astype("uint8")

        output_png_path = output_directory / f"angle_{angle:03d}.png"
        plt.imsave(output_png_path, img_scaled, cmap='gray')
        print(f"      ✅ 可視化PNGを保存しました: {output_png_path}")

    except Exception as e:
        print(f"    ❌ 可視化PNGの保存中にエラーが発生しました: {e}")

# --- メイン処理 ---
def main():
    print(f"Using device: {DEVICE}")
    if not CT_NIFTI_DIR.exists():
        print(f"エラー: コンテナ内の検索ディレクトリが見つかりません: {CT_NIFTI_DIR}")
        print("run_drr.sh の -v マウント設定（NAS_PARENT_DIR）が正しいか確認してください。")
        return

    # 1. 処理対象の全Niftiファイルをリストアップ
    files_to_process = list(CT_NIFTI_DIR.glob("*.nii.gz")) + list(CT_NIFTI_DIR.glob("*.nii"))

    if not files_to_process:
        print(f"エラー: {CT_NIFTI_DIR} 内に .nii.gz または .nii ファイルが見つかりません。")
        return

    print(f"{len(files_to_process)} 件のNiftiファイルが見つかりました。処理を開始します。")

    # 2. 全ファイルで共通の並進ベクトルを定義
    SOD_equivalent = SDD * 0.56
    common_translations = torch.tensor([[0.0, SOD_equivalent, 0.0]], device=DEVICE, dtype=torch.float32)

    # 3. 1ファイルずつループ処理
    for i, nifti_file_path in enumerate(files_to_process):
        print("\n=====================================================")
        print(f"処理中 ({i+1}/{len(files_to_process)}): {nifti_file_path.name}")
        print("=====================================================")

        try:
            # 3-1. Subjectの読み込み
            print(f"  diffdrr.data.read でロード中...")
            subject = read(
                volume=str(nifti_file_path),
                orientation="AP",
                center_volume=True,
            )
            print(f"  ボリュームをロードしました: {subject.density.data.shape}")

            # 3-2. DRRモジュールの初期化 (ファイルごとに実行)
            drr_instance = DRR(
                subject,
                sdd=SDD,
                height=HEIGHT,
                delx=DELX,
                width=WIDTH,
            ).to(DEVICE)

            # 3-3. このファイルのDRRを全角度で生成
            process_single_file(nifti_file_path, drr_instance, common_translations)

            # メモリ解放
            del subject
            del drr_instance
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ❌ ファイル {nifti_file_path.name} の処理中に致命的なエラーが発生しました: {e}")
            continue # 次のファイルへ

    print("\n=====================================================")
    print("すべてのファイル処理が完了しました。")

if __name__ == '__main__':
    main()