# ファイル名: generate_drr.py
import torch
import numpy as np
import os
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # For 3D plotting
import yaml
import argparse

# --- DiffDRR/Torchio Imports ---
try:
    import torchio as tio
    from diffdrr.drr import DRR
    from diffdrr.data import read
    from diffdrr.pose import RigidTransform, make_matrix, euler_angles_to_matrix
    import diffdrr.renderers as renderers_module
    print("PyTorch, DiffDRR, TorchIO のインポートに成功しました。")

except ImportError as e:
    print(f"エラー: 必要なモジュールが見つかりません: {e}")
    print("diffdrr, torchio, matplotlib が正しくインストールされているか確認してください。")
    print("pip install diffdrr torchio matplotlib")
    sys.exit(1)

# ----------------------------------------------------------------------------------
# --- MONKEY-PATCH FOR diffdrr v0.1.3 ---
# diffdrr.renderers._get_xyzs の座標正規化バグを修正
print("🐛 Applying monkey-patch to diffdrr.renderers._get_xyzs to fix coordinate normalization bug...")

def _get_xyzs_patched(alpha, source, target, dims, voxel_shift, eps):
    """Given a set of rays and parametric coordinates, calculates the XYZ coordinates."""
    # Get the world coordinates of every point parameterized by alpha
    xyzs = (
        source.unsqueeze(-2)
        + alpha.unsqueeze(-1) * (target - source + eps).unsqueeze(2)
    ).unsqueeze(1)

    # Normalize coordinates to be in [-1, +1] for grid_sample
    # BUG: `dims` is (Z, Y, X) but `xyzs` is (X, Y, Z).
    # FIX: Reorder dims to (X, Y, Z) before division.
    dims_xyz = dims[[2, 1, 0]]
    xyzs = 2 * (xyzs + voxel_shift) / (dims_xyz - 1) - 1
    return xyzs

renderers_module._get_xyzs = _get_xyzs_patched
# --- END MONKEY-PATCH ---
# ----------------------------------------------------------------------------------

def load_config(path="config.yml"):

    """Load YAML configuration file."""

    with open(path, 'r') as f:

        cfg = yaml.safe_load(f)

    return cfg



# --- ハードコード設定 ---

CT_NIFTI_DIR = Path("/data/CT_Nifti")

BASE_OUTPUT_DIR = Path("drr_dataset")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HEIGHT = 512

WIDTH = 512

DELX = 1.0

VISUALIZE = True

ANGLES_TO_GENERATE = [0, 60, 90]



# --- 関数定義 ---

import torch.nn.functional as F



def process_single_file(nifti_file_path: Path, drr_instance: DRR, sdd: float):



    """



    ライブラリのデフォルト姿勢（orientation="AP"）でDRRを生成する、最終テスト。



    """



    print(f"    --- FINAL TEST: ライブラリのデフォルト姿勢でレンダリングを実行... ---")



    try:
        # an identity matrix pose to the `calibration` keyword argument.
        rotation = torch.eye(3, device=DEVICE)
        translation = torch.zeros(3, 1, device=DEVICE)
        transform_matrix = make_matrix(rotation, translation)
        pose = RigidTransform(transform_matrix)
        img = drr_instance(calibration=pose)

        img_stats = f"min={img.min():.4f}, max={img.max():.4f}, mean={img.mean():.4f}"
        print(f"      📊 DRR画像統計: {img_stats}")

        if img.max() > 0.0:
            print("      ✅✅✅ [BREAKTHROUGH!] 画像が生成されました！黒ではありません！")
        else:
            print("      ❌❌❌ [FAILURE] これでも画像は真っ黒です。問題はライブラリ自体にある可能性が濃厚です。")

        # デバッグ用のPNG画像を保存
        debug_output_dir = Path("debug_outputs")
        debug_output_dir.mkdir(exist_ok=True)
        drr_save_path = debug_output_dir / "debug_drr_default_orientation.png"
        img_to_save = img.squeeze().cpu().detach().numpy()
        if img_to_save.max() > 0:
            img_to_save = (img_to_save - img_to_save.min()) / (img_to_save.max() - img_to_save.min())
        plt.imsave(drr_save_path, img_to_save, cmap='gray')
        print(f"      ✅ [DEBUG] デフォルト姿勢のDRRを画像として保存しました: {drr_save_path}")

    except Exception as e:
        print(f"    ❌ デフォルト姿勢でのレンダリング中にエラーが発生しました: {e}")



def main():

    cfg = load_config()

    

    SDD = cfg['drr']['sdd']

    print(f"  -> [DEBUG] SDD (線源-検出器間距離) を config.yml から読み込みました: {SDD}")



    parser = argparse.ArgumentParser(description="Generate DRRs from NIFTI files.")

    parser.add_argument("--file", type=str, default=None, help="Path to a single NIFTI file to process (within the container).")

    args = parser.parse_args()



    print(f"Using device: {DEVICE}")

    

    if args.file:

        single_file = Path(args.file)

        if not single_file.exists():

            print(f"エラー: 指定されたファイルが見つかりません: {args.file}")

            return

        files_to_process = [single_file]

        print(f"単一ファイルモードで実行します: {args.file}")

    else:

        if not CT_NIFTI_DIR.exists():

            print(f"エラー: コンテナ内の検索ディレクトリが見つかりません: {CT_NIFTI_DIR}")

            return

        files_to_process = list(CT_NIFTI_DIR.glob("*.nii.gz")) + list(CT_NIFTI_DIR.glob("*.nii"))

        if not files_to_process:

            print(f"エラー: {CT_NIFTI_DIR} 内に .nii.gz または .nii ファイルが見つかりません。")

            return

        print(f"{len(files_to_process)} 件のNiftiファイルが見つかりました。処理を開始します。")



    for i, nifti_file_path in enumerate(files_to_process):

        print("\n=====================================================")

        print(f"処理中 ({i+1}/{len(files_to_process)}): {nifti_file_path.name}")

        print("=====================================================")



        try:

            print(f"  diffdrr.data.read でロード中...")

            

            subject = read(

                volume=str(nifti_file_path),

                orientation="AP",

                center_volume=True,

                vmin=-1000.0,

                vmax=1000.0

            )

            print(f"  ボリュームをロードしました: {subject.density.data.shape}")

            print(f"    [DEBUG] Volume Spacing: {subject.spacing}")

            print(f"    [DEBUG] Volume Affine Matrix:\n{subject.volume.affine}")

            density_data = subject.density.data

            print(f"  [DEBUG] Density stats: min={density_data.min():.4f}, max={density_data.max():.4f}, mean={density_data.mean():.4f}")



            if density_data.max() == 0.0:

                print("  ❌ 警告: 密度の最大値が0です。vmin/vmax の設定がCT値の範囲と合っていません。")



            drr_instance = DRR(

                subject,

                sdd=SDD,

                height=HEIGHT,

                delx=DELX,

                width=WIDTH,

                renderer="siddon",

            ).to(DEVICE)



            process_single_file(nifti_file_path, drr_instance, SDD)



            del subject

            del drr_instance

            if torch.cuda.is_available():

                torch.cuda.empty_cache()



        except Exception as e:

            print(f"  ❌ ファイル {nifti_file_path.name} の処理中に致命的なエラーが発生しました: {e}")

            continue



    print("\n=====================================================")

    print("すべてのファイル処理が完了しました。")



if __name__ == '__main__':

    main()


