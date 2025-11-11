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
ANGLES_TO_GENERATE = [0, 30, 60, 90, 120, 150, 180]

# --- 関数定義 ---
import torch.nn.functional as F

def look_at_w2c(eye, target, up, device):
    """
    Pytorch3Dの実装を参考にした、堅牢なlook_at行列計算。
    eye, target, up からワールド→カメラ変換行列（extrinsics）を計算する。
    """
    z_axis = F.normalize(eye - target, eps=1e-5, dim=0)
    if torch.allclose(torch.abs(torch.dot(up, z_axis)), torch.tensor(1.0)):
        up = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=torch.float32)
    x_axis = F.normalize(torch.cross(z_axis, up), eps=1e-5, dim=0)
    y_axis = torch.cross(x_axis, z_axis)
    R = torch.stack([x_axis, y_axis, z_axis], dim=0)
    t = -torch.matmul(R, eye)
    w2c = torch.eye(4, device=device, dtype=torch.float32)
    w2c[:3, :3] = R
    w2c[:3, 3] = t
    return w2c

def save_debug_visualizations(subject, all_cam_params, output_dir):
    """
    デバッグ用の可視化画像を生成・保存する。
    1. CTボリュームの中心スライス
    2. カメラジオメトリの3Dプロット
    """
    debug_output_dir = Path("debug_outputs")
    debug_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  [DEBUG] デバッグ画像を '{debug_output_dir}/' に保存します...")

    # 1. CTスライスの保存
    try:
        ct_slice = subject.density.data[0, :, :, subject.density.shape[-1] // 2].cpu().numpy()
        plt.imsave(debug_output_dir / "debug_ct_slice.png", ct_slice, cmap='gray')
        print(f"    ✅ CTスライス画像を保存しました: {debug_output_dir / 'debug_ct_slice.png'}")
    except Exception as e:
        print(f"    ❌ CTスライスの保存に失敗: {e}")

    # 2. カメラジオメトリの3Dプロット
    try:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter([0], [0], [0], c='r', marker='o', s=100, label='Volume Center')

        for angle, params in all_cam_params.items():
            w2c = params['extrinsics'].cpu()
            try:
                c2w = torch.inverse(w2c)
            except torch.linalg.LinAlgError:
                print(f"    ❌ 警告: {angle}度のExtrinsics行列は特異であり、逆行列を計算できません。プロットから除外します。")
                continue

            cam_pos = c2w[:3, 3]
            forward = -c2w[:3, 2] # カメラの前方ベクトルは-Z軸

            ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2], forward[0], forward[1], forward[2], length=200, normalize=True, color='b')
            ax.text(cam_pos[0], cam_pos[1], cam_pos[2], f' {angle}°')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Camera Positions and View Directions')
        ax.legend()
        # Make axes equal
        max_range = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]).ptp().max() / 2.0
        mean_x = np.mean(ax.get_xlim())
        mean_y = np.mean(ax.get_ylim())
        mean_z = np.mean(ax.get_zlim())
        ax.set_xlim(mean_x - max_range, mean_x + max_range)
        ax.set_ylim(mean_y - max_range, mean_y + max_range)
        ax.set_zlim(mean_z - max_range, mean_z + max_range)

        plt.savefig(debug_output_dir / "debug_camera_geometry.png")
        plt.close(fig)
        print(f"    ✅ カメラジオメトリのプロットを保存しました: {debug_output_dir / 'debug_camera_geometry.png'}")

    except Exception as e:
        print(f"    ❌ カメラジオメトリのプロット作成に失敗: {e}")


def process_single_file(nifti_file_path: Path, drr_instance: DRR, sdd: float):
    """
    単一のNiftiファイルを受け取り、全角度のカメラパラメータを生成・保存する。
    追加で、デバッグ用の可視化も行う。
    """
    
    # SODをSDDに対する比率で計算
    sod = sdd * 0.56
    print(f"  -> [DEBUG] ジオメトリ設定: SDD={sdd:.1f}, SOD={sod:.1f} (ODD={sdd - sod:.1f})")

    filename_stem = nifti_file_path.stem.split('.')[0]
    patient_id = filename_stem.split('_')[0] if '_' in filename_stem else filename_stem

    output_dir_pt = BASE_OUTPUT_DIR / patient_id / "pt"
    output_dir_pt.mkdir(parents=True, exist_ok=True)
    print(f"  -> カメラパラメータ(.pt)の出力先: {output_dir_pt}")

    try:
        intrinsics = drr_instance.detector.intrinsic
    except Exception as e:
        print(f"  ❌ カメラ内部パラメータの取得に失敗: {e}")
        return

    all_cam_params = {}
    for angle in ANGLES_TO_GENERATE:
        print(f"    --- {angle}度のカメラパラメータを生成中... ---")
        try:
            rad = np.deg2rad(angle)
            eye = torch.tensor([sod * np.sin(rad), 0.0, sod * np.cos(rad)], device=DEVICE, dtype=torch.float32)
            target = torch.tensor([0.0, 0.0, 0.0], device=DEVICE, dtype=torch.float32)
            up = torch.tensor([0.0, 1.0, 0.0], device=DEVICE, dtype=torch.float32)
            
            extrinsics = look_at_w2c(eye, target, up, device=DEVICE)
            
            cam_params = {
                "intrinsics": intrinsics.squeeze().cpu().detach(),
                "extrinsics": extrinsics.cpu().detach()
            }
            all_cam_params[angle] = cam_params # デバッグプロット用に保持

            output_params_path = output_dir_pt / f"params_{angle:03d}.pt"
            torch.save(cam_params, output_params_path)
            print(f"      ✅ カメラパラメータを保存しました: {output_params_path}")

        except Exception as e:
            print(f"    ❌ {angle}度のパラメータ生成中にエラーが発生しました: {e}")
            continue
    
    # 全ての角度の処理が終わった後、デバッグ用の可視化を実行
    if all_cam_params:
        save_debug_visualizations(drr_instance.subject, all_cam_params, output_dir_pt.parent)


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
            print(f"  diffdrr.data.read でロード中...")
            
            # ★★★ vmin/vmax は（密度が0でないことを保証するため）必須です ★★★
            subject = read(
                volume=str(nifti_file_path),
                orientation="AP",
                center_volume=True,
                vmin=-1000.0, # CT値（HU）の最小値（空気）
                vmax=3000.0   # CT値（HU）の最大値（骨）
            )
            print(f"  ボリュームをロードしました: {subject.density.data.shape}")
            density_data = subject.density.data
            print(f"  [DEBUG] Density stats: min={density_data.min():.4f}, max={density_data.max():.4f}, mean={density_data.mean():.4f}")

            # 密度が0の場合、警告
            if density_data.max() == 0.0:
                print("  ❌ 警告: 密度の最大値が0です。vmin/vmax の設定がCT値の範囲と合っていません。")

            drr_instance = DRR(
                subject,
                sdd=SDD, # 修正後の SDD (2000.0) を使用
                height=HEIGHT,
                delx=DELX,
                width=WIDTH,
            ).to(DEVICE)

            process_single_file(nifti_file_path, drr_instance, SDD) # 修正後の SDD (2000.0) を使用

            del subject
            del drr_instance
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ❌ ファイル {nifti_file_path.name} の処理中に致命的なエラーが発生しました: {e}")
            continue

    print("\n=====================================================")
    print("すべてのファイル処理が完了しました。")

if __name__ == '__main__':
    main()