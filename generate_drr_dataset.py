# ファイル名: generate_drr.py
# argparse を削除し、sys.argv からパスを受け取る。
# drr() の呼び出し方をユーザー指定の（以前の）形式に戻す。
# ★ 出力先を /pt と /PNG に分割

import torch
import numpy as np
import os
from pathlib import Path
import sys # <-- argparseの代わりにsysをインポート
import matplotlib.pyplot as plt

# --- DiffDRR/Torchio Imports ---
try:
    import torchio as tio
    from diffdrr.drr import DRR
    from diffdrr.data import read
    # ★ パラメータ保存に必須のモジュール
    from diffdrr.pose import RigidTransform, make_matrix, euler_angles_to_matrix

except ImportError as e:
    print(f"エラー: 必要なモジュールが見つかりません: {e}")
    print("diffdrr, torchio, matplotlib が正しくインストールされているか確認してください。")
    print("pip install diffdrr torchio matplotlib")
    sys.exit(1)

# --- ハードコード設定 ---
# 1. CTデータのパス (sys.argv[1] から取得)
if len(sys.argv) > 1:
    CT_RELATIVE_PATH = sys.argv[1]
    CT_DATA_PATH = f"/data/{CT_RELATIVE_PATH}"
    filename_stem = Path(CT_RELATIVE_PATH).stem
    try:
        OUTPUT_PATIENT_ID = filename_stem.split('_')[0] 
    except IndexError:
        OUTPUT_PATIENT_ID = filename_stem
    # ★ ベースとなる患者ディレクトリ
    OUTPUT_BASE_DIR = Path(f"drr_dataset/{OUTPUT_PATIENT_ID}")
else:
    print("エラー: CTファイルへの相対パスが指定されていません。")
    print("run_drr.sh の引数としてパスを渡してください。")
    print("例: ./run_drr.sh CT_Nifti/0001_....nii.gz")
    sys.exit(1)

# 2. レンダリング設定 (ハードコード)
SDD = 1500.0
HEIGHT = 512
WIDTH = 512
DELX = 1.0
VISUALIZE = True # PNGでの可視化を有効にする

# 3. 生成する角度リスト
ANGLES_TO_GENERATE = [0, 30, 60, 90, 120, 150, 180]

def main():
    # --- 1. パラメータ設定 (ハードコードされた値を使用) ---
    nifti_file_path = Path(CT_DATA_PATH)
    
    # ★★★ 修正箇所 1: 出力ディレクトリの作成 ★★★
    # /pt と /PNG のサブディレクトリを作成
    output_dir_pt = OUTPUT_BASE_DIR / "pt"
    output_dir_png = OUTPUT_BASE_DIR / "PNG"
    
    try:
        output_dir_pt.mkdir(parents=True, exist_ok=True)
        output_dir_png.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"出力ディレクトリの作成に失敗しました: {e}")
        exit()
    # ★★★ 修正ここまで ★★★
    
    print(f"処理対象 CTファイル: {nifti_file_path}")
    print(f"  -> .pt 出力先: {output_dir_pt.resolve()}")
    print(f"  -> .png 出力先: {output_dir_png.resolve()}")


    # --- 2. `read()` を使ってSubjectを定義 ---
    print(f"\nLoading subject from: {nifti_file_path}")
    try:
        subject = read(
            volume=str(nifti_file_path),
            orientation="AP",
            center_volume=True,
        )
        print("Subject loaded successfully.")
    except FileNotFoundError:
        print(f"エラー: 指定されたファイルが見つかりません: {nifti_file_path}")
        exit()
    except Exception as e:
        print(f"データの読み込み中にエラーが発生しました: {e}")
        exit()
        
    density_image = subject.density
    print(f"ボリュームをロードしました: {density_image.data.shape}, spacing: {density_image.spacing}")
    print(f"  密度 (Min/Max/Mean): {density_image.data.min()}, {density_image.data.max()}, {density_image.data.mean()}")

    # --- 3. DRRモジュールの初期化 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    drr = DRR(
        subject,
        sdd=SDD,
        height=HEIGHT, 
        delx=DELX,   
        width=WIDTH, 
    ).to(device)

    # --- 4. カメラ内部パラメータを一度だけ取得 ---
    try:
        intrinsics = drr.detector.intrinsic
        print("カメラ内部パラメータ (Intrinsics) を取得しました。")
    except Exception as e:
        print(f"カメラ内部パラメータの取得に失敗しました: {e}")
        return

    # --- 5. 角度リストをループしてDRRとパラメータを生成 ---
    print(f"\n{len(ANGLES_TO_GENERATE)} 角度のDRR生成を開始します...")
    
    # 並進ベクトル (Y軸方向に移動)
    SOD_equivalent = SDD * 0.56
    translations = torch.tensor([[0.0, SOD_equivalent, 0.0]], device=device, dtype=torch.float32)

    for angle in ANGLES_TO_GENERATE:
        print(f"--- {angle}度のDRRを生成中... ---")
        
        try:
            # 1. ポーズ (Extrinsics) の定義
            # X軸周りの回転
            rotations_euler = torch.tensor([[np.deg2rad(angle), 0.0, 0.0]], device=device, dtype=torch.float32)
            
            # 2. レンダリング実行
            img_tensor = drr(
                rotations_euler, 
                translations, 
                parameterization="euler_angles", 
                convention="ZXY"
            )
            
            output_tensor_raw = img_tensor.squeeze().cpu().detach()
            print(f"  レンダリング結果 (Min/Max/Mean): {output_tensor_raw.min()}, {output_tensor_raw.max()}, {output_tensor_raw.mean()}")

            # ★★★ 修正箇所 2: .pt ファイルの保存先変更 ★★★
            # 3. DRRテンソル (.pt) の保存 (生のテンソル)
            output_pt_path = output_dir_pt / f"angle_{angle:03d}.pt"
            torch.save(output_tensor_raw, output_pt_path)
            print(f"  ✅ DRRテンソルを保存しました: {output_pt_path}")

            # 4. カメラパラメータ (.pt) の保存
            R = euler_angles_to_matrix(rotations_euler, convention="ZXY")
            matrix = make_matrix(R, translations)
            extrinsics = matrix # (1, 4, 4) のテンソル

            cam_params = {
                "intrinsics": intrinsics.squeeze().cpu().detach(), 
                "extrinsics": extrinsics.squeeze().cpu().detach()
            }
            output_params_path = output_dir_pt / f"params_{angle:03d}.pt"
            torch.save(cam_params, output_params_path)
            print(f"  ✅ カメラパラメータを保存しました: {output_params_path}")
            # ★★★ 修正ここまで ★★★

            # ★★★ 修正箇所 3: .png ファイルの保存先変更 ★★★
            # 5. 可視化PNGの保存 (VISUALIZEフラグで制御)
            if VISUALIZE:
                # save_visualization には 'output_dir_png' を渡す
                save_visualization(output_tensor_raw.numpy(), output_dir_png, angle)
            # ★★★ 修正ここまで ★★★

        except Exception as e:
            print(f"  ❌ {angle}度の生成中にエラーが発生しました: {e}")

    print("\nすべてのDRR生成が完了しました。")


def save_visualization(img_np, output_directory, angle):
    """
    DRRテンソル（numpy）をPNG画像として保存する
    (output_directory には /PNG サブディレクトリが渡される)
    """
    try:
        img_np[img_np < 0] = 0
        
        # 99.9パーセンタイルでクリッピング (白飛び防止)
        p_99_9 = np.percentile(img_np, 99.9)
        
        if p_99_9 == 0:
            p_99_9 = img_np.max() # 99.9パーセンタイルが0なら、最大値を使う

        if p_99_9 == 0:
            print("  （可視化警告: レンダリング結果がすべて0です）")
            img_scaled = np.zeros(img_np.shape, dtype="uint8")
        else:
            img_np[img_np > p_99_9] = p_99_9
            img_scaled = (img_np / p_99_9) * 255
            img_scaled = img_scaled.astype("uint8")

        output_png_path = output_directory / f"angle_{angle:03d}.png"
        
        # plt.imsave を使う (matplotlib)
        plt.imsave(output_png_path, img_scaled, cmap='gray')
        print(f"  ✅ 可視化PNGを保存しました: {output_png_path}")

    except Exception as e:
        print(f"  ❌ 可視化PNGの保存中にエラーが発生しました: {e}")


if __name__ == '__main__':
    main()
