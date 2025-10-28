import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import yaml # PyYAMLをインポート
from tqdm import tqdm
import os # ディレクトリ作成用
import argparse # <-- コマンドライン引数処理

# --- ローカルモジュール ---
from dataset import DRRDataset
from model import PixelNeRFModel, get_rays

def load_config(path="config.yml"):
    """YAML設定ファイルを読み込む"""
    try:
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)
        print("設定ファイルをロードしました。")
        return cfg
    except FileNotFoundError:
        print(f"エラー: 設定ファイル '{path}' が見つかりません。")
        exit()
    except Exception as e:
        print(f"エラー: 設定ファイルの読み込み中にエラーが発生しました: {e}")
        exit()

def safe_inverse(matrix_batch):
    """バッチ内の各4x4行列の逆行列を計算する（失敗時に警告）"""
    inverses = []
    for i in range(matrix_batch.shape[0]):
        try:
            # .float() を追加してデータ型を保証
            inv = torch.inverse(matrix_batch[i].float())
            inverses.append(inv)
        except Exception as e:
            print(f"警告(safe_inverse): バッチ {i} の逆行列計算に失敗: {e}")
            inverses.append(torch.eye(4, device=matrix_batch.device, dtype=matrix_batch.dtype))
    return torch.stack(inverses, dim=0)

# --- ★★★ 追加: コマンドライン引数パーサー ★★★ ---
def parse_args():
    parser = argparse.ArgumentParser(description="PixelNeRF Training Script")
    parser.add_argument("--config", type=str, default="config.yml",
                        help="設定ファイルへのパス (default: config.yml)")
    parser.add_argument("--debug", action="store_true",
                        help="デバッグモード (各エポック1バッチのみ実行)")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="学習を再開するチェックポイントファイル (.pth) へのパス")
    return parser.parse_args()
# --- ★★★ 追加ここまで ★★★ ---

def train(args): # <-- args を受け取るように変更
    # --- 1. 設定のロード ---
    cfg = load_config(args.config) # <-- 引数から設定ファイルパスを取得
    DEVICE = torch.device(cfg['training']['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # --- 2. ディレクトリ作成 ---
    log_dir = Path(cfg['training']['log_dir'])
    checkpoint_dir = Path(cfg['training']['checkpoint_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # --- 3. データセットとデータローダー ---
    try:
        train_dataset = DRRDataset(cfg=cfg)
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg['training']['batch_size'],
            shuffle=True,
            num_workers=cfg['training'].get('num_workers', 0)
        )
    except Exception as e:
         print(f"エラー: データローダーの初期化に失敗しました: {e}")
         return

    # --- 4. モデル ---
    model = PixelNeRFModel(cfg=cfg).to(DEVICE)

    # --- 5. オプティマイザと損失関数 ---
    optimizer = optim.Adam(model.parameters(), lr=cfg['training']['learning_rate'])
    loss_fn_l1 = nn.L1Loss()

    # --- ★★★ 追加: レジューム処理 ★★★ ---
    start_epoch = 0
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if resume_path.is_file():
            print(f"チェックポイントをロード中: {resume_path}")
            try:
                checkpoint = torch.load(resume_path, map_location=DEVICE)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] # 次のエポックから開始
                print(f"チェックポイントをロードしました。エポック {start_epoch + 1} から再開します。")
            except Exception as e:
                print(f"エラー: チェックポイントのロードに失敗しました: {e}")
                print("学習を最初から開始します。")
                start_epoch = 0 # エラー時は最初から
        else:
            print(f"警告: 指定されたチェックポイントファイルが見つかりません: {resume_path}")
            print("学習を最初から開始します。")
    else:
        print("学習を最初から開始します。")
    # --- ★★★ 追加ここまで ★★★ ---


    # --- 6. 学習ループ ---
    print("\n学習開始...")
    num_epochs = cfg['training']['num_epochs']
    batch_size = cfg['training']['batch_size'] # configから取得 (datasetとは別)
    num_rays_per_batch = cfg['training']['num_rays_per_batch']
    invert_extrinsics = cfg['geometry']['invert_extrinsics']

    # エポックループを start_epoch から開始
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss_coarse = 0.0
        total_loss_fine = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            try:
                # データをGPUへ
                img_0 = batch["img_0"].to(DEVICE)
                img_90 = batch["img_90"].to(DEVICE)
                target_img = batch["target_img"].to(DEVICE)

                params_0 = {k: v.to(DEVICE) for k, v in batch["params_0"].items()}
                params_90 = {k: v.to(DEVICE) for k, v in batch["params_90"].items()}
                target_params = {k: v.to(DEVICE) for k, v in batch["target_params"].items()}

                B, _, H, W = target_img.shape

                # --- レイトレーシングの準備 ---
                K = target_params["intrinsics"][:, :3, :3]
                w2c = target_params["extrinsics"]

                if invert_extrinsics:
                    c2w = safe_inverse(w2c)
                else:
                    c2w = w2c

                rays_o, rays_d = get_rays(H, W, K, c2w)
                target_pixels = target_img.permute(0, 2, 3, 1).reshape(B, -1, 1)

                # レイのランダムサンプリング
                num_pixels = H * W
                actual_batch_size = img_0.shape[0]
                rays_per_image = num_rays_per_batch // actual_batch_size
                selected_indices = []
                for b in range(actual_batch_size):
                     indices = torch.randperm(num_pixels, device=DEVICE)[:rays_per_image]
                     selected_indices.append(indices + b * num_pixels)

                indices = torch.cat(selected_indices)

                rays_o_flat = rays_o.view(-1, 3)
                rays_d_flat = rays_d.view(-1, 3)
                target_pixels_flat = target_pixels.view(-1, 1)

                rays_o_batch = rays_o_flat[indices]
                rays_d_batch = rays_d_flat[indices]
                target_pixels_batch = target_pixels_flat[indices]

                rays_o_batch = rays_o_batch.view(actual_batch_size, rays_per_image, 3)
                rays_d_batch = rays_d_batch.view(actual_batch_size, rays_per_image, 3)
                target_pixels_batch = target_pixels_batch.view(actual_batch_size, rays_per_image, 1)

                # --- モデル実行と損失計算 ---
                optimizer.zero_grad()
                outputs = model(
                    img_0, img_90, params_0, params_90,
                    rays_o_batch, rays_d_batch
                )

                loss_coarse = loss_fn_l1(outputs["coarse"], target_pixels_batch)
                loss = loss_coarse
                loss_fine_val = 0.0

                if model.use_fine_model and "fine" in outputs:
                    loss_fine = loss_fn_l1(outputs["fine"], target_pixels_batch)
                    loss = loss + loss_fine
                    loss_fine_val = loss_fine.item()

                loss.backward()
                optimizer.step()

                total_loss_coarse += loss_coarse.item()
                total_loss_fine += loss_fine_val

                progress_bar.set_postfix({
                    "Loss_C": f"{loss_coarse.item():.4f}",
                    "Loss_F": f"{loss_fine_val:.4f}"
                })

                # --- ★★★ 追加: デバッグモード処理 ★★★ ---
                if args.debug:
                    print("\nデバッグモード: 1バッチ処理後、次のエポックへスキップします。")
                    break # このエポックのループを抜ける
                # --- ★★★ 追加ここまで ★★★ ---


            except RuntimeError as e:
                 # bmm エラーなどがここでキャッチされるはず
                 print(f"\nエラー: バッチ処理中にRuntimeErrorが発生しました: {e}")
                 print("このバッチをスキップします。")
                 torch.cuda.empty_cache()
                 continue # 次のバッチへ
            except Exception as e:
                 print(f"\nエラー: バッチ処理中に予期せぬエラーが発生しました: {e}")
                 torch.cuda.empty_cache()
                 continue # 次のバッチへ

        # --- エポック終了処理 ---
        # イテレーションが0の場合のゼロ除算を回避
        num_batches = len(train_loader)
        if args.debug:
             num_batches = 1 # デバッグモードでは1バッチのみ

        if num_batches > 0:
            avg_loss_coarse = total_loss_coarse / num_batches
            avg_loss_fine = total_loss_fine / num_batches
            print(f"Epoch {epoch+1}完了. Avg Loss Coarse: {avg_loss_coarse:.4f}, Avg Loss Fine: {avg_loss_fine:.4f}")
        else:
             print(f"Epoch {epoch+1}完了. データローダーが空でした。")


        # チェックポイント保存
        save_freq = cfg['training']['save_checkpoint_freq']
        if (epoch + 1) % save_freq == 0:
            checkpoint_path = checkpoint_dir / f"pixelnerf_epoch_{(epoch+1):04d}.pth"
            try:
                torch.save({
                    'epoch': epoch + 1, # ★ 保存するのは完了したエポック数+1
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss_coarse': avg_loss_coarse if num_batches > 0 else 0, # 保存用に値を入れる
                    'loss_fine': avg_loss_fine if num_batches > 0 else 0,
                    'config': cfg # 設定ファイルも保存しておくと便利
                }, checkpoint_path)
                print(f"チェックポイントを保存しました: {checkpoint_path}")
            except Exception as e:
                print(f"エラー: チェックポイントの保存に失敗しました: {e}")

        # TODO: 検証ループ (run_validation)

    print("\n学習完了。")

if __name__ == "__main__":
    # PyYAMLのインストールが必要: pip install pyyaml
    try:
         import yaml
    except ImportError:
         print("エラー: PyYAML がインストールされていません。`pip install pyyaml` を実行してください。")
         exit()

    # コマンドライン引数をパース
    args = parse_args()

    # 学習関数を実行
    train(args)