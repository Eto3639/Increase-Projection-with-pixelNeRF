import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random
from torchvision import transforms
from torchvision.transforms.functional import resize

def normalize_drr_max_val(tensor):
    """最大値で割って [0, 1] に正規化"""
    max_val = tensor.max()
    if max_val > 1e-6: # ほぼ0でない場合
        return tensor / max_val
    else:
        return torch.zeros_like(tensor)

def normalize_drr_min_max(tensor):
    """Min-Max正規化で [0, 1] に"""
    min_val = tensor.min()
    max_val = tensor.max()
    if max_val > min_val + 1e-6:
        return (tensor - min_val) / (max_val - min_val)
    else:
        return torch.zeros_like(tensor)

class DRRDataset(Dataset):
    """
    DiffDRRで生成されたデータセット用のカスタムDatasetクラス。
    0度と90度を入力とし、他の角度をランダムにターゲットとする。
    """
    def __init__(self, cfg: dict, split: str = 'train', patient_dirs: list = None):
        self.root_path = Path(cfg['dataset']['root_dir'])
        self.target_angles = cfg['dataset']['target_angles']
        self.img_norm_type = cfg['dataset']['img_norm_type']
        self.img_size = cfg['dataset'].get('img_size', None)
        self.split = split

        # --- Augmentation ---
        self.transform = None
        aug_cfg = cfg['dataset'].get('augmentation', {})
        if self.split == 'train' and aug_cfg.get('enabled', False):
            self.transform = transforms.ColorJitter(
                brightness=aug_cfg.get('brightness', 0),
                contrast=aug_cfg.get('contrast', 0)
            )
            print("Data augmentation enabled for training split.")

        if patient_dirs is None:
            self.patient_dirs = sorted([d for d in self.root_path.iterdir() if d.is_dir()])
        else:
            self.patient_dirs = patient_dirs
        
        self.file_list = []
        for patient_dir in self.patient_dirs:
            pt_dir = patient_dir / "pt"
            required_files_exist = (
                (pt_dir / "angle_000.pt").exists() and
                (pt_dir / "angle_090.pt").exists() and
                (pt_dir / "params_000.pt").exists() and
                (pt_dir / "params_090.pt").exists()
            )
            if not required_files_exist:
                print(f"警告(Dataset): 患者 {patient_dir.name} に入力ファイルが不足。スキップ。")
                continue
                
            has_target = any(
                (pt_dir / f"angle_{angle:03d}.pt").exists() and
                (pt_dir / f"params_{angle:03d}.pt").exists()
                for angle in self.target_angles
            )
            if has_target:
                self.file_list.append(pt_dir)
            else:
                 print(f"警告(Dataset): 患者 {patient_dir.name} にターゲットファイルが存在せず。スキップ。")

        if not self.file_list:
             raise RuntimeError(f"指定されたディレクトリ '{self.root_path}' に有効なデータが見つかりません。")
        print(f"データセット初期化完了。{len(self.file_list)} 人の患者データが見つかりました。")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        pt_dir = self.file_list[idx]
        valid_targets = [a for a in self.target_angles if (pt_dir / f"angle_{a:03d}.pt").exists()]
        target_angle = random.choice(valid_targets)
        return self._load_and_process_item(pt_dir, target_angle)

    def get_item_for_visualization(self, patient_idx: int, angle: int):
        """指定された患者IDと角度のデータを取得する（可視化用）"""
        pt_dir = self.file_list[patient_idx]
        target_img_path = pt_dir / f"angle_{angle:03d}.pt"
        target_params_path = pt_dir / f"params_{angle:03d}.pt"

        if not (target_img_path.exists() and target_params_path.exists()):
            return None # その患者が指定の角度を持っていない場合はNoneを返す

        return self._load_and_process_item(pt_dir, angle)

    def _load_and_process_item(self, pt_dir: Path, target_angle: int):
        # --- データ読み込み ---
        img_0 = torch.load(pt_dir / "angle_000.pt")
        img_90 = torch.load(pt_dir / "angle_090.pt")
        params_0 = torch.load(pt_dir / "params_000.pt")
        params_90 = torch.load(pt_dir / "params_090.pt")

        target_img = torch.load(pt_dir / f"angle_{target_angle:03d}.pt")
        target_params = torch.load(pt_dir / f"params_{target_angle:03d}.pt")
        target_params['angle'] = torch.tensor(float(target_angle))

        # --- 前処理 ---
        # チャンネル次元を追加 (C, H, W) -> (1, H, W)
        img_0 = img_0.unsqueeze(0)
        img_90 = img_90.unsqueeze(0)
        target_img = target_img.unsqueeze(0)

        # 正規化
        if self.img_norm_type == "max_val":
            img_0 = normalize_drr_max_val(img_0)
            img_90 = normalize_drr_max_val(img_90)
            target_img = normalize_drr_max_val(target_img)
        elif self.img_norm_type == "min_max":
            img_0 = normalize_drr_min_max(img_0)
            img_90 = normalize_drr_min_max(img_90)
            target_img = normalize_drr_min_max(target_img)
        else:
             # 正規化しない場合 (または他のタイプ)
             pass

        # --- Augmentation ---
        if self.transform:
            # 同じ変換を適用するために一度スタックする
            all_images = torch.cat([img_0, img_90, target_img], dim=0)
            all_images_transformed = self.transform(all_images)
            img_0, img_90, target_img = torch.chunk(all_images_transformed, 3, dim=0)

        # オプション: リサイズ
        if self.img_size:
            original_h, original_w = img_0.shape[-2:]
            new_h, new_w = self.img_size

            if original_h != new_h or original_w != new_w:
                # 画像をリサイズ
                img_0 = resize(img_0, self.img_size, antialias=True)
                img_90 = resize(img_90, self.img_size, antialias=True)
                target_img = resize(target_img, self.img_size, antialias=True)
                
                # カメラのIntrinsicsも調整
                scale_h = new_h / original_h
                scale_w = new_w / original_w

                def adjust_intrinsics(params):
                    intrinsics = params['intrinsics']
                    intrinsics[0, 0] *= scale_w  # fx
                    intrinsics[1, 1] *= scale_h  # fy
                    intrinsics[0, 2] *= scale_w  # cx
                    intrinsics[1, 2] *= scale_h  # cy
                    params['intrinsics'] = intrinsics
                    return params

                params_0 = adjust_intrinsics(params_0)
                params_90 = adjust_intrinsics(params_90)
                target_params = adjust_intrinsics(target_params)

        # カメラパラメータの次元を確認し、必要ならバッチ次元を追加 (load後は通常ない)
        if params_0['intrinsics'].dim() == 2:
            params_0['intrinsics'] = params_0['intrinsics'].unsqueeze(0) # (3, 3) -> (1, 3, 3)
        if params_0['extrinsics'].dim() == 2:
            params_0['extrinsics'] = params_0['extrinsics'].unsqueeze(0) # (4, 4) -> (1, 4, 4)
        # params_90, target_params も同様に処理
        if params_90['intrinsics'].dim() == 2:
             params_90['intrinsics'] = params_90['intrinsics'].unsqueeze(0)
        if params_90['extrinsics'].dim() == 2:
             params_90['extrinsics'] = params_90['extrinsics'].unsqueeze(0)
        if target_params['intrinsics'].dim() == 2:
             target_params['intrinsics'] = target_params['intrinsics'].unsqueeze(0)
        if target_params['extrinsics'].dim() == 2:
             target_params['extrinsics'] = target_params['extrinsics'].unsqueeze(0)


        return {
            "img_0": img_0.float(), # float型に変換
            "params_0": params_0,
            "img_90": img_90.float(),
            "params_90": params_90,
            "target_img": target_img.float(),
            "target_params": target_params,
            "info": {"patient": pt_dir.parent.name, "target_angle": target_angle}
        }
