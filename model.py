import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# --- 3.1 ヘルパー: Positional Encoder ---
class PositionalEncoder(nn.Module):
    """標準的なNeRFの位置エンコーダ"""
    def __init__(self, d_input: int, n_freqs: int, log_space: bool = True):
        super().__init__()
        self.d_input = d_input
        self.n_freqs = n_freqs
        self.log_space = log_space
        self.d_output = d_input * (1 + 2 * n_freqs)
        self.embed_fns = [lambda x: x]
        if self.log_space:
            freq_bands = 2.**torch.linspace(0., self.n_freqs - 1, self.n_freqs)
        else:
            freq_bands = torch.linspace(1., 2.**(self.n_freqs - 1), self.n_freqs)
        for freq in freq_bands:
            self.embed_fns.append(lambda x, freq=freq: torch.sin(x * freq))
            self.embed_fns.append(lambda x, freq=freq: torch.cos(x * freq))
    def forward(self, x):
        return torch.cat([fn(x) for fn in self.embed_fns], dim=-1)

import torchvision

# --- 3.2 画像エンコーダ (CNN / Transformer) ---
class ImageEncoder(nn.Module):
    """
    バックボーンネットワークを使用して画像から特徴マップを抽出するエンコーダ。
    'densenet121'が指定された場合は、CheXNetの構造と重みを正しく読み込むための専用処理を実行する。
    それ以外のモデルについてはtimmライブラリを使用する。
    """
    def __init__(self, backbone_cfg: dict, d_out: int):
        super().__init__()
        model_name = backbone_cfg['name']
        
        if model_name == 'densenet121':
            # --- CheXNet (densenet121) 専用の初期化処理 ---
            print("Initializing dedicated CheXNet feature extractor.")
            # 1. ImageNetで事前学習済みのdensenet121をtorchvisionからロード
            densenet = torchvision.models.densenet121(pretrained=True)
            
            # 2. 入力層を1チャネル（グレースケール）に適応させる
            # 元の3チャネル用重みを平均化して、1チャネル用の重みとして利用
            original_conv0_weights = densenet.features.conv0.weight.data
            new_conv0_weights = original_conv0_weights.mean(dim=1, keepdim=True)
            densenet.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            densenet.features.conv0.weight.data = new_conv0_weights
            
            # 3. 分類層を除いた'features'部分をバックボーンとして使用
            self.backbone = densenet.features
            
            # 4. CheXNetの重みをロードしてImageNetの重みを上書き
            self.load_chexnet_weights('checkpoints/chexnet_weights.pth.tar')

            # 5. 出力チャンネル数をd_outに変換するための1x1畳み込み
            d_backbone_out = densenet.classifier.in_features # densenet121では1024
            self.projection = nn.Conv2d(d_backbone_out, d_out, 1)

        else:
            # --- 他モデル用の汎用初期化処理 (timmを使用) ---
            print(f"Initializing generic timm feature extractor for {model_name}.")
            use_pretrained = backbone_cfg.get('pretrained', True)
            self.backbone = timm.create_model(
                model_name,
                pretrained=use_pretrained,
                in_chans=1,          # 1チャンネル入力に対応
                features_only=True,  # 中間層の特徴マップを出力
            )
            d_backbone_out = self.backbone.feature_info.channels()[-1]
            self.projection = nn.Conv2d(d_backbone_out, d_out, 1)

    def load_chexnet_weights(self, path: str):
        """CheXNetのチェックポイントから'features'の重みをロードする"""
        try:
            print(f"Attempting to load custom CheXNet weights from {path}...")
            checkpoint = torch.load(path, map_location='cpu')
            state_dict = checkpoint['state_dict']
            
            # 'features'モジュールにロードするための新しいstate_dictを作成
            new_state_dict = {}
            for k, v in state_dict.items():
                # 'module.densenet121.features.'プレフィックスを削除
                if k.startswith('module.densenet121.features.'):
                    new_k = k.replace('module.densenet121.features.', '')
                    new_state_dict[new_k] = v
            
            # conv0の重みは1チャネル用に調整済みなので、チェックポイントからはロードしない
            new_state_dict.pop('conv0.weight', None)

            # 'features'モジュールに重みをロード
            incompatible_keys = self.backbone.load_state_dict(new_state_dict, strict=False)
            
            print("\n--- CheXNet Weight Loading Report ---")
            num_loaded = len(new_state_dict)
            num_total_features = len(self.backbone.state_dict())
            print(f"Matched and attempted to load {num_loaded} layers into the 'features' module.")
            if incompatible_keys.missing_keys:
                print(f"  - Layers in our model not found in checkpoint: {len(incompatible_keys.missing_keys)} keys. (e.g., {incompatible_keys.missing_keys[:5]})")
            if incompatible_keys.unexpected_keys:
                print(f"  - Keys in checkpoint not used by our model: {len(incompatible_keys.unexpected_keys)} keys. (e.g., {incompatible_keys.unexpected_keys[:5]})")
            
            if num_loaded > 0:
                print(f"Successfully loaded weights for {num_loaded}/{num_total_features} layers in the backbone.")
            else:
                print("ERROR: No CheXNet weights were loaded. Check prefixes and keys in the checkpoint file.")
            print("------------------------------------\n")

        except FileNotFoundError:
            print(f"WARNING: CheXNet weights file not found at '{path}'. Using ImageNet pretrained weights with averaged input layer instead.")
        except Exception as e:
            print(f"ERROR: Failed to load CheXNet weights: {e}. Using ImageNet weights.")

    def forward(self, x):
        # バックボーンから特徴マップを取得
        # densenet121の場合、self.backboneは'features'モジュールそのもの
        # timmの場合、features_only=Trueなので特徴マップのリストが返る
        features = self.backbone(x)
        if isinstance(features, list):
            # timmからの出力の場合、最後の特徴マップを使用
            features = features[-1]
        
        # チャンネル数をプロジェクト
        return self.projection(features)

# --- 3.3 NeRF MLP (★ビュー方向を考慮するように修正) ---
class NeRF_MLP(nn.Module):
    """(位置エンコード, 特徴量1, 特徴量2, ★ビュー方向エンコード) -> 密度σ"""
    def __init__(self, d_pos_emb: int, d_feature: int, d_view_emb: int, depth: int = 8, width: int = 256, skip_layer: int = 4, dropout_rate: float = 0.0):
        super().__init__()
        self.skip_layer = skip_layer
        # 位置と画像特徴量の入力次元
        d_in = d_pos_emb + d_feature * 2
        
        # 密度(σ)を計算する部分のネットワーク
        self.sigma_layers = nn.ModuleList()
        for i in range(depth):
            # 入力層とスキップ層の次元を調整
            layer_in_dim = width + d_in if i == skip_layer else (d_in if i == 0 else width)
            
            block = [nn.Linear(layer_in_dim, width), nn.ReLU(True)]
            if dropout_rate > 0:
                block.append(nn.Dropout(dropout_rate))
            self.sigma_layers.append(nn.Sequential(*block))
        
        # 密度(σ)を出力する部分のネットワーク
        # ビュー方向の情報を結合して最終的な密度を計算
        self.feature_layer = nn.Linear(width, width) # 中間特徴ベクトル
        self.output_layer = nn.Linear(width + d_view_emb, 1) # 中間特徴量とビュー方向を結合
        # 学習初期のsigmaが大きすぎないようにバイアスを初期化し、学習を安定させる
        torch.nn.init.constant_(self.output_layer.bias, -10.0)
        # 学習初期のsigmaが大きすぎないようにバイアスを初期化し、学習を安定させる
        torch.nn.init.constant_(self.output_layer.bias, -10.0)

    def forward(self, pos_emb, feature_0, feature_90, view_emb):
        # 入力を結合
        x_input = torch.cat([pos_emb, feature_0, feature_90], dim=-1)
        x = x_input
        
        # 密度計算ネットワーク
        for i, block in enumerate(self.sigma_layers):
            if i == self.skip_layer:
                x = torch.cat([x, x_input], dim=-1)
            x = block(x)
        
        # 出力ネットワーク
        feature = self.feature_layer(x)
        # 中間特徴量とビュー方向エンコーディングを結合
        combined = torch.cat([feature, view_emb], dim=-1)
        sigma = self.output_layer(combined)
        
        # 密度は非負であるためsoftplusを適用
        return F.softplus(sigma)

# --- 3.4 PixelNeRFモデル本体 (★ビュー方向を考慮するように修正) ---
class PixelNeRFModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.n_samples_coarse = cfg['model']['n_samples_coarse']
        self.n_samples_fine = cfg['model']['n_samples_fine']
        self.use_fine_model = cfg['model']['use_fine_model']
        self.near_plane = cfg['model'].get('near_plane', 0.1)
        self.far_plane = cfg['model'].get('far_plane', 2000.0)
        self.grid_sample_align_corners = cfg['geometry']['grid_sample_align_corners']

        # コンポーネント初期化
        # 位置エンコーダ
        self.pos_encoder = PositionalEncoder(d_input=3, n_freqs=cfg['model']['n_pos_freqs'])
        # ★ビュー方向エンコーダを追加 (configになければデフォルトで4)
        self.view_encoder = PositionalEncoder(d_input=3, n_freqs=cfg['model'].get('n_view_freqs', 4))
        
        # 画像エンコーダ
        self.image_encoder = ImageEncoder(
            backbone_cfg=cfg['model']['backbone'], 
            d_out=cfg['model']['d_feature']
        )
        
        # NeRF MLP (Coarse)
        self.nerf_mlp_coarse = NeRF_MLP(
            d_pos_emb=self.pos_encoder.d_output,
            d_feature=cfg['model']['d_feature'],
            d_view_emb=self.view_encoder.d_output, # ★ビュー方向エンコーダの出力次元
            width=cfg['model']['d_mlp_width'],
            dropout_rate=cfg['model'].get('dropout_rate', 0.0)
        )
        # NeRF MLP (Fine)
        if self.use_fine_model:
             self.nerf_mlp_fine = NeRF_MLP(
                 d_pos_emb=self.pos_encoder.d_output,
                 d_feature=cfg['model']['d_feature'],
                 d_view_emb=self.view_encoder.d_output, # ★ビュー方向エンコーダの出力次元
                 width=cfg['model']['d_mlp_width'],
                 dropout_rate=cfg['model'].get('dropout_rate', 0.0)
             )

    def forward(self, img_0, img_90, params_0, params_90, target_rays_o, target_rays_d):
        B, N_rays, _ = target_rays_o.shape
        F_0 = self.image_encoder(img_0)
        F_90 = self.image_encoder(img_90)

        # Coarseサンプリングと評価
        points_coarse, z_vals_coarse = self.sample_points_stratified(
            target_rays_o, target_rays_d, self.n_samples_coarse,
            near=self.near_plane, far=self.far_plane
        )
        # ★evaluate_mlpに target_rays_d を渡す
        sigma_coarse = self.evaluate_mlp(
            self.nerf_mlp_coarse, points_coarse, F_0, F_90, params_0, params_90, 
            H=img_0.shape[2], W=img_0.shape[3], rays_d=target_rays_d
        )
        pixel_values_coarse, weights_coarse = self.volume_render(sigma_coarse, z_vals_coarse)

        outputs = {
            "coarse": pixel_values_coarse,
            "weights_coarse": weights_coarse,
            "z_vals_coarse": z_vals_coarse,
        }

        if not self.use_fine_model:
            return outputs

        # Fineサンプリングと評価
        points_fine, z_vals_fine = self.sample_points_hierarchical(
             target_rays_o, target_rays_d, z_vals_coarse, weights_coarse, self.n_samples_fine
        )
        points_all = torch.cat([points_coarse, points_fine], dim=-2)
        z_vals_all = torch.cat([z_vals_coarse, z_vals_fine], dim=-1)
        z_vals_sorted, sort_indices = torch.sort(z_vals_all, dim=-1)
        points_sorted = torch.gather(points_all, -2, sort_indices.unsqueeze(-1).expand(-1, -1, -1, 3))

        # ★evaluate_mlpに target_rays_d を渡す
        sigma_fine = self.evaluate_mlp(
            self.nerf_mlp_fine, points_sorted, F_0, F_90, params_0, params_90, 
            H=img_0.shape[2], W=img_0.shape[3], rays_d=target_rays_d
        )
        pixel_values_fine, weights_fine = self.volume_render(sigma_fine, z_vals_sorted)

        outputs["fine"] = pixel_values_fine
        outputs["weights_fine"] = weights_fine
        outputs["z_vals_fine"] = z_vals_sorted
        outputs["F_0"] = F_0
        outputs["F_90"] = F_90
        outputs["sigma_coarse"] = sigma_coarse
        outputs["sigma_fine"] = sigma_fine
        
        return outputs

    # ★evaluate_mlpを修正
    def evaluate_mlp(self, mlp, points_3d, F_0, F_90, params_0, params_90, H, W, rays_d):
        B, N_rays, N_samples, _ = points_3d.shape
        points_flat = points_3d.reshape(-1, 3)

        # --- 特徴量サンプリング ---
        K_0_raw, K_90_raw = params_0['intrinsics'], params_90['intrinsics']
        w2c_0_raw, w2c_90_raw = params_0['extrinsics'], params_90['extrinsics']

        K_0, K_90 = K_0_raw.float(), K_90_raw.float()
        while K_0.dim() > 3: K_0 = K_0.squeeze(1)
        while K_90.dim() > 3: K_90 = K_90.squeeze(1)
        
        w2c_0, w2c_90 = w2c_0_raw.float(), w2c_90_raw.float()
        while w2c_0.dim() > 3: w2c_0 = w2c_0.squeeze(1)
        while w2c_90.dim() > 3: w2c_90 = w2c_90.squeeze(1)

        K_0, K_90 = K_0[:, :3, :3], K_90[:, :3, :3]

        coords_0_flat = self.project_points(points_flat, w2c_0, K_0, H, W, B, N_rays, N_samples)
        coords_90_flat = self.project_points(points_flat, w2c_90, K_90, H, W, B, N_rays, N_samples)

        sampled_F_0_flat = self.sample_features(F_0, coords_0_flat, B, N_rays, N_samples)
        sampled_F_90_flat = self.sample_features(F_90, coords_90_flat, B, N_rays, N_samples)

        # --- ★ビュー方向のエンコーディング ---
        # rays_dをサンプル点数分に拡張 -> (B, N_rays, N_samples, 3)
        rays_d_expanded = rays_d.unsqueeze(-2).expand(-1, -1, N_samples, -1)
        rays_d_flat = rays_d_expanded.reshape(-1, 3)
        view_emb_flat = self.view_encoder(rays_d_flat)

        # --- MLP評価 ---
        pos_emb_flat = self.pos_encoder(points_flat)
        # ★MLPにビュー方向エンコーディングを渡す
        sigma_flat = mlp(pos_emb_flat, sampled_F_0_flat, sampled_F_90_flat, view_emb_flat)
        
        return sigma_flat.view(B, N_rays, N_samples, 1)


    def sample_points_stratified(self, rays_o, rays_d, n_samples, near, far, perturb=True):
        B, N_rays, _ = rays_o.shape
        t_vals = torch.linspace(0., 1., steps=n_samples, device=rays_o.device)
        z_vals = near * (1.-t_vals) + far * (t_vals)
        z_vals = z_vals.expand([B, N_rays, n_samples])
        if perturb:
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            t_rand = torch.rand(z_vals.shape, device=rays_o.device)
            z_vals = lower + (upper - lower) * t_rand
        points = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]
        return points, z_vals

    def sample_points_hierarchical(self, rays_o, rays_d, z_vals_coarse, weights_coarse, n_samples_fine):
         z_vals_mid = .5 * (z_vals_coarse[..., 1:] + z_vals_coarse[..., :-1])
         weights = weights_coarse[..., 1:-1] + 1e-5
         pdf = weights / torch.sum(weights, -1, keepdim=True)
         cdf = torch.cumsum(pdf, -1)
         cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)
         u = torch.rand(list(cdf.shape[:-1]) + [n_samples_fine], device=rays_o.device)
         cdf = cdf.contiguous()
         inds = torch.searchsorted(cdf, u, right=True)
         below = torch.max(torch.zeros_like(inds-1), inds-1)
         above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
         inds_g = torch.stack([below, above], -1) # (B, N_rays, N_samples_fine, 2)

         # gatherのためにcdfとz_vals_midをinds_gの形状にブロードキャスト可能にする
         B, N_rays, N_samples_fine, _ = inds_g.shape
         cdf_expanded = cdf.unsqueeze(2).expand(B, N_rays, N_samples_fine, -1)
         z_vals_mid_expanded = z_vals_mid.unsqueeze(2).expand(B, N_rays, N_samples_fine, -1)

         cdf_g = torch.gather(cdf_expanded, 3, inds_g)
         bins_g = torch.gather(z_vals_mid_expanded, 3, inds_g)

         denom = (cdf_g[..., 1]-cdf_g[..., 0])
         denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
         t = (u-cdf_g[..., 0])/denom
         samples = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])
         z_vals_fine = samples
         points = rays_o[..., None, :] + rays_d[..., None, :] * z_vals_fine[..., :, None]
         return points, z_vals_fine

    def volume_render(self, sigma, z_vals):
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)
        alpha = sigma.squeeze(-1) * dists
        
        # Fine Sampling用のWeightも計算
        transmittance = torch.exp(-torch.cumsum(torch.cat([torch.zeros_like(alpha[..., :1]), alpha], -1), -1)[..., :-1])
        weights = transmittance * (1. - torch.exp(-alpha))

        # ★★★ レンダリング方式を変更 ★★★
        # alphaの直接合計は数値的に不安定になる可能性があるため、
        # [0, 1]に正規化されているweightsの合計値をピクセル値として使用する。
        # これは、レイが物体に遮蔽される確率の合計に相当し、シルエットのような画像を生成する。
        pixel_val = torch.sum(weights, dim=-1) 

        return pixel_val.unsqueeze(-1), weights

    def project_points(self, points_world_flat, w2c, K, H, W, B, N_rays, N_samples):
        """
        ワールド座標系3D点 -> 画像平面正規化座標[-1, 1]
        """
        device = points_world_flat.device
        num_points = points_world_flat.shape[0]

        points_world = points_world_flat.view(B, N_rays * N_samples, 3)
        points_world_h = F.pad(points_world, (0, 1), value=1.0)

        # bmmを使うために転置
        points_cam_h = torch.bmm(w2c, points_world_h.transpose(1, 2)).transpose(1, 2)
        points_cam = points_cam_h[..., :3] / (points_cam_h[..., 3:].clamp(min=1e-8)) # ゼロ除算防止

        points_img_h = torch.bmm(K, points_cam.transpose(1, 2)).transpose(1, 2)
        points_pixel = points_img_h[..., :2] / (points_img_h[..., 2:].clamp(min=1e-8))
        u = points_pixel[..., 0]
        v = points_pixel[..., 1]

        x_norm = (u / (W - 1)) * 2 - 1
        y_norm = (v / (H - 1)) * 2 - 1

        coords_norm_flat = torch.stack([x_norm, y_norm], dim=-1).view(-1, 2)

        return coords_norm_flat

    def sample_features(self, F_map, coords_flat, B, N_rays, N_samples):
        """
        特徴マップ F_map から、正規化座標 coords_flat に対応する特徴ベクトルを抽出。
        """
        D_feat = F_map.shape[1]
        coords_grid = coords_flat.view(B, N_rays * N_samples, 1, 2)

        sampled = F.grid_sample(
            F_map,
            coords_grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=self.grid_sample_align_corners
        ) # (B, D_feat, N_rays * N_samples, 1)

        return sampled.permute(0, 2, 3, 1).reshape(-1, D_feat)

# --- ★★★ 修正: get_rays 関数の形状処理を強化 ★★★ ---
def get_rays(H: int, W: int, K: torch.Tensor, c2w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    ピクセル座標からワールド座標系のレイ原点と方向を計算。
    H, W: 画像サイズ
    K: (B, ..., 3, 3) カメラ内部パラメータ (余分な次元を含む可能性あり)
    c2w: (B, ..., 4, 4) カメラ→ワールド変換行列 (余分な次元を含む可能性あり)
    戻り値: rays_o (B, H*W, 3), rays_d (B, H*W, 3)
    """
    device = K.device
    B = K.shape[0] # Assume the first dimension is batch

    i, j = torch.meshgrid(
        torch.linspace(0.5, W - 0.5, W, device=device),
        torch.linspace(0.5, H - 0.5, H, device=device),
        indexing='xy'
    )
    i = i.reshape(-1)
    j = j.reshape(-1)
    pixel_coords = torch.stack([i, j, torch.ones_like(i)], dim=-1) # (H*W, 3)

    # --- K_inv の形状を (B, 3, 3) に整形 ---
    K_proc = K.float()
    while K_proc.dim() > 3: # 次元数が3より大きい間、 squeeze(1) を繰り返す
        if K_proc.shape[1] == 1:
            K_proc = K_proc.squeeze(1)
        else:
            # 予期しない形状 (例: [B, 2, 3, 3]) の場合エラー
            raise ValueError(f"Cannot squeeze K to 3D tensor: current shape {K_proc.shape}")
    if K_proc.dim() != 3 or K_proc.shape[1:] != (3, 3):
        raise ValueError(f"Failed to reshape K to (B, 3, 3): final shape {K_proc.shape}")
    
    try:
        K_inv = torch.inverse(K_proc)
    except Exception as e:
         print(f"警告(get_rays): Kの逆行列計算に失敗 (Shape: {K_proc.shape}): {e}")
         rays_o = torch.zeros((B, H * W, 3), device=device)
         rays_d = torch.zeros((B, H * W, 3), device=device)
         rays_d[..., 2] = 1.0
         return rays_o, rays_d

    pixel_coords_batched = pixel_coords.t().unsqueeze(0).expand(B, -1, -1) # (B, 3, H*W)

    try:
        cam_coords = torch.bmm(K_inv, pixel_coords_batched)
    except RuntimeError as e:
        print(f"[get_rays DEBUG] torch.bmm (cam_coords) failed!")
        print(f"  K_inv shape: {K_inv.shape}")
        print(f"  pixel_coords_batched shape: {pixel_coords_batched.shape}")
        raise e

    directions_cam = cam_coords.transpose(1, 2) # (B, H*W, 3)

    # --- c2w の形状を (B, 4, 4) に整形 ---
    c2w_proc = c2w.float().reshape(B, 4, 4)

    rotation_c2w = c2w_proc[:, :3, :3] # (B, 3, 3)

    try:
        directions_world_transposed = torch.bmm(rotation_c2w, directions_cam.transpose(1, 2))
        directions_world = directions_world_transposed.transpose(1, 2)
    except RuntimeError as e:
        print(f"[get_rays DEBUG] torch.bmm (world directions) failed!")
        print(f"  rotation_c2w shape: {rotation_c2w.shape}")
        print(f"  directions_cam transposed shape: {directions_cam.transpose(1, 2).shape}")
        raise e

    rays_d = F.normalize(-directions_world, dim=-1) # ★★★ 視線方向を反転させるための修正 ★★★

    # レイの原点
    rays_o = c2w_proc[:, None, :3, 3].expand(-1, H * W, -1) # (B, H*W, 3)

    return rays_o, rays_d

