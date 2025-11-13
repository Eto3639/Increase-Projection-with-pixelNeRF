import torch
import yaml
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from pathlib import Path

from dataset import DRRDataset
from model import ImageEncoder, get_rays, PixelNeRFModel

# --- Helper Functions ---

def load_config(path="config.yml"):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

def set_axes_equal(ax):
    """3Dプロットで各軸のスケールを揃える"""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    plot_radius = 0.5 * max([x_range, y_range, z_range])
    mid_x = 0.5 * (x_limits[1] + x_limits[0])
    mid_y = 0.5 * (y_limits[1] + y_limits[0])
    mid_z = 0.5 * (z_limits[1] + z_limits[0])
    ax.set_xlim3d([mid_x - plot_radius, mid_x + plot_radius])
    ax.set_ylim3d([mid_y - plot_radius, mid_y + plot_radius])
    ax.set_zlim3d([mid_z - plot_radius, mid_z + plot_radius])

def visualize_cameras_and_rays(c2w_0, c2w_90, c2w_target, rays_o, rays_d, sdd, save_path):
    """カメラの姿勢、検出器、レイを3Dで可視化する"""
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    # 原点と被写体の目安となる人型（カプセル）をプロット
    ax.scatter(0, 0, 0, color='k', marker='o', s=50, label='Origin (Subject Center)')
    # --- Plot a capsule to represent the patient phantom ---
    radius = 150 
    cylinder_height = 100 # The cylindrical part of the torso
    
    # Cylinder
    u_cyl = np.linspace(0, 2 * np.pi, 50)
    y_cyl = np.linspace(-cylinder_height / 2, cylinder_height / 2, 2)
    u_cyl, y_cyl = np.meshgrid(u_cyl, y_cyl)
    x_cyl = radius * np.cos(u_cyl)
    z_cyl = radius * np.sin(u_cyl)
    ax.plot_surface(x_cyl, y_cyl, z_cyl, color='gray', alpha=0.1, linewidth=0)

    # Hemispheres (caps)
    u_hemi = np.linspace(0, 2 * np.pi, 50)
    v_hemi = np.linspace(0, np.pi / 2, 25) # Quarter circle
    u_hemi, v_hemi = np.meshgrid(u_hemi, v_hemi)

    # Top cap
    x_top = radius * np.cos(u_hemi) * np.sin(v_hemi)
    z_top = radius * np.sin(u_hemi) * np.sin(v_hemi)
    y_top = radius * np.cos(v_hemi) + cylinder_height / 2
    ax.plot_surface(x_top, y_top, z_top, color='gray', alpha=0.1, linewidth=0)

    # Bottom cap
    x_bot = radius * np.cos(u_hemi) * np.sin(v_hemi)
    z_bot = radius * np.sin(u_hemi) * np.sin(v_hemi)
    y_bot = -radius * np.cos(v_hemi) - cylinder_height / 2
    ax.plot_surface(x_bot, y_bot, z_bot, color='gray', alpha=0.1, linewidth=0)


    # カメラの位置と向き、検出器を描画
    arrow_length = 300.0
    for c2w, color, label in [(c2w_0, 'r', 'Input Cam 0°'), (c2w_90, 'g', 'Input Cam 90°'), (c2w_target, 'b', 'Target Cam')]:
        # カメラ（線源）の位置
        source_pos = c2w[:3, 3]
        # カメラの向き（視線方向）
        view_dir = -c2w[:3, 2]
        ax.quiver(source_pos[0], source_pos[1], source_pos[2], view_dir[0], view_dir[1], view_dir[2], length=arrow_length, color=color, label=label)

        # 検出器の位置と平面を描画
        detector_center = source_pos + view_dir * sdd
        # 検出器平面の基底ベクトル（カメラのX, Y軸）
        x_axis = c2w[:3, 0]
        y_axis = c2w[:3, 1]
        corner_scale = 400 # 検出器のサイズ
        p1 = detector_center - x_axis * corner_scale + y_axis * corner_scale
        p2 = detector_center + x_axis * corner_scale + y_axis * corner_scale
        p3 = detector_center + x_axis * corner_scale - y_axis * corner_scale
        p4 = detector_center - x_axis * corner_scale - y_axis * corner_scale
        verts = [list(zip([p1[0], p2[0], p3[0], p4[0]], [p1[1], p2[1], p3[1], p4[1]], [p1[2], p2[2], p3[2], p4[2]]))]
        ax.add_collection3d(Poly3DCollection(verts, facecolors=color, alpha=0.1))


    # レイを描画
    num_rays_to_show = 5
    step = rays_o.shape[0] // num_rays_to_show
    for i in range(0, rays_o.shape[0], step):
        ro, rd = rays_o[i], rays_d[i]
        ax.plot([ro[0], ro[0] + rd[0] * sdd * 1.2], [ro[1], ro[1] + rd[1] * sdd * 1.2], [ro[2], ro[2] + rd[2] * sdd * 1.2], 'c-', alpha=0.3)

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend()
    ax.set_title('Camera, Detector, and Ray Visualization')
    set_axes_equal(ax)
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved camera visualization to {save_path}")

def main():
    print("--- Starting Visual Debug Script ---")
    output_dir = Path("debug_outputs")
    output_dir.mkdir(exist_ok=True)

    # --- 1. データと設定の読み込み ---
    print("\n[Step 1/5] Loading data and config...")
    cfg = load_config()
    # 1症例のみを読み込むようにデータセットを初期化
    root_dir = Path(cfg['dataset']['root_dir'])
    all_patient_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir()])
    dataset = DRRDataset(cfg=cfg, split='train', patient_dirs=all_patient_dirs[:1])
    if len(dataset) == 0:
        print("ERROR: Dataset is empty. Check 'drr_dataset' directory.")
        return
    sample = dataset[0]
    print("Successfully loaded 1 data sample.")

    # --- 2. 入力画像の可視化 ---
    print("\n[Step 2/5] Visualizing input data...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(sample['img_0'].squeeze(), cmap='gray'); axes[0].set_title('Input Image 0°')
    axes[1].imshow(sample['img_90'].squeeze(), cmap='gray'); axes[1].set_title('Input Image 90°')
    axes[2].imshow(sample['target_img'].squeeze(), cmap='gray'); axes[2].set_title(f"Target Image (Angle: {sample['info']['target_angle']}°)")
    save_path = output_dir / "debug_01_inputs.png"
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved input visualization to {save_path}")

    # --- 3. カメラ姿勢とレイの可視化 ---
    print("\n[Step 3/5] Visualizing camera poses and rays...")
    H, W = cfg['dataset']['img_size']
    # c2w（カメラ→ワールド）行列を取得
    c2w_0 = torch.inverse(sample['params_0']['extrinsics']).squeeze()
    c2w_90 = torch.inverse(sample['params_90']['extrinsics']).squeeze()
    c2w_target = torch.inverse(sample['target_params']['extrinsics']).squeeze()
    # ターゲットビューのレイを計算
    rays_o, rays_d = get_rays(H, W, sample['target_params']['intrinsics'].unsqueeze(0), c2w_target.unsqueeze(0))
    rays_o, rays_d = rays_o.squeeze(), rays_d.squeeze()
    visualize_cameras_and_rays(c2w_0, c2w_90, c2w_target, rays_o, rays_d, cfg['drr']['sdd'], output_dir / "debug_02_cameras.png")

    # --- 4. 特徴マップの可視化 ---
    print("\n[Step 4/5] Visualizing feature maps...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = ImageEncoder(cfg['model']['backbone'], cfg['model']['d_feature']).to(device)
    encoder.eval()
    img_0_tensor = sample['img_0'].unsqueeze(0).to(device)
    img_90_tensor = sample['img_90'].unsqueeze(0).to(device)
    with torch.no_grad():
        F_0 = encoder(img_0_tensor)
        F_90 = encoder(img_90_tensor)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(F_0.mean(dim=1).squeeze().cpu().numpy(), cmap='viridis'); axes[0].set_title('Feature Map from Image 0°')
    axes[1].imshow(F_90.mean(dim=1).squeeze().cpu().numpy(), cmap='viridis'); axes[1].set_title('Feature Map from Image 90°')
    save_path = output_dir / "debug_03_features.png"
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved feature map visualization to {save_path}")

    # --- 5. 3D点→2D投影の可視化 ---
    print("\n[Step 5/5] Visualizing 3D point projections...")
    model = PixelNeRFModel(cfg).to(device) # model全体を初期化して内部関数を利用
    model.eval()
    with torch.no_grad():
        # ターゲットビューの中心レイに沿って3D点をサンプリング
        center_ray_idx = H // 2 * W + W // 2
        rays_o_center, rays_d_center = rays_o[center_ray_idx:center_ray_idx+1], rays_d[center_ray_idx:center_ray_idx+1]
        points_3d, _ = model.sample_points_stratified(
            rays_o_center.unsqueeze(0).to(device), 
            rays_d_center.unsqueeze(0).to(device), 
            64, 0.1, 2000.0
        )
        points_3d_flat = points_3d.reshape(-1, 3)

        # サンプリングした3D点を、入力画像0度と90度のカメラから見た2D座標に逆投影
        coords_0_flat = model.project_points(points_3d_flat, sample['params_0']['extrinsics'].to(device), sample['params_0']['intrinsics'].to(device), H, W, 1, 1, 64)
        coords_90_flat = model.project_points(points_3d_flat, sample['params_90']['extrinsics'].to(device), sample['params_90']['intrinsics'].to(device), H, W, 1, 1, 64)
        
        # 2D座標をピクセル座標に変換
        coords_0_pixel = (coords_0_flat.cpu().numpy() + 1) / 2 * np.array([W, H])
        coords_90_pixel = (coords_90_flat.cpu().numpy() + 1) / 2 * np.array([W, H])

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(sample['img_0'].squeeze(), cmap='gray')
    axes[0].scatter(coords_0_pixel[:, 0], coords_0_pixel[:, 1], c=np.arange(64), cmap='cool', s=5)
    axes[0].set_title('Projected 3D points on Image 0°')
    axes[0].set_xlim(0, W); axes[0].set_ylim(H, 0)

    axes[1].imshow(sample['img_90'].squeeze(), cmap='gray')
    axes[1].scatter(coords_90_pixel[:, 0], coords_90_pixel[:, 1], c=np.arange(64), cmap='cool', s=5)
    axes[1].set_title('Projected 3D points on Image 90°')
    axes[1].set_xlim(0, W); axes[1].set_ylim(H, 0)
    
    save_path = output_dir / "debug_04_projections.png"
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved projection visualization to {save_path}")

    print("\n--- Visual Debug Script Finished ---")
    print(f"Please check the images saved in the '{output_dir.resolve()}' directory.")

if __name__ == '__main__':
    main()
