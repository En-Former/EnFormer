import os
import argparse

import timm
import torch

import cv2
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as TransF
from torchvision import transforms
from timm.models import load_checkpoint
from torchvision.utils import draw_segmentation_masks
from torchvision.io import read_image

# --- Import custom model files ---
import models.ensemble_clustering_for_vis
import models.base_clustering_for_vis
import models.utils
from models.base_clustering_for_vis import PartitionalClustering

# --- Global Hook Storage ---
similarity_maps = []
assignment_maps = []
feature_map_shape = []


def _preprocess(image_path, shape=224):
    """Preprocesses the image."""
    raw_image = cv2.imread(image_path)
    raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)  # to RGB
    image = cv2.resize(raw_image, (shape, shape))
    image_tensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(image)

    img_for_viz = read_image(image_path)
    raw_image_orig = cv2.imread(image_path)
    return image_tensor, img_for_viz, raw_image_orig


# --- Hook Functions ---

def get_assignment_hook(module, input, output):
    """Hook to capture both assignment and similarity maps."""
    aggregation, (assignment_map, similarity_map) = output
    assignment_maps.append(assignment_map.detach().cpu())
    similarity_maps.append(similarity_map.detach().cpu())


def get_input_shape_hook(module, input, output):
    """Hook to capture the input feature map shape to EnsembleClustering."""
    feature_map_shape.append(input[0].shape)


# --- Visualization Helpers ---

def visualize_hard_clusters(img_tensor, mask_bool, save_path, alpha=0.5):
    """Saves a segmentation mask for hard cluster assignments."""
    colors = ["brown", "green", "deepskyblue", "blue", "darkgreen", "darkcyan", "coral", "aliceblue",
              "white", "black", "beige", "red", "tomato", "yellowgreen", "violet", "mediumseagreen"]
    if mask_bool.shape[0] > len(colors):
        colors = colors * (mask_bool.shape[0] // 16 + 1)
    colors = colors[:mask_bool.shape[0]]

    img_with_masks = draw_segmentation_masks(img_tensor, masks=mask_bool, alpha=alpha, colors=colors)
    img_with_masks_pil = TransF.to_pil_image(img_with_masks)
    img_with_masks_cv2 = cv2.cvtColor(np.array(img_with_masks_pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, img_with_masks_cv2)
    print(f"Saved hard cluster visualization to: {save_path}")


def visualize_soft_cluster(raw_image, heatmap_tensor, save_path, center_type='peak', alpha=0.5, draw_center=True):
    """Saves a heatmap overlay and optionally marks its center (peak or centroid)."""
    # raw_image: [H, W, C] (numpy, BGR)
    # heatmap_tensor: [H_img, W_img] (torch.Tensor)

    # --- 1. Calculate Center (only if requested) ---
    if draw_center:
        h, w = heatmap_tensor.shape
        if center_type == 'peak':
            # Calculate Peak (Mode)
            max_flat_index = heatmap_tensor.argmax()
            cy, cx = max_flat_index // w, max_flat_index % w
        else:  # 'centroid'
            # Calculate Centroid (Weighted Average)
            y_coords, x_coords = torch.meshgrid(torch.arange(h, device=heatmap_tensor.device),
                                                torch.arange(w, device=heatmap_tensor.device),
                                                indexing='ij')
            total_weight = heatmap_tensor.sum() + 1e-6
            cy = (y_coords * heatmap_tensor).sum() / total_weight
            cx = (x_coords * heatmap_tensor).sum() / total_weight

        cy = int(cy.item())
        cx = int(cx.item())

    # --- 2. Draw Heatmap (unchanged) ---
    heatmap_np = heatmap_tensor.numpy()
    heatmap = (heatmap_np - np.min(heatmap_np)) / (np.max(heatmap_np) - np.min(heatmap_np) + 1e-6)
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.resize(heatmap_color, (raw_image.shape[1], raw_image.shape[0]))
    overlay = cv2.addWeighted(raw_image, 1 - alpha, heatmap_color, alpha, 0)

    # --- 3. Draw Center Marker (if requested) ---
    if draw_center:
        cv2.drawMarker(overlay, (cx, cy),
                       color=(0, 255, 0),  # Green
                       markerType=cv2.MARKER_TILTED_CROSS,
                       markerSize=20,
                       thickness=2)

    cv2.imwrite(save_path, overlay)


# --- Main Function ---

def main():
    parser = argparse.ArgumentParser(description='Ensemble Cluster Visualization')
    parser.add_argument('--image', type=str, required=True, help='Path to image')
    parser.add_argument('--shape', type=int, default=224, help='Image size for model input')
    parser.add_argument('--model', default='enformer_small', type=str,
                        help='Name of model (must be registered in timm)')
    parser.add_argument('--labels_file', type=str, default="./imagenet1k_id_to_label.txt",
                        help='Path to imagenet1k_id_to_label.txt')
    parser.add_argument('--stage', default=0, type=int, help='Index of visualized stage (0-3)')
    parser.add_argument('--block', default=-1, type=int,
                        help='Index of visualized block. -1 means the last block of the stage.')
    parser.add_argument('--head', default=0, type=int, help='Index of visualized head')
    parser.add_argument('--checkpoint', type=str, default="", metavar='PATH',
                        help='Path to pretrained checkpoint (default: none)')
    parser.add_argument('--alpha', default=0.5, type=float, help='Transparency for overlay (0-1)')
    parser.add_argument('--map_type', type=str, default='assignment', choices=['assignment', 'similarity'],
                        help='Which map to visualize for soft clustering (default: assignment)')
    parser.add_argument('--part_method', type=str, default='forward', choices=['forward', 'forward_fast'],
                        help='Which method to use for PartitionalClustering visualization (default: forward)')
    parser.add_argument('--center_type', type=str, default='peak', choices=['peak', 'centroid'],
                        help='How to mark the soft cluster center (default: peak)')
    parser.add_argument('--interpolation', type=str, default='bilinear', choices=['bilinear', 'nearest'],
                        help='Interpolation mode for upsampling heatmaps (default: bilinear)')
    parser.add_argument('--draw_center', action='store_true', default=True,
                        help='Draw the center marker (X) on soft cluster maps (default: True)')
    parser.add_argument('--no_draw_center', dest='draw_center', action='store_false',
                        help='Do not draw the center marker (X) on soft cluster maps')

    parser.add_argument('--output_dir', default="images/ensemble_vis", type=str,
                        help='Directory to save visualizations')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load ImageNet Categories
    object_categories = []
    try:
        with open(args.labels_file, "r") as f:
            for line in f:
                _, val = line.strip().split(":")
                object_categories.append(val)
    except FileNotFoundError:
        print(f"Warning: Labels file not found at {args.labels_file}. Predictions will be indices only.")
        object_categories = None

    # 1. Load Model
    model = timm.create_model(args.model, pretrained=False)
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, strict=False)
        print(f"==> Loaded checkpoint from {args.checkpoint}")
    else:
        print("==> Warning: No checkpoint loaded. Visualizing with random weights.")
    model.to(device)
    model.eval()

    # 2. Preprocess Image
    image_tensor, img_for_viz, raw_image_orig = _preprocess(args.image, args.shape)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    # 3. Get Target Module and Register Hooks
    if args.block == -1:
        num_blocks = len(model.stages[args.stage].blocks)
        args.block = num_blocks - 1
        print(f"Visualizing last block of stage {args.stage}, index: {args.block}")

    try:
        target_block = model.stages[args.stage].blocks[args.block]
        ensemble_module = target_block.cluster
    except Exception as e:
        print(f"Error accessing stage {args.stage}, block {args.block}: {e}")
        print("Please ensure stage and block indices are valid for your model.")
        return

    hook_handles = []
    module_names = []

    shape_hook_handle = ensemble_module.register_forward_hook(get_input_shape_hook)
    hook_handles.append(shape_hook_handle)

    for i in range(ensemble_module.num_clustering_modules):
        cluster_sub_module = getattr(ensemble_module, f'cluster_module_{i}')
        handle = cluster_sub_module.register_forward_hook(get_assignment_hook)
        hook_handles.append(handle)
        module_name = cluster_sub_module.__class__.__name__
        module_names.append(module_name)
        print(f"Registered hook for: {module_name}")

    # 4. Run Forward Pass(es)
    out = None
    if args.part_method == 'forward_fast':
        # --- TWO-PASS LOGIC ---
        print("\n" + "=" * 30)
        print("==> Running FIRST pass (original model) for prediction...")
        with torch.no_grad():
            out = model(image_tensor)
        print("=" * 30 + "\n")

        global similarity_maps, assignment_maps, feature_map_shape
        similarity_maps.clear()
        assignment_maps.clear()
        feature_map_shape.clear()

        print("\n" + "=" * 30)
        print(f"==> Switching PartitionalClustering to use .forward_fast for visualization...")
        count = 0
        for module in model.modules():
            if isinstance(module, PartitionalClustering):
                module.forward = module.forward_fast
                count += 1
        print(f"==> Patched {count} PartitionalClustering modules.")
        print("=" * 30 + "\n")

        print("\n" + "=" * 30)
        print("==> Running SECOND pass (patched model) for visualization...")
        with torch.no_grad():
            _ = model(image_tensor)
        print("=" * 30 + "\n")

    else:
        # --- ONE-PASS LOGIC (default) ---
        print("\n" + "=" * 30)
        print("==> Running ONE pass (original model) for prediction and visualization...")
        with torch.no_grad():
            out = model(image_tensor)
        print("=" * 30 + "\n")

    # 5. Remove Hooks
    for handle in hook_handles:
        handle.remove()

    # 6. Print Prediction Output
    print("\n" + "=" * 30)
    print("==> Prediction from 'forward' pass:")
    if type(out) is tuple:
        out = out[0]
    possibility = torch.softmax(out, dim=1).max()
    value, index = torch.max(out, dim=1)
    if object_categories:
        print(f'==> Prediction: {object_categories[index.item()]}')
    else:
        print(f'==> Prediction Index: {index.item()}')
    print(f'==> Probability: {possibility.item() * 100:.3f}%')
    print("=" * 30 + "\n")

    # 7. Process and Visualize
    B, C, H_feat, W_feat = feature_map_shape[0]
    if ensemble_module.use_agents:
        h_map, w_map = ensemble_module.agents_hw
    else:
        h_map, w_map = H_feat, W_feat

    print(f"Feature map size (N): {h_map}x{w_map}")

    image_name = os.path.basename(args.image).split(".")[0]
    base_save_dir = os.path.join(args.output_dir, args.model, image_name)
    os.makedirs(base_save_dir, exist_ok=True)

    for i, assign_map in enumerate(assignment_maps):
        module_name = module_names[i]
        sim_map = similarity_maps[i]

        if 'Partitional' in module_name:
            map_to_visualize = assign_map
            is_hard = True
            map_type_suffix = f"MASK_{args.part_method.upper()}"
        else:
            is_hard = False
            if args.map_type == 'assignment':
                map_to_visualize = assign_map
                map_type_suffix = "ASSIGNMENT"
            else:
                map_to_visualize = sim_map
                map_type_suffix = "SIMILARITY"

        num_clusters = getattr(ensemble_module, f'num_clusters_{i}')
        cluster_pool_layer = getattr(ensemble_module, f'clusters_{i}')
        cluster_grid_size = cluster_pool_layer.output_size
        if isinstance(cluster_grid_size, int):
            grid_h, grid_w = cluster_grid_size, cluster_grid_size
        else:
            grid_h, grid_w = cluster_grid_size

        if args.head >= map_to_visualize.shape[0]:
            print(
                f"Warning: Head {args.head} out of range for {module_name} (max {map_to_visualize.shape[0] - 1}). Skipping.")
            continue

        head_map = map_to_visualize[args.head]
        head_map = head_map.view(num_clusters, h_map, w_map)

        head_map_up = F.interpolate(head_map.unsqueeze(0),
                                    size=(img_for_viz.shape[-2], img_for_viz.shape[-1]),
                                    mode=args.interpolation,
                                    align_corners=False if args.interpolation == 'bilinear' else None).squeeze(0)

        save_prefix = f"Stage{args.stage}_Block{args.block}_Head{args.head}_{module_name}"

        if is_hard:
            save_prefix += f"_{map_type_suffix}"
            mask_up_for_viz = F.interpolate(head_map.unsqueeze(0).float(),
                                            size=(img_for_viz.shape[-2], img_for_viz.shape[-1]),
                                            mode=args.interpolation,
                                            align_corners=False if args.interpolation == 'bilinear' else None).squeeze(0)
            save_path = os.path.join(base_save_dir, f"{save_prefix}.png")
            visualize_hard_clusters(img_for_viz, mask_up_for_viz.bool(), save_path, args.alpha)
        else:
            save_prefix += f"_{map_type_suffix}"
            for cluster_idx in range(num_clusters):
                grid_r = cluster_idx // grid_w
                grid_c = cluster_idx % grid_w
                heatmap_tensor = head_map_up[cluster_idx]
                save_path = os.path.join(base_save_dir, f"{save_prefix}_Cluster_Row{grid_r}_Col{grid_c}.png")

                visualize_soft_cluster(raw_image_orig, heatmap_tensor, save_path,
                                       args.center_type, args.alpha, args.draw_center)
                print(f"Saved soft cluster visualization to: {save_path}")


if __name__ == '__main__':
    main()
