from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image
from PIL import Image
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment


def process_and_save(
    processor: AutoImageProcessor,
    image: Image.Image,
    out_path: str = "output.jpg"
) -> None: 
    pixel_values = processor(images=image, return_tensors='pt').pixel_values.squeeze(0).cpu() 
    mean = torch.tensor(processor.image_mean).view(3, 1, 1)
    std = torch.tensor(processor.image_std).view(3, 1, 1)
    image_tensor = pixel_values * std + mean  # undo normalization
    clamped_image = image_tensor.clamp(0, 1)
    save_image(clamped_image, out_path)

def poly_to_mask(
    polygon: list[tuple[int, int]],
    width: int,
    height: int
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)

    cv2.fillPoly(mask, [pts], 1)

    return mask.astype(np.float32)

def mask_to_patch_weights(poly_mask, grid_size):
    m = torch.tensor(poly_mask, dtype=torch.float32)  # (H, W)

    m = m[None, None]  # (1, 1, H, W)

    # downsample mask to patch grid resolution
    # weights are approx polygon coverage per patch
    m_down = F.interpolate(
        m,
        size=(grid_size, grid_size),
        mode="bilinear",
        align_corners=False
    )

    return m_down[0, 0].clamp(min=0)  # (grid_size, grid_size)

def pool_features(patch_features, weights):
    # TODO: try attention pooling
    # patch_features is (D, G, G)
    # weights is (G, G)
    weighted = patch_features * weights.unsqueeze(0)

    denom = weights.sum().clamp(min=1e-6)

    return weighted.sum(dim=(1,2)) / denom

def extract_bbox_features(
    model: AutoModel,
    processor: AutoImageProcessor,
    image: Image.Image,
    polygons: list[tuple[int, int]]
) -> torch.Tensor:
    scaled_w = processor.size.width
    scaled_h = processor.size.height

    def scale_poly(
        polygon: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        width_ratio = scaled_w / image.size[0]
        height_ratio = scaled_h / image.size[1]
        return [
            (int(x * width_ratio), int(y * height_ratio))
            for x, y in polygon
        ]

    model.eval()

    # preprocess image
    inputs = processor(images=image, return_tensors="pt").to(model.device)

    # forward pass
    with torch.inference_mode():
        outputs = model(**inputs)
        features = outputs.last_hidden_state[0]  # [N+5, D]
        patch_features = features[5:]            # drop CLS and 4 registers

    num_patches = patch_features.shape[0]
    grid_size = int(num_patches ** 0.5)
    D = patch_features.shape[-1]

    # reshape to grid
    patch_features = patch_features.reshape(grid_size, grid_size, D)
    patch_features = patch_features.permute(2, 0, 1)  # (D, G, G)

    results = []

    for poly in polygons:
        poly_t = scale_poly(poly)

        # convert polygon to area mask
        poly_mask = poly_to_mask(poly_t, scaled_w, scaled_h)

        # get weighted sum pooling
        weights = mask_to_patch_weights(poly_mask, grid_size).to(patch_features.device)

        feat = pool_features(patch_features, weights)

        results.append(feat)

    return torch.stack(results)

def compute_cost_matrix(
    tracklet_features: torch.Tensor,
    masklet_features: torch.Tensor
) -> np.ndarray:
    # assumes tracklet_features.device == masklet_features.device
    # cost = 1 - cos similarity
    trk_norm = F.normalize(tracklet_features, dim=-1, eps=1e-8)
    mask_norm = F.normalize(masklet_features, dim=-1, eps=1e-8)
    cost = (1 - trk_norm @ mask_norm.T).clamp(0, 2)
    return cost.detach().cpu().numpy().astype(np.float64)

def update_tracklets(
    model: AutoModel,
    processor: AutoImageProcessor,
    frame: Image.Image,
    bboxes: list[tuple[int, int]],
    tracklet_features: torch.Tensor | None,
    ema_alpha: float = 0.1
) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    masklet_features = extract_bbox_features(
        model,
        processor,
        frame,
        bboxes
    )

    if tracklet_features is None:
        tracklet_features = masklet_features.clone()
        return np.arange(len(bboxes)), np.arange(len(bboxes)), F.normalize(tracklet_features, dim=-1, eps=1e-8)

    tracklet_features = tracklet_features.to(masklet_features.device)
    cost = compute_cost_matrix(tracklet_features, masklet_features)
    # TODO: add match threshold + handle # tracks != # dets
    row_ind, col_ind = linear_sum_assignment(cost)

    new_tracklet_features = tracklet_features.clone()
    t_norm = F.normalize(tracklet_features, dim=-1, eps=1e-8)
    m_norm = F.normalize(masklet_features, dim=-1, eps=1e-8)
    for t, m in zip(row_ind, col_ind):
        new_tracklet_features[t] = ema_alpha * m_norm[m] + (1 - ema_alpha) * t_norm[t]
    
    return row_ind, col_ind, F.normalize(new_tracklet_features, dim=-1, eps=1e-8)

    
def main():
    # TODO: change to onnxruntime or tensorrt
    model_name = "camenduru/dinov3-vitl16-pretrain-lvd1689m"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, device_map="auto")
    
    # TODO: change to take actual frames and detections
    frames = []
    frame_dets = [[]]  # one det array per frame

    tracklet_features = None
    for frame, dets in zip(frames, frame_dets):
        row_ind, col_ind, tracklet_features = update_tracklets(
            model,
            processor,
            frame,
            dets,
            tracklet_features
        )

    # TODO: figure out what and how to return

if __name__ == '__main__':
    main()