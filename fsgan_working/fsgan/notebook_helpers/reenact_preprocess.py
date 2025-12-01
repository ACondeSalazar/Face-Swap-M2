"""Notebook helper: reproduce canonical preprocessing for reenactment.

Provides functions to detect & crop faces, compute landmarks heatmaps, build the
generator input pyramid and run the reenactment generator. Intended to be
imported/used from `main.ipynb` or other interactive sessions.

Example (in a notebook cell):
    from fsgan.notebook_helpers.reenact_preprocess import run_reenactment_simple
    run_reenactment_simple('a.jpg', 'j.jpg', reenactment_ckpt='fsgan/weights/nfv_msrunet_256_1_2_reenactment_v2.1.pth')
"""
from typing import Optional, Tuple
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from fsgan.utils.utils import load_model
from fsgan.utils.img_utils import bgr2tensor, create_pyramid, tensor2bgr
from fsgan.utils.landmarks_utils import LandmarksHeatMapDecoder, filter_landmarks
from fsgan.utils.bbox_utils import get_main_bbox, crop_img
from fsgan.utils.seg_utils import SoftErosion, blend_seg_label

# Face detector is optional (repo-local). Import lazily so module import won't fail
def _import_detector():
    try:
        from fsgan.face_detection_dsfd.face_detector import FaceDetector
        return FaceDetector
    except Exception:
        print("no detector")
        return None


def detect_face_bbox(img_bgr: np.ndarray, detector=None, detection_model_path: Optional[str] = None, use_detector : bool = True,
                     verbose: int = 0) -> Optional[np.ndarray]:
    """Detect main face bbox in `img_bgr`.

    Returns bbox in [left, top, width, height] format or None on failure.
    """
    FaceDetector = _import_detector()
    # Create repo-local detector instance only when requested (use_detector=True)
    if FaceDetector is not None and detector is None and use_detector:
        det_model_path = detection_model_path if detection_model_path is not None else os.path.join(
            os.path.dirname(__file__), '..', 'weights', 'WIDERFace_DSFD_RES152.pth')
        #print("MODEL PATH IS ", det_model_path)
        try:
            detector = FaceDetector(detection_model_path=det_model_path, verbose=0)
        except Exception:
            print( "detector is none")
            detector = None

    if detector is not None:
        dets = detector.detect(img_bgr)
        if dets is None or len(dets) == 0:
            return None
        # detections are x1,y1,x2,y2 -> convert to left,top,width,height
        # choose main bbox by centrality/size
        bboxes = []
        for d in dets:
            x1, y1, x2, y2 = d[:4]
            bboxes.append(np.array([x1, y1, x2 - x1, y2 - y1], dtype=int))
        main = get_main_bbox(bboxes, img_bgr.shape[:2])
        print("face detected")
        return main

    # Fallback: center crop (half of min dimension)
    h, w = img_bgr.shape[:2]
    size = int(min(h, w) * 0.6)
    left = (w - size) // 2
    top = (h - size) // 2
    #print("face cropped")
    return np.array([left, top, size, size], dtype=int)


def preprocess_image_for_generator(img_bgr: np.ndarray, landmarks_model: torch.nn.Module, g_model: torch.nn.Module,
                                   device: torch.device, resolution: int = 256, crop_scale: float = 1.2,
                                   detector=None,use_detector : bool = True) -> Tuple[list, torch.Tensor]:
    """Perform canonical preprocessing for a single image.

    Returns (input_pyramid_list, original_cropped_bgr)
    - input_pyramid_list: list of torch.Tensor ready to be passed to reenactment generator
    - original_cropped_bgr: cropped+resized BGR numpy image (for reference / compositing)
    """
    # 1) Detect and crop
    bbox = detect_face_bbox(img_bgr, detector=detector, use_detector=use_detector)
    if bbox is None:
        raise RuntimeError('No face detected and no fallback available')

    # Scale bbox and crop (crop_img expects [left,top,width,height])
    bbox_scaled = np.round((bbox[:2] + bbox[2:] / 2) - (bbox[2:] * crop_scale) / 2).astype(int)
    # Simpler use of crop_img: use scale_bbox-like behaviour by recomputing center/size via get_main_bbox equivalent
    # Use crop_img via utility: provide scale via computing scaled bbox using bounding box utils
    # Here reuse simple approach: compute square bbox centered on original center
    cx, cy = bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0
    new_size = int(np.round(max(bbox[2], bbox[3]) * crop_scale))
    left = int(np.round(cx - new_size / 2.0))
    top = int(np.round(cy - new_size / 2.0))
    bbox_for_crop = np.array([left, top, new_size, new_size], dtype=int)
    cropped = crop_img(img_bgr, bbox_for_crop, border=cv2.BORDER_REPLICATE)

    # Resize to desired resolution
    cropped_resized = cv2.resize(cropped, (resolution, resolution), interpolation=cv2.INTER_CUBIC)

    # 2) Prepare image tensors
    img_gen = bgr2tensor(cropped_resized, normalize=True).to(device)  # [-1,1], shape [1,3,H,W]

    # Landmarks model expects imagenet normalization (see training code): mean/std
    context_mean = torch.as_tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    context_std = torch.as_tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    # torch.from_numpy cannot handle negative-stride views (e.g. reversed arrays). Make an explicit copy.
    img_context = torch.from_numpy(cropped_resized[:, :, ::-1].copy().transpose(2, 0, 1)).float().div_(255.0).unsqueeze(0).to(device)
    img_context = img_context.sub(context_mean).div(context_std)

    # 3) Landmarks prediction
    with torch.no_grad():
        landmarks_pred = landmarks_model(img_context)

    # 4) Build pyramid for generator inputs
    # Determine number of pyramid levels expected by G
    n_levels = getattr(g_model, 'n_local_enhancers', 0) + 1
    pyd = create_pyramid(img_gen, n_levels)

    # Prepare contexts per level
    ctx_list = []
    # landmarks_pred can be either heatmaps (B,C,H,W) or points (B,C,2)
    for p, p_img in enumerate(pyd):
        Hp = p_img.shape[2]
        Wp = p_img.shape[3]
        if landmarks_pred.dim() == 4:
            # heatmaps -> interpolate to desired size
            heat = F.interpolate(landmarks_pred, size=(Hp, Wp), mode='bilinear', align_corners=False)
            heat = filter_landmarks(heat)
            ctx = heat
        else:
            # points -> decode to heatmap of desired size
            decoder = LandmarksHeatMapDecoder(Hp).to(device)
            # ensure landmarks_pred is (B, C, 2)
            pts = landmarks_pred
            if pts.dim() == 2:
                pts = pts.unsqueeze(0)
            ctx = decoder(pts.to(device))
            ctx = filter_landmarks(ctx)

        ctx_list.append(ctx)

    # 5) Concatenate image + context per level -> input list
    input_pyd = [torch.cat((pyd[i], ctx_list[i]), dim=1) for i in range(len(pyd))]

    return input_pyd, cropped_resized


def run_reenactment_simple(src_path: str, tgt_path: str, reenactment_ckpt: Optional[str] = None,
                           resolution: int = 256, crop_scale: float = 1.2, out_path: Optional[str] = None,
                           device: Optional[torch.device] = None, use_detector: bool = True):
    """Load models and run a single-frame reenactment using canonical preprocessing.

    This function aims to reproduce preprocessing used by the repo's pipeline.
    It does NOT run segmentation / inpainting / blending (postprocessing) — only Gr.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

    # Locate default checkpoint if not provided
    if reenactment_ckpt is None:
        reenactment_ckpt = os.path.join(os.path.dirname(__file__), '..', '..', 'weights',
                                        'nfv_msrunet_256_1_2_reenactment_v2.1.pth')

    # Load Reenactment generator
    G, ckpt = load_model(reenactment_ckpt, 'reenactment', device, return_checkpoint=True)
    G.eval()

    # Load landmarks model: try to infer from checkpoint names in repo
    # Common weight present in repo: 'hr18_wflw_landmarks.pth'
    lms_weights = os.path.join(os.path.dirname(__file__), '..', '..', 'weights', 'hr18_wflw_landmarks.pth')
    if not os.path.isfile(lms_weights):
        raise RuntimeError('Could not find landmarks weights at: %s' % lms_weights)

    # Load landmarks model via load_model if checkpoint stores arch, else load state dict into hrnet factory
    L, _ = load_model(lms_weights, 'landmarks', device, return_checkpoint=True)
    L.eval()

    # optional detector
    detector = None
    if use_detector:
        FaceDetector = _import_detector()
        if FaceDetector is not None:
            try:
                detector = FaceDetector(detection_model_path=os.path.join(os.path.dirname(__file__), '..', '..', 'weights', 'WIDERFace_DSFD_RES152.pth'))
            except Exception:
                detector = None

    # Read images
    src_bgr = cv2.imread(src_path)
    tgt_bgr = cv2.imread(tgt_path)
    if src_bgr is None or tgt_bgr is None:
        raise RuntimeError('Failed reading input images: %s, %s' % (src_path, tgt_path))

    # Preprocess source: build input pyramid list
    inp_pyd_src, src_crop = preprocess_image_for_generator(src_bgr, L, G, device, resolution, crop_scale, detector, use_detector=use_detector)
    # Preprocess target (we need target landmarks context): use target to get landmarks and contexts
    inp_pyd_tgt, tgt_crop = preprocess_image_for_generator(tgt_bgr, L, G, device, resolution, crop_scale, detector, use_detector=use_detector)

    # For reenactment, generator expects source pyramid + target context. We already built input as image+context per level
    # Run generator: use source image but replace context with target context
    # Build final input list where for each level we cat source image channels and target context channels
    final_input = []
    for ps, pt in zip(inp_pyd_src, inp_pyd_tgt):
        # ps: [1, 3 + C_ctx_src, H, W]  (but C_ctx_src and C_ctx_tgt should match)
        # split image (first 3 channels) and keep target context (remaining channels)
        img_ch = ps[:, :3, :, :]
        ctx_ch = pt[:, 3:, :, :]
        final_input.append(torch.cat((img_ch, ctx_ch), dim=1))

    with torch.no_grad():
        out = G(final_input)

    # Convert to BGR numpy and save
    out_bgr = tensor2bgr(out).astype('uint8')
    if out_path is None:
        out_path = 'reenactment_out.jpg'
    cv2.imwrite(out_path, out_bgr)

    return out_bgr, src_crop, tgt_crop


def load_postprocessing_models(seg_ckpt: Optional[str] = None,
                               inpaint_ckpt: Optional[str] = None,
                               blend_ckpt: Optional[str] = None,
                               device: Optional[torch.device] = None):
    """Load segmentation, inpainting (completion) and blending models.

    Returns (S, Gc, Gb, smooth_mask) where smooth_mask is an instance of SoftErosion.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

    # default weight locations (relative to repo root)
    repo_root = os.path.join(os.path.dirname(__file__), '..')
    if seg_ckpt is None:
        seg_ckpt = os.path.join(repo_root, 'weights', 'celeba_unet_256_1_2_segmentation_v2.pth')
    if inpaint_ckpt is None:
        inpaint_ckpt = os.path.join(repo_root, 'weights', 'ijbc_msrunet_256_1_2_inpainting_v2.pth')
    if blend_ckpt is None:
        blend_ckpt = os.path.join(repo_root, 'weights', 'ijbc_msrunet_256_1_2_blending_v2.pth')

    S = load_model(seg_ckpt, 'segmentation', device)
    S.eval()
    Gc = load_model(inpaint_ckpt, 'completion', device)
    Gc.eval()
    Gb = load_model(blend_ckpt, 'blending', device)
    Gb.eval()

    smooth_mask = SoftErosion(kernel_size=21, threshold=0.6).to(device)

    return S, Gc, Gb, smooth_mask


def postprocess_reenactment(reenact_tensor: torch.Tensor, tgt_bgr: np.ndarray,
                            S: torch.nn.Module, Gc: torch.nn.Module, Gb: torch.nn.Module,
                            smooth_mask: SoftErosion, device: torch.device):
    """Run segmentation -> inpainting -> blending and compose final image.

    Inputs/outputs use tensors in the same conventions as the repo (images in range [-1,1]).

    Args:
        reenact_tensor: torch.Tensor, shape (B,3,H,W), values in [-1,1]
        tgt_bgr: numpy BGR image (H,W,3) in uint8 or float
        S, Gc, Gb: models loaded and set to eval on `device`
        smooth_mask: instance of SoftErosion on `device`
        device: torch.device

    Returns:
        result_tensor (torch.Tensor): composited final tensor in same range as models ([-1,1])
        intermediate dict with tensors (completion, transfer, blend, masks) for debugging
    """
    # Prepare target frame tensor in [-1,1] expected by postprocessing models
    tgt_frame = bgr2tensor(tgt_bgr, normalize=True).to(device)  # shape [1,3,H,W]

    with torch.no_grad():
        # Segmentation of reenacted image
        reenact_seg = S(reenact_tensor)
        reenact_bg_mask = (reenact_seg.argmax(1) != 1).unsqueeze(1)  # background where label != face

        # Remove background from reenacted face (fill with -1 background)
        reenact_tensor = reenact_tensor.clone()
        reenact_tensor.masked_fill_(reenact_bg_mask, -1.0)

        # Segment target to obtain face mask
        tgt_seg = S(tgt_frame)
        tgt_mask = (tgt_seg.argmax(1) == 1).unsqueeze(1).int()  # binary mask where face label == 1

        # Soften/erode target mask
        soft_tgt_mask, eroded_tgt_mask = smooth_mask(tgt_mask)

        # Inpainting / completion: cat reenact + eroded mask
        inpainting_input = torch.cat((reenact_tensor, eroded_tgt_mask.float()), dim=1)
        inpainting_input_pyd = create_pyramid(inpainting_input, 2)
        completion = Gc(inpainting_input_pyd)

        # Transfer completed face onto target crop using eroded mask
        mask_f = eroded_tgt_mask.float().repeat(1, 3, 1, 1)
        transfer = completion * mask_f + tgt_frame * (1.0 - mask_f)

        # Blend: cat transfer, target, eroded mask
        blend_in = torch.cat((transfer, tgt_frame, eroded_tgt_mask.float()), dim=1)
        blend_in_pyd = create_pyramid(blend_in, 2)
        blend_out = Gb(blend_in_pyd)

        # Final composite: soft mask applied to blend_out
        result = blend_out * soft_tgt_mask + tgt_frame * (1.0 - soft_tgt_mask)

    intermediates = {
        'reenact_seg': reenact_seg.detach().cpu(),
        'tgt_seg': tgt_seg.detach().cpu(),
        'soft_tgt_mask': soft_tgt_mask.detach().cpu(),
        'eroded_tgt_mask': eroded_tgt_mask.detach().cpu(),
        'completion': completion.detach().cpu(),
        'transfer': transfer.detach().cpu(),
        'blend_out': blend_out.detach().cpu(),
    }

    return result, intermediates


def run_full_pipeline(src_path: str, tgt_path: str,
                      reenactment_ckpt: Optional[str] = None,
                      seg_ckpt: Optional[str] = None,
                      inpaint_ckpt: Optional[str] = None,
                      blend_ckpt: Optional[str] = None,
                      resolution: int = 256, crop_scale: float = 1.2,
                      out_path: Optional[str] = None, device: Optional[torch.device] = None,
                      use_detector: bool = True, reenact: bool = True):
    """Run full pipeline: preprocessing -> Gr (optional) -> segmentation -> Gc -> Gb -> composite.

    If `reenact` is False the pipeline will skip running the reenactment generator and
    use the source crop directly as the reenactment tensor (useful for plain faceswap
    without reenactment transformations).

    Returns: (result_bgr, intermediates, cropped_src, cropped_tgt)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

    # Load models
    if reenactment_ckpt is None:
        reenactment_ckpt = os.path.join(os.path.dirname(__file__),'..', 'weights', 'nfv_msrunet_256_1_2_reenactment_v2.1.pth')
    G, ckpt = load_model(reenactment_ckpt, 'reenactment', device, return_checkpoint=True)
    G.eval()

    # Landmarks
    lms_weights = os.path.join(os.path.dirname(__file__), '..', 'weights', 'hr18_wflw_landmarks.pth')
    L, _ = load_model(lms_weights, 'landmarks', device, return_checkpoint=True)
    L.eval()

    # Preprocess (note: returns pyramid inputs and cropped RGB images)
    src_bgr = cv2.imread(src_path)
    tgt_bgr = cv2.imread(tgt_path)
    inp_pyd_src, src_crop = preprocess_image_for_generator(src_bgr, L, G, device, resolution, crop_scale,
                                                           detector=None, use_detector=use_detector)
    inp_pyd_tgt, tgt_crop = preprocess_image_for_generator(tgt_bgr, L, G, device, resolution, crop_scale,
                                                           detector=None, use_detector=use_detector)

    # Build final input with source image channels and target context (same as earlier helper)
    final_input = []
    for ps, pt in zip(inp_pyd_src, inp_pyd_tgt):
        img_ch = ps[:, :3, :, :]
        ctx_ch = pt[:, 3:, :, :]
        final_input.append(torch.cat((img_ch, ctx_ch), dim=1))

    # Run reenactment generator unless disabled. When reenact is False we'll reuse the
    # source crop (converted to model range) as the reenactment tensor so postprocessing
    # (segmentation/inpainting/blending) can still run and composite the source face into
    # the target frame.
    with torch.no_grad():
        if reenact:
            out = G(final_input)
            if isinstance(out, (list, tuple)):
                out_tensor = out[-1]
            else:
                out_tensor = out
        else:
            # convert src_crop (BGR numpy) to tensor in same range expected by postprocessing
            src_frame = bgr2tensor(src_crop, normalize=True).to(device)  # shape [1,3,H,W]
            out_tensor = src_frame

    # Load postprocessing models
    S, Gc, Gb, smooth_mask = load_postprocessing_models(seg_ckpt, inpaint_ckpt, blend_ckpt, device)

    # Compute source segmentation (for visualization) using same S model
    try:
        src_frame = bgr2tensor(src_crop, normalize=True).to(device)
        with torch.no_grad():
            src_seg = S(src_frame)
    except Exception:
        src_seg = None

    # Postprocess and compose using cropped target so sizes match
    result_tensor, intermediates = postprocess_reenactment(out_tensor, tgt_crop, S, Gc, Gb, smooth_mask, device)

    # Add reenactment output and source segmentation to intermediates for visualization
    try:
        intermediates['reenact_tensor'] = out_tensor.detach().cpu()
    except Exception:
        pass
    if src_seg is not None:
        try:
            intermediates['src_seg'] = src_seg.detach().cpu()
        except Exception:
            pass

    # Save result as BGR uint8
    result_bgr = tensor2bgr(result_tensor[0].cpu())
    if out_path is None:
        out_path = 'reenact_full_out.jpg'
    cv2.imwrite(out_path, result_bgr)

    return result_bgr, intermediates, src_crop, tgt_crop
