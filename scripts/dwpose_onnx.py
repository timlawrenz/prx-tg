"""Standalone DWPose ONNX inference — no mmpose/mmcv dependencies.

Uses YOLOX-L for person detection and DWPose-L (384x288) for whole-body
keypoint estimation (133 COCO-WholeBody keypoints).

ONNX models are downloaded from HuggingFace on first use:
  - yzd-v/DWPose/yolox_l.onnx
  - yzd-v/DWPose/dw-ll_ucoco_384.onnx

Based on the DWPose ONNX branch by IDEA-Research (Apache 2.0 license).
"""

import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# YOLOX detection helpers
# ---------------------------------------------------------------------------

def _nms(boxes, scores, nms_thr):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]
    return keep


def _multiclass_nms(boxes, scores, nms_thr, score_thr):
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid = cls_scores > score_thr
        if valid.sum() == 0:
            continue
        valid_scores = cls_scores[valid]
        valid_boxes = boxes[valid]
        keep = _nms(valid_boxes, valid_scores, nms_thr)
        if keep:
            cls_inds = np.ones((len(keep), 1)) * cls_ind
            dets = np.concatenate(
                [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1)
            final_dets.append(dets)
    if not final_dets:
        return None
    return np.concatenate(final_dets, 0)


def _yolox_postprocess(outputs, img_size):
    grids, expanded_strides = [], []
    strides = [8, 16, 32]
    hsizes = [img_size[0] // s for s in strides]
    wsizes = [img_size[1] // s for s in strides]
    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        expanded_strides.append(np.full((*grid.shape[:2], 1), stride))
    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides
    return outputs


def _yolox_preprocess(img, input_size, swap=(2, 0, 1)):
    padded = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized = cv2.resize(
        img, (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR).astype(np.uint8)
    padded[:int(img.shape[0] * r), :int(img.shape[1] * r)] = resized
    padded = padded.transpose(swap)
    return np.ascontiguousarray(padded, dtype=np.float32), r


def _detect_persons(session, img):
    """Run YOLOX to detect person bounding boxes."""
    input_shape = (640, 640)
    preprocessed, ratio = _yolox_preprocess(img, input_shape)
    inp = preprocessed[None, :, :, :].astype(np.float32)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: inp})
    predictions = _yolox_postprocess(output[0], input_shape)[0]
    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]
    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
    boxes_xyxy /= ratio
    dets = _multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
    if dets is None:
        return None
    final_boxes = dets[:, :4]
    final_scores = dets[:, 4]
    final_cls = dets[:, 5]
    mask = (final_scores > 0.3) & (final_cls == 0)  # class 0 = person
    return final_boxes[mask] if mask.any() else None


# ---------------------------------------------------------------------------
# DWPose keypoint helpers
# ---------------------------------------------------------------------------

def _bbox_xyxy2cs(bbox, padding=1.25):
    dim = bbox.ndim
    if dim == 1:
        bbox = bbox[None, :]
    x1, y1, x2, y2 = np.hsplit(bbox, [1, 2, 3])
    center = np.hstack([x1 + x2, y1 + y2]) * 0.5
    scale = np.hstack([x2 - x1, y2 - y1]) * padding
    if dim == 1:
        center, scale = center[0], scale[0]
    return center, scale


def _fix_aspect_ratio(bbox_scale, aspect_ratio):
    w, h = np.hsplit(bbox_scale, [1])
    return np.where(w > h * aspect_ratio,
                    np.hstack([w, w / aspect_ratio]),
                    np.hstack([h * aspect_ratio, h]))


def _get_3rd_point(a, b):
    direction = a - b
    return b + np.r_[-direction[1], direction[0]]


def _get_warp_matrix(center, scale, rot, output_size):
    shift = np.zeros(2)
    src_w = scale[0]
    dst_w, dst_h = output_size
    rot_rad = np.deg2rad(rot)
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_dir = np.array([0., src_w * -0.5]) @ np.array([[cs, sn], [-sn, cs]])
    dst_dir = np.array([0., dst_w * -0.5])
    src = np.zeros((3, 2), dtype=np.float32)
    src[0] = center + scale * shift
    src[1] = center + src_dir + scale * shift
    src[2] = _get_3rd_point(src[0], src[1])
    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0] = [dst_w * 0.5, dst_h * 0.5]
    dst[1] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2] = _get_3rd_point(dst[0], dst[1])
    return cv2.getAffineTransform(np.float32(src), np.float32(dst))


def _pose_preprocess(img, bboxes, input_size=(288, 384)):
    out_imgs, out_centers, out_scales = [], [], []
    if len(bboxes) == 0:
        bboxes = [[0, 0, img.shape[1], img.shape[0]]]
    w, h = input_size
    for bbox in bboxes:
        center, scale = _bbox_xyxy2cs(np.array(bbox), padding=1.25)
        scale = _fix_aspect_ratio(scale, aspect_ratio=w / h)
        warp_mat = _get_warp_matrix(center, scale, 0, output_size=(w, h))
        resized = cv2.warpAffine(img, warp_mat, (int(w), int(h)),
                                 flags=cv2.INTER_LINEAR)
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        resized = (resized - mean) / std
        out_imgs.append(resized)
        out_centers.append(center)
        out_scales.append(scale)
    return out_imgs, out_centers, out_scales


def _simcc_maximum(simcc_x, simcc_y):
    N, K, _ = simcc_x.shape
    sx = simcc_x.reshape(N * K, -1)
    sy = simcc_y.reshape(N * K, -1)
    x_locs = np.argmax(sx, axis=1)
    y_locs = np.argmax(sy, axis=1)
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    max_val_x = np.amax(sx, axis=1)
    max_val_y = np.amax(sy, axis=1)
    mask = max_val_x > max_val_y
    max_val_x[mask] = max_val_y[mask]
    vals = max_val_x
    locs[vals <= 0.] = -1
    return locs.reshape(N, K, 2), vals.reshape(N, K)


def _decode_simcc(simcc_x, simcc_y, split_ratio=2.0):
    keypoints, scores = _simcc_maximum(simcc_x, simcc_y)
    keypoints /= split_ratio
    return keypoints, scores


def _pose_postprocess(outputs, model_input_size, centers, scales):
    all_kpts, all_scores = [], []
    for i, (simcc_x, simcc_y) in enumerate(outputs):
        keypoints, scores = _decode_simcc(simcc_x, simcc_y)
        keypoints = keypoints / model_input_size * scales[i] + centers[i] - scales[i] / 2
        all_kpts.append(keypoints[0])
        all_scores.append(scores[0])
    return np.array(all_kpts), np.array(all_scores)


def _infer_pose(session, bboxes, img, model_input_size=(288, 384)):
    resized_imgs, centers, scales = _pose_preprocess(img, bboxes, model_input_size)
    inp = np.stack(resized_imgs, axis=0).transpose(0, 3, 1, 2).astype(np.float32)
    input_name = session.get_inputs()[0].name
    all_outputs = session.run(None, {input_name: inp})
    outputs = []
    for idx in range(len(all_outputs[0])):
        outputs.append([all_outputs[j][idx:idx + 1, ...] for j in range(len(all_outputs))])
    keypoints, scores = _pose_postprocess(outputs, model_input_size, centers, scales)
    return keypoints, scores


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

DWPOSE_CACHE_DIR = Path(os.environ.get(
    "DWPOSE_CACHE_DIR",
    Path.home() / ".cache" / "dwpose"))


def _download_model(filename: str) -> Path:
    """Download ONNX model from HuggingFace if not cached."""
    cache_dir = DWPOSE_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / filename
    if model_path.exists():
        return model_path
    from huggingface_hub import hf_hub_download
    hf_hub_download(
        repo_id="yzd-v/DWPose",
        filename=filename,
        local_dir=str(cache_dir),
    )
    return model_path


class DWPoseDetector:
    """Lightweight DWPose whole-body detector using ONNX Runtime.

    Usage::

        detector = DWPoseDetector()
        keypoints, scores, bboxes = detector(image_rgb_hwc)
    """

    def __init__(self, device: str = "cpu"):
        import onnxruntime as ort

        det_path = _download_model("yolox_l.onnx")
        pose_path = _download_model("dw-ll_ucoco_384.onnx")

        device_str = str(device)
        providers = ["CPUExecutionProvider"]
        if "cuda" in device_str or "rocm" in device_str:
            # ROCm uses the same provider name as CUDA in onnxruntime
            if ort.get_available_providers():
                gpu_providers = [p for p in ort.get_available_providers()
                                 if p != "CPUExecutionProvider"]
                if gpu_providers:
                    providers = gpu_providers + providers

        self.det_session = ort.InferenceSession(str(det_path), providers=providers)
        self.pose_session = ort.InferenceSession(str(pose_path), providers=providers)
        self.pose_input_size = (288, 384)  # (w, h) for dw-ll_ucoco_384

    def __call__(
        self, img: np.ndarray, single_person: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Detect whole-body keypoints.

        Args:
            img: RGB image as HxWx3 uint8 numpy array.
            single_person: If True, return only the largest detection.

        Returns:
            keypoints: (N, 133, 2) pixel coordinates
            scores: (N, 133) confidence scores
            bboxes: (N, 4) bounding boxes [x1, y1, x2, y2]
        """
        bboxes = _detect_persons(self.det_session, img)

        if bboxes is None or len(bboxes) == 0:
            empty = np.zeros
            return empty((0, 133, 2)), empty((0, 133)), empty((0, 4))

        if single_person:
            # Select largest bounding box by area
            areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            best = np.argmax(areas)
            bboxes = bboxes[best:best + 1]

        keypoints, scores = _infer_pose(
            self.pose_session, bboxes, img, self.pose_input_size)

        return keypoints, scores, bboxes
