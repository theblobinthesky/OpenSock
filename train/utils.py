import numpy as np
import jax.numpy as jnp

def batch_intersection_over_union(bbox: jnp.ndarray, rem_bboxes: jnp.ndarray) -> float:
    bbox_left, bbox_right, bbox_top, bbox_bottom = bbox[0], bbox[1], bbox[2], bbox[3]
    [rem_left, rem_right, rem_top, rem_bottom] = rem_bboxes.T

    left = jnp.maximum(bbox_left, rem_left)
    right = jnp.minimum(bbox_right, rem_right)
    top = jnp.maximum(bbox_top, rem_top)
    bottom = jnp.minimum(bbox_bottom, rem_bottom)

    inters_width = jnp.clip(right - left, min=0.0)
    inters_height = jnp.clip(bottom - top, min=0.0)
    intersection = inters_width * inters_height

    union = (bbox_right - bbox_left) * (bbox_bottom - bbox_top) \
        + (rem_right - rem_left) * (rem_bottom - rem_top) \
        - intersection
    
    iou = intersection / union
    return iou

def non_maximum_supression(
        bboxes: jnp.ndarray,
        scores: jnp.ndarray,
        threshold: float
        ) -> jnp.ndarray:
    
    order = scores.argsort()
    keep_bboxes = []
    keep_scores = []

    while len(order) > 0:
        idx = order[-1]
        order = order[:-1]

        bbox = bboxes[idx]
        keep_bboxes.append(bbox[None, :])
        keep_scores.append(order)

        rem_bboxes = bboxes[order]

        iou = batch_intersection_over_union(bbox, rem_bboxes)
        order = order[iou < threshold]

    return jnp.concat(keep_bboxes, axis=0), jnp.concat(keep_scores)

# TODO: Support multiple classes.
def average_precision(
        gt_bboxes: np.ndarray,
        pred_bboxes: np.ndarray,
        pred_scores: np.ndarray,
        iou_thresholds: list[float]=list(np.arange(start=0.5, stop=1.0, step=0.05))
) -> float:
    num_gt_bboxes = len(gt_bboxes)
    total_area = 0.0
    
    for iou_threshold in iou_thresholds: 
        true_pos, false_pos = 0, 0
        order = pred_scores.argsort()
        # Add starting and end point to match standard implementations.
        recall_prec_curve = [(0.0, 1.0), (1.0, 0.0)]
        mask = np.ones((len(gt_bboxes),), np.bool_)

        while len(order) > 0:
            idx = order[-1]
            pred_bbox = pred_bboxes[idx]
            pred_score = pred_scores[idx]
            order = order[:-1]

            iou = batch_intersection_over_union(pred_bbox, gt_bboxes)
            gt_idxs = np.nonzero((iou >= iou_threshold) & mask)[0]

            if gt_idxs.size > 0:
                max_gt = np.argmax(iou[gt_idxs])
                mask[gt_idxs[max_gt]] = False
                true_pos += 1
            else:
                false_pos += 1 

            precision = true_pos / (true_pos + false_pos) 
            recall = true_pos / num_gt_bboxes
            recall_prec_curve.append((recall, precision))

        recall_prec_curve = sorted(recall_prec_curve, key=lambda item: item[0], reverse=True)
        rolling_precision_max = float('-inf')
        recall_prec_curve = [(recall, (rolling_precision_max := max(rolling_precision_max, precision))) for (recall, precision) in recall_prec_curve]
        area = 0.0
        for (r1, p1), (r2, p2) in zip(recall_prec_curve[:-1], recall_prec_curve[1:]):
            area += p2 * (r1 - r2) + (p1 - p2) * (r1 - r2) / 2.0

        total_area += area

    return total_area / len(iou_thresholds)

def focal_loss(target: jnp.ndarray, pred: jnp.ndarray, focal_loss_config: dict) -> jnp.ndarray: 
    assert target.shape[-1] == 1
    assert target.dtype == jnp.bool_
    exponent = focal_loss_config['exponent']
    pos_class_weight = focal_loss_config['pos_class_weight']

    pred_t = jnp.where(target, pred, 1.0 - pred)
    weight = jnp.where(target, pos_class_weight, 1.0 - pos_class_weight)
    loss = -weight * jnp.log(pred_t + 1e-8) * jnp.abs(1 - pred_t) ** exponent
    return loss.mean()

def quality_focal_loss(target: jnp.ndarray, pred: jnp.ndarray, exponent: float) -> jnp.ndarray:
    cross_entropy = (1.0 - target) * jnp.log(1.0 - pred) + target * jnp.log(pred)
    loss = cross_entropy * jnp.abs(target - pred) ** exponent
    return -loss.mean()

def distribution_focal_loss(cx, cy, target: jnp.ndarray, pred: jnp.ndarray) -> jnp.ndarray:
    loss = 0.0

    for i, (bin, fract) in enumerate(target):
        b = cx.shape[0]
        loss += fract * jnp.log(pred[jnp.arange(b)[:, None], cy, cx, i, bin])

    return -loss.mean()

def general_iou_loss():
    # TODO: Dew it.
    return 0.0