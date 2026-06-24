import numpy as np
import onnxruntime as ort


def create_session(path):
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _sigmoid(logits):
    positive = logits >= 0.0
    negative = ~positive
    result = np.empty_like(logits, dtype=np.float32)
    result[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[negative])
    result[negative] = exp_logits / (1.0 + exp_logits)
    return result


def _compute_loss(logits, targets, bce_weight=1.0, pos_weight=1.0):
    probabilities = np.clip(_sigmoid(logits), 1e-7, 1.0 - 1e-7)
    targets = targets.astype(np.float32, copy=False)
    dice_intersection = (probabilities * targets).sum()
    dice_denominator = probabilities.sum() + targets.sum()
    dice_loss = 1.0 - ((2.0 * dice_intersection + 1.0) / (dice_denominator + 1.0))
    bce_loss = -(
        (pos_weight * targets * np.log(probabilities))
        + ((1.0 - targets) * np.log(1.0 - probabilities))
    ).mean()
    return float(dice_loss + (float(bce_weight) * bce_loss))


def evaluate_session(session, dataset, threshold=0.5, bce_weight=1.0, pos_weight=1.0):
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    sample_iou = sample_dice = correct = pixels = total_loss = 0.0
    for image, mask in dataset:
        logits = session.run([output_name], {input_name: image.numpy()[None, ...]})[0][0]
        prediction = logits > 0.0 if threshold == 0.5 else _sigmoid(logits) > threshold
        target = mask.numpy() > 0.5
        intersection = float(np.logical_and(prediction, target).sum())
        prediction_sum, mask_sum = float(prediction.sum()), float(target.sum())
        sample_iou += (intersection + 1.0) / (prediction_sum + mask_sum - intersection + 1.0)
        sample_dice += (2.0 * intersection + 1.0) / (prediction_sum + mask_sum + 1.0)
        correct += float((prediction == target).sum())
        pixels += target.size
        total_loss += _compute_loss(logits, mask.numpy(), bce_weight=bce_weight, pos_weight=pos_weight)
    count = len(dataset)
    if not count:
        raise RuntimeError("evaluation dataset is empty")
    return {
        "samples": count,
        "loss": total_loss / count,
        "iou": sample_iou / count,
        "dice": sample_dice / count,
        "pixel_accuracy": correct / pixels,
    }
