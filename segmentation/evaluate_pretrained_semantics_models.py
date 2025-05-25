import time
import random
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
from torchvision.models.segmentation import (
    deeplabv3_resnet50, DeepLabV3_ResNet50_Weights,
    fcn_resnet50, FCN_ResNet50_Weights,
    lraspp_mobilenet_v3_large, LRASPP_MobileNet_V3_Large_Weights
)
import segmentation_models_pytorch as smp
import torch.nn.functional as F


def get_transforms(resize_to=(256, 256)):
    transform = transforms.Compose([
        transforms.Resize(resize_to),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    target_transform = transforms.Compose([
        transforms.Resize(resize_to, interpolation=Image.NEAREST),
        transforms.PILToTensor()
    ])
    return transform, target_transform


def load_semantics_models(device):
    return {
        "deeplabv3_resnet50": deeplabv3_resnet50(
            weights=DeepLabV3_ResNet50_Weights.DEFAULT
        ).to(device),

        "fcn_resnet50": fcn_resnet50(
            weights=FCN_ResNet50_Weights.DEFAULT
        ).to(device),

        "lraspp_mobilenet_v3_large": lraspp_mobilenet_v3_large(
            weights=LRASPP_MobileNet_V3_Large_Weights.DEFAULT
        ).to(device),

        "unet_resnet50": smp.Unet(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=3,
            classes=21
        ).to(device),

        "pspnet_resnet50": smp.PSPNet(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=3,
            classes=21
        ).to(device)
    }


def segment_image(model, input_tensor, target_shape=(256, 256)):
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        if isinstance(output, dict) and 'out' in output:
            output = output['out']
        if output.shape[-2:] != target_shape:
            output = F.interpolate(output, size=target_shape, mode='bilinear', align_corners=False)
        pred_mask = output.argmax(dim=1).squeeze().cpu().numpy()
    return pred_mask


def compute_mean_iou(gt_mask, pred_mask, num_classes):
    ious = []
    for cls in range(num_classes):
        pred_inds = (pred_mask == cls)
        gt_inds = (gt_mask == cls)
        intersection = np.logical_and(pred_inds, gt_inds).sum()
        union = np.logical_or(pred_inds, gt_inds).sum()
        if union == 0:
            continue
        ious.append(intersection / union)
    return np.mean(ious) if ious else 0.0


def evaluation_metrics(gt_mask, pred_mask, num_classes=21):
    valid = gt_mask < num_classes
    acc = accuracy_score(gt_mask[valid], pred_mask[valid])
    prec = precision_score(gt_mask[valid], pred_mask[valid], average='macro', zero_division=0)
    iou = compute_mean_iou(gt_mask[valid], pred_mask[valid], num_classes)
    return acc, prec, iou


def evaluate_semantics_models(models, dataset, num_images=20, device=None):
    for name, model in models.items():
        print(f"\nEvaluating model: {name}")
        print(f"  Running on device: {next(model.parameters()).device}")

        accs, precs, ious, times = [], [], [], []
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # warm-up
        image, mask = dataset[0]
        input_tensor = image.unsqueeze(0).contiguous().to(device)
        _ = segment_image(model, input_tensor, target_shape=mask.shape[-2:])

        for i in range(num_images):
            image, mask = dataset[i]
            input_tensor = image.unsqueeze(0).contiguous().to(device)
            gt_mask = mask.squeeze().numpy()

            start = time.time()
            pred_mask = segment_image(model, input_tensor, target_shape=mask.shape[-2:])
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.time() - start

            acc, prec, iou = evaluation_metrics(gt_mask, pred_mask)
            accs.append(acc)
            precs.append(prec)
            ious.append(iou)
            times.append(elapsed)

        print(f"  Accuracy:   {np.mean(accs):.4f}")
        print(f"  Precision:  {np.mean(precs):.4f}")
        print(f"  IoU:        {np.mean(ious):.4f}")
        print(f"  FPS:        {1 / np.mean(times):.2f}")
        print(f"  Parameters: {total_params / 1e6:.2f}M")


if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)

    print("CUDA available:", torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resize_to = (256, 256)
    transform, target_transform = get_transforms(resize_to=resize_to)

    dataset = VOCSegmentation(
        root='./VOC2012_dataset',
        year='2012',
        image_set='val',
        download=True,
        transform=transform,
        target_transform=target_transform
    )

    models = load_semantics_models(device)
    evaluate_semantics_models(models, dataset, num_images=50, device=device)