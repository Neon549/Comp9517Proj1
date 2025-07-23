import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix as sk_confusion_matrix, classification_report
import seaborn as sns


class MetricsCalculator:
    """Metrics Calculator"""

    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        """Reset metrics"""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def update(self, pred, target):
        """Update confusion matrix"""
        pred = pred.flatten()
        target = target.flatten()
        cm = sk_confusion_matrix(target, pred, labels=list(range(self.num_classes)))
        self.confusion_matrix += cm

    def get_iou(self):
        """Calculate IoU"""
        intersection = np.diag(self.confusion_matrix)
        union = (self.confusion_matrix.sum(axis=1) +
                 self.confusion_matrix.sum(axis=0) - intersection)

        # Avoid division by zero
        iou = intersection / (union + 1e-8)
        mean_iou = np.nanmean(iou)
        return iou, mean_iou

    def get_pixel_accuracy(self):
        """Calculate Pixel Accuracy"""
        total = self.confusion_matrix.sum()
        if total == 0:
            return 0.0
        acc = np.diag(self.confusion_matrix).sum() / total
        return acc

    def get_class_accuracy(self):
        """Calculate Class Accuracy"""
        class_totals = self.confusion_matrix.sum(axis=1)
        acc = np.diag(self.confusion_matrix) / (class_totals + 1e-8)
        mean_acc = np.nanmean(acc)
        return acc, mean_acc


def denormalize_image(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize image"""
    mean = np.array(mean)
    std = np.array(std)
    img = img * std[:, None, None] + mean[:, None, None]
    img = np.clip(img, 0, 1)
    return img


def visualize_predictions(model, dataloader, device, num_samples=5, save_path=None):
    """Visualize prediction results"""
    model.eval()

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    sample_count = 0

    with torch.no_grad():
        for images, masks in dataloader:
            if sample_count >= num_samples:
                break

            images = images.to(device)
            masks = masks.to(device)

            # Predict
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            pred_masks = torch.argmax(outputs, dim=1)

            # Take the first sample for visualization
            img = images[0].cpu().numpy()
            true_mask = masks[0].cpu().numpy()
            pred_mask = pred_masks[0].cpu().numpy()

            # Denormalize image
            img = denormalize_image(img)
            img = img.transpose(1, 2, 0)

            # Plot original image
            axes[sample_count, 0].imshow(img)
            axes[sample_count, 0].set_title('Original Image')
            axes[sample_count, 0].axis('off')

            # Plot true mask
            axes[sample_count, 1].imshow(true_mask, cmap='gray')
            axes[sample_count, 1].set_title('True Mask')
            axes[sample_count, 1].axis('off')

            # Plot predicted mask
            axes[sample_count, 2].imshow(pred_mask, cmap='gray')
            axes[sample_count, 2].set_title('Predicted Mask')
            axes[sample_count, 2].axis('off')

            # Plot overlay
            overlay = img.copy()
            overlay[pred_mask == 1] = [1, 0, 0]  # Red for predicted dead trees
            axes[sample_count, 3].imshow(overlay)
            axes[sample_count, 3].set_title('Prediction Overlay')
            axes[sample_count, 3].axis('off')

            sample_count += 1

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(cm, class_names, save_path=None):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def calculate_inference_time(model, dataloader, device, num_batches=10):
    """Calculate inference time"""
    model.eval()
    import time

    times = []

    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if i >= num_batches:
                break

            images = images.to(device)

            # Warm-up GPU
            if i == 0:
                _ = model(images)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                continue

            start_time = time.time()
            _ = model(images)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()

            batch_time = (end_time - start_time) / images.size(0)  # Time per image
            times.append(batch_time)

    avg_time = np.mean(times)
    std_time = np.std(times)

    print(f"Average inference time: {avg_time:.4f}±{std_time:.4f} seconds/image")
    return avg_time, std_time


def save_sample_predictions(model, dataloader, device, save_dir, num_samples=20):
    """Save sample predictions"""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    sample_count = 0

    with torch.no_grad():
        for images, masks in dataloader:
            if sample_count >= num_samples:
                break

            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            pred_masks = torch.argmax(outputs, dim=1)

            for i in range(images.size(0)):
                if sample_count >= num_samples:
                    break

                # Save predicted mask
                pred_mask = pred_masks[i].cpu().numpy().astype(np.uint8) * 255
                cv2.imwrite(
                    os.path.join(save_dir, f'pred_{sample_count:03d}.png'),
                    pred_mask
                )

                sample_count += 1

    print(f"Saved {sample_count} predictions to {save_dir}")


def create_evaluation_report(metrics_calc, class_names=['Background', 'Dead Tree']):
    """Create evaluation report"""
    iou_per_class, mean_iou = metrics_calc.get_iou()
    pixel_acc = metrics_calc.get_pixel_accuracy()
    class_acc, mean_class_acc = metrics_calc.get_class_accuracy()

    report = f"""
=== Model Evaluation Report ===

Overall Metrics:
- Mean IoU: {mean_iou:.4f}
- Pixel Accuracy: {pixel_acc:.4f}
- Mean Class Accuracy: {mean_class_acc:.4f}

IoU per Class:
"""

    for i, (name, iou) in enumerate(zip(class_names, iou_per_class)):
        report += f"- {name}: {iou:.4f}\n"

    report += "\nAccuracy per Class:\n"
    for i, (name, acc) in enumerate(zip(class_names, class_acc)):
        report += f"- {name}: {acc:.4f}\n"

    print(report)
    return report