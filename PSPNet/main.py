#!/usr/bin/env python3
"""
COMP9517 计算机视觉项目 - PSPNet森林分割
主程序文件
"""

import torch
import argparse
import os
import sys
from config import Config
from dataset import DataManager
from trainer import Trainer
from model import create_model
from utils import (
    MetricsCalculator,
    visualize_predictions,
    calculate_inference_time,
    save_sample_predictions,
    create_evaluation_report,
    plot_confusion_matrix
)


def main():
    parser = argparse.ArgumentParser(description='PSPNetSplitTrainer')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test', 'demo'],
                        help='run mode: train, test, demo')
    parser.add_argument('--resume', action='store_true',
                        help='train from the observation point')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='check the root path')
    parser.add_argument('--model_path', type=str, default='models/best_model.pth',
                        help='check the train path')

    args = parser.parse_args()

    # Initialization
    config = Config()
    config.create_directories()

    print("=" * 60)
    print("COMP9517 Project")
    print("model: PSPNet")
    print(f"runtime: {args.mode}")
    print("=" * 60)

    # Create data manager
    data_manager = DataManager(config)

    try:
        if args.mode == 'train':
            train_model(config, data_manager, args)
        elif args.mode == 'test':
            test_model(config, data_manager, args)
        elif args.mode == 'demo':
            demo_model(config, data_manager, args)
    except Exception as e:
        print(f"error: {e}")
        sys.exit(1)


def train_model(config, data_manager, args):
    """Train model"""
    print("start training model...")

    # Create data loader
    train_loader, val_loader = data_manager.create_dataloaders()

    # Create trainer
    trainer = Trainer(config, train_loader, val_loader)

    # Train model
    best_iou = trainer.train(
        resume=args.resume,
        checkpoint_path=args.checkpoint
    )

    print(f"complete！best IoU: {best_iou:.4f}")


def test_model(config, data_manager, args):
    """Test model"""
    print("start testing model...")

    # Check the existence of file
    if not os.path.exists(args.model_path):
        print(f"error: files {args.model_path} does not exist！")
        print("please check the root path。")
        return

    # Device
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"use device: {device}")

    # Create model and update weights
    model = create_model(config).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("model loaded!！")

    # Create data loader caculator
    _, val_loader = data_manager.create_dataloaders()

    # Create index caculator
    metrics_calc = MetricsCalculator(config.NUM_CLASSES)

    print("start inference time...")

    # Evaluation
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            pred_masks = torch.argmax(outputs, dim=1)

            # update index
            for i in range(pred_masks.size(0)):
                pred = pred_masks[i].cpu().numpy()
                target = masks[i].cpu().numpy()
                metrics_calc.update(pred, target)

    # Report
    report = create_evaluation_report(metrics_calc)

    # save
    report_path = os.path.join(config.RESULTS_SAVE_PATH, 'evaluation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    # confusion metrix
    cm_path = os.path.join(config.RESULTS_SAVE_PATH, 'confusion_matrix.png')
    plot_confusion_matrix(
        metrics_calc.confusion_matrix,
        ['bg', 'tree'],
        save_path=cm_path
    )

    # visualization
    vis_path = os.path.join(config.RESULTS_SAVE_PATH, 'predictions_visualization.png')
    visualize_predictions(model, val_loader, device, num_samples=5, save_path=vis_path)

    # calculate the run time
    avg_time, std_time = calculate_inference_time(model, val_loader, device)

    # save samples
    pred_dir = os.path.join(config.RESULTS_SAVE_PATH, 'sample_predictions')
    save_sample_predictions(model, val_loader, device, pred_dir, num_samples=20)

    print(f"test finished！save the result to {config.RESULTS_SAVE_PATH}")


def demo_model(config, data_manager, args):
    """Model demo"""
    print("Start demo model...")

    if not os.path.exists(args.model_path):
        print(f"Error: files {args.model_path} does not exist！")
        return

    # Device
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')

    # Create model and load weights
    model = create_model(config).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create data loader
    _, val_loader = data_manager.create_dataloaders()

    # Visualization
    print("Visualization...")
    visualize_predictions(model, val_loader, device, num_samples=3)

    # calculator
    metrics_calc = MetricsCalculator(config.NUM_CLASSES)

    sample_count = 0
    max_samples = 50  # A few samples to demo

    with torch.no_grad():
        for images, masks in val_loader:
            if sample_count >= max_samples:
                break

            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            pred_masks = torch.argmax(outputs, dim=1)

            for i in range(pred_masks.size(0)):
                if sample_count >= max_samples:
                    break
                pred = pred_masks[i].cpu().numpy()
                target = masks[i].cpu().numpy()
                metrics_calc.update(pred, target)
                sample_count += 1

    # show demo
    _, mean_iou = metrics_calc.get_iou()
    pixel_acc = metrics_calc.get_pixel_accuracy()

    print(f"result ( {sample_count} samples):")
    print(f"average : {mean_iou:.4f}")
    print(f"pixel accuracy: {pixel_acc:.4f}")

    print("finish demo model!！")


if __name__ == "__main__":
    main()