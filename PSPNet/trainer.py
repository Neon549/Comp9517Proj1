import torch
import torch.optim as optim
from torch.optim.lr_scheduler import PolynomialLR
import time
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import jaccard_score
import matplotlib.pyplot as plt
from config import Config
from model import create_model, PSPNetLoss


class Trainer:
    """Trainer class"""

    def __init__(self, config, train_loader, val_loader):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Device setup
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")

        # Model creation
        self.model = create_model(config).to(self.device)

        # Loss function and optimizer setup
        self.criterion = PSPNetLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=1e-4
        )

        # Learning rate scheduler
        self.scheduler = PolynomialLR(
            self.optimizer,
            total_iters=config.NUM_EPOCHS,
            power=0.9
        )

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_ious = []
        self.best_iou = 0.0

        # Create directories for saving files
        config.create_directories()

    def calculate_iou(self, pred, target):
        """Calculate Intersection over Union (IoU)"""
        pred = torch.argmax(pred, dim=1).cpu().numpy()
        target = target.cpu().numpy()

        # Calculate IoU for each image in the batch
        ious = []
        for i in range(pred.shape[0]):
            pred_flat = pred[i].flatten()
            target_flat = target[i].flatten()

            # Calculate Jaccard score (IoU)
            # Handle cases where both pred and target are empty (IoU is 1.0)
            if target_flat.max() == 0 and pred_flat.max() == 0:
                iou = 1.0
            else:
                iou = jaccard_score(target_flat, pred_flat, average='macro', zero_division=1)
            ious.append(iou)

        return np.mean(ious)

    def train_epoch(self):
        """Train a single epoch"""
        self.model.train()
        running_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Training")
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        return running_loss / len(self.train_loader)

    def validate(self):
        """Perform validation"""
        self.model.eval()
        running_loss = 0.0
        total_iou = 0.0

        with torch.no_grad():
            for images, masks in tqdm(self.val_loader, desc="Validating"):
                images = images.to(self.device)
                masks = masks.to(self.device)

                # Forward pass
                outputs = self.model(images)
                # If PSPNetLoss returns a tuple (main_output, aux_output), take the main output
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                loss = self.criterion(outputs, masks)
                iou = self.calculate_iou(outputs, masks)

                running_loss += loss.item()
                total_iou += iou

        avg_loss = running_loss / len(self.val_loader)
        avg_iou = total_iou / len(self.val_loader)

        return avg_loss, avg_iou

    def save_model(self, epoch, iou, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'iou': iou,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_ious': self.val_ious
        }

        # Save the latest model
        torch.save(checkpoint, os.path.join(self.config.MODEL_SAVE_PATH, 'latest_model.pth'))

        # Save the best model if current IoU is the highest
        if is_best:
            torch.save(checkpoint, os.path.join(self.config.MODEL_SAVE_PATH, 'best_model.pth'))
            print(f"Saved the best model (IoU: {iou:.4f})")

    def load_model(self, checkpoint_path):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_ious = checkpoint.get('val_ious', [])
        self.best_iou = checkpoint.get('iou', 0.0)

        return checkpoint['epoch']

    def plot_training_history(self):
        """Plot training history (losses and IoU)"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot loss history
        ax1.plot(self.train_losses, label='Training Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # Plot IoU history
        ax2.plot(self.val_ious, label='Validation IoU', color='green')
        ax2.set_title('Validation IoU')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('IoU')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.config.RESULTS_SAVE_PATH, 'training_history.png'))
        plt.show()

    def train(self, resume=False, checkpoint_path=None):
        """Train the model"""
        start_epoch = 0

        if resume and checkpoint_path:
            start_epoch = self.load_model(checkpoint_path) + 1
            print(f"Resuming training from epoch {start_epoch}")

        print(f"Starting PSPNet model training")
        print(f"Training set size: {len(self.train_loader.dataset)}")
        print(f"Validation set size: {len(self.val_loader.dataset)}")

        for epoch in range(start_epoch, self.config.NUM_EPOCHS):
            start_time = time.time()

            # Train an epoch
            train_loss = self.train_epoch()

            # Validate after epoch
            val_loss, val_iou = self.validate()

            # Update learning rate
            self.scheduler.step()

            # Record history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_ious.append(val_iou)

            # Check if current model is the best
            is_best = val_iou > self.best_iou
            if is_best:
                self.best_iou = val_iou

            # Save model checkpoint
            self.save_model(epoch, val_iou, is_best)

            # Print epoch results
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch + 1}/{self.config.NUM_EPOCHS}")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f}")
            print(f"Learning Rate: {self.scheduler.get_last_lr()[0]:.6f} | Time: {epoch_time:.2f}s")
            print("-" * 60)

        print(f"Training complete! Best IoU: {self.best_iou:.4f}")
        self.plot_training_history()

        return self.best_iou