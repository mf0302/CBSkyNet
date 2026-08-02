import os
import time
import json
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional, List
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    classification_report
)
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import ExperimentConfig


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(log_path: str, name: str = "train") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))

    logger.addHandler(fh)
    return logger


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


class WarmupScheduler:
    def __init__(self, optimizer, warmup_epochs: int, base_lr: float):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.current_epoch = 0

    def step(self, epoch: int):
        self.current_epoch = epoch
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

    def get_last_lr(self) -> List[float]:
        return [pg['lr'] for pg in self.optimizer.param_groups]


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: ExperimentConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: Optional[torch.Tensor] = None,
        log_dir: Optional[str] = None
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(config.train.device if torch.cuda.is_available() else "cpu")

        self.model = self.model.to(self.device)

        if config.train.use_class_weights and class_weights is not None:
            class_weights = class_weights.to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay
        )

        self.warmup_scheduler = WarmupScheduler(
            self.optimizer,
            config.train.warmup_epochs,
            config.train.learning_rate
        )

        if config.train.lr_scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config.train.num_epochs - config.train.warmup_epochs,
                eta_min=1e-7
            )
        elif config.train.lr_scheduler == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5
            )
        else:
            self.scheduler = StepLR(self.optimizer, step_size=30, gamma=0.1)

        self.early_stopping = EarlyStopping(
            patience=config.train.patience,
            min_delta=config.train.min_delta,
            mode='max'
        )

        self.use_amp = self.device.type == 'cuda'
        self.scaler = GradScaler(enabled=self.use_amp)

        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            self.logger = setup_logger(os.path.join(log_dir, "train_log.txt"))
        else:
            self.logger = None

        self.history = {
            'train_loss': [], 'train_acc': [], 'train_f1': [],
            'val_loss': [], 'val_acc': [], 'val_f1': [],
            'learning_rate': []
        }

        self.best_val_f1 = 0
        self.best_model_state = None

    def _log(self, msg: str):
        print(msg)
        if self.logger:
            self.logger.info(msg)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()

        total_loss = 0
        all_preds = []
        all_labels = []

        pbar = tqdm(self.train_loader, desc=f"Train Epoch {epoch}")

        for batch_idx, (features, labels) in enumerate(pbar):
            features = features.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            with autocast(enabled=self.use_amp):
                outputs = self.model(features)
                logits = outputs['logits']
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()

            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='binary')

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1': f1
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()

        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []

        for features, labels in tqdm(self.val_loader, desc="Validating"):
            features = features.to(self.device)
            labels = labels.to(self.device)

            with autocast(enabled=self.use_amp):
                outputs = self.model(features)
                logits = outputs['logits']
                loss = self.criterion(logits, labels)

            total_loss += loss.item()
            probs = F.softmax(logits.float(), dim=1)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)

        try:
            auc = roc_auc_score(all_labels, all_probs)
            ap = average_precision_score(all_labels, all_probs)
        except:
            auc = 0
            ap = 0

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'ap': ap,
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs
        }

    def train(self) -> Dict[str, List[float]]:
        self._log(f"\n{'='*60}")
        self._log(f"Training: {self.config.exp_name}")
        self._log(f"Feature type: {self.config.feature_type}")
        self._log(f"Device: {self.device}")
        self._log(f"AMP (mixed precision): {'ON' if self.use_amp else 'OFF'}")
        self._log(f"{'='*60}\n")

        for epoch in range(1, self.config.train.num_epochs + 1):
            epoch_start = time.time()

            if epoch <= self.config.train.warmup_epochs:
                self.warmup_scheduler.step(epoch - 1)

            train_metrics = self.train_epoch(epoch)

            val_metrics = self.validate()

            current_lr = self.optimizer.param_groups[0]['lr']
            if epoch > self.config.train.warmup_epochs:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()

            epoch_time = time.time() - epoch_start

            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['train_f1'].append(train_metrics['f1'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['learning_rate'].append(current_lr)

            gap_loss = abs(train_metrics['loss'] - val_metrics['loss'])
            gap_acc = abs(train_metrics['accuracy'] - val_metrics['accuracy'])
            gap_f1 = abs(train_metrics['f1'] - val_metrics['f1'])

            is_best = val_metrics['f1'] > self.best_val_f1
            if is_best:
                self.best_val_f1 = val_metrics['f1']
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            self._log(f"\nEpoch {epoch}/{self.config.train.num_epochs} ({epoch_time:.1f}s)")
            self._log(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}")
            self._log(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")
            self._log(f"  Val   - Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, AUC: {val_metrics['auc']:.4f}")
            self._log(f"  LR: {current_lr:.6f}")
            self._log(f"  Gap   - Loss: {gap_loss:.4f}, Acc: {gap_acc:.4f}, F1: {gap_f1:.4f}")
            if is_best:
                self._log(f"  ★ New best model! Val F1: {self.best_val_f1:.4f}")

            if self.early_stopping(val_metrics['f1']):
                self._log(f"\nEarly stopping at epoch {epoch}")
                break

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        self._log(f"\nTraining finished. Best Val F1: {self.best_val_f1:.4f}")
        return self.history

    def save_checkpoint(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history,
            'best_val_f1': self.best_val_f1
        }, path)

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        self.best_val_f1 = checkpoint['best_val_f1']


@torch.no_grad()
def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    save_path: Optional[str] = None
) -> Dict[str, float]:
    model.eval()
    use_amp = device.type == 'cuda'

    all_preds = []
    all_labels = []
    all_probs = []

    for features, labels in tqdm(test_loader, desc="Testing"):
        features = features.to(device)

        with autocast(enabled=use_amp):
            outputs = model(features)
            logits = outputs['logits']

        probs = F.softmax(logits.float(), dim=1)
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)

    try:
        auc = roc_auc_score(all_labels, all_probs)
        ap = average_precision_score(all_labels, all_probs)
    except:
        auc = 0
        ap = 0

    cm = confusion_matrix(all_labels, all_preds)

    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'ap': ap,
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(all_labels, all_preds, target_names=['Non-Gibbon', 'Gibbon'])
    }

    print("\n" + "="*60)
    print("Test Results")
    print("="*60)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")
    print(f"AP:        {ap:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(results['classification_report'])

    if save_path:
        with open(save_path, 'w') as f:
            json.dump({k: v for k, v in results.items() if k != 'classification_report'}, f, indent=2)
        print(f"\nResults saved to {save_path}")

    return results


def plot_training_history(history: Dict[str, List[float]], save_path: str):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epochs = range(1, len(history['train_loss']) + 1)

    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True)

    ax = axes[0, 1]
    ax.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    ax.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training and Validation Accuracy')
    ax.legend()
    ax.grid(True)

    ax = axes[1, 0]
    ax.plot(epochs, history['train_f1'], 'b-', label='Train F1')
    ax.plot(epochs, history['val_f1'], 'r-', label='Val F1')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1 Score')
    ax.set_title('Training and Validation F1 Score')
    ax.legend()
    ax.grid(True)

    ax = axes[1, 1]
    ax.plot(epochs, history['learning_rate'], 'g-')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True)
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training history plot saved to {save_path}")


def plot_comparison(results: Dict[str, Dict], save_path: str):
    features = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(features))
    width = 0.15

    ax = axes[0]
    for i, metric in enumerate(metrics):
        values = [results[f]['test'][metric] for f in features]
        ax.bar(x + i * width, values, width, label=metric.upper())

    ax.set_xlabel('Feature Type')
    ax.set_ylabel('Score')
    ax.set_title('Test Metrics Comparison')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([f.upper() for f in features])
    ax.legend()
    ax.grid(True, axis='y')

    ax = axes[1]
    train_f1s = [results[f]['history']['train_f1'][-1] for f in features]
    val_f1s = [results[f]['history']['val_f1'][-1] for f in features]

    x = np.arange(len(features))
    width = 0.35

    ax.bar(x - width/2, train_f1s, width, label='Train F1')
    ax.bar(x + width/2, val_f1s, width, label='Val F1')

    ax.set_xlabel('Feature Type')
    ax.set_ylabel('F1 Score')
    ax.set_title('Train vs Validation F1 (Overfitting Check)')
    ax.set_xticks(x)
    ax.set_xticklabels([f.upper() for f in features])
    ax.legend()
    ax.grid(True, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Comparison plot saved to {save_path}")
