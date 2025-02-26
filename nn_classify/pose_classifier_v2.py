import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import logging
import math
import matplotlib.pyplot as plt
import argparse

BATCH_SIZE = 32
LR = 0.001

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=3, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes
        
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

class PoseDataset(Dataset):
    def __init__(self, data_path, label_file):
        self.samples = []
        self.labels = []
        
        # Read label file
        with open(label_file, 'r') as f:
            for line in f:
                pose_path, label = line.strip().split()
                if os.path.exists(pose_path):
                    # Read pose file
                    pose_data = np.zeros((51, 4))  # 17 keypoints * 3 params * 4 groups
                    with open(pose_path, 'r') as pf:
                        lines = pf.readlines()
                        group_idx = 0
                        kpt_idx = 0
                        for line in lines:
                            if line.strip():
                                kpt_id, x, y, conf = line.strip().split('\t')
                                if group_idx < 4:  # Only take first 4 groups
                                    pose_data[kpt_idx*3:kpt_idx*3+3, group_idx] = [float(x), float(y), float(conf)]
                                    kpt_idx += 1
                                    if kpt_idx == 17:
                                        kpt_idx = 0
                                        group_idx += 1
                    
                    self.samples.append(pose_data)
                    self.labels.append(int(label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.samples[idx]), self.labels[idx]

class PoseClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(PoseClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.features = nn.Sequential(
            nn.Linear(51 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.classifier = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.flatten(x)
        features = self.features(x)
        return self.classifier(features)

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def plot_training_curves(train_losses, train_accs, val_losses, val_accs):
    plt.figure(figsize=(12, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('nn_classify/training_curves.png')
    plt.close()

def train(model, train_loader, val_loader, num_epochs=100, device='cuda', checkpoint_path=None):
    # Initialize model from checkpoint if provided
    start_epoch = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Loaded model from epoch {checkpoint['epoch']} with validation accuracy {checkpoint.get('val_acc', 'N/A')}")
    
    criterion = LabelSmoothingLoss(classes=3, smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Warmup steps = 5% of total steps
    num_warmup_steps = int(0.05 * len(train_loader) * num_epochs)
    num_training_steps = len(train_loader) * num_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    
    best_val_acc = 0.0
    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        logger.info(f'Epoch {epoch+1}/{num_epochs}:')
        logger.info(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        logger.info(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        
        # Save model checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, f'nn_classify/checkpoints/model_epoch_{epoch+1}.pth')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
            }, 'nn_classify/checkpoints/model_best.pth')
    
    # Plot training curves
    plot_training_curves(train_losses, train_accs, val_losses, val_accs)
    return model

def evaluate(model, test_loader, device='cuda'):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    acc = 100. * correct / total
    logger.info(f'Test Accuracy: {acc:.2f}%')
    return acc

def main():
    parser = argparse.ArgumentParser(description='Pose Classification Training')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint to resume training from')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create checkpoints directory
    os.makedirs('nn_classify/checkpoints', exist_ok=True)
    
    # Load and split data
    dataset = PoseDataset('data', 'data/label.txt')
    train_data, val_data = train_test_split(dataset, test_size=0.1, random_state=42)
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
    
    # Initialize model
    model = PoseClassifier().to(device)
    
    # Train model
    model = train(model, train_loader, val_loader, checkpoint_path=args.checkpoint)
    
    # Evaluate on validation set
    evaluate(model, val_loader)

if __name__ == '__main__':
    main()
