import os
import torch.optim as optim
from .loss import *
from .utils import *

class BaseTrainer:
    def __init__(self, logger):
        self.logger = logger

    def train(self, model, train_loader, val_loader, config, output_dir, device='cuda', checkpoint_path=None):
        # Initialize model from checkpoint if provided
        start_epoch = 0
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            self.logger.info(f"Loaded model from epoch {checkpoint['epoch']} with validation accuracy {checkpoint.get('val_acc', 'N/A')}")
        
        # 使用配置文件中的损失函数设置
        criterion = MixedLoss(config)
        optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
        
        # Warmup steps = 5% of total steps
        num_warmup_steps = int(0.05 * len(train_loader) * config['training']['num_epochs'])
        num_training_steps = len(train_loader) * config['training']['num_epochs']
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
        
        best_val_acc = 0.0
        train_losses, train_accs, val_losses, val_accs = [], [], [], []
        
        checkpoint_dir = os.path.join(output_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 记录训练参数
        self.logger.info("Training parameters:")
        self.logger.info(f"Batch size: {config['training']['batch_size']}")
        self.logger.info(f"Learning rate: {config['training']['learning_rate']}")
        self.logger.info(f"Number of epochs: {config['training']['num_epochs']}")
        self.logger.info(f"Loss weights - Focal: {config['loss']['loss_weights']['focal']}, Smoothing: {config['loss']['loss_weights']['smoothing']}")
        
        for epoch in range(start_epoch, config['training']['num_epochs']):
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
            
            self.logger.info(f'Epoch {epoch+1}:')
            self.logger.info(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            self.logger.info(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
            
            # Save model checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pth'))
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_acc': val_acc,
                }, os.path.join(checkpoint_dir, 'model_best.pth'))
        
        # Plot training curves
        plot_training_curves(train_losses, train_accs, val_losses, val_accs, output_dir)
        return model

    def evaluate(self, model, test_loader, device='cuda'):
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
        self.logger.info(f'Test Accuracy: {acc:.2f}%')
        return acc
