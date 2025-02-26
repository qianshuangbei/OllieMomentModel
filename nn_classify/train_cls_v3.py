import argparse
import os
from sklearn.model_selection import train_test_split
import sys
import torch
from torch.utils.data import DataLoader
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.dataset import PoseDataset
from model.classifier import PoseClassifierV1
from model.logger import setup_logger

from model.utils import *
from model.train import BaseTrainer

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Pose Classification Training')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint to resume training from')
    parser.add_argument('--config', type=str, default='nn_classify/config.yaml', help='Path to config file')
    parser.add_argument('--version', type=str, default='v1', help='Output version for checkpoints and plots')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create output directory with version
    output_dir = os.path.join('outputs', args.version)
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置日志记录器
    logger = setup_logger(output_dir)
    
    # 记录训练开始和配置信息
    logger.info("="*50)
    logger.info(f"Training started - Version: {args.version}")
    logger.info(f"Config file: {args.config}")
    logger.info(f"Device: {device}")
    logger.info("="*50)
    
    # Save config to output directory
    output_config_path = os.path.join(output_dir, 'config.yaml')
    with open(output_config_path, 'w') as f:
        yaml.dump(config, f)
        logger.info(f"Config saved to {output_config_path}")
    
    # Load and split data
    dataset = PoseDataset('data', 'data/label.txt')
    train_data, val_data = train_test_split(dataset, test_size=0.1, random_state=42)
    
    # Create data loaders using config
    train_loader = DataLoader(train_data, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config['training']['batch_size'])
    
    # Initialize model
    model = PoseClassifierV1(config['training']['num_classes']).to(device)
    
    base_trainer = BaseTrainer(logger)
    # Train model with output directory
    model = base_trainer.train(model, train_loader, val_loader, config, output_dir, checkpoint_path=args.checkpoint)
    
    # Evaluate on validation set
    base_trainer.evaluate(model, val_loader)

if __name__ == '__main__':
    main()
