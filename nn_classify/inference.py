import torch
from nn_classify.pose_classifier_v2 import PoseClassifier, PoseDataset
from torch.utils.data import DataLoader
import argparse
import logging
import numpy as np
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_path, device):
    model = PoseClassifier().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded model from epoch {checkpoint['epoch']} with validation accuracy {checkpoint.get('val_acc', 'N/A')}")
    return model

def predict_single_pose(model, pose_file, device):
    model.eval()
    
    # Read and process pose file
    pose_data = np.zeros((51, 4))
    try:
        with open(pose_file, 'r') as f:
            lines = f.readlines()
            group_idx = 0
            kpt_idx = 0
            for line in lines:
                if line.strip():
                    kpt_id, x, y, conf = line.strip().split('\t')
                    if group_idx < 4:
                        pose_data[kpt_idx*3:kpt_idx*3+3, group_idx] = [float(x), float(y), float(conf)]
                        kpt_idx += 1
                        if kpt_idx == 17:
                            kpt_idx = 0
                            group_idx += 1
    except Exception as e:
        logger.error(f"Error reading pose file: {e}")
        return None, 0.0
    
    # Make prediction
    with torch.no_grad():
        input_tensor = torch.FloatTensor(pose_data).unsqueeze(0).to(device)
        output = model(input_tensor)
        prob = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(output, dim=1).item()
        confidence = prob[0][predicted_class].item()
    
    return predicted_class, confidence

def evaluate_dataset(model, data_path, label_file, batch_size=16, device='cuda'):
    dataset = PoseDataset(data_path, label_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * 3
    class_total = [0] * 3
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Per-class accuracy
            for i in range(len(targets)):
                label = targets[i].item()
                class_total[label] += 1
                if predicted[i].item() == label:
                    class_correct[label] += 1
    
    # Print results
    logger.info(f"Overall Accuracy: {100.0 * correct / total:.2f}%")
    for i in range(3):
        if class_total[i] > 0:
            logger.info(f"Class {i} Accuracy: {100.0 * class_correct[i] / class_total[i]:.2f}%")

def main():
    parser = argparse.ArgumentParser(description='Pose Classification Inference')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--mode', choices=['single', 'dataset'], required=True, 
                      help='Inference mode: single pose file or entire dataset')
    parser.add_argument('--input', type=str, required=True, 
                      help='Path to pose file (single mode) or data directory (dataset mode)')
    parser.add_argument('--label_file', type=str, help='Path to label file (required for dataset mode)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model, device)
    
    if args.mode == 'single':
        if not os.path.exists(args.input):
            logger.error(f"Pose file not found: {args.input}")
            return
            
        predicted_class, confidence = predict_single_pose(model, args.input, device)
        logger.info(f"Predicted class: {predicted_class}")
        logger.info(f"Confidence: {confidence:.4f}")
        
    else:  # dataset mode
        if not args.label_file:
            logger.error("Label file is required for dataset mode")
            return
            
        evaluate_dataset(model, args.input, args.label_file, device=device)

if __name__ == '__main__':
    main()
