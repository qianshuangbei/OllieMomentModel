import cv2
import os
import sys
import numpy as np
from ultralytics import YOLO
import torch
import torch.nn as nn
from scipy.signal import medfilt
import logging
import matplotlib.pyplot as plt
sys.path.append('/home/shq/yolo')
from nn_classify.data_generate_v3 import get_person_crops
from Tmoment.cut_analyzer import analyze_video_cuts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def process_frame(frame, seg_model, pose_model, max_persons=4, frame_idx=None, isdebug=False):
    """Process a single frame to detect persons and their poses."""
    # Get segmentation results
    seg_results = seg_model(frame)[0]
    pose_features = np.zeros((51, 4))  # Initialize with zeros
    
    i = 0
    img_height, img_width = frame.shape[:2]  # Get image dimensions

    person_crops, orig_coords =get_person_crops(seg_results, frame, img_width, img_height)
    for i, (person_crop, orig_coord) in enumerate(zip(person_crops, orig_coords)):
        if isdebug:
            cv2.imwrite(f'Tmoment/debug/segment/frame_{frame_idx:04d}_person_{i}.jpg', person_crop)
    
        # Get pose for this person
        pose_results = pose_model(person_crop)[0]
        keypoints = pose_results.keypoints.data[0]  # Get first (and should be only) pose
        posed_crop = pose_results.plot()
    
        frame[orig_coord[1]:orig_coord[3], orig_coord[0]:orig_coord[2]] = posed_crop
    
        # Fill in available keypoints
        for kp_idx, kp in enumerate(keypoints):
            x = float(kp[0].cpu().numpy())
            y = float(kp[1].cpu().numpy())
            conf = float(kp[2].cpu().numpy())
            pose_features[kp_idx*3:kp_idx*3+3, i] = [x, y, conf]
            
        if isdebug:      
            cv2.imwrite(f'Tmoment/debug/pose/frame_{frame_idx:04d}_person_{i}.jpg', frame)
    return pose_features

def apply_smoothing(probs, window_size=5):
    """
    Apply median filter smoothing to predictions and probabilities.
    
    Args:
        predictions: numpy array of shape (n_frames,) containing class predictions
        probs: numpy array of shape (n_frames, n_classes) containing class probabilities
        window_size: size of the median filter window
    
    Returns:
        tuple: (smoothed predictions, smoothed probabilities)
    """
    # Smooth probabilities first (shape: n_frames x n_classes)
    smoothed_probs = np.apply_along_axis(lambda x: medfilt(x, window_size), axis=0, arr=probs[:, 0, :])
    
    # Get predictions from smoothed probabilities
    smoothed_preds = np.argmax(smoothed_probs, axis=1)
    
    return smoothed_preds, smoothed_probs



def main(target_fps=5, rotate=0, isdebug=False):
    # Initialize models
    seg_model = YOLO("yolo11s-seg.pt")
    pose_model = YOLO("yolo11n-pose.pt")
    
    # Load classifier
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier = PoseClassifier().to(device)
    checkpoint = torch.load('/data/shq/yolo_data/V2/checkpoints/model_best.pth', map_location=device)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.eval()
    
    # Process video
    video_path = 'demo/test1.mp4'    
    all_predictions = []
    all_prod = []    
    all_time_stamp = []
    """Extract frames from video at specified FPS."""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps / target_fps)
    frame_idx = 0
    output_prefix = os.path.join('Tmoment/debug/image', f"{video_name}_")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        if frame_idx % interval != 0:
            continue
        
        time_stamp = frame_idx / fps
        # frame = cv2.resize(frame, (720, 1080), interpolation=cv2.INTER_AREA)
        output_path = output_prefix + f"{ frame_idx :06d}.jpg"
        if rotate == 1:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotate == -1:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        all_time_stamp.append(time_stamp)
        if isdebug:
            cv2.imwrite(output_path, frame)
        
        # Process each frame
        if frame_idx % 100 == 0:
            logger.info(f"Processing frame {frame_idx}")
            
        # Get pose data for all persons in frame
        pose_data = process_frame(frame, seg_model, pose_model, frame_idx=frame_idx, isdebug=isdebug)
        
        # Prepare batch for classifier
        batch = torch.FloatTensor(pose_data).to(device)
        # Get predictions
        with torch.no_grad():
            outputs = classifier(batch.unsqueeze(0))
            probs = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probs, dim=1)
        
        # Store predictions
        all_predictions.append(predictions.cpu().numpy())
        all_prod.append(probs.cpu().numpy())
    cap.release()
    
    # Convert predictions to numpy array
    all_predictions = np.array(all_predictions)
    all_prod = np.array(all_prod)
    # Apply smoothing to each person's predictions
    smoothed, smoothed_prob = apply_smoothing(all_prod)
    
    # Analyze cuts and get metrics
    cut_results = analyze_video_cuts(smoothed, target_fps, min_consecutive=5)
    
    # Save predictions and metrics to text files
    with open('Tmoment/raw_predictions.txt', 'w') as f:
        f.write("Frame\tPrediction\tSmoothed\tProb0\tProb1\tProb2\n")
        for i, (pred, prod) in enumerate(zip(all_predictions, all_prod)):
            f.write(f"{i}\t{pred[0]}\t{smoothed[i]}\t{prod[0][0]:.3f}\t{prod[0][1]:.3f}\t{prod[0][2]:.3f}\n")
    
    # Save cut analysis results
    with open('Tmoment/cut_analysis.txt', 'w') as f:
        f.write(f"Precision: {cut_results['precision']:.3f}\n")
        f.write(f"Recall: {cut_results['recall']:.3f}\n")
        f.write(f"F1 Score: {cut_results['f1']:.3f}\n")
        f.write("\nCut Points (timestamps):\n")
        for cp in cut_results['cut_points']:
            f.write(f"{cp:.2f}\n")
    
    logger.info(f"Cut Analysis Results:")
    logger.info(f"Precision: {cut_results['precision']:.3f}")
    logger.info(f"Recall: {cut_results['recall']:.3f}")
    logger.info(f"F1 Score: {cut_results['f1']:.3f}")
    # Plot results
    plt.figure(figsize=(15, 8))
    colors = ['b', 'g', 'r', 'c']
    labels = ['Standing/Waiting', 'Moving Through', 'Leaving']
    plt.plot(smoothed, 
            color=colors[0], 
            alpha=0.7)
    
    plt.grid(True, alpha=0.3)
    plt.xlabel('Frame Number')
    plt.ylabel('Action State')
    plt.yticks([0, 1, 2], labels)
    plt.legend()
    plt.title('Action States Over Time (Smoothed)')
    plt.savefig('Tmoment/predictions_plot.png')
    logger.info("Analysis complete. Results saved to Tmoment/predictions.npy and predictions_plot.png")
    
    
    

if __name__ == '__main__':
    main(target_fps=5, rotate=-1, isdebug=False)
