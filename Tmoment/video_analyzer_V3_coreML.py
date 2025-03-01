import argparse
import cv2
import os
import sys
import numpy as np
from ultralytics import YOLO
import torch
import torch.nn as nn
import logging
import matplotlib.pyplot as plt
import shutil
import coremltools  # added for CoreML model

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nn_classify.data_generate_v3 import get_person_crops
from Tmoment.cut_analyzer import analyze_video_cuts
from model.classifier import PoseClassifierV1
from Tmoment.postprocessing import apply_smoothing

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
LABEL_size = 5


def process_frame(frame, seg_model, pose_model, max_persons=4, frame_idx=None, isdebug=False):
    """Process a single frame to detect persons and their poses."""
    # Get segmentation results
    seg_results = seg_model(frame)[0]
    pose_features = np.zeros((34, 4))  # Initialize with zeros
    
    i = 0
    img_height, img_width = frame.shape[:2]  # Get image dimensions

    person_crops, orig_coords =get_person_crops(seg_results, frame, img_width, img_height)
    for i, (person_crop, orig_coord) in enumerate(zip(person_crops, orig_coords)):
        # if isdebug:
        #     cv2.imwrite(f'Tmoment/debug_coreML/segment/frame_{frame_idx:04d}_person_{i}.jpg', person_crop)
    
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
            pose_features[kp_idx*2:kp_idx*2+2, i] = [x, y]
            
    if isdebug:      
        cv2.imwrite(f'Tmoment/debug_coreML/pose/frame_{frame_idx:04d}_.jpg', frame)
    return pose_features


def cut_video(video_path, cut_points, output_dir):
    """Cut video into segments based on cut points."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Add 0 as the start point and video duration as the end point
    probe = os.popen(f'ffprobe -v quiet -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{video_path}"')
    duration = float(probe.read())
    cut_times = [0] + cut_points + [duration]
    
    # Cut video into segments
    for i in range(len(cut_times) - 1):
        start_time = cut_times[i]
        end_time = cut_times[i + 1]
        duration = end_time - start_time
        output_path = os.path.join(output_dir, f'segment_{i:03d}.mp4')
        
        # FFmpeg command to cut video without re-encoding
        cmd = f'ffmpeg -y -ss {start_time} -i "{video_path}" -t {duration} -c copy "{output_path}"'
        
        # Execute FFmpeg command
        os.system(cmd)
        logger.info(f"Created segment {i}: {start_time:.2f}s to {end_time:.2f}s")
    
    logger.info(f"Video segments saved to {output_dir}")

def main(target_fps=5, rotate=0, isdebug=False):
    VIDEO_URL = "demo/test1.mp4"
    # Load CoreML classifier model (replace PyTorch model)
    classifier = coremltools.models.MLModel("outputs/iosModel/cls.mlpackage")
    
    # Initialize models with CoreML package paths
    seg_model = YOLO("outputs/iosModel/yolo11sseg.mlpackage")
    pose_model = YOLO("outputs/iosModel/yolo11npose.mlpackage")
    
    # Process video
    all_predictions = []
    all_prod = []    
    all_time_stamp = []
    """Extract frames from video at specified FPS."""
    video_name = os.path.splitext(os.path.basename(VIDEO_URL))[0]
    if os.path.exists('Tmoment/debug_coreML'):
        shutil.rmtree('Tmoment/debug_coreML')
    os.makedirs('Tmoment/debug_coreML/image', exist_ok=True)
    os.makedirs('Tmoment/debug_coreML/segment', exist_ok=True)
    os.makedirs('Tmoment/debug_coreML/pose', exist_ok=True)
    output_prefix = os.path.join('Tmoment/debug_coreML/image', f"{video_name}_")
    
    cap = cv2.VideoCapture(VIDEO_URL)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps / target_fps)
    frame_idx = 0
    
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
            
        # Get pose data for all persons in frame
        pose_data = process_frame(frame, seg_model, pose_model, frame_idx=frame_idx, isdebug=isdebug)
        
        # Prepare input for CoreML classifier
        batch_np = np.array(pose_data, dtype=np.float32)
        input_data = {"input": batch_np[np.newaxis, ...]}  # add batch dimension
        
        # Get predictions from CoreML model
        result = classifier.predict(input_data)
        probs = result["probabilities"]  # assume output key is "probabilities"
        weights = np.array([0.8, 1.2, 0.8, 0.8, 1.0], dtype=np.float32)
        weighted_outputs = probs * weights
        prediction = int(np.argmax(weighted_outputs, axis=1)[0])
        
        # Store predictions
        all_predictions.append(prediction)
        all_prod.append(probs)
        all_time_stamp.append(time_stamp)
    cap.release()
    
    # Convert predictions to numpy array
    all_predictions = np.array(all_predictions)
    all_prod = np.array(all_prod)
    # Apply smoothing to each person's predictions
    smoothed, smoothed_prob = apply_smoothing(all_prod)
    
    # Analyze cuts and get metrics
    cut_results = analyze_video_cuts(smoothed, timestamp=all_time_stamp, min_consecutive=2, cut_label=1)
    
    # Save predictions and metrics to text files
    with open('Tmoment/raw_predictions.txt', 'w') as f:
        f.write("Frame\tPrediction\tSmoothed\tProb0\tProb1\tProb2\n")
        for i, (pred, prod) in enumerate(zip(all_predictions, all_prod)):
            f.write(f"{i}\t{pred[0]}\t{smoothed[i]}\t{prod[0][0]:.3f}\t{prod[0][1]:.3f}\t{prod[0][2]:.3f}\t{prod[0][3]:.3f}\t{prod[0][4]:.3f}\n")
    
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
    labels = ['before start', 'start', 'Moving Through', 'Leaving', 'waiting']
    plt.plot(smoothed, 
            color=colors[0], 
            alpha=0.7)
    
    plt.grid(True, alpha=0.3)
    plt.xlabel('Frame Number')
    plt.ylabel('Action State')
    plt.yticks([0, 1, 2, 3, 5], labels)
    plt.legend()
    plt.title('Action States Over Time (Smoothed)')
    plt.savefig('Tmoment/predictions_plot.png')
    logger.info("Analysis complete. Results saved to Tmoment/predictions.npy and predictions_plot.png")

    # After analyzing cuts and saving results
    output_dir = 'Tmoment/output_coreML'
    cut_video(VIDEO_URL, cut_results['cut_points'], output_dir)
    
    logger.info("Analysis and video cutting complete")

if __name__ == '__main__':
    main(target_fps=1, rotate=-1, isdebug=True)
