import cv2
import os
import numpy as np
from glob import glob
from tqdm import tqdm
from ultralytics import YOLO
import random

# Set parameters
VIDEO_DIR = 'video'
VIDEO_DIR_2 = 'video_long'
IMAGE_DIR = 'image'
PREDICT_DIR = 'predict'
LABEL_DIR = 'data/label.txt'
TARGET_FPS = 5

# Create output directories if they don't exist
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LABEL_DIR), exist_ok=True)
os.makedirs(PREDICT_DIR, exist_ok=True)  # For pose result files
os.makedirs('data/train', exist_ok=True)  # For pose result files

# Clear the label file at the start
with open(LABEL_DIR, 'w') as f:
    pass  # Create empty file

def apply_augmentation(image_path, output_prefix, label, wf, aug_count=5):
    """
    Apply data augmentation to an image and save the augmented versions
    Returns: Number of augmented images created
    """
    img = cv2.imread(image_path)
    if img is None:
        return 0
        
    h, w = img.shape[:2]
    center = (w/2, h/2)
    
    # Define augmentation parameters
    rotations = [-5, -2.5, 0, 2.5, 5]  # 6 rotation angles
    scales = [1.0, 1.05, 1.1, 1.15]   # 5 scale factors
    
    count = 0
    # Apply rotations and scales
    for i in range(aug_count):
        angle = random.choice(rotations)
        scale = random.choice(scales)
        # Calculate the rotation matrix
        rot_mat = cv2.getRotationMatrix2D(center, angle, scale)
        
        # Apply the transformation
        augmented = cv2.warpAffine(img, rot_mat, (w, h), flags=cv2.INTER_LINEAR)
        
        # Save augmented image
        aug_path = f"{output_prefix}_aug_{count:02d}.jpg"
        cv2.imwrite(aug_path, augmented)
        
        aug_path_new = aug_path.replace('image/', 'data/train/').replace('.jpg', '.jpg_pose.txt')
        # Write label for augmented image
        wf.write(f"{aug_path_new}\t{label}\n")
        
        count += 1
            
    return count

def extract_frames(video_path, output_dir):
    # Get video name without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Read video
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame interval to achieve target FPS
    interval = int(fps / TARGET_FPS)
    
    frame_idx = 0
    saved_count = 0
    
    output_prefix = os.path.join(output_dir, f"{video_name}_")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % interval == 0:
            # Save frame
            output_path = output_prefix + f"{saved_count:06d}.jpg"
            
            cv2.imwrite(output_path, frame)
            saved_count += 1
            
        frame_idx += 1
    
    cap.release()
    return saved_count, output_prefix

def process_video(video_path, output_dir, label_dir):
    # Get list of video files
    video_files = glob(video_path)
    print(f"Found {len(video_files)} video files")

    wf = open(label_dir, 'a')
    
    start_time_0 = int(1.0 * TARGET_FPS)
    start_time_1 = int(1.7 * TARGET_FPS)
    if 'long' in video_path:
        end_time_2 = int(3.0 * TARGET_FPS)
        end_time_1 = int(2.0 * TARGET_FPS)
    else:
        end_time_2 = int(2.0 * TARGET_FPS)
        end_time_1 = int(1.0 * TARGET_FPS)
        
    # Process each video
    for video_path in tqdm(video_files):
        print(f"\nProcessing {video_path}...")
        frames_saved, prefix = extract_frames(video_path, output_dir)
        
        print(f"Saved {frames_saved} frames")

        # Process initial frames (label 0)
        for i in range(start_time_0):
            frame_path = f"{prefix}{i:06d}.jpg"
            apply_augmentation(frame_path, frame_path[:-4], 0, wf)
            frame_path_new = frame_path.replace('image/', 'data/train/').replace('.jpg', '.jpg_pose.txt')
            wf.write(f"{frame_path_new}\t0\n")
            
        # Process middle frames (label 1)
        for i in range(start_time_0, start_time_1):
            frame_path = f"{prefix}{i:06d}.jpg"
            apply_augmentation(frame_path, frame_path[:-4], 1, wf)
            frame_path_new = frame_path.replace('image/', 'data/train/').replace('.jpg', '.jpg_pose.txt')
            wf.write(f"{frame_path_new}\t1\n")
            
        # Process middle frames (label 2)
        for i in range(start_time_1, frames_saved - end_time_2):
            frame_path = f"{prefix}{i:06d}.jpg"
            apply_augmentation(frame_path, frame_path[:-4], 2, wf)
            frame_path_new = frame_path.replace('image/', 'data/train/').replace('.jpg', '.jpg_pose.txt')
            wf.write(f"{frame_path_new}\t2\n")
            
        # Process end frames (label 3)
        for i in range(frames_saved - end_time_2, frames_saved-end_time_1):
            frame_path = f"{prefix}{i:06d}.jpg"
            apply_augmentation(frame_path, frame_path[:-4], 3, wf)
            frame_path_new = frame_path.replace('image/', 'data/train/').replace('.jpg', '.jpg_pose.txt')
            wf.write(f"{frame_path_new}\t3\n")
            
        # Process end frames (label 2)
        for i in range(frames_saved - end_time_1, frames_saved):
            frame_path = f"{prefix}{i:06d}.jpg"
            apply_augmentation(frame_path, frame_path[:-4], 4, wf)
            frame_path_new = frame_path.replace('image/', 'data/train/').replace('.jpg', '.jpg_pose.txt')
            wf.write(f"{frame_path_new}\t4\n")

    wf.close()

def parse_pose_result(result, result_file):
    for r in result:
    # Save keypoints information for each person
        if r.keypoints is None:
            continue
        keypoints = r.keypoints.data
        for person_idx, person_keypoints in enumerate(keypoints):
            # Create text file for each person
         
            with open(result_file, 'a') as f:
                # Each keypoint contains [x, y, confidence]
                for kp_idx, kp in enumerate(person_keypoints):
                    x = float(kp[0].cpu().numpy())
                    y = float(kp[1].cpu().numpy())
                    conf = float(kp[2].cpu().numpy())
                    f.write(f"{kp_idx}\t{x:.2f}\t{y:.2f}\t{conf:.2f}\n")

def get_person_crops(seg_results, img, img_width, img_height):
    orig_coords = []
    person_crops = []
    # Extract each person using segmentation masks
    for i, mask in enumerate(seg_results.masks.data):
        if seg_results.boxes.cls[i] == 0:  # class 0 is person in COCO
            mask = mask.cpu().numpy()
            
            # Get original bounding box coordinates
            x1, y1, x2, y2 = map(int, seg_results.boxes.xyxy[i])
            
            # Calculate box enlargement (20%)
            width = x2 - x1
            height = y2 - y1
            dx = int(width * 0.1)  # 10% each side
            dy = int(height * 0.1)  # 10% each side
            
            # Apply enlargement with boundary checking
            new_x1 = max(0, x1 - dx)
            new_y1 = max(0, y1 - dy)
            new_x2 = min(img_width, x2 + dx)
            new_y2 = min(img_height, y2 + dy)
            
            # Store enlarged coordinates
            orig_coords.append((new_x1, new_y1, new_x2, new_y2))
            
            # Create enlarged person crop
            person_crop = img[new_y1:new_y2, new_x1:new_x2].copy()
            person_crops.append(person_crop)
        
        if len(person_crops) == 4:
            break
    
    return person_crops, orig_coords

def generate_pose_label(imgs_path, output_dir):
    seg_model = YOLO("yolo11s-seg.pt")
    enhance_seg_model = YOLO("yolo11x-seg.pt")
    pose_model = YOLO("yolo11n-pose.pt")
    
    imgs = glob(imgs_path)
    print(f"Found {len(imgs)} video files")
    diff_size = 0
    for img_path in tqdm(imgs):
        print(f"\nProcessing {img_path}...")
        img = cv2.imread(img_path)
        img_height, img_width = img.shape[:2]  # Get image dimensions
        
        seg_results = seg_model(img_path, conf=0.3)[0]
        orig_filename = os.path.basename(img_path)
        result_file = os.path.join('data', 'train', f'{orig_filename}_pose.txt')

        if os.path.exists(result_file):
            print(f"skip: {img_path}")
            continue

        person_crops, orig_coords = get_person_crops(seg_results, img, img_width, img_height)
            
        if len(person_crops) < 4:
            seg_results = enhance_seg_model(img_path, conf=0.3)[0]
            person_crops_size = len(person_crops)
            person_crops, orig_coords = get_person_crops(seg_results, img, img_width, img_height)
            diff_size += len(person_crops) - person_crops_size

        # Process each person crop with pose detection
        final_image = img.copy()
        for i, (crop, (x1, y1, x2, y2)) in enumerate(zip(person_crops, orig_coords)):
            # Predict pose on the crop
            pose_results = pose_model(crop)[0]
            parse_pose_result(pose_results, result_file)
            
            # Draw pose on the crop
            posed_crop = pose_results.plot()
            
            # Place the posed crop back in the original image
            final_image[y1:y2, x1:x2] = posed_crop

        # Save and display the final result
        save_path = os.path.join(output_dir, f'pose_seg_{orig_filename}')
        cv2.imwrite(save_path, final_image)
    print(f"Final increased person box by largest model: {diff_size}")

if __name__ == '__main__':
    video_path = os.path.join(VIDEO_DIR, '*.*4')
    video_path_long = os.path.join(VIDEO_DIR_2, '*.*4')
    image_path = os.path.join(IMAGE_DIR, '*.jpg')
    # extract frames from video
    process_video(video_path, IMAGE_DIR, LABEL_DIR)
    process_video(video_path_long, IMAGE_DIR, LABEL_DIR)
    
    # generate pose label for each image
    # generate_pose_label(image_path, PREDICT_DIR)
