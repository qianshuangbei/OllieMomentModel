import cv2
import os
from glob import glob
from tqdm import tqdm
from ultralytics import YOLO

# Set parameters
VIDEO_DIR = 'video'
IMAGE_DIR = 'image'
PREDICT_DIR = 'predict'
LABEL_DIR = 'data/train/label.txt'
TARGET_FPS = 10

# Create output directory if it doesn't exist
os.makedirs(IMAGE_DIR, exist_ok=True)

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

            if os.path.exists(output_path):
                print(f"skip: {video_path}")
                return 0, output_prefix
            
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
    
    start_time = int(1.3 * TARGET_FPS)
    end_time =int( 1.3 * TARGET_FPS)
    
    # Process each video
    for video_path in tqdm(video_files):
        print(f"\nProcessing {video_path}...")
        frames_saved, prefix = extract_frames(video_path, output_dir)
        
        if frames_saved == 0:
            continue
        
        print(f"Saved {frames_saved} frames")

        for i in range(start_time):
            wf.write(f"{prefix}{i:06d}.jpg\t0\n")
            
        for i in range(start_time, frames_saved - end_time):
            wf.write(f"{prefix}{i:06d}.jpg\t1\n")
            
        for i in range(frames_saved - end_time, frames_saved):
            wf.write(f"{prefix}{i:06d}.jpg\t2\n")
    
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

def generate_pose_label(imgs_path, output_dir):
    seg_model = YOLO("yolo11n-seg.pt")  # load segmentation model
    pose_model = YOLO("yolo11n-pose.pt")  # load pose model

    imgs = glob(imgs_path)
    
    print(f"Found {len(imgs)} video files")

    # Process each video
    for img_path in tqdm(imgs):
        print(f"\nProcessing {img_path}...")
    
        # Input image path
        img = cv2.imread(img_path)

        # First get segmentation results
        seg_results = seg_model(img_path)[0]
        orig_filename = os.path.basename(img_path)
        result_file = os.path.join('data', 'train', f'{orig_filename}_pose.txt')

        if os.path.exists(result_file):
            print(f"skip: {img_path}")
            continue
        # Create a list to store individual person crops
        person_crops = []
        orig_coords = []

        # Extract each person using segmentation masks
        for i, mask in enumerate(seg_results.masks.data):
            if seg_results.boxes.cls[i] == 0:  # class 0 is person in COCO
                # Convert mask to numpy array
                mask = mask.cpu().numpy()
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, seg_results.boxes.xyxy[i])
                orig_coords.append((x1, y1, x2, y2))
                
                # Create person crop
                person_crop = img[y1:y2, x1:x2].copy()
                person_crops.append(person_crop)
            
            if i > 3:
                break

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

if __name__ == '__main__':
    video_path = os.path.join(VIDEO_DIR, '*.mp4')
    image_path = os.path.join(IMAGE_DIR, '*.jpg')
    # extract frames from video
    process_video(video_path, IMAGE_DIR, LABEL_DIR)
    
    # generate pose label for each image
    generate_pose_label(image_path, PREDICT_DIR)