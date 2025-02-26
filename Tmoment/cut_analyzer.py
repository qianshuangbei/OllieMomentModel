import numpy as np

def find_cut_points(smoothed_preds, min_consecutive=3, cut_label=0):
    """Find cut points where there are consecutive '1's."""
    cut_points = []
    count = 0
    start_idx = -1
    
    for i, pred in enumerate(smoothed_preds):
        if pred == cut_label:
            if count == 0:
                start_idx = i
            count += 1
        else:
            if count >= min_consecutive:
                mid_point = start_idx + count // 2
                cut_points.append(mid_point)
            count = 0
            start_idx = -1
    
    if count >= min_consecutive:
        mid_point = start_idx + count // 2
        cut_points.append(mid_point)
    
    return np.array(cut_points)

def frame_to_timestamp(frame_idx, target_fps, original_fps):
    """Convert frame index to timestamp in seconds."""
    interval = int(original_fps / target_fps)
    original_frame = frame_idx * interval
    return original_frame / original_fps

def load_ground_truth(label_file):
    """Load ground truth intervals from label file."""
    intervals = []
    with open(label_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            start, end = map(float, line.strip().split())
            intervals.append((start, end))
    return intervals

def is_point_in_any_interval(point, intervals, tolerance=0.5):
    """Check if a point falls within any of the intervals."""
    for start, end in intervals:
        if start - tolerance <= point <= end + tolerance:
            return True
    return False

def evaluate_cuts(predicted_timestamps, ground_truth_intervals):
    """Calculate precision and recall for cut points."""
    true_positives = sum(1 for pt in predicted_timestamps 
                        if is_point_in_any_interval(pt, ground_truth_intervals))
    false_positives = len(predicted_timestamps) - true_positives
    false_negatives = sum(1 for start, end in ground_truth_intervals 
                         if not any(start <= pt <= end for pt in predicted_timestamps))
    
    precision = true_positives / len(predicted_timestamps) if predicted_timestamps else 0
    recall = true_positives / len(ground_truth_intervals) if ground_truth_intervals else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'cut_points': predicted_timestamps
    }

def analyze_video_cuts(smoothed_predictions, timestamp, label_file='demo/text1.mp4.label', cut_label=0, min_consecutive=3):
    """Analyze video cuts and evaluate against ground truth."""
    # Find cut points
    cut_points = find_cut_points(smoothed_predictions, min_consecutive, cut_label)
    
    # Convert to timestamps
    cut_timestamps = [timestamp[cp] for cp in cut_points]
    
    # Load ground truth and evaluate
    ground_truth = load_ground_truth(label_file)
    results = evaluate_cuts(cut_timestamps, ground_truth)
    
    return results
