import numpy as np
from scipy.signal import medfilt

def apply_smoothing(probs, window_size=3):
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