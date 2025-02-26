import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader


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
                    pose_data = np.zeros((34, 4))  # 17 keypoints * 2 params(x,y) * 4 groups
                    with open(pose_path, 'r') as pf:
                        lines = pf.readlines()
                        group_idx = 0
                        kpt_idx = 0
                        for line in lines:
                            if line.strip():
                                kpt_id, x, y, conf = line.strip().split('\t')
                                if group_idx < 4:  # Only take first 4 groups
                                    pose_data[kpt_idx*2:kpt_idx*2+2, group_idx] = [float(x), float(y)]
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