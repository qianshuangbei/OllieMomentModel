import torch.nn as nn

class PoseClassifierV1(nn.Module):
    def __init__(self, num_classes=3):
        super(PoseClassifierV1, self).__init__()
        self.flatten = nn.Flatten()
        self.features = nn.Sequential(
            nn.Linear(34 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.classifier = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.flatten(x)
        features = self.features(x)
        return self.classifier(features)