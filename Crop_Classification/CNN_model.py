class CNN_model(nn.Module):
    """
    A simple Convolutional Neural Network model:
    - 2 convolutional layers, each followed by batch-normalization, ReLU activation and MaxPool
    - 3 fully connected layer, each followed by dropout and ReLU activation
    - Output: logaritmic softmax (you could also use sigmoid for binary classification, but for multi-class classification problems you should use softmax)
    
    Please note that the following model was implemented for an input image having 64x64x3 dimensions
    More general implementation will be updated soon.
    """
    
        def __init__(self): 
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1),  # (N, 3, 64, 64) -> (N,  16, 62, 62)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (N, 16, 62, 62) -> (N,  16, 31, 31)
            nn.Conv2d(16, 32, 3, 1), # (N,16,31,31) -> (N, 32, 29, 29)
            nn.BatchNorm2d(32), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2) # (N,32,29,29) -> (N,32,14,14); 
            # Please note, the final dimension of the features is 32x14x14, that the reason why nn.Linear has 14x14x32 as its first parameter
            # Please note that N is the number of images in a batch (batch_size defined create datasets function)
        )
        self.classifier = nn.Sequential(
            nn.Linear(14*14*32, 120),         # (N, 14x14x32) -> (N, 120)
            nn.Dropout(0.5), 
            nn.ReLU(),
            nn.Linear(120, 84),         # (N, 120) -> (N, 84)
            nn.Dropout(0.5), 
            nn.ReLU(),
            nn.Linear(84,32),         # (N, 84) -> (N, 32)
            nn.Dropout(0.5), 
            nn.ReLU(),
            nn.Linear(32,2)          # (N, 32) -> (N, 2)
       )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 14*14*32)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)
