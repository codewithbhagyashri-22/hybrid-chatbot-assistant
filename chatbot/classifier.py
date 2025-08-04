import torch.nn as nn

# Create a class for intent classification with neural network's custom architecture
class IntentClassifier(nn.Module):
    def __init__(self, input_size,hidden_size,output_size):
        super(IntentClassifier,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,output_size)
        )

    def forward(self,x):
        return self.model(x)