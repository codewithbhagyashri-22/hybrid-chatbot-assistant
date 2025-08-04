import torch
import pickle
import torch.nn as nn
from .intent_dataset import IntentDataSet
from .classifier import IntentClassifier

# Create a class for traning the model
class ChatbotTrainer:
    def __init__(self,input_size,hidden_size,intents_path,model_name):
        self.dataset = IntentDataSet(intents_path=intents_path,model_name=model_name)
        self.label_encoder = self.dataset.label_encoeder
        self.classes = self.label_encoder.classes_

        self.intent_embeddings = self.dataset.sentences_embeddings
        self.model = IntentClassifier(input_size,hidden_size,len(self.classes))
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.001)

    # Create a function to train the model with prepared dataset
    def train_model(self,epochs = 50):
        loader = self.dataset.get_data_loader()

        for epoch in range(epochs):
            for batch_X, batch_y in loader:
                self.optimizer.zero_grad()      # Set Grads to None to improve performance 
                output = self.model(batch_X)
                loss  = self.criterion(output,batch_y)  # Calculate loss by comparing the predicted ouput with actual value
                loss.backward()                 # Back propagation
                self.optimizer.step()           # Closure that re-evaluates the model and return the loss

            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    # Create a function to save tranied model and tags
    def save_model(self, model_path, labels_path):

        # Save trained model
        torch.save(self.model.state_dict(),model_path)      
        

        # Save intent embeddings and labels
        with open(labels_path, "wb") as f:
            pickle.dump({
                "embeddings": self.intent_embeddings,
                "labels":self.dataset.labels,
                "dimensions": self.classes
                }, f)