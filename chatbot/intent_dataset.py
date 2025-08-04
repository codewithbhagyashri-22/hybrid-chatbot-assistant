import torch
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder

# Create a class with Senetence Transformer for senetnce embeddings
class IntentDataSet:
    def __init__(self,intents_path,model_name):
        # Open and Load json data 
        with open(intents_path) as f:
            intents = json.load(f)
        
        self.model = SentenceTransformer(model_name) # Create instance of Sentence Transformer
        self.sentences = []     # List of sentences
        self.labels =[]         # List of tags

        # Add Sentences and tags in list
        for intent in intents['intents']:
            for pattern in intent['patterns']:
                self.sentences.append(pattern)
                self.labels.append(intent['tag'])

        self.sentences_embeddings_np = self.model.encode(self.sentences)
        self.X = np.array(self.sentences_embeddings_np)     # Sentence converted to numbers
        self.sentences_embeddings = torch.tensor(self.sentences_embeddings_np , dtype=torch.float32)
        

        self.label_encoeder = LabelEncoder()                     # Encode target labels with value between 0 and n_classes-1
        self.y = np.array(self.label_encoeder.fit_transform(self.labels)) #Fit label encoder and return encoded labels.


    # Create a function to build efficient training data
    def get_data_loader(self, batch_size = 8):
        X_tensor = torch.tensor(self.X,dtype=torch.float32)
        y_tensor = torch.tensor(self.y,dtype=torch.long)
        dataset = TensorDataset(X_tensor,y_tensor)
        dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
        return dataloader