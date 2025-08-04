import torch
import torch.nn.functional as F
import json
import numpy as np
from sentence_transformers import SentenceTransformer,util
from chatbot.classifier import IntentClassifier
import pickle


class ChatbotAssistant:
    def __init__(self,intents_path,model_path,labels_path,model_name,input_size=768,hidden_size=128):
        self.intent_data=self.load_intents(intents_path)
        self.model_name = model_name
        self.embedding_model = SentenceTransformer(model_name)

        # Load intents embeddings and labels/tags
        with open(labels_path, "rb") as f:
            data = pickle.load(f)
            self.intent_embeddings = data["embeddings"]
            self.intent_labels = data["labels"]
            self.classes = data["dimensions"]



        #self.classes = torch.load(labels_path,weights_only=False)
        self.model = IntentClassifier(input_size,hidden_size,len(self.classes)) # Initizlize module
        self.model.load_state_dict(torch.load(model_path))                      # Load the saved pretarined module
        self.model.eval()                                                       # Set the module in evaluation mode

    # Create a function to load data from json
    @staticmethod
    def load_intents(intents_path):
        with open(intents_path) as f:
            return json.load(f)
        
    # Create a function to predict the intents
    def predict_intents(self,sentence,sim_threshold =0.7,clf_threshold=0.6):
        sentence_embedding_np = self.embedding_model.encode(sentence)
        sentence_embedding= torch.tensor(sentence_embedding_np, dtype=torch.float32)     # Perform sentence embeddings
        sentence_tensor = torch.tensor( np.array([sentence_embedding_np]),dtype=torch.float32)             # Create a tensor for embeddings

         # ========== CLASSIFIER BRANCH ==========
        with torch.no_grad():
            output = self.model(sentence_tensor)
            probs = F.softmax(output, dim=1)      # Convert to probabilities
            confidence, prediction = torch.max(probs, dim=1)
            predicted_tag_clf = self.classes[prediction.item()]
        
        # ========== SEMANTIC BRANCH ==========
        cosine_scores = util.cos_sim(sentence_embedding,self.intent_embeddings)[0]
        best_idx = torch.argmax(cosine_scores)
        best_score = cosine_scores[best_idx].item()
        predicted_tag_sim = self.intent_labels[best_idx]

        # Prefer SEMANTIC if it's confident, otherwise use CLASSIFIER
        if best_score >= sim_threshold:
            return predicted_tag_sim
        elif confidence >= clf_threshold:
            return predicted_tag_clf
        else:
            return -1
            
            
        
    # Create a function to return actual response using predicted label/tag
    def get_response(self, tag):
       
        if tag == -1:
            return "I'm not sure I understand. Can you rephrase that?"

        for intent in self.intent_data['intents']:
            if intent['tag'] == tag:
                return np.random.choice(intent['responses'])

        return "Sorry, I don't understand that."
    

    # Create a function to Chat
    def chat(self):
         print("Your Chatbot is ready! Type 'quit' to exit.")

         while True:
             human_message = input("You: ")
             if human_message.lower() in ["quit","exit"]:
                 print("Bot: Goodbye!")
                 break
             tag = self.predict_intents(human_message)
             bot_message = self.get_response(tag)
             print("Bot: ",bot_message)