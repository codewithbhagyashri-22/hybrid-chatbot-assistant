# 🧠 Hybrid Chatbot Assistant (Intent Classifier + Semantic Search)

This repository contains a proof of concept (PoC) for a chatbot that uses a **hybrid approach** combining **intent classification** (via a neural network classifier) and **semantic search** (via sentence embeddings).

## 🔧 Features

- 🎯 **Intent Classifier** using `torch.nn.Sequential` inside a custom `IntentClassifier` (`nn.Module`).
- 🧬 **Sentence Embedding** generation via [Sentence Transformers](https://www.sbert.net/) in a custom `IntentDataset`.
- 🏋️ **Chatbot Trainer** for training, saving model weights, embeddings, and class labels.
- 🤖 **Chatbot Assistant** combines **classifier + semantic similarity** to respond more accurately to user queries.

---


## 🚀 How It Works

1. **Preprocessing**:
   - Loads intent dataset (e.g. from `intents.json`).
   - Generates embeddings using `sentence-transformers`.
   - Assigns class labels.

2. **Training**:
   - A PyTorch-based feed-forward neural network (`IntentClassifier`) is trained using embeddings and labels.
   - Trained model, class mappings, and sentence embeddings are saved.

3. **Answering Queries**:
   - The chatbot first tries to classify the user’s query.
   - If confidence is low or multiple intents match closely, it performs semantic similarity matching.
   - Based on scores of both results, choose the best response.

---

## 🛠 Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/codewithbhagyashri-22/hybrid-chatbot-poc.git
cd hybrid-chatbot-poc
pip install -r requirements.txt
```

---

## 🧪 Training the Model

To train the intent classifier:

```bash
python train.py
```

---
## 🗂 intents.json Format

The chatbot is trained on a structured `intents.json` file that defines the different user intents and example phrases. Here is a sample format:

```json
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": [
        "Hi",
        "Hello",
        "Hey there",
        "Good morning"
      ],
      "responses": [
        "Hello! How can I assist you today?",
        "Hi there, what can I help you with?"
      ]
    },
    {
      "tag": "goodbye",
      "patterns": [
        "Bye",
        "See you later",
        "Goodbye"
      ],
      "responses": [
        "Goodbye! Have a great day!",
        "See you soon!"
      ]
    }
  ]
}
```

- `tag`: The unique identifier for the intent class.
- `patterns`: Example user queries used to train the intent classifier.
- `responses`: Possible responses the chatbot can give when that intent is detected.

📌 **Note**: You can expand the list of intents and examples to better cover your domain.


---

## 💬 Using the Assistant

To start the chatbot assistant:

```bash
python run_chat.py
```

You can enter your queries in the terminal. The assistant will classify the intent and provide a relevant response.

---

## 🧾 Requirements

```
torch
numpy
sentence-transformers
scikit-learn
python-dotenv
```

---

## 📌 Next Steps (Optional Enhancements)

- Add web/chat UI (e.g., Streamlit, Flask, or React frontend).
- Integrate memory (short-term or per-session).
- Plug into real-world APIs for dynamic responses.
- Extend to multi-language support using multilingual embeddings.

---

## 📄 License

MIT License – feel free to use, modify, and share with attribution.

---

## 🤝 Contributing

Contributions are welcome!  
If you'd like to contribute, please fork the repo and open a pull request.  
For major changes, open an issue first to discuss what you'd like to propose.

---

## 📄 License

This project is licensed under the **MIT License**. Feel free to use, modify, and distribute with attribution.

---

## ✨ Credits

Built with ❤️ using PyTorch and Sentence Transformers.

---

## 💬 Contact

Created by **[Bhagyashri Salgare]**  
GitHub: [github.com/codewithbhagyashri-22](https://github.com/codewithbhagyashri-22)  
LinkedIn: [linkedin.com/in/BhagyashriSalgare](https://www.linkedin.com/in/bhagyashri-salgare-485b5b146/)