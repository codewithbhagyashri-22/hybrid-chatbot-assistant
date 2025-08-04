from pathlib import Path
from dotenv import load_dotenv
import os

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from chatbot.chatbot_trainer import ChatbotTrainer

if __name__  == '__main__':
    # Load .env from root
    ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path=ENV_PATH,override=True)

    intents_path = Path(os.getenv("INTENTS_PATH"))
    if not intents_path.is_absolute():
        intents_path = ENV_PATH.parent / intents_path

    model_path = Path(os.getenv("MODEL_PATH"))
    if not model_path.is_absolute():
        model_path = ENV_PATH.parent / model_path

    labels_path = Path(os.getenv("LABELS_PATH"))
    if not labels_path.is_absolute():
        labels_path = ENV_PATH.parent / labels_path

    model_name = str(os.getenv("MODEL_NAME"))

    # To Train and Save Model
    trainer = ChatbotTrainer(
        input_size=768,
        hidden_size=128,
        intents_path=intents_path,
        model_name=model_name
        )
    trainer.train_model(epochs=50)
    trainer.save_model(
        model_path=model_path,
        labels_path=labels_path
        )
    print("✅ Model training complete and saved.")