# download_model.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Use the Simplified Chinese model for better results
model_name = "Helsinki-NLP/opus-mt-zh-en" 
output_dir = "./offline_model" # This folder will contain the model files

print(f"Downloading model '{model_name}' to '{output_dir}'...")

# Download and save the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(output_dir)

# Download and save the model
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.save_pretrained(output_dir)

print("Model downloaded and saved successfully!")