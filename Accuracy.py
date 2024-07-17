import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch

# Load your trained model
model_path = '/content/drive/My Drive/Colab Notebooks/steam_reviews_model'
model = DistilBertForSequenceClassification.from_pretrained(model_path)

# Load the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Define your validation data and labels
# For this example, I'm using some dummy data. Replace this with your actual validation data.
validation_data = [
    "This is a great game!", 
    "I didn't like this game much.", 
    "It's an okay game."
]
validation_labels = [1, 0, 1]  # Corresponding labels (e.g., 1 for positive, 0 for negative)

# Tokenize the validation data
validation_encodings = tokenizer(validation_data, truncation=True, padding=True, max_length=64)

# Convert tokenized data to tensors
inputs = torch.tensor(validation_encodings['input_ids'])
masks = torch.tensor(validation_encodings['attention_mask'])
labels = torch.tensor(validation_labels)

# Create a DataLoader for the validation set
validation_dataset = torch.utils.data.TensorDataset(inputs, masks, labels)
validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=8)

# Evaluate the model
model.eval()
predictions = []
true_labels = []

with torch.no_grad():
    for batch in validation_dataloader:
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions.append(logits.argmax(dim=-1).cpu().numpy())
        true_labels.append(labels.cpu().numpy())

# Flatten the predictions and true labels lists
predictions = [item for sublist in predictions for item in sublist]
true_labels = [item for sublist in true_labels for item in sublist]

# Calculate accuracy
accuracy = sum([pred == true for pred, true in zip(predictions, true_labels)]) / len(true_labels)
print(f'Validation Accuracy: {accuracy:.4f}')
