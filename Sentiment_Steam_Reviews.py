import pandas as pd
from transformers import DistilBertTokenizer
import torch
from transformers import DistilBertForSequenceClassification
import torch.nn as nn
from transformers import Trainer
from transformers import TrainingArguments

# Load the CSV file into a pandas DataFrame
csv_file_path = '/content/drive/My Drive/Colab Notebooks/steam_reviews.csv'
data = pd.read_csv(csv_file_path)

# Display the first few rows of the dataset
print(data.head())

# Ensure column names are correct
print(data.columns)

# Convert 'Recommended/Not Recommended' to binary labels
data['Recommended/Not Recommended'] = data['Recommended/Not Recommended'].map({'Recommended': 1, 'Not Recommended': 0})

# Ensure all review texts are strings and handle missing values
data['Review Text'] = data['Review Text'].fillna('').astype(str)

# Additional cleaning: remove extra whitespace and special characters (optional)
data['Review Text'] = data['Review Text'].str.strip()  # Remove leading/trailing whitespace

# Tokenize the review text
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
encodings = tokenizer(data['Review Text'].tolist(), truncation=True, padding=True)

# Prepare labels
labels = data['Recommended/Not Recommended'].values

# Convert to PyTorch tensors
class SteamReviewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

dataset = SteamReviewsDataset(encodings, labels)

# Split the dataset into train and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Load pre-trained DistilBERT model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',            # Output directory
    num_train_epochs=3,                # Number of training epochs
    per_device_train_batch_size=16,    # Batch size for training
    per_device_eval_batch_size=64,     # Batch size for evaluation
    warmup_steps=500,                  # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,                 # Strength of weight decay
    logging_dir='./logs',              # Directory for storing logs
    logging_steps=10,
)

# Create a Trainer instance
trainer = Trainer(
    model=model,                       # The pre-trained DistilBERT model
    args=training_args,                # Training arguments
    train_dataset=train_dataset,       # Training dataset
    eval_dataset=val_dataset           # Evaluation dataset
)

# Train the model
trainer.train()

# Save the trained model
model.save_pretrained('/content/drive/My Drive/Colab Notebooks/steam_reviews_model')
tokenizer.save_pretrained('/content/drive/My Drive/Colab Notebooks/steam_reviews_model')
