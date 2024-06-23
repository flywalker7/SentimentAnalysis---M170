# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch

# Load your dataset
# Specify the path to your CSV file
csv_file_path = r'C:\Users\Γιωργος\Downloads\IMDB Dataset.csv'  # Use a raw string for the file path

# Load the CSV file into a pandas DataFrame
data = pd.read_csv(csv_file_path)

# Print the first few rows of the dataset to verify it loaded correctly
print(data.head())

# Convert string labels to integers
label_mapping = {"positive": 1, "negative": 0}
data['label'] = data['sentiment'].map(label_mapping)

# Split the dataset into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(data['review'], data['label'], test_size=0.2)

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the training and validation texts
train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True, max_length=32)  # Adjust max_length if necessary
val_encodings = tokenizer(val_texts.tolist(), truncation=True, padding=True, max_length=32)

# Define a custom dataset class for PyTorch
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings  # Store the tokenized encodings
        self.labels = labels  # Store the labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}  # Get the tokenized data
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)  # Get the label
        return item
    def __len__(self):
        return len(self.labels)  # Return the length of the dataset

# Create PyTorch datasets for training and validation
train_dataset = Dataset(train_encodings, train_labels.tolist())
val_dataset = Dataset(val_encodings, val_labels.tolist())

# Load the pre-trained BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',  # Directory to save results
    num_train_epochs=3,  # Number of training epochs, start with 1
    per_device_train_batch_size=1,  # Reduce batch size for training
    per_device_eval_batch_size=1,  # Reduce batch size for evaluation
    warmup_steps=100,  # Fewer warmup steps
    weight_decay=0.01,  # Weight decay for regularization
    logging_dir='./logs',  # Directory to save logs
    logging_steps=10,  # Log every 10 steps
    eval_strategy="epoch",  # Evaluate at the end of each epoch
)

# Create a Trainer instance
trainer = Trainer(
    model=model,  # The BERT model to be trained
    args=training_args,  # Training arguments
    train_dataset=train_dataset,  # Training dataset
    eval_dataset=val_dataset  # Evaluation dataset
)

# Train the model
trainer.train()
