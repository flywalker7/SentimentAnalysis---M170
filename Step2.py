import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Load the preprocessed data
preprocessed_csv_file_path = '/content/drive/My Drive/steam_reviews_preprocessed.csv'
df = pd.read_csv(preprocessed_csv_file_path)

# Ensure all entries in 'Review Text' are strings
df['Review Text'] = df['Review Text'].astype(str)

# Load the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Custom dataset class
class SteamReviewsDataset(Dataset):
    def __init__(self, encodings, labels, pos_scores, neg_scores, total_scores):
        self.encodings = encodings
        self.labels = labels
        self.pos_scores = pos_scores
        self.neg_scores = neg_scores
        self.total_scores = total_scores

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).long()
        item['pos_scores'] = torch.tensor(self.pos_scores[idx]).float()
        item['neg_scores'] = torch.tensor(self.neg_scores[idx]).float()
        item['total_scores'] = torch.tensor(self.total_scores[idx]).float()
        return item

    def __len__(self):
        return len(self.labels)

# Tokenize the review texts
encodings = tokenizer(df['Review Text'].tolist(), truncation=True, padding=True, max_length=512)

# Create dataset
dataset = SteamReviewsDataset(encodings, df['labels'].tolist(), df['Positive Score'].tolist(), df['Negative Score'].tolist(), df['Total Score'].tolist())

# Define model
class WeightedDistilBERT(DistilBertForSequenceClassification):
    def __init__(self, config, class_weights):
        super().__init__(config)
        self.class_weights = class_weights
        self.pre_classifier = torch.nn.Linear(config.dim + 3, config.dim)  # Adding 3 for the sentiment scores

    def forward(self, input_ids=None, attention_mask=None, labels=None, pos_scores=None, neg_scores=None, total_scores=None):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[0][:, 0]  # Take [CLS] token
        # Concatenate the sentiment scores
        combined_output = torch.cat((pooled_output, pos_scores.unsqueeze(1), neg_scores.unsqueeze(1), total_scores.unsqueeze(1)), dim=1)
        combined_output = self.pre_classifier(combined_output)
        logits = self.classifier(combined_output)
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return (loss, logits)
#Step 2: Train and Save the Model
# Define class weights (assuming binary classification)
class_weights = torch.tensor([0.1, 0.9], dtype=torch.float).to('cuda')  # Adjust as per your class imbalance

# Load the model
model = WeightedDistilBERT.from_pretrained('distilbert-base-uncased', num_labels=2, class_weights=class_weights)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy='epoch',
    logging_dir='./logs',
    logging_steps=10,
    save_strategy='epoch',
    load_best_model_at_end=True,
    report_to='all'  # Ensure logs are available in the output
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,
    compute_metrics=lambda p: {"accuracy": (p.predictions.argmax(-1) == p.label_ids).astype(float).mean().item()}
)

# Train the model
trainer.train()

# Save the model
model_save_path = '/content/drive/My Drive/weighted_distilbert'
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

print("Model training complete and model saved.")
