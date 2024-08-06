import pandas as pd
import re
import torch
from transformers import DistilBertTokenizer, pipeline, DistilBertForSequenceClassification

# Define model class
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

# Load the tokenizer and model
model_save_path = '/content/drive/My Drive/weighted_distilbert'
tokenizer = DistilBertTokenizer.from_pretrained(model_save_path)
model = WeightedDistilBERT.from_pretrained(model_save_path, num_labels=2, class_weights=torch.tensor([0.1, 0.9], dtype=torch.float).to('cuda'))

# Define a function to calculate sentiment scores
def get_sentiment_scores(review):
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=0)
    parts = re.split(r',|\.|\bbut\b|\bhowever\b|\balthough\b|\bthough\b|\bwhile\b|\byet\b|\bnevertheless\b|\bon the other hand\b', review, flags=re.IGNORECASE)
    sentiments = [sentiment_analyzer(part.strip()[:512])[0] for part in parts if part.strip()]
    positive_score = sum(sent['score'] for sent in sentiments if sent['label'] == 'POSITIVE')
    negative_score = sum(sent['score'] for sent in sentiments if sent['label'] == 'NEGATIVE')
    total_score = positive_score - negative_score
    return positive_score, negative_score, total_score

# Example reviews for prediction
reviews = [
    "The game has an interesting concept, but the execution is poor.",
    "I enjoyed the gameplay, however, the graphics could be better.",
    "Great mechanics, but the storyline is quite weak."
]

# Tokenize the example reviews
inputs = tokenizer(reviews, truncation=True, padding=True, max_length=512, return_tensors='pt').to('cuda')

# Get sentiment scores for the example reviews
sentiment_scores = [get_sentiment_scores(review) for review in reviews]
positive_scores = torch.tensor([score[0] for score in sentiment_scores]).float().to('cuda')
negative_scores = torch.tensor([score[1] for score in sentiment_scores]).float().to('cuda')
total_scores = torch.tensor([score[2] for score in sentiment_scores]).float().to('cuda')

# Perform inference
model.to('cuda')
model.eval()
with torch.no_grad():
    outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                    pos_scores=positive_scores, neg_scores=negative_scores, total_scores=total_scores)
    predictions = torch.argmax(outputs[1], dim=-1)

# Map the predictions to 'Recommended' or 'Not Recommended'
predicted_labels = ['Recommended' if pred == 1 else 'Not Recommended' for pred in predictions]
for review, label in zip(reviews, predicted_labels):
    print(f"Review: {review}\nPrediction: {label}\n")
