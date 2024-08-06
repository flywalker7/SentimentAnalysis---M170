#Step1Preprocess_Data_and_Calculate_Sentiment_Scores
import pandas as pd
import re
from transformers import pipeline
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Load the CSV file
csv_file_path = '/content/drive/My Drive/steam_reviews.csv'
df = pd.read_csv(csv_file_path)

# Load the sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=0)

# Define a function to calculate sentiment scores
def get_sentiment_scores(review):
    if not isinstance(review, str):
        return 0, 0, 0
    parts = re.split(r',|\.|\bbut\b|\bhowever\b|\balthough\b|\bthough\b|\bwhile\b|\byet\b|\bnevertheless\b|\bon the other hand\b', review, flags=re.IGNORECASE)
    sentiments = [sentiment_analyzer(part.strip()[:512])[0] for part in parts if part.strip()]
    positive_score = sum(sent['score'] for sent in sentiments if sent['label'] == 'POSITIVE')
    negative_score = sum(sent['score'] for sent in sentiments if sent['label'] == 'NEGATIVE')
    total_score = positive_score - negative_score
    return positive_score, negative_score, total_score

# Apply sentiment analysis to each review
df[['Positive Score', 'Negative Score', 'Total Score']] = df['Review Text'].apply(lambda x: pd.Series(get_sentiment_scores(x)))
df['labels'] = df['Recommended/Not Recommended'].apply(lambda x: 1 if x == 'Recommended' else 0)

# Save the preprocessed dataset
preprocessed_csv_file_path = '/content/drive/My Drive/steam_reviews_preprocessed.csv'
df.to_csv(preprocessed_csv_file_path, index=False)

print("Preprocessing complete and data saved.")
