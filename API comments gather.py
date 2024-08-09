import requests
import csv
import time
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from requests.exceptions import SSLError

# Steam API key
api_key = '79ADD83F80A7BDABDBB56FB3D9A5563B'  # Replace with your actual Steam API key

# App ID for Europa Universalis IV
app_id = 236850

# URL to fetch reviews
url = f'https://store.steampowered.com/appreviews/{app_id}?json=1&key={api_key}'

# Parameters for fetching reviews
params = {
    'json': 1,
    'num_per_page': 100,
    'filter': 'recent'
}

# Set up a session with retry logic
session = requests.Session()
retry = Retry(
    total=5,
    read=5,
    connect=5,
    backoff_factor=0.3,
    status_forcelist=(500, 502, 504),
    allowed_methods=["HEAD", "GET", "OPTIONS"]
)
adapter = HTTPAdapter(max_retries=retry)
session.mount("https://", adapter)
session.mount("http://", adapter)

# Function to fetch reviews
def fetch_reviews(url, params, total_reviews):
    reviews = []
    count = 0
    cursor = '*'
    
    while count < total_reviews:
        params['cursor'] = cursor
        try:
            response = session.get(url, params=params)
            data = response.json()
            
            if 'reviews' not in data:
                break
            
            for review in data['reviews']:
                review_text = review.get('review', 'N/A')
                votes_up = review.get('votes_up', 0)
                votes_funny = review.get('votes_funny', 0)
                playtime_forever = review.get('author', {}).get('playtime_forever', 0)
                playtime_hours = playtime_forever / 60  # Convert minutes to hours
                voted_up = review.get('voted_up', False)
                
                reviews.append([
                    'Recommended' if voted_up else 'Not Recommended',  # Recommended or Not Recommended
                    votes_funny,  # Funny votes
                    round(playtime_hours, 2),  # Playtime in hours, rounded to 2 decimal places
                    votes_up,  # Thumbs Up
                    review_text  # Review Text
                ])
                count += 1
                if count >= total_reviews:
                    break
            
            print(f"Fetched {count}/{total_reviews} reviews so far...")
            cursor = data.get('cursor', '*')
            time.sleep(1)  # Sleep to avoid hitting rate limit
        except SSLError as e:
            print(f"SSL Error: {e}. Retrying...")
            continue
    
    return reviews

# Parameters
total_reviews = 30000

# Fetch reviews
reviews = fetch_reviews(url, params, total_reviews)

# Write reviews to a CSV file
csv_file_path = 'steam_reviews.csv'
with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow([
        'Recommended/Not Recommended', 'Funny Votes', 'Playtime (Hours)', 
        'Thumbs Up', 'Review Text'
    ])
    writer.writerows(reviews)

print(f"Reviews have been written to {csv_file_path}")
