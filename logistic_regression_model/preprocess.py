import pandas as pd
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as stop_words

# Cleans up the text for model to process
def clean_text(text):
    text = text.lower()
    # remove urls, @mentions and #hashtags from text
    text = re.sub(r"http\S+|@\w+|#\w+", "", text)
    # remove punctuation and special characters
    text = re.sub(r"[^\w\s]", "", text)
    return text

# Tokenization + remove unneeded words like "the", "than", etc.
def tokenize(text):
    return [tok for tok in text.split() if tok not in stop_words]

#################################################

# Load the data
df = pd.read_csv('../data/train_subset.csv')

# Clean data by stripping URLs and punctuation
df['clean_comment'] = df['comment'].astype(str).apply(clean_text)

# Tokenize data and remove unneeded words
df['tokens'] = df['clean_comment'].apply(tokenize)

# Ensure all labels are integers
for col in ['toxic', 'insult', 'threat', 'discrimination']:
    df[col] = df[col].astype(int)

# Save cleaned data
df.to_csv('../data/train_subset_clean.csv', index=False)

print("Preprocessing finished! First few rows:")
print(df[['comment','clean_comment','tokens','toxic','insult','threat','discrimination']].head())
