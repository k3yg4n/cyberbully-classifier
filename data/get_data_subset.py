import pandas as pd

CSV_IN = "train.csv" # original data file
CSV_OUT = "train_subset.csv"
MIN_POS = 100 # Minimum positives per label
N_NEG = 500 # hNumber of all-zero rows to include
RNG = 42

# Load the full CSV
df = pd.read_csv(CSV_IN)

# Combine 'toxic' and 'severe_toxic' into a single 'toxic' column
df['toxic'] = ((df['toxic'] + df['severe_toxic']) > 0).astype(int)

# Drop 'severe_toxic' and 'obscene' columns
df = df.drop(columns=['severe_toxic', 'obscene', 'id'])

# Rename 'identity_hate' to 'discrimination'
df = df.rename(columns={'identity_hate': 'discrimination'})
df = df.rename(columns={'comment_text': 'comment'})

# Define updated label columns
label_cols = ['toxic', 'threat', 'insult', 'discrimination']

# Collect/oversample positives per label
selected = []

for lbl in label_cols:
    pos_rows = df[df[lbl] == 1]
    if len(pos_rows) >= MIN_POS:
        pos_sample = pos_rows.sample(n=MIN_POS, random_state=RNG, replace=False)
    else:
        # Oversample with replacement to reach MIN_POS
        pos_sample = pos_rows.sample(n=MIN_POS, random_state=RNG, replace=True)
    selected.append(pos_sample)

# Union all positive samples (may include overlaps)
positives_union = pd.concat(selected).drop_duplicates()

# Sample rows with all zeros across all labels
neg_rows = df[df[label_cols].sum(axis=1) == 0]

# Avoid picking negatives already in positives_union
neg_rows = neg_rows.loc[~neg_rows.index.isin(positives_union.index)]

neg_sample = neg_rows.sample(n=min(N_NEG, len(neg_rows)), random_state=RNG, replace=False)

# Combine the samples
combined = pd.concat([positives_union, neg_sample], axis=0)

# Shuffle the combined dataset
combined = combined.sample(frac=1, random_state=RNG).reset_index(drop=True)

# Save to a new CSV
combined.to_csv(CSV_OUT, index=False)
print(f"Saved subset to {CSV_OUT}")

print("\nLabel counts in subset:")
print(combined[label_cols].sum())
