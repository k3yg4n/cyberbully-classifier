import pandas as pd

# Load the full CSV
df = pd.read_csv("train.csv")

# Combine 'toxic' and 'severe_toxic' into a single 'toxic' column
df['toxic'] = ((df['toxic'] + df['severe_toxic']) > 0).astype(int)

# Drop 'severe_toxic' and 'obscene' columns
df = df.drop(columns=['severe_toxic', 'obscene', 'id'])

# Rename 'identity_hate' to 'discrimination'
df = df.rename(columns={'identity_hate': 'discrimination'})
df = df.rename(columns={'comment_text': 'comment'})

# Define updated label columns
label_cols = ['toxic', 'threat', 'insult', 'discrimination']

# Get rows with at least one '1' in the label columns
toxic_rows = df[df[label_cols].sum(axis=1) >= 1]

# Get rows with all zeros in the label columns
non_toxic_rows = df[df[label_cols].sum(axis=1) == 0]

# Sample the desired number of rows
toxic_sample = toxic_rows.sample(n=1500, random_state=42)
non_toxic_sample = non_toxic_rows.sample(n=500, random_state=42)

# Combine the samples
combined = pd.concat([toxic_sample, non_toxic_sample])

# Shuffle the combined dataset (optional)
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to a new CSV
combined.to_csv("train_subset.csv", index=False)

print("Saved 1500 toxic + 500 non-toxic rows to train_subset.csv")
