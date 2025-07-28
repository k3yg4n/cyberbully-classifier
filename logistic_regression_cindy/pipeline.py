import pandas as pd
from model import CyberbullyingDetector
# Load data
df = pd.read_csv("output/clean_data.csv")
modelDetector = cyberbullyingDetection()

df['text'] = df['text'].apply(modelDetector.clean)
# Build vocab and vectorize
vocab = modelDetector.build_vocab(df['text'], vocab_size=1000)
X = np.array([modelDetector.vectorize(text, vocab) for text in df['text']])
y = df['label'].values

# Train/test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Train model
weights = modelDetector.train(X_train, y_train)

# Evaluate
acc = modelDetector.evaluate(X_test, y_test, weights)
print(f"Test Accuracy: {acc:.4f}")
