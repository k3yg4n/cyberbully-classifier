import re

class cyberbullyingDetection:
    
    # cleans up the text for model to process
    def clean(self, text): 
        text = text.lower()
        # remove urls, @mentions and #hashtags from text
        text = re.sub(r"http\S+|@\w+|#\w+", "", text)
        # remove punctuation and special characters
        text = re.sub(r"[^\w\s]", "", text)
        return text

    # creates vocaulary and map each sentence to a vector manually
    def build_vocab(self, texts, vocab_size = 1000): 
        all_words = []
        for text in texts: 
            all_words.extend(text.split())
        
        most_common = Counter(all_words).most_common(vocab_size)
        vocab = {word: i for i, (word, _) in enumerate(most_common)}
        return vocab
    
    def vectorize(self, text, vocab): 
        vec = np.zeros(len(vocab))
        for word in text.split(): 
            if word in vocab: 
                vec[vocab[word]] += 1 
        return vec
    
    # implement the model using logistic regression
    def sigmoid(self, z): 
        return 1 /(1+ np.exp(-z))
    
    def predict(self,X, weights): 
        return self.sigmoid(np.dot(X,weights))
    
    # perform gradient descent training
    def train(X, y, lr = 0.01, epochs=100): 
        weights = np.zeros(X.shape[1])
        for epoch in range(epochs): 
            predictions = predict(X, weights)
            errors = predictions - y 
            graident = np.dot(X.t, errors) / len(y)
            weights -= lr*graident
            
            if epoch % 10 == 0:
                loss = -np.mean(y * np.log(predictions + 1e-9) + (1 - y) * np.log(1 - predictions + 1e-9))
                print(f"Epoch {epoch} | Loss: {loss:.4f}")
    return weights

    def evaluate(self, X, y, weights):
        probs = self.predict(X, weights)
        preds = (probs >= 0.5).astype(int)
        accuracy = np.mean(preds == y)
        return accuracy
