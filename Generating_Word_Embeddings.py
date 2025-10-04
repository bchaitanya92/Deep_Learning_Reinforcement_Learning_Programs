import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import re
import random

# ------------------------------
# 1. Preprocess Corpus
# ------------------------------
def tokenize(text):
    # very simple tokenizer: lowercase + split on non-letters
    return re.findall(r"\b\w+\b", text.lower())

corpus = """Natural language processing enables computers to understand human language.
Word embeddings map words into continuous vector space.
Neural networks can be used to learn these embeddings from context words in a document corpus."""
tokens = tokenize(corpus)

# ------------------------------
# 2. Build Vocabulary
# ------------------------------
word_counts = Counter(tokens)
vocab = {w: i for i, (w, _) in enumerate(word_counts.items())}
id2word = {i: w for w, i in vocab.items()}
vocab_size = len(vocab)

# ------------------------------
# 3. Generate Training Data (Skip-gram)
# ------------------------------
def generate_pairs(tokens, window_size=2):
    pairs = []
    for i, center in enumerate(tokens):
        for j in range(max(0, i - window_size), min(len(tokens), i + window_size + 1)):
            if i != j:
                pairs.append((center, tokens[j]))
    return pairs

pairs = generate_pairs(tokens, window_size=2)
print("Sample training pairs:", pairs[:10])

# Convert to indices
training_data = [(vocab[c], vocab[o]) for c, o in pairs]

# ------------------------------
# 4. Define Neural Network
# ------------------------------
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(Word2Vec, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)  # input layer
        self.output = nn.Linear(embed_dim, vocab_size)        # output layer

    def forward(self, center_word):
        embed = self.embeddings(center_word)
        out = self.output(embed)
        return out

# ------------------------------
# 5. Training
# ------------------------------
embed_dim = 50
model = Word2Vec(vocab_size, embed_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(50):
    total_loss = 0
    for center, context in training_data:
        center_tensor = torch.tensor([center], dtype=torch.long)
        context_tensor = torch.tensor([context], dtype=torch.long)

        optimizer.zero_grad()
        output = model(center_tensor)
        loss = criterion(output, context_tensor)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# ------------------------------
# 6. Extract Word Embeddings
# ------------------------------
embeddings = model.embeddings.weight.data
print("\nWord Embedding for 'language':\n", embeddings[vocab["language"]])
print("\nWord Embedding for 'neural':\n", embeddings[vocab["neural"]])


/*
Sample training pairs: [('natural', 'language'), ('natural', 'processing'), ('language', 'natural'), ('language', 'processing'), ('language', 'enables'), ('processing', 'natural'), ('processing', 'language'), ('processing', 'enables'), ('processing', 'computers'), ('enables', 'language')]
Epoch 10, Loss: 264.6419
Epoch 20, Loss: 248.5941
Epoch 30, Loss: 238.4213
Epoch 40, Loss: 235.1496
Epoch 50, Loss: 230.1036

Word Embedding for 'language':
 tensor([-0.5211, -0.0398, -0.1866, -0.0288, -0.4527, -0.0603, -0.0144, -0.0196,
         0.1168, -0.3704, -0.0470,  0.7273, -0.1065, -0.0066,  0.0832,  0.3182,
         0.0571, -0.1028, -0.1645, -0.4021,  0.0821,  0.0520,  0.2955, -0.1912,
        -0.0548, -1.0331,  0.2676, -0.4810, -0.5650, -0.0550, -0.5612, -1.7615,
        -0.5346, -0.0154, -0.2718,  0.0328, -0.6433, -0.3753, -0.0761, -0.7315,
        -0.1847, -0.0831, -0.2612,  0.0465, -0.1019, -0.0099,  0.0117, -0.0886,
         0.0460, -0.2623])

Word Embedding for 'neural':
 tensor([ 3.1592e-01,  3.4966e-01,  7.2536e-01, -1.6453e-01,  6.2357e-01,
        -6.7021e-04,  9.6454e-01,  9.2496e-01, -5.9064e-01, -1.9390e-02,
        -1.2982e-01, -4.0040e-01,  2.2320e-01,  2.1080e-01,  7.8963e-02,
         3.8069e-01,  6.1177e-01,  5.3831e-02,  6.1975e-01, -4.6652e-01,
         9.4187e-01, -3.6157e-02, -8.6640e-01,  1.0516e+00,  1.1266e-01,
        -2.0185e-02,  4.1761e-01,  1.5600e-01,  4.9597e-01,  5.4737e-01,
        -2.0418e+00,  4.1971e-01, -9.3843e-02, -6.3190e-01,  4.0373e-01,
         8.7371e-01, -4.5783e-01, -1.0898e+00,  7.4455e-02, -8.4540e-01,
         1.5338e+00, -5.6010e-01, -6.4116e-02, -1.7250e-01,  3.5546e-01,
         1.2032e-01, -3.7513e-01,  6.1328e-01, -1.3033e+00,  1.2831e+00])
*/