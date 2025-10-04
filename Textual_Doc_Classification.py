import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

print("Step 1: Loading dataset ...")
data = fetch_20newsgroups(subset='all')
texts, labels = data.data, data.target
num_classes = len(set(labels))
print(f"Loaded {len(texts)} documents across {num_classes} classes.")

print("\nStep 2: Preprocessing text ...")
MAX_WORDS = 10000   # vocabulary size
MAX_LEN = 300       # max tokens per document

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

X = tokenizer.texts_to_sequences(texts)
X = pad_sequences(X, maxlen=MAX_LEN)
y = to_categorical(labels, num_classes=num_classes)
print("Text converted to padded sequences.")

print("\nStep 3: Splitting into train and test ...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

print("\nStep 4: Building deep learning model ...")
model = Sequential([
    Embedding(input_dim=MAX_WORDS, output_dim=128, input_length=MAX_LEN),
    LSTM(128),
    Dense(num_classes, activation="softmax")
])
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])
print(model.summary())

print("\nStep 5: Training model ...")
history = model.fit(X_train, y_train,
          epochs=5,
          batch_size=64,
          validation_split=0.1,
          verbose=2)

print("\nStep 6: Evaluating model on test set ...")
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {acc:.4f}")

/*
Step 1: Loading dataset ...
Loaded 18846 documents across 20 classes.

Step 2: Preprocessing text ...
Text converted to padded sequences.

Step 3: Splitting into train and test ...
Training samples: 15076, Testing samples: 3770

Step 4: Building deep learning model ...

Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding (Embedding)       (None, 300, 128)          1280000   
                                                                 
 lstm (LSTM)                 (None, 128)               131584    
                                                                 
 dense (Dense)               (None, 20)                2580      
                                                                 
=================================================================
Total params: 1414164 (5.39 MB)
Trainable params: 1414164 (5.39 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
None

Step 5: Training model ...
Epoch 1/5
212/212 - 76s - loss: 2.6723 - accuracy: 0.1621 - val_loss: 2.2644 - val_accuracy: 0.3150 - 76s/epoch - 357ms/step
Epoch 2/5
212/212 - 86s - loss: 1.9426 - accuracy: 0.3839 - val_loss: 1.8901 - val_accuracy: 0.4198 - 86s/epoch - 407ms/step
Epoch 3/5
212/212 - 92s - loss: 1.5213 - accuracy: 0.5242 - val_loss: 1.9722 - val_accuracy: 0.4244 - 92s/epoch - 433ms/step
Epoch 4/5
212/212 - 92s - loss: 1.3967 - accuracy: 0.5716 - val_loss: 1.6435 - val_accuracy: 0.4894 - 92s/epoch - 433ms/step
Epoch 5/5
212/212 - 90s - loss: 1.0232 - accuracy: 0.6859 - val_loss: 1.4114 - val_accuracy: 0.5590 - 90s/epoch - 423ms/step

Step 6: Evaluating model on test set ...
Test Accuracy: 0.5804
*/