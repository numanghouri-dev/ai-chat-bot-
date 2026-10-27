"""AI Customer Support Chatbot - Intent Classification using Bi-LSTM
(plain-script version of DL_Case_Study_Chatbot.ipynb)
"""

# # AI Customer Support Chatbot — Intent Classification using Bidirectional LSTM
# 
# **Deep Learning Case Study** · CSE (AI & ML), Chaitanya Bharathi Institute of Technology
# **Submitted by:** Md Numan Gouri (160124748122) and Malige Afridi (160124748119)
# **Submitted to:** Mr. Srikant
# 
# The chatbot reads a customer's message, classifies it into one of **27 intents** with a Bi-LSTM, and replies with a suitable support response. If the model's confidence is below 0.5 it asks the user to rephrase.
# 
# **Dataset:** [Bitext Gen AI Chatbot Customer Support Dataset (Kaggle)](https://www.kaggle.com/datasets/bitext/bitext-gen-ai-chatbot-customer-support-dataset) — 26,872 queries, 27 intents, 11 categories.

# ## 1. Import libraries

import os, re, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Fix random seeds so results are repeatable
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TensorFlow version:", tf.__version__)

# ## 2. Load the dataset
# 
# Download `Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv` from Kaggle and put it in the same folder as this notebook (or in a `data/` folder).
# On Google Colab you can upload it with the file panel on the left.

FILE_NAME = "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
DATA_PATH = None
for p in [FILE_NAME, os.path.join("data", FILE_NAME), os.path.join("/content", FILE_NAME)]:
    if os.path.exists(p):
        DATA_PATH = p
        break
if DATA_PATH is None:
    raise FileNotFoundError("Dataset not found. Download it from Kaggle and place it next to this notebook.")

df = pd.read_csv(DATA_PATH)
print("Using file:", os.path.basename(DATA_PATH))
print("Shape:", df.shape)
print(df.head())

# ## 3. Data exploration

print("Columns:", list(df.columns))
print("Number of intents    :", df['intent'].nunique())
print("Number of categories:", df['category'].nunique())
print("Missing values:\n", df[['instruction','intent','response']].isnull().sum())

words = df['instruction'].str.split().str.len()
print(f"Average words per query: {words.mean():.1f} (maximum {words.max()})")

# Samples per intent
intent_counts = df['intent'].value_counts()
plt.figure(figsize=(9, 8))
ax = intent_counts.sort_values().plot(kind='barh', color='#1f3a5f')
for i, v in enumerate(intent_counts.sort_values()):
    ax.text(v + 5, i, str(v), va='center', fontsize=8, color='gray')
plt.title("Samples per intent (27 intents)", fontweight='bold')
plt.xlabel("Number of user queries")
plt.tight_layout()
plt.show()

# Samples per category
cat_counts = df['category'].value_counts()
plt.figure(figsize=(10, 4))
ax = cat_counts.plot(kind='bar', color='#2a9d8f')
for i, v in enumerate(cat_counts):
    ax.text(i, v + 60, str(v), ha='center', fontsize=8, color='gray')
plt.title("Samples per category (11 categories)", fontweight='bold')
plt.ylabel("Queries")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# ## 4. Text preprocessing

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\{\{.*?\}\}", " placeholder ", text)   # {{Order Number}} -> placeholder
    text = re.sub(r"[^a-z0-9\s]", " ", text)                  # remove symbols
    text = re.sub(r"\s+", " ", text).strip()                  # remove extra spaces
    return text

df['clean_text'] = df['instruction'].apply(clean_text)
print(df[['instruction', 'clean_text', 'intent']].head(6))

# ## 5. Label encoding, train/test split and tokenization

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['intent'])
num_classes = len(label_encoder.classes_)

X_train_text, X_test_text, y_train, y_test = train_test_split(
    df['clean_text'].values, y, test_size=0.2, random_state=SEED, stratify=y)

MAX_WORDS = 10000   # vocabulary size
MAX_LEN   = 30      # max words per sentence

vectorizer = layers.TextVectorization(max_tokens=MAX_WORDS,
                                      output_sequence_length=MAX_LEN)
vectorizer.adapt(X_train_text)

X_train = vectorizer(X_train_text).numpy()
X_test  = vectorizer(X_test_text).numpy()

print("Classes:", num_classes)
print("Train:", len(X_train_text), " Test:", len(X_test_text))
print("Vocabulary size:", len(vectorizer.get_vocabulary()))
print("Example:", X_train_text[0], "->", X_train[0])

# ## 6. Build the Bi-LSTM model

model = models.Sequential([
    layers.Input(shape=(MAX_LEN,)),
    layers.Embedding(input_dim=MAX_WORDS, output_dim=128),
    layers.Bidirectional(layers.LSTM(64)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# ## 7. Train with EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(X_train, y_train,
                    validation_split=0.1,
                    epochs=15,
                    batch_size=64,
                    callbacks=[early_stop])

# Accuracy and loss curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
epochs_ran = range(1, len(history.history['loss']) + 1)

axes[0].plot(epochs_ran, history.history['accuracy'], 'o-', label='Train', color='#1f3a5f')
axes[0].plot(epochs_ran, history.history['val_accuracy'], 's-', label='Validation', color='#e76f51')
axes[0].set_title("Accuracy", fontweight='bold'); axes[0].set_xlabel("Epoch"); axes[0].legend()

axes[1].plot(epochs_ran, history.history['loss'], 'o-', label='Train', color='#1f3a5f')
axes[1].plot(epochs_ran, history.history['val_loss'], 's-', label='Validation', color='#e76f51')
axes[1].set_title("Loss", fontweight='bold'); axes[1].set_xlabel("Epoch"); axes[1].legend()

best_epoch = int(np.argmin(history.history['val_loss'])) + 1
for ax in axes:
    ax.axvline(best_epoch, linestyle='--', color='gray', linewidth=1)
plt.tight_layout()
plt.show()
print("Best epoch (lowest val_loss):", best_epoch)

# ## 8. Evaluate on the test data

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

print(f"Test Accuracy: {test_acc*100:.2f}%   Test Loss: {test_loss:.4f}")
print("Wrongly classified:", int((y_pred != y_test).sum()), "out of", len(y_test))
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={'size': 7},
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title(f"Confusion matrix on {len(y_test):,} test queries", fontweight='bold')
plt.xlabel("Predicted intent"); plt.ylabel("Actual intent")
plt.tight_layout()
plt.show()

# Most common mistakes (actual -> predicted)
mistakes = {}
for a, p in zip(y_test, y_pred):
    if a != p:
        key = (label_encoder.classes_[a], label_encoder.classes_[p])
        mistakes[key] = mistakes.get(key, 0) + 1
for (a, p), n in sorted(mistakes.items(), key=lambda x: -x[1])[:6]:
    print(f"{a:28s} -> {p:28s} : {n}")

# ## 9. Chatbot reply function

# All dataset responses grouped by intent
responses = df.groupby('intent')['response'].apply(list).to_dict()

def fill_placeholders(reply):
    # {{Order Number}} -> [Order Number], to be filled from a real database later
    return re.sub(r"\{\{(.*?)\}\}", r"[\1]", reply)

def chatbot_reply(message, threshold=0.5):
    seq = vectorizer(np.array([clean_text(message)])).numpy()
    probs = model.predict(seq, verbose=0)[0]
    idx = int(np.argmax(probs))
    confidence = float(probs[idx])
    intent = label_encoder.inverse_transform([idx])[0]

    if confidence < threshold:
        return (intent, confidence,
                "Sorry, I didn't understand that. Could you please rephrase your question?")

    reply = fill_placeholders(random.choice(responses[intent]))
    return intent, confidence, reply

# ## 10. Test the chatbot with new messages

samples = [
    "I want to cancel my order",
    "how can i get my money back",
    "i forgot my password",
    "where is my package",
    "I need to talk to a real person",
    "can I pay with paypal",
    "I want to change my delivery address",
    "what's the weather today",
]

for s in samples:
    intent, conf, reply = chatbot_reply(s)
    print(f"User : {s}\nIntent: {intent} ({conf:.2f})\nBot  : {reply[:250]}...\n")

# ## 11. Interactive chat session
# Type `quit` to end the chat.

print("Bot: Hello! I am your customer support assistant. How can I help you?")
while True:
    user = input("You: ")
    if user.strip().lower() in ("quit", "exit", "bye"):
        print("Bot: Thank you for contacting us. Have a great day!")
        break
    intent, conf, reply = chatbot_reply(user)
    print(f"Bot [{intent}, {conf:.2f}]: {reply}\n")

# ## 12. Save the model (optional)

model.save("bilstm_intent_model.keras")
import pickle
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
print("Saved model and label encoder.")

