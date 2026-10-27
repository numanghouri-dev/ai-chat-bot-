# AI Customer Support Chatbot — Intent Classification using Bi-LSTM

Deep Learning case study, CSE (AI & ML), Chaitanya Bharathi Institute of Technology.

**Submitted by:** Md Numan Gouri (160124748122) and Malige Afridi (160124748119)
**Submitted to:** Mr. Srikant · 3rd year / 5th sem / CSM-2

A customer-support chatbot that reads a free-text message, classifies it into one of
**27 intents** with a Bidirectional LSTM, and replies with a matching support response.
When its confidence is below 0.5 it asks the user to rephrase.

## Results

| Metric | Value |
|---|---|
| Test accuracy | **98.99%** |
| Test loss | 0.0371 |
| Macro precision / recall / F1 | 0.99 / 0.99 / 0.99 |
| Test queries | 5,375 (54 misclassified) |
| Parameters | 1,388,827 |

## How it works

```
User message → Text cleaning → Vectorize (30 word IDs) → Bi-LSTM model → Intent + confidence
                                                                           ├─ < 0.5 → "Please rephrase"
                                                                           └─ ≥ 0.5 → reply from dataset
```

**Model:** Embedding (10,000 × 128) → Bidirectional LSTM (64 + 64) → Dropout 0.3 →
Dense 64 ReLU → Dropout 0.3 → Dense 27 Softmax.
Trained with Adam, batch size 64, up to 15 epochs, EarlyStopping (val_loss, patience 3).

## Dataset

[Bitext Gen AI Chatbot Customer Support Dataset](https://www.kaggle.com/datasets/bitext/bitext-gen-ai-chatbot-customer-support-dataset)
— 26,872 queries, 27 intents, 11 categories.
The CSV is not included in this repository. Download
`Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv` and place it
next to the notebook (or in a `data/` folder).

## Run it

**Google Colab (easiest):** open `DL_Case_Study_Chatbot.ipynb` in Colab, upload the CSV
using the file panel, then *Runtime → Run all*.

**Locally:**

```bash
pip install -r requirements.txt
jupyter notebook DL_Case_Study_Chatbot.ipynb
# or run the plain script version
python chatbot.py
```

## Files

| File | Contents |
|---|---|
| `DL_Case_Study_Chatbot.ipynb` | Full notebook: exploration, preprocessing, training, evaluation, chatbot |
| `chatbot.py` | Same code as a plain Python script |
| `requirements.txt` | Python libraries needed |

## Limitations and future scope

- Out-of-scope messages are still forced into one of the 27 intents → add an `out_of_scope` intent.
- Rare words (e.g. "paypal") are not learned well → pretrained embeddings (GloVe) or BERT.
- Single-turn only → add conversation memory; connect to a real order database.
- Deploy with Flask/FastAPI as a web or WhatsApp bot; support Telugu and Hindi.

## References

1. Bitext, Customer Support LLM Chatbot Training Dataset — https://github.com/bitext/customer-support-llm-chatbot-training-dataset
2. Hochreiter & Schmidhuber, "Long Short-Term Memory", Neural Computation, 1997.
3. Schuster & Paliwal, "Bidirectional Recurrent Neural Networks", IEEE Trans. Signal Processing, 1997.
4. F. Chollet, *Deep Learning with Python*, 2nd ed., Manning, 2021.
