# 🤖 AI-Driven NLP Project — BERT Across Four Tasks

An end-to-end exploration of **BERT** and its fine-tuned variants across four core NLP tasks. Each task is stress-tested with both standard and deliberately adversarial inputs — surfacing where these models genuinely struggle, not just where they look good.

---

## 🧠 Tasks Covered

| # | Task | Model Used |
|---|---|---|
| 1 | **Masked Language Modeling** | `bert-base-uncased` |
| 2 | **Sentiment Analysis** | `distilbert-base-uncased-finetuned-sst-2-english` |
| 3 | **Named Entity Recognition (NER)** | `dslim/bert-base-NER` |
| 4 | **Semantic Similarity** (via `[CLS]` embeddings) | `bert-base-uncased` |

---

## 🔬 Adversarial Test Inputs

Each task is evaluated on edge cases designed to expose model limitations:

- 🙃 **Sarcasm** — Sentiment that is inverted in intent
- ❌ **Negation** — Sentences with grammatical negation that flip meaning
- 🌐 **Code-Mixed Text** — Mixed-language inputs (e.g., Hinglish)
- 🏥 **Medical Jargon** — Domain-specific terminology outside typical training distribution
- 🔁 **Long Repetitive Input** — Stress test for attention span and token limits

---

## ❓ Why BERT over GPT-3?

| Factor | BERT | GPT-3 |
|---|---|---|
| **Cost** | Free | Requires paid API |
| **API Key** | ❌ Not needed | ✅ Required |
| **Reproducibility** | Fully reproducible on free Colab | Depends on billing account |
| **Model Access** | Open-source via Hugging Face | Closed-source via OpenAI |
| **Suitability** | Ideal for encoding & classification tasks | Ideal for open-ended generation |

> BERT is **open-source**, free to run, and fully reproducible on a free Google Colab instance — making it the practical choice for this project.

---

## 📁 File Structure

```
advanced lvl task/
├── NLP_Project_BERT.ipynb   # Main notebook — all four NLP experiments
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation (this file)
```

---

## 🚀 How to Run

### ✅ Option A: Google Colab *(Recommended — no local setup required)*

1. Go to [colab.research.google.com](https://colab.research.google.com) and upload `NLP_Project_BERT.ipynb`  
   *(File → Upload notebook)*
2. Run the **first cell** (`!pip install ...`) — Colab has internet access, so Hugging Face model downloads happen automatically.
3. Run all remaining cells top to bottom: **Runtime → Run all**

> 💡 A GPU is **not required** for these models (they're small). If you want one anyway:  
> *Runtime → Change runtime type → GPU*

---

### 🖥️ Option B: Local Jupyter

```bash
pip install -r requirements.txt
jupyter notebook NLP_Project_BERT.ipynb
```

---

## 📦 Dependencies

```
transformers>=4.40.0
torch>=2.0.0
pandas>=2.0.0
scikit-learn>=1.3.0
numpy>=1.24.0
```

> All models are downloaded **automatically** the first time you run the notebook (~a few hundred MB total). An internet connection is required for the first run.

---

## ✍️ What You Still Need to Do

The notebook is fully coded and ready to run, but **two sections are intentionally left blank** for you to complete after seeing your actual runtime output:

| Section | What to Fill In |
|---|---|
| **Section 6 — Research Questions** | 5 questions are pre-defined; add your own 2–3 observations under each based on what you see printed |
| **Section 8 — Conclusion & Insights** | A structure/outline is provided; fill it in with your genuine takeaways after running everything |

> ⚠️ **This is intentional.** A strong internship submission should reflect *your own analysis* of the outputs, not a pre-written conclusion.

---

## 🔑 Notes

- ✅ No API keys required anywhere in this notebook
- 🌐 Internet connection needed on first run for model downloads
- 🆓 Fully reproducible on a free Google Colab instance

---

## 👤 Author

**Rishi Chaurasia**  
ShadowFox Internship — Advanced Level Task
