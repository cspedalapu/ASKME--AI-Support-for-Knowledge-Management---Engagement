# ASKME – AI Support for Knowledge Management & Engagement

**ASKME** is an intelligent GenAI-powered assistant designed to transform university support services. It delivers real-time, context-aware, and multilingual responses to student queries — with a special focus on international student needs — through advanced NLP, Retrieval-Augmented Generation (RAG), and fine-tuned Large Language Models (LLMs).

---

## Problem Statement

University departments are overwhelmed by thousands of repetitive queries each semester, resulting in:

- Long response delays during peak periods
- Inconsistent or incomplete information from different departments
- High dependency on limited staff hours (8 AM – 5 PM)
- Lack of 24/7, personalized assistance for critical student issues (visa, scholarship, course selection, etc.)

**CampusNavigator-AI** addresses these issues by providing an AI-driven, always-available support system tailored to the academic environment.

---

## Objectives

- ✅ Automate academic and administrative FAQ responses using LLMs
- ✅ Support personalized and multilingual responses with context retention
- ✅ Reduce human staff workload and response wait times
- ✅ Build a scalable and accurate AI assistant for university environments
- ✅ Enable document-level understanding from PDF, web, and JSON content
- ✅ Track and log queries for analytics, feedback, and improvement

---

## Tech Stack

| Component             | Tools / Frameworks                                      |
|----------------------|----------------------------------------------------------|
| **LLMs**             | BERT, T5, Sentence-BERT, LLaMA (Phase-2)                |
| **NLP Frameworks**   | HuggingFace Transformers, LangChain                     |
| **Embeddings**       | Word2Vec, TF-IDF, SentenceTransformers                  |
| **RAG Framework**    | FAISS / ChromaDB + LLM-based Generator                  |
| **Data Collection**  | BeautifulSoup, Selenium, OCR (Tesseract)                |
| **Preprocessing**    | spaCy, NLTK, Regex-based cleaning                       |
| **Backend/API**      | FastAPI                                                  |
| **Frontend (UI)**    | Streamlit (initial), React (optional future phase)      |
| **Deployment**       | AWS EC2, S3, Lambda (Future), GitHub Actions (CI/CD)    |
| **Datasets**         | UNT scraped content (.pdf/.json), SQuAD, internal logs  |

---

## Project Modules

1. **Data Scraping & Cleaning**
   - Web scraping with BeautifulSoup/Selenium
   - OCR from scanned PDFs using Tesseract
   - Store data in structured JSON/PDF formats

2. **Preprocessing Pipeline**
   - Tokenization, Lemmatization, Stopword removal
   - POS tagging, Named Entity Recognition (NER)
   - Extractive and abstractive summarization

3. **Knowledge Base Builder**
   - Index structured documents using FAISS or ChromaDB
   - Embed documents using Sentence-BERT for similarity search

4. **RAG Pipeline**
   - Combine retrieval with LLM (T5/BERT)
   - Contextual matching + dynamic answer generation

5. **Evaluation**
   - Metrics: F1 Score, Exact Match, Precision, Recall
   - User satisfaction surveys and feedback loop
   - Accuracy benchmarking using external datasets

6. **Deployment**
   - Backend with FastAPI and REST endpoints
   - Initial UI with Streamlit; optional migration to React
   - Hosting on AWS with future Lambda integration

---
CampusNavigator-AI/
├── README.md                   ← Project overview and documentation
├── requirements.txt            ← Python dependencies
├── .gitignore                  ← Files to ignore in version control
│
├── data/                       ← Raw and processed data
│   ├── raw/                    ← Scraped or unprocessed PDFs, JSONs
│   └── processed/              ← Cleaned and structured text data
│
├── notebooks/                  ← EDA, model training experiments
│   └── RAG_Prototype.ipynb     ← First prototype of RAG model
│
├── src/                        ← Core source code
│   ├── scraping/               ← Scripts for web/PDF scraping
│   │   └── scrape_unt.py       ← Scraper for UNT website
│   │
│   ├── preprocessing/          ← Text cleaning and normalization
│   │   └── clean_text.py       ← Tokenizer, lemmatizer, etc.
│   │
│   ├── embeddings/             ← Word/sentence embedding generators
│   │   └── embed_text.py       ← Word2Vec, SBERT
│   │
│   ├── rag_pipeline/           ← RAG retrieval and response logic
│   │   ├── retriever.py        ← Document retriever using FAISS/Chroma
│   │   └── generator.py        ← T5/BERT-based answer generation
│   │
│   ├── api/                    ← FastAPI app and endpoints
│   │   ├── main.py             ← Entry point for FastAPI server
│   │   └── routes.py           ← Defines routes for querying
│   │
│   └── models/                 ← Trained / fine-tuned model files
│       └── bert_tuned.pt       ← Saved model checkpoint
│
├── ui/                         ← Streamlit or React frontend
│   └── streamlit_app.py        ← Prototype user interface
│
├── tests/                      ← Unit & integration tests
│   └── test_pipeline.py        ← Pipeline and RAG tests
│
└── docs/                       ← (Optional) Sphinx/Markdown documentation
    └── architecture.md         ← System design and flow diagrams


