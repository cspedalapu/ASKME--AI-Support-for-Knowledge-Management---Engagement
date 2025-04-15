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
├── README.md                           ← Project documentation overview
├── requirements.txt                    ← Python dependencies
├── .env.example                        ← Example environment variables
├── .gitignore                          ← Files to ignore
│
├── backend/                            ← Core AI/NLP backend logic
│   ├── api/                            ← FastAPI app with routes
│   │   ├── main.py                     ← FastAPI entrypoint
│   │   ├── routes.py                   ← Endpoint definitions
│   │   └── models.py                   ← Pydantic models for request/response
│   │
│   ├── rag_engine/                     ← Retrieval-Augmented Generation
│   │   ├── retriever.py                ← Vector search logic (FAISS/Chroma)
│   │   ├── generator.py                ← LLM response generation (T5/BERT)
│   │   └── pipeline.py                 ← End-to-end RAG pipeline
│   │
│   ├── embeddings/                     ← Word/sentence embedding logic
│   │   ├── embed_utils.py              ← Sentence-BERT, TF-IDF, Word2Vec loaders
│   │   └── faiss_index/                ← Saved vector indices
│   │
│   ├── preprocessing/                  ← Data cleaning & text preparation
│   │   └── cleaner.py                  ← Tokenization, Lemmatization, etc.
│   │
│   ├── data/                           ← Local dataset handling
│   │   ├── raw/                        ← Scraped PDFs, HTMLs
│   │   └── processed/                  ← Cleaned structured data
│   │
│   ├── models/                         ← Fine-tuned and custom-trained models
│   │   ├── bert_tuned/
│   │   └── t5_adapter/
│   │
│   └── notebooks/                      ← Research & prototyping (Colab ready)
│       ├── 1_scraping_colab.ipynb
│       ├── 2_preprocessing.ipynb
│       ├── 3_embeddings.ipynb
│       ├── 4_rag_pipeline.ipynb
│       ├── 5_evaluation.ipynb
│       └── 6_interface_streamlit.ipynb
│
├── frontend/                           ← UI layer (Streamlit, React later)
│   ├── streamlit_app/                  ← Initial chatbot with Streamlit
│   │   ├── app.py                      ← Streamlit UI script
│   │   └── components/                 ← Optional reusable UI blocks
│   └── react_app/                      ← (Optional) Production-grade React UI
│       ├── public/
│       ├── src/
│       └── package.json
│
├── tests/                              ← Unit + integration tests
│   ├── test_rag.py
│   ├── test_api.py
│   └── test_preprocessing.py
│
├── scripts/                            ← ⚙️ Deployment / automation scripts
│   ├── deploy_aws.sh                   ← AWS deployment setup
│   └── ingest_data.py                  ← CLI script to scrape and prepare data
│
├── .dockerignore                       ← 🐳 Docker ignore
├── Dockerfile                          ← 🐳 Backend containerization
├── docker-compose.yml                  ← 🐳 Local full-stack deployment
└── LICENSE                             ← 📄 License
