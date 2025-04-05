# Text Summarization, Keyword Identification, and Title Generation App

## Description
This Streamlit-based NLP application allows users to upload or input text and generate summaries using two different models (BART and T5).  
It also identifies keywords, generates a title, and compares the generated summaries using similarity metrics.



## Features

- **Summarization**
  - Facebook BART Model
  - Google FLAN-T5 Model
- **Keyword Extraction** (using KeyBERT)
- **Title Generation** (using FLAN-T5)
- **Text Input Options**
  - PDF (`.pdf`)
  - Word Document (`.docx`)
  - Plain Text File (`.txt`)
  - Manual Text Entry
- **Comparison Metrics**
  - Cosine Similarity (TF-IDF)
  - ROUGE (R1, R2, RL)
  - BERT-based Semantic Similarity

