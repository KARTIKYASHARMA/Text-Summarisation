import streamlit as st
from transformers import pipeline
import PyPDF2
import docx
from keybert import KeyBERT
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd

# Text extraction functions
def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def read_word(file):
    doc = docx.Document(file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text


st.markdown(
    """
    <h1 style='text-align: center; color: #2A9D8F;'>Text Summarization, Keyword Identification, and Title Generation</h1>
    <hr style='border: 1px solid #264653;' >
    """, 
    unsafe_allow_html=True
)


# Load models
def load_models():
    summarizer_bart = pipeline("summarization", model="facebook/bart-base", truncation=True)
    summarizer_t5 = pipeline("summarization", model="google/flan-t5-small", truncation=True)

    keyword_model = KeyBERT()
    title_generator = pipeline("text2text-generation", model="google/flan-t5-small")
    return summarizer_bart, summarizer_t5, keyword_model, title_generator


# Comparison functions
def calculate_cosine_similarity(original_text, summary):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([original_text, summary])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return cosine_sim[0][0]

def calculate_rouge_score(original_text, summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(original_text, summary)
    return scores

def calculate_bert_similarity(original_text, summary):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode([original_text, summary])
    cosine_sim = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
    return cosine_sim


# Sidebar options
st.sidebar.markdown(
    """
    <h2 style='text-align: center; color: #E76F51;'>Options</h2>
    """, 
    unsafe_allow_html=True
)

generate_keywords = st.sidebar.checkbox("Enable Keyword Generation", value=True)
generate_title = st.sidebar.checkbox("Enable Title Generation", value=True)
summary_length = st.sidebar.slider("Summary Length", min_value=10, max_value=150, value=50, step=10)


st.subheader("Input Section")
uploaded_file = st.file_uploader("Upload a text file (PDF, DOCX, TXT):", type=['pdf', 'docx', 'txt'])
input_text = st.text_area("Or enter your text here:", "", height=200, max_chars=2000)

uploaded_text = ""
if uploaded_file is not None:
    file_type = uploaded_file.type
    if file_type == "application/pdf":
        uploaded_text = read_pdf(uploaded_file)
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        uploaded_text = read_word(uploaded_file)
    elif file_type == "text/plain":
        uploaded_text = StringIO(uploaded_file.read().decode("utf-8")).read()

input_text = input_text or uploaded_text

if st.button("Generate Output"):
    if not input_text.strip():
        st.error("Please provide input text or upload a file.")
    else:
        
        with st.spinner("🔍 Generating summary with both models..."):
            try:
                # Load the models
                summarizer_bart, summarizer_t5, keyword_model, title_generator = load_models()

                # Generate summaries with both models (full text without truncation)
                summary_bart = summarizer_bart(input_text, max_length=summary_length, min_length=20, truncation=False)
                summary_t5 = summarizer_t5(input_text, max_length=summary_length, min_length=20, truncation=False)

                # Extract the summary texts
                bart_summary = summary_bart[0]['summary_text']
                t5_summary = summary_t5[0]['summary_text']

                st.subheader("Comparison of Summaries")
                st.text_area(f"Summary using BART:", bart_summary, height=200)
                st.text_area(f"Summary using T5:", t5_summary, height=200)
                if generate_keywords:
                    with st.spinner("🔑 Extracting keywords..."):
                        try:
                            keywords = keyword_model.extract_keywords(input_text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=5)
                            keyword_list = [kw[0] for kw in keywords]
                            st.text_area("Keywords:", ", ".join(keyword_list), height=50)
                        except Exception as e:
                            st.error(f"Keyword Generation Error: {str(e)}")

        
                if generate_title:
                    with st.spinner("📝 Generating title..."):
                        try:
                            title_prompt = f"generate a title for the following text: {input_text[:512]}"
                            title_output = title_generator(title_prompt, max_length=10, num_return_sequences=1)
                            generated_title = title_output[0]['generated_text'].strip()
                            st.text_area("Generated Title:", generated_title, height=50)
                        except Exception as e:
                            st.error(f"Title Generation Error: {str(e)}")

                # Compare the summaries using different metrics
                cosine_sim_bart = calculate_cosine_similarity(input_text, bart_summary)
                cosine_sim_t5 = calculate_cosine_similarity(input_text, t5_summary)

                rouge_bart = calculate_rouge_score(input_text, bart_summary)
                rouge_t5 = calculate_rouge_score(input_text, t5_summary)

                bert_sim_bart = calculate_bert_similarity(input_text, bart_summary)
                bert_sim_t5 = calculate_bert_similarity(input_text, t5_summary)

                # Prepare data for the table
                st.title("Comparison Result")
                data = {
                    "Model": ["BART", "T5"],
                    "Cosine Similarity": [cosine_sim_bart, cosine_sim_t5],
                    "ROUGE Score (R1)": [rouge_bart['rouge1'].fmeasure, rouge_t5['rouge1'].fmeasure],
                    "ROUGE Score (R2)": [rouge_bart['rouge2'].fmeasure, rouge_t5['rouge2'].fmeasure],
                    "ROUGE Score (RL)": [rouge_bart['rougeL'].fmeasure, rouge_t5['rougeL'].fmeasure],
                    "BERT Similarity": [bert_sim_bart, bert_sim_t5],
                }

                # Convert the data into a DataFrame and display the table
                df = pd.DataFrame(data)
                st.table(df)

            except Exception as e:
                st.error(f"Error generating summary: {e}")