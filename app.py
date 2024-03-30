import streamlit as st
from sentence_transformers import SentenceTransformer, util

fine_tuned_model = SentenceTransformer('fine_tuned_model/')

def calculate_similarity(sentence1, sentence2):
    embedding1 = fine_tuned_model.encode(sentence1, convert_to_tensor=True)
    embedding2 = fine_tuned_model.encode(sentence2, convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(embedding1, embedding2)
    return similarity_score.item()

st.title("Sentence Similarity Calculator")
sentence1 = st.text_input("Enter the first sentence:")
sentence2 = st.text_input("Enter the second sentence:")

if sentence1 and sentence2:
    similarity_score = calculate_similarity(sentence1, sentence2)
    st.write("Similarity Score:", similarity_score)