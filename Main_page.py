import streamlit as st
import pandas as pd

st.set_page_config(page_title="Main page")

st.title("WebApp data science - EDA & Machine learning models")

st.markdown("Bienvenue sur cette application de data science ! + explanation")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state["df"] = df
    st.subheader("Data preview :")
    st.dataframe(df.head())


