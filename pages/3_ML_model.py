import streamlit as st
import src.eda as eda


st.set_page_config(page_title="Describe and Summaries")

st.title("Describe and Summaries")

df = st.session_state.get("df")

if df is None:
    st.warning("Load a file on the main page")
    st.stop()

num_cols = df.select_dtypes(include='number').columns.tolist()
target = st.selectbox("Predicted variable :", num_cols)
features = st.multiselect("Explicative variables :", [col for col in num_cols if col != target])
model_type = st.radio("Choose a model :", ["OLS", "XGBoost"])