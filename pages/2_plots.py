import streamlit as st
import src.eda as eda


st.set_page_config(page_title="Plots")
st.title("Plots")

df = st.session_state.get("df")

if df is None:
    st.warning("Load a file on the main page")
    st.stop()


num_cols = df.select_dtypes(include='number').columns.tolist()
categorical_cols = df.select_dtypes(include='string').columns.tolist()
# Certaines variable snumériques sont catégorielles ? 
# Plot var_num selon var_cat c'est possible 
# sélectionner x, y, z tq y et z sont forcément numériques, x peut être catégorielle 