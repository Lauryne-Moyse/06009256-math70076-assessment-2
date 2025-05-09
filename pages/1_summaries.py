import streamlit as st
import src.eda as eda


st.set_page_config(page_title="Describe and Summaries")

st.title("Describe and Summaries")

df = st.session_state.get("df")

if df is None:
    st.warning("Load a file on the main page")
    st.stop()

st.subheader("Desciptives statistics")
st.write(df.describe())

st.subheader("Correlation matrix")
st.write(df.corr())
fig = eda.corr_heatmap(df)
st.pyplot(fig)

for col in df.columns:
    col_data = df[col]

    summary = eda.analyse_column(col_data)
    st.markdown(f"### {col}")
    st.write("**Type :**", summary["dtype"])
    st.write("**Valeurs manquantes :**", summary["missing"])
    if summary["skewness"] is not None:
        st.write("**Skewness :**", round(summary["skewness"], 2))

    fig = eda.plot_column(col_data)
    st.pyplot(fig)