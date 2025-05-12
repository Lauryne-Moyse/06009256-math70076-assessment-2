import streamlit as st
import src.plot as plot


st.title("Plots")

df = st.session_state.get("df")

if df is None:
    st.warning("Please, load a file on the main page")
    st.stop()


num_cols = df.select_dtypes(include='number').columns.tolist()
categorical_cols = df.select_dtypes(include='string').columns.tolist()

x = st.selectbox("Numeric variable X", ["---"]+num_cols)
if x!="---":
    y = st.selectbox("Variable Y (optional)", ["---"]+num_cols)
else:
    y = "---"
if y!="---":
    z = st.selectbox("Variable Z (optional)", ["---"]+num_cols)
else:
    z = "---"
hue = st.selectbox("Coloring categorical variable (optional)", ["---"]+list(df.columns))


# Trace plots once at least x is specified
if x!="---":
    fig = plot.plot_chart(df, x, y, z, hue)
    st.plotly_chart(fig)