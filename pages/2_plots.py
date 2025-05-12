import streamlit as st
import src.plot as plot


st.title("Plots")

st.markdown(
    """
    In this section, you can generate a scatter plot of up to three variables. 
    <br> You can also optionally choose a variable to color the data points. 
    """, 
    unsafe_allow_html=True
    )

# Retrieve datatset 
df = st.session_state.get("df")
if df is None:
    st.warning("Please, load a file on the main page")
    st.stop()


# Sort columns by type
num_cols = df.select_dtypes(include='number').columns.tolist()
categorical_cols = df.select_dtypes(include='string').columns.tolist()

# Select up to 3 numeric variables to plot together and 1 optional color variable
x = st.selectbox("Numeric variable X", ["---"]+num_cols)
if x!="---":
    y = st.selectbox("Variable Y (optional)", ["---"]+num_cols)
else:
    y = "---"
if y!="---":
    z = st.selectbox("Variable Z (optional)", ["---"]+num_cols)
else:
    z = "---"
hue = st.selectbox("Variable for coloring(optional)", ["---"]+list(df.columns))


# Trace plots when at least x is specified
if x!="---":
    fig = plot.plot_chart(df, x, y, z, hue)
    st.plotly_chart(fig)