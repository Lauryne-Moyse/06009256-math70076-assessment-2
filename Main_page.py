import streamlit as st
import pandas as pd
import src.main as main

CAT_THRESHOLD = 20


st.title("WebApp data science - EDA & Machine learning models")

st.markdown(
    """
    ### Welcome to this data science application!
    This app allows you to **load a dataset**, **explore**, **plot variables**, and then **train and evaluate various machine learning models**.
    <br> Load your dataset on this page and then move to other pages to perform your analysis. 
    <br> The dataset must be loaded as a **CSV file** and contain only **numerical or categorical columns**. 
    """, 
    unsafe_allow_html=True
    )

# Load data
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])


# Select file separator
sep = st.radio("Choose a separator:", [
    "Comma (',')",
    "Semicolon (';')",
    "Tab ('\\t')",
    "Pipe ('|')"
    ])
sep = sep.split("'")[1]


if uploaded_file is not None:

    # Load and read df
    try:
        df = pd.read_csv(uploaded_file, sep=sep, on_bad_lines='skip')
    except Exception as e:
        st.error(f"Error reading file: {e}")
    st.write("Original data preview:", df.head(2))

    # Clean df for analysis
    numeric_cols, categorical_cols, problematic_cols = main.clean_dataset_for_app(df, CAT_THRESHOLD)
    if problematic_cols !=[]:
        st.warning(
            f"The following columns appear to have too many unique values "
            f"(>{CAT_THRESHOLD}) and may not be suitable for encoding or modeling: "
            f"{', '.join(problematic_cols)}"
        )
        # Give possibility to exclude the columns
        if st.checkbox("Exclude these columns from the dataset"):
            df = df[numeric_cols + categorical_cols]
        else:
            st.info("Columns will remain in the dataset, but may cause issues later.")
    else:
        st.success("All columns seem usable (either numerical or low-cardinality categorical).")

    # Display (new) df
    st.subheader("Data preview for analysis:")
    st.write(df.head())

    # Store df for other pages
    st.session_state["df"] = df 



