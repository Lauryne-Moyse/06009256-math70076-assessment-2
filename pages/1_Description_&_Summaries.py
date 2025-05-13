import streamlit as st
import src.eda as eda
import pandas as pd


st.title("Description and Summaries")

# Retrieve dataset
df = st.session_state.get("df")
if df is None:
    st.warning("Please, load a file on the main page")
    st.stop()


# Descriptive statistics 
st.subheader("Descriptive statistics")
st.write(df.describe())

# Correlation matrix
st.subheader("Correlation matrix")
fig = eda.corr_heatmap(df)
st.pyplot(fig)

# Missing values
st.subheader("Missing values")
st.write(df.isna().sum())


# Data visualization 
st.markdown(
    """
    ### Variables Distribution Visualization

    In this interface, variables distribution are displayed using **histograms** for numerical variables and **bar plots** for categorical ones.

    Sometimes, a numerical variable can meaningfully be treated as **categorical** — for example, a score out of 10, a year, or a binary flag.
    <br> By default, numerical variables with more than **20 unique values** are treated as continuous. Below this threshold, users can choose to visualize them either as a histogram or as a bar plot.

    This threshold can be adjusted on a per-variable basis using the slider, allowing flexibility to better match the characteristics of your dataset.
    """, 
    unsafe_allow_html=True
    )

# Display histogram or bar plot and short summary for each column
for col in df.columns:

    col_data = df[col]
    st.markdown(f"### {col}")

    # Choose max number of categories for a variable
    max_categories = st.slider(
    f"Maximum number of unique values to treat {col} as categorical",
    min_value=1,
    max_value=200,
    value=20,
    step=1,
    key=f"{col}_max_cat"
    )

    if pd.api.types.is_numeric_dtype(col_data):

        if col_data.nunique()<=max_categories: # Limit number of categories for clarity 
            type = st.radio(
                f"Type of numerical column", 
                ["continuous", "categorical"],
                key=col
                )
        else:
            type = "continuous"
    
    else:
        type = "categorical"

    
    # Generate summary and plot
    summary = eda.analyse_column(col_data, type)
    fig = eda.plot_column(col_data, type)
    
    st.write("**Type :**", summary["dtype"])
    st.write("**Missing values :**", summary["missing"])
    if summary["skewness"] is not None:
        st.write("**Skewness :**", round(summary["skewness"], 2))
    st.pyplot(fig)
