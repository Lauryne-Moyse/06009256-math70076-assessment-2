import streamlit as st
import pandas as pd
import src.ML as ML

ONE_HOT_THRESHOLD = 15


st.title("Describe and Summaries")

st.markdown(
    f"""
    In this section, you can **train and compare different machine learning models**, each using default parameters.  
    <br> The goal is to quickly identify which type of model performs best on your dataset.  

    You can select the **target variable**, the **explanatory variables**, and the **proportion of data to use for training**.  
    <br> Non-numeric columns can be transformed using either **label encoding or one-hot encoding**.  
    <br> When using one-hot encoding, a warning will appear if a column has more than {ONE_HOT_THRESHOLD} unique values, as it may lead to high dimensionality issues.
    """,
    unsafe_allow_html=True
)



# Retrieve dataset
df = st.session_state.get("df")
if df is None:
    st.warning("Please, load a file on the main page")
    st.stop()

# Variable to save model results 
if "model_results" not in st.session_state:
    st.session_state["model_results"] = []


# Possibility of one-hot and label encoding for categorical variables
cat_cols = df.select_dtypes(include='object').columns.tolist()

# Columns to encode
to_label_cols = st.multiselect(
    "Categorical variables to transform with label encoding (optional):", 
    [col for col in cat_cols]
)
to_one_hot_cols = st.multiselect(
    "Categorical variables to transform with one-hot encoding (optional):", 
    [col for col in cat_cols if col not in to_label_cols]
)

# Warning for big dimensionality in one-hot encoding
warning_cols_names, warning_cols_length = ML.one_hot_warning(
    df, to_one_hot_cols, ONE_HOT_THRESHOLD
    )

if warning_cols_names:
    col_msgs = ", ".join(
        f"{name} ({n} unique)" for name, n in zip
        (warning_cols_names, warning_cols_length)
    )
    st.warning(
        f"High-cardinality columns: {col_msgs}.\n\n"
        f"Consider using Label Encoding instead for these."
    )


# Update and display new df if encoding, and eventual label mapping
df_reg = df.copy()
if to_label_cols or to_one_hot_cols:
    df_reg, mappings = ML.get_df_encoded(df_reg, to_label_cols, to_one_hot_cols)
    st.subheader("Encoded dataset:")
    st.write(df_reg.head(5))
if to_label_cols:
    st.write("Label encoding mapping:", mappings)


# Select predicted and explanatory variables
num_cols = df_reg.select_dtypes(include='number').columns.tolist()
target = st.selectbox("Predicted variable :", num_cols)
features = st.multiselect("Explanatory variables :", [
    col for col in num_cols if col != target
    ])


# Select machine learning model and train/test ratio
model_type = st.radio("Choose a model:", [
    "OLS", "Lasso", "Ridge", "Random Forest", "XGBoost"
    ])
split = st.slider("Percentage of training set size", 0.1, 0.95, 0.8)


# Train model and display results
if st.button("Launch training"):

    if features==[]: 
        st.warning("Please, select regressive variables")
    else: 
        res, fig, rmse, mae = ML.run_model(df_reg, target, features, model_type, split)
        st.markdown("""### Training results:""")
        st.write(res)
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.plotly_chart(fig)
        
        # Save regression results for comparison 
        st.session_state["model_results"].append({
            "Model": model_type,
            "Target": target,
            "RMSE": rmse,
            "MAE": mae, 
            "Train_split": split,
            "Features": features
        })


# Display model comparison 
if st.session_state["model_results"]:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.subheader("Model Comparison")
    df_results = pd.DataFrame(st.session_state["model_results"])
    st.write(df_results)
    
    # Reset model results
    if st.button("Reset model results"):
        st.session_state["model_results"] = []
        st.success("Model results have been reset")
