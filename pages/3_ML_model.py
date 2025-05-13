import streamlit as st
import pandas as pd
import src.ML as ML

ONE_HOT_THRESHOLD = 15


st.title("Describe and Summaries")

st.markdown(
    f"""
    In this section, you can **train and compare different machine learning models**, each using default parameters.  
    <br> The goal is to quickly identify which type of model performs best on your dataset.  
    <br> Regression and classification models are available, choose an appropriate one according to the selected target variable.

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
if "regression_results" not in st.session_state:
    st.session_state["regression_results"] = []
if "classification_results" not in st.session_state:
    st.session_state["classification_results"] = []


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
    st.write("Encoded dataset:", df_reg.head(2))
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
    "OLS regression", "Lasso regression", "Ridge regression", "Random-Forest-r regression", "XGBoost-r regression", 
    "Logistic classification", "Random-Forest-c classification", "XGBoost-c classification"
    ])
model, type = model_type.split(' ')
st.warning(f'You chose a {type} model, make sure the selected target variable is adapted.')
split = st.slider("Percentage of training set size", 0.1, 0.95, 0.8)


# Train model and display results
if st.button("Launch training"):

    if features==[]: 
        st.warning("Please, select regressive variables")

    else:
        if type=="regression": 
            res, fig, rmse, mae = ML.run_model(df_reg, target, features, model, type, split)
            st.markdown("""### Training results:""")
            st.write(res)
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.plotly_chart(fig)
            
            # Save model results for comparison 
            st.session_state["regression_results"].append({
                "Model": model_type,
                "Target": target,
                "RMSE": rmse,
                "MAE": mae, 
                "Train_split": split,
                "Features": features
            })

        else: 
            res, fig, acc, prec, rec, f1 = ML.run_model(df_reg, target, features, model, type, split)
            st.markdown(f"### {target} classification report:")
            st.text(res)
            if model=='XGBoost-c':
                matrix, metrics = fig
                st.markdown("### Confusion matrix:")
                st.pyplot(matrix)
                st.markdown("<br><br>", unsafe_allow_html=True)
                st.markdown("### Training results:")
                st.pyplot(metrics)

            else: 
                st.markdown("### Confusion matrix:")
                st.pyplot(fig)
            
            # Save model results for comparison 
            st.session_state["classification_results"].append({
                "Model": model_type,
                "Target": target,
                "Accuracy": acc,
                "Precision": prec, 
                "Recall": rec,
                "F1-score": f1,
                "Train_split": split,
                "Features": features
            })


# Display model comparison 
if st.session_state["regression_results"]:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.subheader("Regression Comparison")
    df_results = pd.DataFrame(st.session_state["regression_results"])
    st.write(df_results)

    # Reset model results
    if st.button("Reset regression results"):
        st.session_state["regression_results"] = []
        st.success("Regression models results have been reset")

if st.session_state["classification_results"]:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.subheader("Classification Comparison")
    df_results = pd.DataFrame(st.session_state["classification_results"])
    st.write(df_results)

    if st.button("Reset classification results"):
        st.session_state["classification_results"] = []
        st.success("Classification models results have been reset")


    
    
