import streamlit as st
import pandas as pd
import src.ML as ML



st.title("Describe and Summaries")

df = st.session_state.get("df")

if "model_results" not in st.session_state:
    st.session_state["model_results"] = []

if df is None:
    st.warning("Please, load a file on the main page")
    st.stop()


cat_cols = df.select_dtypes(include='object').columns.tolist()
to_label_cols = st.multiselect(
    "Categorical variables to transform with label encoding (optional):", 
    [col for col in cat_cols]
)
to_one_hot_cols = st.multiselect(
    "Categorical variables to transform with one-hot encoding (optional):", 
    [col for col in cat_cols if col not in to_label_cols]
)



warning_cols_names, warning_cols_length = ML.one_hot_warning(
    df, cols=to_one_hot_cols, threshold=3
    )

if warning_cols_names:
    col_msgs = ", ".join(f"{name} ({n} unique)" for name, n in zip(
        warning_cols_names, warning_cols_length
        ))
    st.warning(
        f"High-cardinality columns: {col_msgs}.\n\n"
        f"Consider using Label Encoding instead for these."
    )


df_reg = df.copy()
if to_label_cols or to_one_hot_cols:
    df_reg, mappings = ML.get_df_encoded(df_reg, to_label_cols, to_one_hot_cols)
st.subheader("Preview of dataset")
st.write(df_reg.head(5))
if to_label_cols:
    st.write("Label encoding mapping", mappings)


num_cols = df_reg.select_dtypes(include='number').columns.tolist()
target = st.selectbox("Predicted variable :", num_cols)
features = st.multiselect("Explicative variables :", [
    col for col in num_cols if col != target
    ])


model_type = st.radio("Choose a model:", [
    "OLS", "Lasso", "Ridge", "Random Forest", "XGBoost"
    ])
split = st.slider("Train/Test split ratio", 0.1, 0.95, 0.8)


if st.button("Launch training"):

    if features==[]: 
        st.warning("Please, select regressive variables")
    else: 
        res, fig, rmse, mae = ML.run_model(df_reg, target, features, model_type, split)
        st.markdown("""### Training results:""")
        st.write(res)
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.plotly_chart(fig)

        #st.download_button("Download summary", res, file_name="ols_summary.txt")

        
        # Save regression results for comparison 
        st.session_state["model_results"].append({
            "Model": model_type,
            "Target": target,
            "RMSE": rmse,
            "MAE": mae, 
            "Features": features
        })

if st.session_state["model_results"]:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.subheader("Model Comparison")
    df_results = pd.DataFrame(st.session_state["model_results"])
    st.write(df_results)
    
    if st.button("Reset model results"):
        st.session_state["model_results"] = []
        st.success("Model results have been reset")
