def clean_dataset_for_app(df, cat_threshold):
    """
    Evaluates if the dataset is clean for the app analysis:
    - Identifies numeric columns
    - Identifies type 'object' columns considered as categorical
    - Identifies problematic columns

    Parameters:
        df (pd.DataFrame): input dataset
        cat_threshold (int): max number of modalities to have a type 'object' column as categorical

    Returns:
        numeric_cols (list): numeric columns of df
        categorical_cols (list): categorical columns of df with num_cat<cat_threshold
        problematic_cols (list): identified problematic columns of df for analysis
    """

    # Sort columns
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    object_cols = df.select_dtypes(include='object').columns
    categorical_cols = [col for col in object_cols if df[col].nunique() <= cat_threshold]
    problematic_cols = [col for col in object_cols if col not in categorical_cols]

    return numeric_cols, categorical_cols, problematic_cols

    