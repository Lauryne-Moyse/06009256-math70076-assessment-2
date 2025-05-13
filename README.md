# 06009256-math70076-assessment-2

## WebApp data science - EDA & Machine learning models
This Streamlit application enables users to load, explore, preprocess and model their own datasets through an intuitive and interactive interface, without needing to write code. It is intended for users with some prior experience in data science to help them get a quick overview of their dataset and gain relevant insight before carrying deeper analysis. 

## Project structure

- `Main_page.py` : main Streamlit page
- `pages/` : secondary pages of the app
- `src/` : source code (preprocessing, plots, modelization etc.)
- `tests/` : unit tests
- `data/` : example datasets
- `README.md` : this file
- `requirements.txt` : dependencies
- `.gitignore`: files and directories to ignore in version control.

## Launch app

1. Create and activate virtual environment:
- $ python -m venv venv
- $ source venv/bin/activate # For Linux/Mac
- $ venv\Scripts\activate     # For Windows

2. Install dependencies:
$ pip install -r requirements.txt

3. Run unit tests:
$ pytest test/

4. Run app:
$ streamlit run Main_page.py

5. Load an example dataset '.csv' on main page from 'data/' folder


## Features
- Upload and validate your own .csv dataset (numerical or categorical variables).

- Automatic detection and warning for non-usable columns (e.g., free text).

- Data visualization:
    - Descriptive statistics, dtypes and missing values 
    - Heatmap of correlations
    - Histograms of distribution and bar plots
    - Scatter plot of 1 to 3 variables with color appearance parameter. 

- Model Training and Comparison:
    - Preprocessing options: label encoding or one-hot encoding for categorical variables
    - Interactive variable selection for target and features
    - Training/test split control 
    - Training and comparison of multiple regression and classification models, including OLS, Lasso, Ridge, Random forest, XGBoost and Logistic
    - Model results caching and reseting. 


## Tech Stack
Python:

- Pandas and Numpy: for data manipulation and computations.

- Matplotlib, Seaborn, and Plotly: for data visualization.

- Statsmodels, Scikit-learn and XGBoost: for machine learning models.

- Streamlit: for building the interactive web application.

- Pytest: for function testing and test automation.

- Sys and Os: for file and directory management.