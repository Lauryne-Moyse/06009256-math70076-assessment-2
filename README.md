# 06009256-math70076-assessment-2

## WebApp data science - EDA & Machine learning models
This Streamlit application allows users to load, explore, preprocess and model their own datasets using an intuitive and interactive interface. The tool is designed for quick experimentation and comparison of multiple regression models. It globally and aims to save time on the preliminary stages of a data study.

## Project structure

- `Main_page.py` : main Streamlit page
- `pages/` : secondary pages of the app
- `src/` : source code (preprocessing, plots, modelization etc.)
- `tests/` : unit tests
- `data/` : example datasets
- `README.md` : this file
- `requirements.txt` : dependencies

## Launch app

1. Install dependancies:
pip install -r requirements.txt

2. Run app
streamlit run Main_page.py

3. Load a dataset '.csv' on main page from 'data/' folder


## Features
<li> Upload and validate your own .csv dataset (numerical or categorical variables).

<li> Automatic detection and warning for non-usable columns (e.g., free text).

<li> Visualizations:
    • Descriptive statistics, dtypes and missing values 
    • Heatmap of correlations
    • Histograms of distribution and bar plots
    • Scatter plot of 1 to 3 variables with possibility of color appearance parameter. 

<li> Preprocessing options: label encoding or one-hot encoding for categorical variables.

<li> Interactive variable selection for target and features.

<li> Training/test split control.

<li> Train and compare multiple regression models including:
    • OLS
    • Lasso
    • Ridge
    • Random forest
    • XGBoost.

<li> Model results caching and reseting. 


## Tech Stack
Python:

• Pandas
• Numpy
for computations and data management 

• Matplotlib
• Seaborn 
• Plotly
for visualization 

• Statsmodels
• Scikit-learn 
• XGBoost 
for machine learning

• Streamlit
for app designing 

• Pytest
for function testing

• Sys 
• Os
for file navigation. 

## Test source files

1. Write in a terminal: 
pytest tests/

2. Interact with the app