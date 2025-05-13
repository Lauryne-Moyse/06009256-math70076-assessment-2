# 06009256-math70076-assessment-2

## WebApp data science - EDA & Machine learning models
This Streamlit application allows users to load, explore, preprocess and model their own datasets using an intuitive and interactive interface. It is designed for already skilled users to help them get a quick overview of their data set and gain relevant insight before carrying deeper analysis. 

## Project structure

- `Main_page.py` : main Streamlit page
- `pages/` : secondary pages of the app
- `src/` : source code (preprocessing, plots, modelization etc.)
- `tests/` : unit tests
- `data/` : example datasets
- `README.md` : this file
- `requirements.txt` : dependencies
- `.gitignore`

## Launch app

1. Create and activate virtual environment:
- $ python -m venv venv
- $ source venv/bin/activate

2. Install dependencies:
$ pip install -r requirements.txt

3. Run app
$ streamlit run Main_page.py

4. Load an example dataset '.csv' on main page from 'data/' folder


## Features
- Upload and validate your own .csv dataset (numerical or categorical variables).

- Automatic detection and warning for non-usable columns (e.g., free text).

- Visualizations:
    - Descriptive statistics, dtypes and missing values 
    - Heatmap of correlations
    - Histograms of distribution and bar plots
    - Scatter plot of 1 to 3 variables with possibility of color appearance parameter. 

- Preprocessing options: label encoding or one-hot encoding for categorical variables.

- Interactive variable selection for target and features.

- Training/test split control.

- Train and compare multiple regression models including:
    - OLS
    - Lasso
    - Ridge
    - Random forest
    - XGBoost.

- Model results caching and reseting. 


## Tech Stack
Python:

- Pandas
- Numpy
for computations and data management 

- Matplotlib
- Seaborn 
- Plotly
for visualization 

- Statsmodels
- Scikit-learn 
- XGBoost 
for machine learning

- Streamlit
for app designing 

- Pytest
for function testing

- Sys 
- Os
for file navigation. 

## Test source files

1. Write in a terminal: 
$ pytest tests/

2. Interact with the app