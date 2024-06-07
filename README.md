# Spotify Track Popularity Project

## Description

The aim of this project is to 
- Find out if it is possible to predict the popularity of a track using only its metadata and audio features (e.g. track genre, tempo, instrumentality, etc.);
- Identify the most informative data for this purpose.

The hypothesis is that accurate predictions of popularity are impossible if we limit ourselves to tabular data. However, ML can work wonders, so a full analysis is necessary.

## Technology used

Most of the code is written in the Jupyter notebook main.ipynb, with all steps of the data analysis commented.

In the process of creating the project, 
1. **REST API** (specifically the Spotify API) was used to retrieve the data;
2. **Selenium** was used for web-scraping (retrieving information about the dataset from the Kaggle website);
3. **Pandas** was used extensively  to visualise and transform the tabular data (transformations, grouping, etc.);
4. **NumPy** was used throughout for various calculations;
5. **Folium** was used to visualise the geodata (countries where Spotify operates);
6. **Matplotlip** and **Seaborn** were used for non-trivial data visualisation;
7. **SQL** for scoring in the popularity prediction game;
8. **Streamlit** for hosting the popularity prediction game;
9. **Sklearn** and **Catboost** were used for machine learning (linear regression and gradient boosting);
10. Additional technology: 
- **opendatasets** to retrieve a dataset from Kaggle, 
- **typing** for documentation, 
- **tqdm** for progress bars,
- **pycountry** to convert 2-letter country codes to 3-letter codes,
- **matplotlib.colors** to create a custom color gradient,
- various **sklearn** technologies for statistical tests and data pre-processing, as well as regression metrics.

## Usage

To explore the data analysis part of the project, read main.ipynb.

To try playing against my algotithm, go to