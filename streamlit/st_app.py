import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

st.title('Spotify Track Popularity Project')

st.write("*The idea of the project is that it might be possible \
    to predict the popularity of the track before its release \
        simply based on its metadata and audio features.*")

st.write('I started off with a dataframe from **Kaggle**:')

df = pd.read_csv('-spotify-tracks-dataset/dataset.csv', index_col=0)
st.dataframe(df.head(10))

st.write('Then I added some data via **Spotify API**:')

df_ext = pd.read_csv('streamlit/data/df_ext.csv', index_col=0)
st.dataframe(df_ext.head(10))

st.write("Then I selected the updated_pop column as the **target** - \
    the variable we'd like to be able to predict. \
        I had to also do some encoding on the categorical data in the table. \
            After that was done, we had an updated version of the dataset:")

X_upd = pd.read_csv('streamlit/data/X_upd.csv', index_col=0)
st.dataframe(X_upd.head(10))

st.write('I decided to look at where Spotify operates \
    (greeen countries are where you can use Spotify):')

with open('streamlit/data/markets.html') as f:
    map = f.read()
    
components.html(map, height=600)

st.write("Then I calculated and visualized the dataset's columns correlations:")

st.image('streamlit/data/correlations.png')

st.write("Overall, I've done 4 different feature importance tests, \
    after which I found the score for all of them \
        and rated them based on it: ")

st.image('streamlit/data/scores.png')

st.write('For more details, see main.ipynb.')