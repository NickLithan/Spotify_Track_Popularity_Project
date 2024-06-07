import streamlit as st
import sqlite3
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from data_prep import DataPrep

# data
df_ext = pd.read_csv('data/df_ext.csv', index_col=0)
df_ext = df_ext[df_ext['track_name'] != '']

# catboost
data_prep = DataPrep()
_, _, X_upd, y_upd = data_prep.fit_transform(df_ext)
cat = CatBoostRegressor(verbose=False).fit(X_upd, y_upd)
y_pred = cat.predict(X_upd)
y_pred = np.round(np.clip(y_pred, 0, 100))

# SQLite database
conn = sqlite3.connect('data/game_results.db')
c = conn.cursor()

c.execute('''
          CREATE TABLE IF NOT EXISTS game_results
          (row_id INT,
          user_prediction INT,
          predicted_popularity INT,
          actual_popularity INT)
          ''')

# update score board
def save_game_result(row_id, user_prediction, predicted_popularity, actual_popularity):
    global score_str
    c.execute('''
              INSERT INTO game_results 
              (row_id, user_prediction, predicted_popularity, actual_popularity) 
              VALUES (?, ?, ?, ?)
              ''', (row_id, user_prediction, predicted_popularity, actual_popularity))
    conn.commit()
    c.execute('''
              SELECT sum(
                  0.5*(ABS(user_prediction - actual_popularity) < ABS(predicted_popularity - actual_popularity)) + 
                  0.5*(ABS(user_prediction - actual_popularity) <= ABS(predicted_popularity - actual_popularity))
                  ) 
              FROM game_results''')
    score = c.fetchone()[0]
    c.execute('''
              SELECT count(*) 
              FROM game_results''')
    max_score = c.fetchone()[0]
    score_str = f"{score}/{max_score}"
    conn.commit()
    
col1, col2, col3 = st.columns(3)
    
with col2:
    st.title('Guess the Track Popularity (Game)')

# add option to reset the score
with col1:
    if st.button('Reset Score'):
        c.execute('DELETE FROM game_results')

# displaying the score     
score_str = None


selected_row = None

def generate_row():
    global selected_row
    selected_row = df_ext.sample(1)
    
if selected_row is None:
    generate_row()
    
track_name = selected_row['track_name'].values[0]
artists = selected_row['artists'].values[0]
artists = ', '.join(artists.split(';'))

with col2:
    st.write(f'**Track Name:** {track_name}.')
    st.write(f'**Artists:** {artists}.')

save_game_result(0,0,0,0)