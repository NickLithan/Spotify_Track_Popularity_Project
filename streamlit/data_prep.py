import numpy as np
import pandas as pd

artists_popularity = pd.read_csv('data/artists_popularity.csv')

class TargetEncoder:
    """Does target encoding for categorical features with several categories in one row.
    """    
    def fit(self, data: pd.DataFrame, categorical_feature: str, target: str, cat_sep: str=';') -> None:
        """Creates target encoding based on passed data, even if there are several categories for each row.

        Args:
            data (pd.DataFrame): DataFrame from which to take the data for calculating encoding.
            cateorical_feature (str): The column you want to encode.
            target (str): The target column, based on which the target encoding is done.
            cat_sep (str, optional): Separator of categories in a 'cateorical_feature' cell. Defaults to ';'.
        """
        # remember target values for each category        
        category_targets = {}
        for i, categories in enumerate(data[categorical_feature].transform(lambda x: x.split(cat_sep))):
            for category in categories:
                if category not in category_targets.keys():
                    category_targets[category] = [data[target][i]]
                else:
                    category_targets[category].append(data[target][i])
        # each category is encoded with mean target value
        self.encoding = {category: np.mean(targets) for category, targets in category_targets.items()}
        # what to substitute the unknown category with: average target value            
        self.default = data[target].mean()
        
    def encode_cell(self, categories: str, cat_sep: str=';') -> np.float64:
        """Encodes a category cell.

        Args:
            categories (str): The cell you want to encode.
            cat_sep (str, optional): Separator of categories in a 'cateorical_feature' cell. Defaults to ';'.

        Returns:
            np.float64: Mean of target encodings of categories in cell.
        """
        return np.mean(list(map(
            lambda x: self.encoding.get(x, self.default), # encode each category; if not encoded, defaults to self.na
            categories.split(cat_sep)))) # separates categories in cell
        
    def transform(self, data: pd.DataFrame, categorical_feature: str) -> pd.Series:
        """Applies encoding. Unknown values are replaced with average target value from train data.

        Args:
            data (pd.DataFrame): DataFrame from which to take the data to encode.
            cateorical_feature (str): The column you want to encode.

        Returns:
            pd.Series:  The new column of your DataFrame with target encoded categorical feature.
        """
        return data[categorical_feature].transform(self.encode_cell)
        
    def fit_transform(self, data: pd.DataFrame, categorical_feature: str, target: str, cat_sep: str=';') -> pd.Series:
        """Fits the encoder and transforms the given data at once.

        Args:
            data (pd.DataFrame): DataFrame from which to take the data for calculating encoding.
            cateorical_feature (str): The column you want to encode.
            target (str): The target column, based on which the target encoding is done.

        Returns:
            pd.Series: The new column of your DataFrame with target encoded categorical feature.
        """
        self.fit(data, categorical_feature, target, cat_sep)
        return self.transform(data, categorical_feature)
    
class DataPrep:
    def __init__(self, to_encode: str='track_genre', target1: str='popularity', target2: str='updated_pop') -> None:
        """Prepares the data for further analysis. Selects data to encode. Makes an artist-populairty dictionary.

        Args:
            to_encode (str): Which column to do target encoding on. Defaults to 'track_genre'.
            target1 (str): The first target for target encoding. Defaults to 'popularity'.
            target2 (str): The second target for target encoding. Defaults to 'updated_pop'.
        """
        # target encoders to encode the genres by: 1. popularity, 2. updated_pop
        self.encoder_1 = TargetEncoder()
        self.encoder_2 = TargetEncoder()
        
        # storing genre column, as well as 'popularity' and 'updated_pop' in variables
        self.to_encode = to_encode
        self.target1 = target1
        self.target2 = target2
        
        # we'll use this dictionary for easy encoding of artists' popularity
        self.artists_pop_dict = {}
        for _, row in artists_popularity.iterrows(): # iterrating over data/artists_popularity.csv
            self.artists_pop_dict[row['artist_id']] = row['popularity'] # { (artist): (popularity), }
        
    def fit(self, data: pd.DataFrame) -> None:
        """Does target encoding fitting.

        Args:
            data (pd.DataFrame): Where to take data from.
        """
        # fitting target encoders
        self.encoder_1.fit(data, self.to_encode, self.target1)
        self.encoder_2.fit(data, self.to_encode, self.target2)
        
    def transform(self, data: pd.DataFrame) -> tuple:
        """Does target encoding transformation and separates data to features and targets.

        Args:
            data (pd.DataFrame): Where to take data from.

        Returns:
            tuple: A tuple with the features tables, the first target, and the second target.
        """
        # separating targets
        y1, y2 = data[self.target1], data[self.target2]
        
        # forming the dataset with no reference to up-to-date data
        X1 = data.drop(self.target2, axis=1) # remove 'updated_pop'
        X1[f'{self.to_encode}_{self.target1}'] = self.encoder_1.transform(X1, self.to_encode) # encode genres
        X1['n_artists'] = X1['artist_ids'].str.split(';').transform(len)
        X1 = X1[
            X1.columns[X1.dtypes != np.dtype('O')] # remove all 'object' columns
            ].drop(self.target1, axis=1) # remove targets
        
        
        X2 = data.drop(self.target1, axis=1)
        X2[f'{self.to_encode}_{self.target2}'] = self.encoder_2.transform(X2, self.to_encode)
        X2['n_artists'] = X2['artist_ids'].str.split(';').transform(len)
        X2['mean_artist_pop'] = X2['artist_ids'].str.split(';').transform(lambda x: np.mean([
            self.artists_pop_dict[artist] for artist in x
        ]))
        X2['max_artist_pop'] = X2['artist_ids'].str.split(';').transform(lambda x: max(
            self.artists_pop_dict[artist] for artist in x
        ))        
        X2 = X2[
            X2.columns[X2.dtypes != np.dtype('O')]
            ].drop(self.target2, axis=1)
        
        return X1, y1, X2, y2
    
    def fit_transform(self, data: pd.DataFrame) -> tuple:
        """Fits and transforms the data.

        Args:
            data (pd.DataFrame): Where to take data from.

        Returns:
            tuple: A tuple with the features table, the first target, and the second target.
        """
        self.fit(data)
        return self.transform(data)
    
if __name__ == '__main__':
    print('\n')
    print("This file is only used to import the functions we've constructed in main.ipynb. :)")
    print('\n')