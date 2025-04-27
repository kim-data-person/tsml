import numpy as np
import pandas as pd
import pickle

def load_data():
    # Load prefecture name data
    prefecture_names = pd.read_csv("data/Japan_prefecture_name.csv")
    # We need the second row only
    prefecture_names = prefecture_names.iloc[1]

    # Load air pollution data of each prefecture
    features = ['date','pm10','pm25','so2','no2','o38h']
    df = { prefecture_names['V{}'.format(i)] : pd.read_csv("data/Japan_jap1219_{}.csv".format(i))[features] for i in range(1, 48, 1) }

    return df

def fill_missing_values(df):
    # Remove first N rows with NaN values
    first_row_index = {}
    for prefecture in df:
        # Get the first row index without NaN values
        first_row_index[prefecture] = df[prefecture].dropna().index[0]

    # Get the maximum index across all prefectures
    max_index = max(first_row_index.values())

    # Remove rows with index smaller than max_index
    for prefecture in df:
        df[prefecture] = df[prefecture].iloc[max_index:]

    # Interpolate NaN values
    for prefecture in df:  
        # Set the date as DatetimeIndex
        df[prefecture]['date'] = pd.to_datetime(df[prefecture]['date'])
        df[prefecture] = df[prefecture].set_index('date')
        # Interpolate NaN values
        df[prefecture] = df[prefecture].interpolate(method='time')

    return df

def save_data(df):
    # Save the cleaned data as pickle file
    with open('output/cleaned_data.pkl', 'wb') as f:
        pickle.dump(df, f)

if __name__ == "__main__":
    df = load_data()
    df = fill_missing_values(df)
    save_data(df)
