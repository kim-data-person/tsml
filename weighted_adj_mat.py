import numpy as np
import pandas as pd
import pickle
import math

def load_prefecture_coordinates_data():

    # Load the prefecture coordinates
    coordinates = pd.read_csv('data/japanese_prefectures_coordinates.csv').set_index('Prefecture')

    # Load the cleaned data
    with open('output/cleaned_data.pkl', 'rb') as f:
        df = pickle.load(f)

    # Get the list of prefecture names
    prefecture_names = list(df.keys())

    # Sanity check (both lists are ordered)
    assert coordinates.index.tolist() == prefecture_names # Always follow this order

    return coordinates, prefecture_names

# Calculate the distance between two points using lat and long
def get_distance_km(diff_lat_degree, diff_long_degree):
    diff_lat_rad = math.radians(diff_lat_degree)
    diff_long_rad = math.radians(diff_long_degree)

    earth_radius_km = 6371.0 # Globally-averaged earth's radius in kilometers

    distance_km = earth_radius_km * (diff_lat_rad**2 + diff_long_rad**2)**0.5
    return distance_km

# Define the matrix weight using distance
def get_weight(distance_km, sigma, epsilon):
    weight = np.exp(-distance_km**2 / sigma**2)
    if weight < epsilon:
        return 0
    else:
        return weight
    
def create_weighted_adjacency_matrix(coordinates, prefecture_names, sigma, epsilon):
    # Create an empty weighted adjacency matrix
    W = np.zeros((len(prefecture_names), len(prefecture_names)))

    for i in range(W.shape[0]):
        for j in range(W.shape[0]):
            if i != j:
                lat1 = coordinates.loc[prefecture_names[i]]['Latitude']
                long1 = coordinates.loc[prefecture_names[i]]['Longitude']
                lat2 = coordinates.loc[prefecture_names[j]]['Latitude']
                long2 = coordinates.loc[prefecture_names[j]]['Longitude']

                distance_km = get_distance_km(lat1-lat2, long1-long2)
                W[i, j] = get_weight(distance_km, sigma, epsilon)

    W = W.astype(np.float32)


    return W

def get_weighted_adjacency_matrix(sigma=100, epsilon=10e-4):
    coordinates, prefecture_names = load_prefecture_coordinates_data()
    W = create_weighted_adjacency_matrix(coordinates, prefecture_names, sigma, epsilon)

    return W, prefecture_names