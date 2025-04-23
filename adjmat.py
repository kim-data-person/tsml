import numpy as np
import pandas as pd
import pickle
import math

def load_data():

    # Load the prefecture coordinates
    coordinates = pd.read_csv('data/japanese_prefectures_coordinates.csv').set_index('Prefecture')

    # Load the cleaned data
    with open('data/cleaned_data.pkl', 'rb') as f:
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
def get_weight(distance_km, sigma=100, epsilon=10e-5):
    weight = np.exp(-distance_km**2 / sigma**2)
    if weight < epsilon:
        return 0
    else:
        return weight
    
def create_adjacency_matrix(coordinates, prefecture_names):
    # Create an empty adjacency matrix
    A_adj = np.zeros((len(prefecture_names), len(prefecture_names)))

    for i in range(A_adj.shape[0]):
        for j in range(A_adj.shape[0]):
            if i != j:
                lat1 = coordinates.loc[prefecture_names[i]]['Latitude']
                long1 = coordinates.loc[prefecture_names[i]]['Longitude']
                lat2 = coordinates.loc[prefecture_names[j]]['Latitude']
                long2 = coordinates.loc[prefecture_names[j]]['Longitude']

                distance_km = get_distance_km(lat1-lat2, long1-long2)
                A_adj[i, j] = get_weight(distance_km, sigma=100, epsilon=10e-4)

    return A_adj


if __name__ == "__main__":
    coordinates, prefecture_names = load_data()
    A_adj = create_adjacency_matrix(coordinates, prefecture_names)

    with open('output/adjacency_matrix.pkl', 'wb') as f:
        pickle.dump(A_adj, f)