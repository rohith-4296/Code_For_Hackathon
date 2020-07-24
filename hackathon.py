# Importing important packages
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import csv

# Function for K-means clustering implementation.
# Used to group the people with similar interest.
def cluster_data():
    # Getting dataset
    dataset = pd.read_csv("hackathon_data.csv")
    # Considering important part of data for algorithm implementation.
    x = dataset.iloc[:, 3:11].values
    
    # One hot encoding.
    ct = ColumnTransformer(
        [('oh_enc', OneHotEncoder(sparse=False), [0, 1, 2, 3, 4]), ],  
        remainder='passthrough' 
    )
    x = ct.fit_transform(x)
    
    # Implementing K-means algorithm
    kmeans = KMeans(n_clusters=4, init="k-means++")
    y = kmeans.fit_predict(x)
    
    # Storing the output data in the shown file.
    filename = 'Cluster_Category.csv'
    with open(filename, 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        # writing the fields
        csvwriter.writerow(y)

cluster_data()
    
