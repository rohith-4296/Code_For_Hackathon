import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import csv


def cluster_data():
    dataset = pd.read_csv("hackathon_dataset.csv")
    x = dataset.iloc[:, 3:11].values

    ct = ColumnTransformer(
        [('oh_enc', OneHotEncoder(sparse=False), [0, 1, 2, 3, 4]), ],  # the column numbers I want to apply this to
        remainder='passthrough'  # This leaves the rest of my columns in place
    )
    x = ct.fit_transform(x)

    kmeans = KMeans(n_clusters=4, init="k-means++")
    y = kmeans.fit_predict(x)

    filename = 'Cluster_Category.csv'
    with open(filename, 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        # writing the fields
        csvwriter.writerow(y)


cluster_data()

