import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn import decomposition


iris = datasets.load_iris()

pca = decomposition.PCA()
pca.fit(iris.data)

#Ajout sur le remote
