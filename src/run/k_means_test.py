#coding:utf-8

from sklearn.cluster import KMeans
import numpy as np

import pandas as pd

X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

y_pre=kmeans.labels_

df=pd.DataFrame(y_pre)
print('df.dtypes')
print(df.dtypes)

lambda_result=df*100

print(lambda_result)

print('df= %s') %df



print kmeans.labels_

kmeans.predict([[0, 0], [4, 4]])

print kmeans.cluster_centers_
