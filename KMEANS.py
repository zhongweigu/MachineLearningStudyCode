import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

'''
X = np.array([[1,2],[1,4],[1,0],
              [4,2],[4,4],[4,0]])

kmeans = KMeans(n_clusters=2, random_state=0)

kmeans.fit(X)

print(kmeans.labels_)

print(kmeans.cluster_centers_)
'''

def kmeans(X,k,max_iter=100):
    centers = X[np.random.choice(X.shape[0], k, replace=False)]
    labels = np.zeros(X.shape[0])

    for i in range(max_iter):
        distances = np.sqrt(((X - centers[:,np.newaxis])**2).sum(axis=2))
        new_labels = np.argmin(distances, axis=0)

        for j in range(k):
             centers[j] = X[new_labels == j].mean(axis=0)

        if(labels == new_labels).all():
            break
        else:
            labels = new_labels
    return labels,centers


X = np.vstack((
    np.random.randn(100,2)*0.75 + [1,0],
    np.random.randn(100,2)*0.25 + [-0.5,0.5],
    np.random.randn(100,2)*0.5 + [-0.5,-0.5]
))
labels,centers = kmeans(X,k=3)

plt.scatter(X[:,0],X[:,1],c=labels)
plt.scatter(centers[:,0],centers[:,1],marker='x',s=200,linewidths=3,color='red')
plt.show()