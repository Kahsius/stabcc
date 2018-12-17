import numpy as np
from sklearn.cluster import KMeans
from sklearn import mixture

class Kmeans_algo:
    def __init__(self, K, random_state=42):
        self.kmeans = KMeans(n_clusters=K, random_state=random_state)
        
    def fit_predict(self, X):
        return self.kmeans.fit_predict(X)
    
    def complexity(self, X, S):
        centers = self.kmeans.cluster_centers_
        XminusM = X - centers[S[:],:]
        return 1 + np.sum(np.log2(1 + np.abs(XminusM)))


class GMM_algo:
    def __init__(self, K, random_state=42):
        self.gmm = mixture.GaussianMixture(n_components=K, covariance_type='full')
        
    def fit_predict(self, X):
        self.gmm.fit(X)
        return self.gmm.predict(X)
    
    def complexity(self, X, S):
        return - self.gmm.score(X) * X.shape[0]


class constant_complexity_algo:
    def complexity(self, X, S):
        return 0



X = np.random.rand(100, 3)

kmeans = Kmeans_algo(3)
S = kmeans.fit_predict(X)
K_kmeans = kmeans.complexity(X, S)

gmm = GMM_algo(3)
S = gmm.fit_predict(X)
K_gmm = gmm.complexity(X, S)
