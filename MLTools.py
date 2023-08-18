def column_ratio(X):
    return X[:, [0]]/X[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    return["ratio"]

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None, n_init=10):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state
        self.n_init = 10
        
    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state, n_init=self.n_init)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self # Necessary, must always be returned for fit
    
    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)
    
    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]

from sklearn.preprocessing import PowerTransformer
from sklearn.metrics.pairwise import linear_kernel
class AgeSimilarity(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.powertransformer_ = PowerTransformer(method='box-cox')
        # self.powertransformer_.fit(X.values.reshape(-1, 1))
        self.powertransformer_.fit(X)
        return self
    
    def transform(self, X):
        # return linear_kernel(X.values.reshape(-1, 1))
        return linear_kernel(X)

from pathlib import Path
import tarfile
import pandas as pd
def load_housing_data():
    tarball_path=Path("./datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("./datasets").mkdir(parents=True, exist_ok=True)
        url="https://github.com/ageron/data/raw/main/housing.tgz"

        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    print(Path("./datasets/housing/housing.csv"))
    return pd.read_csv(Path("./datasets/housing/housing.csv"))