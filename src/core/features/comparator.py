import numpy as np
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

class FeatureComparison:
    def __init__(self, threshold=0.9):
        self.threshold = threshold

    def cosine_similarity_matrix(self, target_feature_vector, feature_vectors):
        return cosine_similarity([target_feature_vector], feature_vectors)[0]

    def euclidian_distance(self, target_feature_vector, feature_vectors):
        return euclidean_distances(target_feature_vector, feature_vectors)[0]

    def top_similar(self, target_feature_vector, all_feature_vectors):
        neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute',
                                     metric='euclidean').fit(all_feature_vectors)

        distances, indices = neighbors.kneighbors(target_feature_vector)
        return distances, indices

    def compress_features(self, num_feature_dimensions, feature_list):
        pca = PCA(n_components=num_feature_dimensions)
        pca.fit(feature_list)
        return pca.transform(feature_list)
