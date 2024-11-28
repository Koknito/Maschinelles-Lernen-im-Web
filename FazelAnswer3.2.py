import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AffinityPropagation, SpectralClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score

# Function to create grid-structured data
def create_grid_data(grid_size=4, points_per_cluster=300):
    data = []
    labels = []
    for i in range(grid_size):
        for j in range(grid_size):
            center = [i * 7, j * 7]
            spread = [[2, 0], [0, 2]]  # Covariance matrix for multivariate normal distribution
            cluster_points = np.random.multivariate_normal(center, spread, points_per_cluster)
            data.append(cluster_points)
            labels.extend([i * grid_size + j] * points_per_cluster)
    return np.vstack(data), np.array(labels)

# Function to create circular/ring data
def create_circle_data(num_rings=10, points_per_ring=300):
    data = []
    labels = []
    for i in range(1, num_rings + 1):
        radii = np.random.normal(loc=i * 5, scale=1.5, size=points_per_ring)
        angles = np.random.uniform(0, 2 * np.pi, points_per_ring)
        x_coords = radii * np.cos(angles)
        y_coords = radii * np.sin(angles)
        ring_points = np.column_stack((x_coords, y_coords))
        data.append(ring_points)
        labels.extend([i] * points_per_ring)
    return np.vstack(data), np.array(labels)

# Function to perform clustering using a specified method
def cluster_data(data, method, n_clusters=None):
    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=42)
    elif method == 'affinity':
        model = AffinityPropagation(random_state=42)
    elif method == 'spectral':
        model = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)
    else:
        raise ValueError("Invalid clustering method. Choose from 'kmeans', 'affinity', or 'spectral'.")
    
    labels = model.fit_predict(data)
    return labels

# Function to visualize clustering results
def visualize_clusters(data, labels, title):
    unique_labels = np.unique(labels)
    for label in unique_labels:
        cluster_points = data[labels == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {label}", s=10)
    plt.title(title)
    plt.legend(fontsize=8)
    plt.show()

# Function to evaluate clustering performance
def evaluate_clustering(true_labels, predicted_labels, data):
    silhouette = silhouette_score(data, predicted_labels)
    ari = adjusted_rand_score(true_labels, predicted_labels)
    print(f"Silhouette Score: {silhouette:.3f}")
    print(f"Adjusted Rand Index: {ari:.3f}")
    return silhouette, ari

# Main function to execute clustering and evaluation
def main():
    # Generate datasets
    print("Generating datasets...")
    grid_data, grid_labels = create_grid_data(grid_size=4)
    circle_data, circle_labels = create_circle_data(num_rings=10)

    # Clustering and evaluation on grid data
    print("\nClustering on grid data...")
    for method in ['kmeans', 'affinity', 'spectral']:
        clusters = 16 if method != 'affinity' else None
        cluster_labels = cluster_data(grid_data, method, clusters)
        visualize_clusters(grid_data, cluster_labels, f"{method.capitalize()} - Grid Data")
        print(f"Evaluation for {method.capitalize()} - Grid Data:")
        evaluate_clustering(grid_labels, cluster_labels, grid_data)

    # Clustering and evaluation on circular data
    print("\nClustering on circular data...")
    for method in ['kmeans', 'affinity', 'spectral']:
        clusters = 10 if method != 'affinity' else None
        cluster_labels = cluster_data(circle_data, method, clusters)
        visualize_clusters(circle_data, cluster_labels, f"{method.capitalize()} - Circular Data")
        print(f"Evaluation for {method.capitalize()} - Circular Data:")
        evaluate_clustering(circle_labels, cluster_labels, circle_data)

# Execute the script
if __name__ == "__main__":
    main()