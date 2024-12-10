import os
default_n_threads = 16
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import joblib
from sklearn.cluster import KMeans, MiniBatchKMeans
from vlannotator.utils import prepare_future_trajectories_with_x_forward


ANNOTATED_DATA_PATH = '/home/shaoze.yang/data/nuplan_to_vad/output/trainval_1010.pkl'
NUM_CLASSES = 4096

os.makedirs(f'traj_clustering/{NUM_CLASSES}', exist_ok=True)
base_root = f'traj_clustering/{NUM_CLASSES}'

data = pickle.load(open(ANNOTATED_DATA_PATH, 'rb'))
data = data["infos"] if "infos" in data else data


all_trajs = []
for i in range(0, len(data), 5):
    all_trajs.append(prepare_future_trajectories_with_x_forward(data, i))

all_trajs = [traj for traj in all_trajs if traj.shape[0] == 18]

selected_trajs = []
for traj in all_trajs:
    if max(traj[:, 0]) > 150 or max(traj[:, 1]) > 150:
        continue
    selected_trajs.append(traj)

all_trajs = selected_trajs

# Flatten trajectories
trajectories_flattened = [traj.flatten() for traj in all_trajs]

kmeans = MiniBatchKMeans(n_clusters=NUM_CLASSES, random_state=42, n_init=10)
kmeans.fit(trajectories_flattened)
labels = kmeans.labels_

cluster_centers = kmeans.cluster_centers_
# Reshape the cluster centers back to (num_points, 2)
num_points = all_trajs[0].shape[0]  # Get the number of points used
cluster_centers_reshaped = cluster_centers.reshape(NUM_CLASSES, num_points, 2)

# residual offset for VAD training
zero_append = np.zeros((cluster_centers_reshaped.shape[0], 1, cluster_centers_reshaped.shape[2]))
cluster_centers = np.concatenate((zero_append, cluster_centers_reshaped), axis=1)
cluster_centers_residual = cluster_centers[:, 1:, :] - cluster_centers[:,:-1,:] # compute offset

# Save the cluster centers to a file
np.save(f'{base_root}/cluster_centers_residual.npy', cluster_centers_residual)
np.save(f'{base_root}/cluster_centers_ori.npy', cluster_centers_reshaped)
# Save the K-Means model to a file
joblib.dump(kmeans, f'{base_root}/kmeans_model.pkl')


# Plot the clustered trajectories
import matplotlib.colors as mcolors

# Extract centroids and reshape them
centroids = kmeans.cluster_centers_
centroids_reshaped = [centroid.reshape(18, 2) for centroid in centroids]

# Plot cluster centroids
plt.figure(figsize=(10, 8))
for centroid in centroids_reshaped:
    plt.plot(centroid[:, 0], centroid[:, 1], alpha=0.7)
plt.title('Cluster Centroids')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.savefig(f'{base_root}/cluster_centroids.png')



