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
# from vlannotator.utils import prepare_future_trajectories_with_x_forward
from pyquaternion import Quaternion
import numpy.typing as npt


def _restore_trajectory(trajectory_deltas: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Restore the trajectory deltas to get the future trajectory."""
    initial_position = trajectory_deltas[0]
    recovered_ego_fut_trajs = np.vstack(
        [initial_position, np.cumsum(trajectory_deltas, axis=0)]
    )
    return recovered_ego_fut_trajs

def _transform_to_global(
        points: npt.NDArray[np.float32],
        x: float,
        y: float,
        z: float,
        q: Quaternion,
        inv: bool = False,
        ) -> npt.NDArray[np.float32]:
        
    """Transform points to global coordinate frame."""
    if inv:
        return (points - np.array([x, y, z])) @ q.rotation_matrix
    else:
        return points @ q.rotation_matrix.T + np.array([x, y, z])
    
def prepare_future_trajectories_with_x_forward(key_frame_data, frame_idx: int) -> npt.NDArray[np.float32]:
    """Prepare future trajectories for ego vehicle."""
    ego_fut_trajs_restored = []
    for future_offset in [0, 6, 12]:
        future_idx = frame_idx + future_offset
        scene_token = key_frame_data[frame_idx]["scene_token"]
        if future_idx < len(key_frame_data):
            future_data = key_frame_data[future_idx]
            if future_data["scene_token"] != scene_token:
                continue
            trajectory_deltas = future_data["gt_ego_fut_trajs"]
            future_positions = _restore_trajectory(trajectory_deltas)
            x, y, z = future_data["can_bus"][0:3]
            qx, qy, qz, qw = future_data["can_bus"][3:7]
            q = Quaternion([qw, qx, qy, qz])
            future_positions = np.c_[
                future_positions, np.zeros(future_positions.shape[0])
            ]
            future_positions_global = _transform_to_global(
                future_positions, x, y, z, q
            )
            ego_fut_trajs_restored.append(future_positions_global[1:, :])
    # Transform to ego coordinate frame
    x, y, z = key_frame_data[frame_idx]["can_bus"][0:3]
    qx, qy, qz, qw = key_frame_data[frame_idx]["can_bus"][3:7]
    q = Quaternion([qw, qx, qy, qz])
    ego_future_positions = []
    for traj in ego_fut_trajs_restored:
        traj_in_ego = _transform_to_global(traj, x, y, z, q, inv=True)
        # traj_in_ego = traj_in_ego @ np.array([[0, -1, 0], [-1, 0, 0], [0, 0, 1]]) # rotate 90 degrees, make y-axis the forward direction

        ego_future_positions.append(traj_in_ego[:, :2])
    if ego_future_positions:
        return np.vstack(ego_future_positions)
    else:
        return np.zeros((2, 0))


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



