# VADv2

## Env setup

## Train & Test

### Data Preparation

The same as VAD data format. 

### Trajectory Clustering

Refer to `tools/traj_clustering.py`, which will cluster all trajectories to 4096 classes. The following component will be saved:

- Cluster centroids: saved in `npy` format
- MiniBatchKMeans sklearn model: saved in `pkl` format
- Visualization: as follows: 

<p align="center">
<img src="data/traj_clusters/4096/cluster_centroids.png" alt="traj cluster" width="400">
</p>

### Train script

Assume run on 2-GPUs machine. RTX4090 or higher is recommended.

```bash
torchrun --nproc_per_node=2 \
    --master_port=28510 \
    adzoo/vad/train.py \
    adzoo/vad/configs/VAD/VADv2_voca4096_config.py \
    --launcher=pytorch \
    --deterministic
```

### Test script

```bash
torchrun --nproc_per_node=1 --master_port=28512 \
    ./adzoo/vad/test.py \
    ./adzoo/vad/configs/VAD/VADv2_voca4096_config.py \
    path-2-your-model \
    --launcher=pytorch \
    --eval=bbox
```


## Open-loop Eval

Nuplan offers ruled-based tags for each driving scenario. 


## Reference

- [VAD: Vectorized Scene Representation for Efficient Autonomous Driving](https://github.com/hustvl/VAD?tab=readme-ov-file)
- [VADv2: End-to-End Vectorized Autonomous Driving via Probabilistic Planning](https://hgao-cv.github.io/VADv2/)
- [ThinkLab@SJTU: Bench2Drive Zoo](https://github.com/Thinklab-SJTU/Bench2DriveZoo)