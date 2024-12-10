import multiprocessing as mp
import os
import shutil
from mmcv.runner import get_dist_info
from mmcv.datasets import build_dataset
from torch.utils.data.distributed import DistributedSampler

# TODO: 
# 1. build_dataloader picks a random sampler. Confirm if the sampler knows the info in advance. 
# 2. Separate the prefetcher to its own thread. 
# 3. prefetcher gives idx, to the loader, normally we use the idx to get the data. Now needs to use SQL (already overloaded). 
# 4. Know the image path
# 5. /data/local/, look for the images
# 6. Design a cache for SQL

# for i in enumerate(data_loader): know the idx before that

class ImagePrefetcher(mp.Process):
    def __init__(self, cfg, local_cache_path, distributed, seed, epoch=0):
        super().__init__()
        self.cfg = cfg
        self.local_cache_path = local_cache_path
        self.distributed = distributed
        self.seed = seed
        self.epoch = epoch
        self.stop_event = mp.Event()

    def run(self):
        # Build dataset
        dataset = build_dataset(self.cfg.data.train)
        # Build sampler
        if self.distributed:
            rank, world_size = get_dist_info()
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=self.seed)
            sampler.set_epoch(self.epoch)
        else:
            sampler = None

        indices = list(iter(sampler)) if sampler else list(range(len(dataset)))

        # Prefetch images
        for idx in indices:
            if self.stop_event.is_set():
                break
            data = dataset[idx]
            filenames = data['img_filename']
            for remote_img_path in filenames:
                local_img_path = os.path.join(self.local_cache_path, osp.basename(remote_img_path))
                if not os.path.exists(local_img_path):
                    # Ensure the directory exists
                    os.makedirs(os.path.dirname(local_img_path), exist_ok=True)
                    # Copy image to local cache
                    try:
                        shutil.copy(remote_img_path, local_img_path)
                    except Exception as e:
                        print(f"Failed to copy {remote_img_path} to {local_img_path}: {e}")

    def stop(self):
        self.stop_event.set()