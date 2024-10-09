import os
import numpy as np
import regex as re

IM_SIZE = 137

WORLD_MAT_RE = re.compile(r"world_mat_(\d+)")
CAMERA_MAT_RE = re.compile(r"camera_mat_(\d+)")

def load_shapenet(root):
    npz = np.load(os.path.join(root, "img_choy2016", "cameras.npz"))

    world_mat_ids = set()
    camera_mat_ids = set()

    for key in npz.keys():
        m = WORLD_MAT_RE.match(key)
        if m is not None:
            world_mat_ids.add(int(m.group(1)))
            continue
        m = CAMERA_MAT_RE.match(key)
        if m is not None:
            camera_mat_ids.add(int(m.group(1)))
            continue

    assert world_mat_ids == camera_mat_ids

    indices = np.array(sorted(list(world_mat_ids)))
    if (indices != np.arange(24)).all():
        raise AssertionError("Bad shapenet model")

    world_mats = np.stack([npz[f"world_mat_{i}"] for i in indices])
    camera_mats = np.stack([npz[f"camera_mat_{i}"] for i in indices])

    # normalize camera matrices
    # print("normalize")
    # print(camera_mats)
    camera_mats /= np.array([IM_SIZE + 1, IM_SIZE + 1, 1]).reshape(3, 1)
    # print(camera_mats)

    world_mats = world_mats.astype(np.float32)
    camera_mats = camera_mats.astype(np.float32)
    # print(f"wmats shape {self.wmats.shape}") # (24, 3, 4)
    # print(f"cmats shape {self.cmats.shape}") # (24, 3, 3)
    return world_mats, camera_mats