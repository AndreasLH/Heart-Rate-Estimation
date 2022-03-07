from utils.dataset import DataLoaderRhythmNet 

import os
import matplotlib.pyplot as plt
import torch
import numpy as np

video_files_train = ["data/st_maps/test/cpi_rotation.npy"]

train_set = DataLoaderRhythmNet(st_maps_path=video_files_train, target_signal_path="data/test/")

data = next(iter(train_set))
print(data)
data['st_maps'].size()
data['st_maps'] = data['st_maps'].to(torch.uint8)
arr = data['st_maps'].permute(0,3,2,1).numpy()

plt.figure()
plt.imshow(arr[0,:,:,:])
plt.show()