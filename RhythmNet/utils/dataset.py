# import albumentations
import torch
import numpy as np
from PIL import Image
from PIL import ImageFile
from torch.utils.data import Dataset
from utils.signal_utils import get_hr_data

ImageFile.LOAD_TRUNCATED_IMAGES = True


class DataLoaderRhythmNet(Dataset):
    """
        Dataset class for RhythmNet
    """
    # The data is now the SpatioTemporal Maps instead of videos

    def __init__(self, st_maps_path, target_signal_path):
        self.st_maps_path = st_maps_path
        self.target_path = target_signal_path
        self.maps = None

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    def __len__(self):
        return len(self.st_maps_path)

    def __getitem__(self, index):
        # identify the name of the video file so as to get the ground truth signal
        self.video_file_name = self.st_maps_path[index].split('/')[-1].split('.')[0]

        # Load the maps for video at 'index'
        self.maps = np.load(self.st_maps_path[index])
        map_shape = self.maps.shape
        self.maps = self.maps.reshape((-1, map_shape[3], map_shape[1], map_shape[2]))

        target_hr = get_hr_data(self.video_file_name)
        # To check the fact that we dont have number of targets greater than the number of maps
        # target_hr = target_hr[:map_shape[0]]
        self.maps = self.maps[:target_hr.shape[0], :, :, :]
        return {
            "st_maps": torch.tensor(self.maps, dtype=torch.float),
            "target": torch.tensor(target_hr, dtype=torch.float)
        }

