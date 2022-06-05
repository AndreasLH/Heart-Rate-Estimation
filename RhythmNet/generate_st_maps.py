# For Generating the spatial temporal maps
# this scripts is heavily multi threaded, it benefits greatly from a lot of cores

from utils import video2st_maps as v2m
import glob

allvideo = glob.glob('../*/*/*/*.mp4', recursive=True)

for v in range(len(allvideo)):
  print(v, allvideo[v])
  v2m.get_spatio_temporal_map_threaded(allvideo[v], 'st_maps')
