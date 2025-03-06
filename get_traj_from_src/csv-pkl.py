import json
import numpy as np
import copy
import os
from shapely.wkt import dumps
from os import path as osp
from matplotlib import pyplot as plt
import cv2
import torch
import mmcv
from tqdm import tqdm
from maptr_av2_utils1 import  VectorizedAV2LocalMap
import random
from torchvision import transforms
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from shapely import affinity, ops
from shapely.geometry import Polygon, LineString, box, MultiPolygon, MultiLineString
from pathlib import Path
from av2.geometry.se3 import SE3
import pandas as pd
from shapely.geometry import Point, Polygon
import matplotlib.collections as mc
import csv
from collections import defaultdict
import matplotlib.patches as patches
import mmcv
import random
from torchvision import transforms
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from shapely import affinity, ops
# from mmdet.datasets.pipelines import to_tensor
from shapely.wkt import loads
from matplotlib.collections import PatchCollection
import pickle

def main():
    path1 = '/data/lzy/csv_val'
    trajectories = defaultdict(list)
    flag=1
    for filename in os.listdir(path1):
        if filename.endswith('.csv'):
            csv_file_path = os.path.join(path1, filename)
            grouped_data = {}
            with open(csv_file_path, 'r') as csvfile:
                csvreader = csv.DictReader(csvfile)
                for row in csvreader:
                    track_id = row['track_id']
                    if track_id not in grouped_data:
                        grouped_data[track_id] = []
                    grouped_data[track_id].append(row)            
                for track_id_, group in grouped_data.items():
                    for i in range(len(group)-1):
                        track_id = group[i]['track_id']
                        position_x = float(group[i]['position_x'])
                        position_y = float(group[i]['position_y'])
                        position_z=float(group[i]['position_z'])
                        velocity_x=float(group[i]['velocity_x'])
                        velocity_y=float(group[i]['velocity_y'])
                        velocity=[velocity_x,velocity_y]
                        acceration_x=float(10*(float(group[i+1]['velocity_x'])-float(group[i]['velocity_x'])))
                        acceration_y=float(10*(float(group[i+1]['velocity_y'])-float(group[i]['velocity_y'])))
                        acceration=[acceration_x,acceration_y]
                        city=group[i]['city']
                        object_type=group[i]['object_type']
                        if object_type=='vehicle' or object_type=='pedestrian' or object_type=='motorcyclist' or object_type=='bus' or object_type=='cyclist':
                            trajectories[track_id+str(flag)].append([track_id+str(flag),city,object_type,position_x, position_y,position_z,velocity,acceration])
        print(flag)
        flag+=1
    formatted_data = {'Trajectory': list(trajectories.values())}
    with open('/data/lzy/val_csv_new.pkl','wb') as f:
        pickle.dump(formatted_data,f)


if __name__ == "__name__":
    main()
