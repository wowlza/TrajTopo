import os
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

id2map = {
    'train':'/data/argoverse2/argo2_sensor/sensor/av2_map_infos_train.pkl'
}
import pickle 
infos={}  
for key in id2map:
    infos[key]=pickle.load(open(id2map[key],"rb"))['samples']
for key in id2map:
    id2map[key] = pickle.load(open(id2map[key],"rb"))['id2map']

AV2_ROOT='/data/argoverse2/argo2_sensor/sensor'
def get_city_name(log_id, origin_split_='train'):
    vector_map_path = os.path.join(AV2_ROOT, origin_split_, log_id, 'map')
    # 获取目录中的所有文件和目录
    vector_map_fpaths = os.listdir(vector_map_path)
    if len(vector_map_fpaths) == 0:
        raise RuntimeError(f"Vector map file is missing for {log_id}.")
    for vector_map_fpath in vector_map_fpaths:
        if 'log_map_archive' in vector_map_fpath:
            log_city_name = vector_map_fpath.split("____")[1].split("_")[0]
            return log_city_name
json_filename = 'ouput/log_id_city_name_train.json'


data = {}

# 遍历infos['train']中的每个元素
for info in tqdm(infos['train']):
    log_id = info['log_id']
    city_name_mapping = {
        'MIA': 'miami',
        'PIT': 'pittsburgh',
        'DTW': 'dearborn',
        'WDC': 'washington-dc',
        'ATX': 'austin',
        'PAO': 'palo-alto'
        }
    city_name = city_name_mapping.get(get_city_name(log_id))
    
    if city_name:
        data[log_id] = {'source_id': log_id, 'city_name': city_name}
    else:
        print(f"No city name found for log_id: {log_id}")

with open(json_filename, 'w') as jsonfile:
    json.dump(data, jsonfile, indent=4)