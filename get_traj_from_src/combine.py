import os
import json
import time
import shutil
import multiprocessing
import json
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
import time
import numpy as np
import random
from torchvision import transforms
import matplotlib.pyplot as plt
from shapely.wkt import loads
from matplotlib.collections import PatchCollection
import pickle
import glob

paths = ['/data/jpj/val/miami4-csv', '/data/jpj/val/miami3-csv', '/data/jpj/val/miami2-csv','/data/jpj/val/miami1-csv']
merged_path = '/data/lzy/val_all_traj_fromtrain/'
subfolders = [f.name for f in os.scandir(paths[0]) if f.is_dir()]
#print(type(subfolders))
state_folder='/home/jpj/lzy/output/state_output/miami-csv'
os.makedirs(state_folder, exist_ok=True)
def process_info(subfolder):
    try:
        print(subfolder)
        state_file_path = os.path.join(state_folder, f'{subfolder}.json')
        if os.path.exists(state_file_path):
            with open(state_file_path, 'r') as f:
                state = json.load(f)
                if state['status'] == 'processed':
                    print("pass!")
                    return
        subfolder_path = os.path.join(merged_path, subfolder)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)

        # 合并JSON内容
        json_files = glob.glob(os.path.join(paths[0], subfolder, '*.json'))
        json_names = [os.path.splitext(os.path.basename(file))[0] for file in json_files]           
        for json_name in json_names:
            print(json_name)
            merged_json = {"lane_centerline": None, "trajectory": []}
        # 读取四个大文件夹中对应子文件夹的JSON文件
            for path in paths:   
                json_file_path = os.path.join(path, subfolder, f"{json_name}.json")
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if merged_json["lane_centerline"] is None:
                        merged_json["lane_centerline"] = data["lane_centerline"]
                    merged_json["trajectory"].extend(data["trajectory"])
            trajectories = merged_json["trajectory"]
            map_elements=merged_json["lane_centerline"]
            fig, ax = plt.subplots()
            for element in map_elements:
                map_data=element["points"]
                x = [point[0] for point in map_data]
                y = [point[1] for point in map_data]
                ax.plot(x, y,linewidth=0.5,color='black')
            for trajectory in trajectories:
                data = trajectory["data"]
                x = [point[0] for point in data]
                y = [point[1] for point in data]
                ax.plot(x, y,linewidth=0.1)
            os.makedirs(os.path.join(subfolder_path, "image"), exist_ok=True)
            os.makedirs(os.path.join(subfolder_path, "info"), exist_ok=True)
            
            plt.savefig(os.path.join(subfolder_path, "image",f"{json_name}.jpg"), format='jpg',dpi=500)
            plt.close()
            # 将合并后的JSON内容写入新文件
            merged_json_file_path = os.path.join(subfolder_path,"info",f"{json_name}.json")
            with open(merged_json_file_path, 'w', encoding='utf-8') as f:
                json.dump(merged_json, f)
            # for path in paths:
            #     subfolder_path = os.path.join(path, subfolder)
            #     if os.path.exists(subfolder_path):
            #         shutil.rmtree(subfolder_path)
            with open(state_file_path, 'w') as f:
                json.dump({'info_id':subfolder , 'status': 'processed'}, f)
            print("complete!")
    except Exception as e:
        print(f"Error processing info: {e}")



def parallel_process(subfolders):
    num_processes = os.cpu_count()
    pool =multiprocessing.Pool(processes=num_processes-20)
    try:
        for _ in tqdm(pool.imap_unordered(process_info, subfolders), total=len(subfolders)):
            pass
    finally:
        pool.close()
        pool.join()

    print("所有任务已完成。")


parallel_results = parallel_process(subfolders)