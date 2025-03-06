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
import multiprocessing
formatted_data1={'washington-dc4':'/data/pkl/washington-dc4-csv.pkl'}
formatted_data=pickle.load(open(formatted_data1['washington-dc4'],"rb"))
filtered_data = [entry for entry in formatted_data['Trajectory'] if entry[0][2] == 'vehicle']
filtered_dict={'Trajectory':filtered_data}
print(type(filtered_dict))
print("hello")
id2map = {
    'train':'/data/OpenLane-V2/data_dict_subset_A_val.pkl'
}
infos={}  
infos['train']=pickle.load(open(id2map['train'],"rb"))
pc_range= [-100, -50, -2.0, 100.0, 50.0, 2.0]
patch_h = pc_range[4]-pc_range[1]
patch_w = pc_range[3]-pc_range[0]
patch_size = (patch_h, patch_w)

vector_map = VectorizedAV2LocalMap(
    patch_size=(200,100), 
    map_classes=['divider', 'ped_crossing','boundary'], 
    fixed_ptsnum_per_line=20,
    padding_value=-10000)

def convert_linestring_to_list(item):
    key, linestring = item
    return {
        "id":key,
        
        "data":list(linestring.coords)
    }

def get_city_name_by_log_id(log_id, data):
    entry = data.get(log_id)
    if entry:
        return entry['city_name']
    else:
        return None
    
# def get_city_name_by_log_id(log_id, data):
#     entry = data.get(log_id)
#     if entry:
#         return entry['city_name']
#     else:
#         return None
    
# json_filename = '/home/jpj/lzy/output/log_id_info_train.json'
# with open(json_filename, 'r') as jsonfile:
#     data = json.load(jsonfile)
# miami_dict={'315967208949927222.json': {'segment_id': '00196', 'city_name': 'miami'}, '315967209949927210.json': {'segment_id': '00196', 'city_name': 'miami'}, '315967214449927219.json': {'segment_id': '00196', 'city_name': 'miami'}, '315967221949927219.json': {'segment_id': '00196', 'city_name': 'miami'}, '315969926549927214.json': {'segment_id': '00197', 'city_name': 'miami'}, '315969927049927211.json': {'segment_id': '00197', 'city_name': 'miami'}, '315969930049927218.json': {'segment_id': '00197', 'city_name': 'miami'}, '315969931049927218.json': {'segment_id': '00197', 'city_name': 'miami'}, '315969931549927215.json': {'segment_id': '00197', 'city_name': 'miami'}, '315972017549927219.json': {'segment_id': '00202', 'city_name': 'miami'}, '315967570249927216.json': {'segment_id': '00254', 'city_name': 'miami'}, '315967572749927211.json': {'segment_id': '00254', 'city_name': 'miami'}, '315970417349927213.json': {'segment_id': '00265', 'city_name': 'miami'}, '315973188449927219.json': {'segment_id': '00294', 'city_name': 'miami'}, '315973192949927209.json': {'segment_id': '00294', 'city_name': 'miami'}, '315965801949927215.json': {'segment_id': '00354', 'city_name': 'miami'}, '315966120749927220.json': {'segment_id': '00391', 'city_name': 'miami'}, '315966121749927208.json': {'segment_id': '00391', 'city_name': 'miami'}, '315966123249927216.json': {'segment_id': '00391', 'city_name': 'miami'}, '315968291349927213.json': {'segment_id': '00392', 'city_name': 'miami'}, '315968293349927215.json': {'segment_id': '00392', 'city_name': 'miami'}, '315970179949927223.json': {'segment_id': '00400', 'city_name': 'miami'}, '315970187949927217.json': {'segment_id': '00400', 'city_name': 'miami'}, '315970122449927216.json': {'segment_id': '00411', 'city_name': 'miami'}, '315969133149927218.json': {'segment_id': '00424', 'city_name': 'miami'}, '315968855849927215.json': {'segment_id': '00437', 'city_name': 'miami'}, '315968858849927215.json': {'segment_id': '00437', 'city_name': 'miami'}, '315972146549927220.json': {'segment_id': '00468', 'city_name': 'miami'}, '315973181249927213.json': {'segment_id': '00493', 'city_name': 'miami'}, '315973184249927215.json': {'segment_id': '00493', 'city_name': 'miami'}, '315973185249927215.json': {'segment_id': '00493', 'city_name': 'miami'}, '315973189249927219.json': {'segment_id': '00493', 'city_name': 'miami'}, '315970627649927217.json': {'segment_id': '00526', 'city_name': 'miami'}, '315967651549927216.json': {'segment_id': '00528', 'city_name': 'miami'}, '315970122949927221.json': {'segment_id': '00555', 'city_name': 'miami'}, '315970123449927215.json': {'segment_id': '00555', 'city_name': 'miami'}, '315970126449927216.json': {'segment_id': '00555', 'city_name': 'miami'}, '315976786449927212.json': {'segment_id': '00562', 'city_name': 'miami'}, '315976788449927218.json': {'segment_id': '00562', 'city_name': 'miami'}, '315976791949927223.json': {'segment_id': '00562', 'city_name': 'miami'}, '315968230349927217.json': {'segment_id': '00582', 'city_name': 'miami'}, '315968235349927216.json': {'segment_id': '00582', 'city_name': 'miami'}, '315967872149927216.json': {'segment_id': '00585', 'city_name': 'miami'}, '315973815049927220.json': {'segment_id': '00608', 'city_name': 'miami'}, '315973817549927218.json': {'segment_id': '00608', 'city_name': 'miami'}, '315967276549927221.json': {'segment_id': '00610', 'city_name': 'miami'}, '315968021949927216.json': {'segment_id': '00633', 'city_name': 'miami'}, '315970290049927216.json': {'segment_id': '00651', 'city_name': 'miami'}, '315970026749927214.json': {'segment_id': '00660', 'city_name': 'miami'}, '315973050249927219.json': {'segment_id': '00671', 'city_name': 'miami'}, '315968425349927217.json': {'segment_id': '00687', 'city_name': 'miami'}, '315968428349927210.json': {'segment_id': '00687', 'city_name': 'miami'}}
# print(len(miami_dict))
json_filename = '/home/jpj/lzy/output/log_id_info_val.json'
with open(json_filename, 'r') as jsonfile:
    data = json.load(jsonfile)
    
trajectories_new = defaultdict(list) 
for trajectory in formatted_data['Trajectory']:
    for position in trajectory:
        x, y, z = position[3], position[4], position[5]
        trajectories_new[position[0]].append([position[2],x,y,z])
extracted_data={'Trajectory':list(trajectories_new.values())}   
#生成文件夹
state_folder='/home/jpj/lzy/output/state_output/washington-dc4-csv'
def process_info(info):
    try:
        
        log_id = info['meta_data']['source_id']
        token = str(info['timestamp'])
        json_name=f"{token}.json"
        segment_id=info['segment_id']
        city_name=get_city_name_by_log_id(log_id,data)
        print(token)
        if city_name == "washington-dc":
        # if json_name in miami_dict and segment_id == miami_dict[json_name]['segment_id']:
        #     state_file_path = os.path.join(state_folder, f'{token}_{segment_id}.json')
        #     if os.path.exists(state_file_path):
        #         with open(state_file_path, 'r') as f:
        #             state = json.load(f)
        #             if state['status'] == 'processed':
        #                 print("pass!")
        #                 return
            os.makedirs(state_folder, exist_ok=True)
            state_file_path = os.path.join(state_folder, f'{token}_{segment_id}.json')
            subfolder_path = os.path.join('/data/jpj/val/washington-dc4-csv', segment_id)
            image_path = os.path.join(subfolder_path, f'{token}.jpg')
            json_path = os.path.join(subfolder_path, f'{token}.json')
            os.makedirs(subfolder_path, exist_ok=True)
            e2g_translation = info['pose']['translation']
            e2g_rotation = info['pose']['rotation']
            map_elements=info['annotation']['lane_centerline']
            trajectories_new = defaultdict(list)           
            map_pose = e2g_translation[:2]
            rotation = Quaternion._from_matrix(e2g_rotation)
            patch_box = (map_pose[0], map_pose[1], patch_size[0],patch_size[1])
            patch_angle = quaternion_yaw(rotation) / np.pi * 180
            city_SE2_ego = SE3(e2g_rotation, e2g_translation)
            ego_SE3_city = city_SE2_ego.inverse()
            
            map_data_Trajectory = vector_map.get_map_trajectory_geom(patch_box, patch_angle, extracted_data['Trajectory'], ego_SE3_city)
            fig, ax = plt.subplots()
            for element in map_elements:
                points = element['points'][:, :2]    
                ax.plot(points[:, 0], points[:, 1],color='black',linewidth=0.5)
            for i in range(1,len(map_data_Trajectory[0])):
                for j in range(len(map_data_Trajectory[0][1])):
                    x,y=map_data_Trajectory[0][i][j][1].coords.xy
                    ax.plot(x, y, linewidth=0.1)
            plt.savefig(image_path, format='jpg',dpi=500)
            plt.close(fig)
            for element in map_elements:
                element['points'] = element['points'].tolist()
            converted_trajectory =[convert_linestring_to_list(inner_item) for inner_item in map_data_Trajectory[0][1]]
            combined_data = {
                "lane_centerline":map_elements,
                "trajectory": converted_trajectory
            }
            for item in combined_data['lane_centerline']:
                item['points'] = [[round(coord, 2) for coord in point] for point in item['points']]
            for item in combined_data['trajectory']:
                item['data'] = [[round(coord, 2) for coord in point] for point in item['data']]
            with open(json_path, 'w') as f:
                json.dump(combined_data, f)
            with open(state_file_path, 'w') as f:
                json.dump({'token': token,'segment_id':segment_id,'status': 'processed'}, f)
            print("complete!")
    except Exception as e:
        print(f"Error processing info: {e}")
    
            


def parallel_process(infos):
    num_processes = os.cpu_count()
    pool =multiprocessing.Pool(processes=num_processes-20)
    try:
        for _ in tqdm(pool.imap_unordered(process_info, infos['train'].values()), total=len(infos['train'])):
            pass
    finally:
        pool.close()
        pool.join()

    print("所有任务已完成。")


parallel_results = parallel_process(infos)