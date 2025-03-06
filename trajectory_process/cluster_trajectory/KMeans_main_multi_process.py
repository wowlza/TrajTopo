# -*- coding: utf-8 -*-

"""
file: run_trajectory_clustering.py
"""

import argparse

# import RoadUserPathways as rup
from clustering import Intersection, Clusters
import os
import json
import numpy as np
from openlanev2.lanesegment.io import io
from tqdm import tqdm
import multiprocessing

#Parameters
def create_args():
    parser = argparse.ArgumentParser(prog = 'road user pathways',
                                     description = 'trajectory clustering to find common pathway types')

    parser.add_argument('--dataset_dir', default="data/lzy/filter_val_new",
                        help="Path to directory that contains the trajectory SQLite files and geometric information.", type=str)
    parser.add_argument('--approaches', default=['N','E','S','W'],
                        help="List with labels for the arms of the intersection (eg. ['N','E','S','W']", type=list)  
    parser.add_argument('--num_points', default=20,
                        help="The number of points in each feature vector.", type=int)
    parser.add_argument('--traj_min_length', default=20,
                        help="The minimum length for trajectories to be included in clustering in timesteps.", type=int)    
    parser.add_argument('--num_SQL', default=20,
                        help="The maximum number of trajectory SQLite files to include in clustering analysis.", type=int)     
    parser.add_argument('--trim', default=False,
                        help="Use polygon to trim trajectories (recommended if starting and ending position points vary in space).", type=bool)    
    parser.add_argument('--delete', default=True,
                        help="Use polygon to delete trajectories starting or ending in polygon.", type=bool)  
    parser.add_argument('--road_user_types', default=[1,2,4],
                        help="List with road user types to analyse (eg [1,4]), from types = {car: 1, pedestrian: 2, motorcycle: 3, bicycle: 4, truck_bus: 5", type=list)  
    parser.add_argument('--define_use', default='use',
                        help="Set to 'use' if geometry already defined, otherwise set to 'define'", type=str)      
    parser.add_argument('--cluster_omit', default=2,
                        help="Minimum number of trajectories in a cluster to include in output", type=int)     
      

    return vars(parser.parse_args())

def load_trajectory_data(file_name):
    """Load trajectory data from json file."""
    if not os.path.exists(file_name):
        raise FileNotFoundError("File not found at {}".format(file_name))
    with open(file_name, 'r') as file:
        data = json.load(file)
    
    return data

# 定义数据处理函数
def process_segment(args):
    split, segment_id, timestamp, root_path, road_user_types, traj_min_length, num_points, trim, delete, num_SQL, cluster_omit = args
    target_path = '/data/lza/cluster_traj_project/RoadUserPathways/cluster_results'
    target = os.path.join(f'{target_path}/{split}/{segment_id}/{timestamp}.json')
    if os.path.exists(target):
        return True
    identifier = (split, segment_id, timestamp)
    file_path = os.path.join(f'{root_path}/{segment_id}/info/{timestamp}.json')
    
    # 加载轨迹数据
    trajectory_data = load_trajectory_data(file_path)
    
    # 处理轨迹数据
    trajectories = []
    traj_data = trajectory_data['trajectory']
    for traj in traj_data:
        traj_coords = np.array(traj['data'])
        traj_coords = traj_coords[:, :2]
        trajectories.append(traj_coords)
    
    # 对每种 road_user_type 进行轨迹聚类
    # for road_user_type in road_user_types:
    observations = Clusters(trajectories, traj_min_length, num_points, trim, delete, num_SQL, cluster_omit, obs_list=[])
    # for approach in ['all'] + approaches:
    for approach in ['all']:
        # observations.cluster_trajectories(approach, segment_id, timestamp, plot = True, table = False)
        observations.cluster_trajectories_KMeans(approach, segment_id, timestamp, plot = True, table = True)
    

def main():
    config = create_args()
    
    dataset_dir = config["dataset_dir"] + "/"
    approaches = config["approaches"]
    num_points = config["num_points"]
    traj_min_length = config["traj_min_length"]
    num_SQL = config["num_SQL"]
    trim = config["trim"]
    delete = config["delete"]
    road_user_types = config["road_user_types"]
    define_use = config["define_use"]
    cluster_omit = config["cluster_omit"]
    
    val_file = '/data/lzy/output_val_new.json'
    root_path = '/data/lzy/filter_val_new'
    
    # 加载数据列表
    data_list = []
    for split, segments in io.json_load(val_file).items():
        for segment_id, timestamps in segments.items():
            for timestamp in timestamps:
                # 确保将 11 个参数正确地传递给 process_segment
                data_list.append((split, segment_id, timestamp.split('.')[0], root_path, road_user_types, traj_min_length, num_points, trim, delete, num_SQL, cluster_omit))
    # 使用多进程池进行并行处理
    num_processes = max(1, os.cpu_count() - 1)  # 保留一个 CPU 供系统使用
    pool = multiprocessing.Pool(processes=num_processes)
    try:
        # tqdm 结合 imap_unordered 进行进度展示
        for _ in tqdm(pool.imap_unordered(process_segment, data_list), total=len(data_list), ncols=100):
            pass
    finally:
        pool.close()
        pool.join()
    print("所有任务已完成。")

    
    


              
if __name__ == '__main__':
    main()
    