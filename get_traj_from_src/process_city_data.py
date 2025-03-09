import json
import os
import pickle
import multiprocessing
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from av2.geometry.se3 import SE3
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from maptr_av2_utils1 import VectorizedAV2LocalMap

CONFIG = {
"pkl_path_template": "data/pkl/{city_name}-csv.pkl",
"map_data_path": "data/OpenLane-V2/data_dict_subset_A_val.pkl",
"log_id_info_path": "output/log_id_info_val.json",
"output_root": "output",
"pc_range": [-100, -50, -2.0, 100.0, 50.0, 2.0],
"map_classes": ['divider', 'ped_crossing', 'boundary'],
"fixed_ptsnum_per_line": 20
}

class CityDataProcessor:
    def __init__(self, city_name):
        self.city_name = city_name
        self.vector_map = VectorizedAV2LocalMap(
            patch_size=(200, 100),
            map_classes=CONFIG['map_classes'],
            fixed_ptsnum_per_line=CONFIG['fixed_ptsnum_per_line'],
            padding_value=-10000
        )
        
        with open(CONFIG['log_id_info_path'], 'r') as f:
            self.log_id_data = json.load(f)
            
        self._load_data()
        
    def _load_data(self):
        pkl_path = CONFIG['pkl_path_template'].format(city_name=self.city_name)
        with open(pkl_path, 'rb') as f:
            formatted_data = pickle.load(f)
            
        self.filtered_data = {
            'Trajectory': [entry for entry in formatted_data['Trajectory'] 
                          if entry[0][2] == 'vehicle']
        }
        
        with open(CONFIG['map_data_path'], 'rb') as f:
            self.infos = {'train': pickle.load(f)}
            
        self._preprocess_trajectories()
        
    def _preprocess_trajectories(self):
        trajectories_new = defaultdict(list)
        for trajectory in self.filtered_data['Trajectory']:
            for position in trajectory:
                x, y, z = position[3], position[4], position[5]
                trajectories_new[position[0]].append([position[2], x, y, z])
        self.extracted_data = {'Trajectory': list(trajectories_new.values())}
        
    def _get_city_name(self, log_id):
        entry = self.log_id_data.get(log_id)
        return entry['city_name'] if entry else None
        
    def process_info(self, info):
        try:
            log_id = info['meta_data']['source_id']
            token = str(info['timestamp'])
            segment_id = info['segment_id']
            
            if self._get_city_name(log_id) != self.city_name:
                return
                
            state_folder = os.path.join(
                CONFIG['output_root'], 
                'state_output', 
                f'{self.city_name}-csv'
            )
            os.makedirs(state_folder, exist_ok=True)
            
            state_path = os.path.join(state_folder, f'{token}_{segment_id}.json')
            if self._check_processed(state_path):
                print(f"跳过已处理文件: {token}")
                return
                
            self._process_single_entry(info, token, segment_id)
            
        except Exception as e:
            print(f"处理 {token} 时出错: {str(e)}")
            
    def _check_processed(self, state_path):
        if os.path.exists(state_path):
            with open(state_path, 'r') as f:
                state = json.load(f)
                return state.get('status') == 'processed'
        return False
        
    def _process_single_entry(self, info, token, segment_id):
        subfolder = os.path.join(
            CONFIG['output_root'], 
            'val', 
            f'{self.city_name}-csv', 
            segment_id
        )
        os.makedirs(subfolder, exist_ok=True)
        
        pose = info['pose']
        city_SE2_ego = SE3(pose['rotation'], pose['translation'])
        ego_SE3_city = city_SE2_ego.inverse()
        
        map_pose = pose['translation'][:2]
        rotation = Quaternion._from_matrix(pose['rotation'])
        patch_box = (
            map_pose[0], 
            map_pose[1], 
            CONFIG['pc_range'][4]-CONFIG['pc_range'][1], 
            CONFIG['pc_range'][3]-CONFIG['pc_range'][0]
        )
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        
        map_traj = self.vector_map.get_map_trajectory_geom(
            patch_box, 
            patch_angle, 
            self.extracted_data['Trajectory'], 
            ego_SE3_city
        )
        
        self._generate_visualization(
            info['annotation']['lane_centerline'], 
            map_traj, 
            os.path.join(subfolder, f'{token}.jpg')
        )
        
        self._save_json_data(
            info['annotation']['lane_centerline'], 
            map_traj, 
            os.path.join(subfolder, f'{token}.json')
        )
        
        self._update_processing_state(token, segment_id, state_path)
        
    def _generate_visualization(self, lane_data, traj_data, img_path):
        fig, ax = plt.subplots()
        
        for element in lane_data:
            points = np.array(element['points'])[:, :2]
            ax.plot(points[:, 0], points[:, 1], color='black', linewidth=0.5)
            
        if traj_data and traj_data[0]:
            for traj in traj_data[0][1]:
                x, y = traj['data'].coords.xy
                ax.plot(x, y, linewidth=0.1)
                
        plt.savefig(img_path, dpi=500, bbox_inches='tight')
        plt.close()
        
    def _save_json_data(self, lane_data, traj_data, json_path):
        processed_lanes = [
            {
                'points': [
                    [round(coord, 2) for coord in point] 
                    for point in element['points']
                ]
            } for element in lane_data
        ]
        
        processed_trajs = [
            {
                'id': item[0],
                'data': [
                    [round(coord, 2) for coord in point] 
                    for point in item[1].coords
                ]
            } for item in traj_data[0][1]
        ]
        
        with open(json_path, 'w') as f:
            json.dump({
                'lane_centerline': processed_lanes,
                'trajectory': processed_trajs
            }, f)
            
    def _update_processing_state(self, token, segment_id, state_path):
        with open(state_path, 'w') as f:
            json.dump({
                'token': token,
                'segment_id': segment_id,
                'status': 'processed'
            }, f)
            
    def parallel_process(self):
        num_cores = multiprocessing.cpu_count()
        print(f"可用CPU核心数: {num_cores}")
        
        with multiprocessing.Pool(processes=max(1, num_cores-2)) as pool:
            tasks = list(self.infos['train'].values())
            for _ in tqdm(pool.imap_unordered(self.process_info, tasks), 
                         total=len(tasks)):
                pass
                
        print(f"{self.city_name} 数据处理完成")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('city', type=str, 
                       help='要处理的城市名称（如：dearborn, miami等）')
    args = parser.parse_args()
    
    processor = CityDataProcessor(args.city)
    processor.parallel_process()