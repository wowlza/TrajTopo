import os
from shapely.geometry import LineString
from multiprocessing import Pool
from tqdm import tqdm

import numpy as np
from scipy.spatial.distance import cdist, euclidean
from openlanev2.lanesegment.io import io
import json
import cv2
from similaritymeasures import frechet_dist
import matplotlib.pyplot as plt

def calculate_distance_error(arrs, array2):
    dists = []
    for arr in arrs:
        array1 = np.array(arr)
        dist_mat = cdist(array1, array2)

        dist_pred = dist_mat.min(-1).mean()
        dist_gt = dist_mat.min(0).mean()
        dist = (dist_pred + dist_gt) / 2
        fre = frechet_dist(arr, array2, p=2)
        dists.append(fre)
    return np.array(dists)

def farthest_point_sampling(arr, n_sample, segment_id, timestamp, start_idx=None):
    """Farthest Point Sampling without the need to compute all pairs of distance.

    Parameters
    ----------
    arr : numpy array
        The positional array of shape (n_points, n_dim)
    n_sample : int
        The number of points to sample.
    start_idx : int, optional
        If given, appoint the index of the starting point,
        otherwise randomly select a point as the start point.
        (default: None)

    Returns
    -------
    numpy array of shape (n_sample,)
        The sampled indices.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.rand(100, 1024)
    >>> point_idx = farthest_point_sampling(data, 3)
    >>> print(point_idx)
        array([80, 79, 27])

    >>> point_idx = farthest_point_sampling(data, 5, 60)
    >>> print(point_idx)
        array([60, 39, 59, 21, 73])
    """
    n_lines, n_points, n_dim = arr.shape

    if (start_idx is None) or (start_idx < 0):
        start_idx = np.random.randint(0, n_lines)

    sampled_indices = [start_idx]
    min_distances = np.full(n_lines, np.inf)
    
    for _ in range(n_sample - 1):
        current_point = arr[sampled_indices[-1]]
        dist_to_current_point = calculate_distance_error(arr, current_point)
        # np.linalg.norm(arr - current_point, axis=1)
        min_distances = np.minimum(min_distances, dist_to_current_point)
        farthest_point_idx = np.argmax(min_distances)
        sampled_indices.append(farthest_point_idx)
    
    
    root_path = '/data/lza/cluster_traj_project/RoadUserPathways/far_sampling_results/vis/val'
    path = os.path.join(root_path, segment_id)
    os.makedirs(path, exist_ok=True)
    
    json_path = '/data/lza/cluster_traj_project/RoadUserPathways/far_sampling_results/val/vecterize'
    new_path = os.path.join(json_path, segment_id)
    os.makedirs(new_path, exist_ok=True)
    json_final_path = os.path.join(new_path, f'{timestamp}.json')
    with open(json_final_path, 'w') as f:
        json.dump(arr[sampled_indices].tolist(), f, indent=4)
    
    fig_all, ax_all = plt.subplots(figsize=(10, 10))
    for t in arr[sampled_indices]:
        pts_x = t[:, 0]  # 原始 x 坐标
        pts_y = t[:, 1]  # 原始 y 坐标
        
        # 直接使用 plt.plot() 绘制轨迹线条
        ax_all.plot(pts_x, pts_y, color='black', linewidth=1)
    plt_path = os.path.join(root_path, f'{segment_id}/{timestamp}_plt.png')
    plt.savefig(plt_path)
    plt.close()
    
    pic_np = np.zeros((100,200,3),dtype=np.uint8)
    for t in arr:
        pts_x = (t[:,0]+50)/0.5
        pts_y = (t[:,1]+25)/0.5
        for i in range(len(pts_x)-1):
            cv2.arrowedLine(pic_np,(int(pts_x[i]),int(pts_y[i])),(int(pts_x[i+1]),int(pts_y[i+1])),1,1,tipLength=0.3)
    img_path = os.path.join(root_path, f'{segment_id}/{timestamp}_pic.png')
    cv2.imwrite(img_path, pic_np*255)

    pic_np_sample = np.zeros((100,200,3),dtype=np.uint8)
    for t in arr[sampled_indices]:
        pts_x = (t[:,0]+50)/0.5
        pts_y = (t[:,1]+25)/0.5
        for i in range(len(pts_x)-1):
            cv2.arrowedLine(pic_np_sample,(int(pts_x[i]),int(pts_y[i])),(int(pts_x[i+1]),int(pts_y[i+1])),1,1,tipLength=0.3)
    img_fre_path = os.path.join(root_path, f'{segment_id}/{timestamp}_pic_sample_fre.png')
    cv2.imwrite(img_fre_path, pic_np_sample*255)
            

    return np.array(sampled_indices)

def load_trajectory_data(file_name):
    """Load trajectory data from json file."""
    if not os.path.exists(file_name):
        raise FileNotFoundError("File not found at {}".format(file_name))
    with open(file_name, 'r') as file:
        data = json.load(file)
    
    return data

def process_data(item):
    split, segment_id, timestamp, root_path = item
    file_path = os.path.join(f'{root_path}/{segment_id}/info/{timestamp}.json')
    
    target = '/data/lza/cluster_traj_project/RoadUserPathways/far_sampling_results/val/vecterize'
    test_path = os.path.join(f'{target}/{segment_id}/{timestamp}.json')
    if os.path.exists(test_path):
        print(f'{segment_id}/{timestamp}.json already exists')
        return
    
    # Load the trajectory data
    trajectory_data = load_trajectory_data(file_path)

    # process the traj data to list
    trajectories = []
    traj_data = trajectory_data['trajectory']
    for traj in traj_data:
        traj_coords = np.array(traj['data'])
        traj_coords = traj_coords[:, :2]  # 只保留 2D 坐标
        ls = LineString(traj_coords)
        distances = np.linspace(0, ls.length, 10)
        line = np.array([ls.interpolate(distance).coords[0] for distance in distances])
        trajectories.append(line)

    point_idx = farthest_point_sampling(np.stack(trajectories), n_sample=30, segment_id=segment_id, timestamp=timestamp)
    return point_idx

def main():
    val_file = '/data/lzy/output_val_new.json'
    root_path = '/data/lzy/filter_val_new'
    target = '/data/lza/cluster_traj_project/RoadUserPathways/far_sampling_results/val/vecterize'
    
    
    # 加载数据
    data_dict = {}
    for split, segments in io.json_load(val_file).items():
        data_dict[split] = segments

    data_list = [(split, segment_id, timestamp.split('.')[0], root_path) \
                 for split, segment_ids in data_dict.items() \
                 for segment_id, timestamps in segment_ids.items() \
                 for timestamp in timestamps]

    # 使用多进程池处理数据
    with Pool() as pool:
        results = list(tqdm(pool.imap(process_data, data_list), total=len(data_list), ncols=100))

    # 处理结果
    # for point_idx in results:
    #     print(point_idx)

if __name__ == "__main__":
    main()