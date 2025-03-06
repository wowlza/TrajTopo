import os
import json
import pandas as pd
from shapely.geometry import Polygon, LineString, box, MultiPolygon, MultiLineString
from matplotlib import pyplot as plt
import numpy as np
from skimage.draw import line as skiline
from shapely.geometry import Point
import pandas as pd
import matplotlib.pyplot as plt
from pyproj import Transformer, CRS
from matplotlib.ticker import ScalarFormatter
from scipy.interpolate import splprep, splev
from shapely.geometry import LineString
import numpy as np
from shapely.geometry import box
import alphashape
import geopandas as gpd
from shapely.ops import unary_union,nearest_points
from scipy.spatial import Delaunay
import json
from sklearn.cluster import DBSCAN
from shapely.geometry import Point, LineString, MultiPoint, GeometryCollection, MultiLineString
from scipy.spatial.distance import cdist
from math import sqrt
from matplotlib.patches import FancyArrowPatch
from matplotlib.collections import PatchCollection
from scipy.interpolate import splprep, splev, interp1d
from collections import defaultdict
from scipy.interpolate import CubicSpline
import matplotlib.patches as patches
from heapq import heappush, heappop
from scipy.ndimage import gaussian_filter1d
import math
import heapq
from multiprocessing import Pool
import chardet
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import math
import json
import cv2
from skimage.draw import line as skiline
from shapely.geometry import LineString, box
import pandas as pd
import multiprocessing
import os
from tqdm import tqdm
import json
import cv2
import warnings
warnings.filterwarnings("ignore")
from av2.map.map_api import ArgoverseStaticMap
import pickle

folderA_path = '/data/lza/olv2-traj-data/train_all_traj'
json_file_list = []
for segment_id in os.listdir(folderA_path):
    # if segment_id == '10095':
    segment_id_path = os.path.join(folderA_path, segment_id)
    if os.path.isdir(segment_id_path):
        info_path = os.path.join(segment_id_path, 'info')
        if os.path.exists(info_path):
            for file_name in os.listdir(info_path):
                if file_name.endswith('.json'):
                    # if file_name =='315974146749927211.json':
                    json_file_path = os.path.join(info_path, file_name)
                    json_file_list.append(json_file_path)
# id2map = {
#     'val':'/data/OpenLane-V2/data_dict_subset_A_train.pkl'
# }
# infos={}
# infos['val']=pickle.load(open(id2map['val'],"rb"))
def point_in_range(x, y, x_range=(-50, 50), y_range=(-25, 25)):
    return x_range[0] <= x <= x_range[1] and y_range[0] <= y <= y_range[1]

def moving_average_filter(group, window_size=10):
    group['x'] = group['x'].rolling(window=window_size, min_periods=1, center=True).mean()
    group['y'] = group['y'].rolling(window=window_size, min_periods=1, center=True).mean()
    return group

def count_density_and_direction(trajectory, initdict, grid_size, x_min1, y_min1):
    for i in range(len(trajectory) - 1):
        point1 = trajectory[i]
        point2 = trajectory[i+1]
        list1=point1.tolist()
        list2=point2.tolist()
        if isinstance(list1, list):
            x1, y1 = list1[0][0],list1[0][1]
        else:
            x1, y1 = list1.x,list1.y
        if isinstance(list2, list):
            x2, y2 = list2[0][0],list2[0][1]
        else:
            x2, y2 = list2.x,list2.y
        angle = math.atan2(y2 - y1, x2 - x1)
        grid_x1, grid_y1 = int((x1 - x_min1) / grid_size), int((y1 - y_min1) / grid_size)
        grid_x2, grid_y2 = int((x2 - x_min1) / grid_size), int((y2 - y_min1) / grid_size)
        rr, cc = skiline(grid_y1, grid_x1, grid_y2, grid_x2)
        cross_pixels = np.array([rr, cc])
        # cv2.line(img, (grid_x1, grid_y1), (grid_x2, grid_y2), 255, 3)
        # cross_pixels = np.array(np.where(img == 255))
        for cross_pixel in cross_pixels.T:
            cross_pixel = tuple(cross_pixel)
            initdict[cross_pixel]['counts'] += 1
            initdict[cross_pixel]['angles'].append(angle)
    return initdict

def custom_sigmoid(x):
    return 1/(1+np.exp(-10*(x-0.3)))
# 1/(1+np.exp(-10*(x-0.1)))
# 1.1/(1+np.exp(-10*(x-0.2)))-0.1
# 0.1 + 0.9 / (1 + np.exp(-6 * (x-0.5)))

def map_to_unit_interval(x):
    return 0.5 + np.arctan(x) / np.pi

def draw_heatmap(data,resolution, x_min1, x_max1, y_min1, y_max1, min_len, num_sample_points,bev_h, bev_w,segment_id,token):
    trajs = data ['trajectory']
    density, direction = calculate_grid_density_and_direction(trajs, resolution, x_min1, x_max1, y_min1, y_max1, min_len, num_sample_points)
    a = [density[key] for key in density.keys()]
    a.sort()
    max_value = np.percentile(a,80)
    arrow_img = np.zeros((bev_h, bev_w, 2))  # 创建一个3通道的图像，以便我们可以使用颜色

    for (x, y), dir in direction.items():
        if x < 0 or y < 0 or x >= bev_h or y >= bev_w:
            continue
        density_value = density[(x, y)]
        color = custom_sigmoid(density_value/max_value)
        color_dir = map_to_unit_interval(dir)
        arrow_img[...,0][x, y] = color
        arrow_img[...,1][x, y] = color_dir
    # os.makedirs(f'{root_path}/{split}/{segment_id}', exist_ok=True)
    os.makedirs(f"/data/lza/olv2-traj-data/traj_train_Final_no_filter/{segment_id}",exist_ok=True)
    np.save(f'/data/lza/olv2-traj-data/traj_train_Final_no_filter/{segment_id}/{token}.npy', arrow_img)

def calculate_grid_density_and_direction( df, grid_size, x_min, x_max, y_min, y_max, min_len, num_sample_points):
    """计算网格的密度和方向"""
    # 初始化density，给每个网格初始密度为1，初始角度为0
    initdict = defaultdict(lambda: {'counts': 0, 'angles': [0]})
    patch = box(x_min, y_min, x_max, y_max)
    line_list = []
    for traj in df:
        data = np.array(traj['data'])
        # data = pd.DataFrame({'x': data[:, 0], 'y': data[:, 1]})
        # df_smoothed = moving_average_filter(data)
        line = LineString(data)
        if line.is_empty or line.length < min_len:
            continue
        new_line = line.intersection(patch)
        if not new_line.is_empty:
            if new_line.geom_type == 'MultiLineString':
                for single_line in new_line.geoms:
                    if single_line.is_empty or single_line.length < min_len:
                        continue
                    distances = np.linspace(0, single_line.length, num_sample_points)
                    trajectory = [np.array(line.interpolate(distance).coords) for distance in distances]
                    initdict = count_density_and_direction(trajectory, initdict, grid_size, x_min, y_min)
                    line_list.append(trajectory)
            else:                                            
                if new_line.length < min_len:
                    continue       
                distances = np.linspace(0, new_line.length, num_sample_points)
                trajectory = [np.array(line.interpolate(distance)) for distance in distances]
                initdict = count_density_and_direction(trajectory, initdict, grid_size, x_min, y_min)
                line_list.append(trajectory)
    direction = {k: np.mean(v['angles']) for k, v in initdict.items()}
    density = {k: v['counts'] for k, v in initdict.items()}
    return density, direction



def process_info(json_name):
    # try:
    dir_name, file_name = os.path.split(json_name)
    _, segment_id = os.path.split(os.path.dirname(dir_name))
    token, _ = os.path.splitext(file_name)
    
    destination_folder_path = os.path.join('/data/lza/olv2-traj-data/filter_train_new', segment_id, 'info')
    image_folder_path = os.path.join('/data/lza/olv2-traj-data/filter_train_new',segment_id, 'image')
    json_path = os.path.join(destination_folder_path, f"{token}.json")
    image_path = os.path.join(image_folder_path, f"{token}.jpg")
    # 如果已经生成过该帧数据，则跳过
    # if os.path.exists(json_path) and os.path.exists(image_path):

    if os.path.exists(json_path):
        # print(f"Skipping {token}, already processed.")
        return  # 跳过已处理的场景
    
    try:
        with open(json_name, 'r', encoding='utf-8') as f:
            data = json.load(f)  # 尝试读取 JSON 数据
    except (UnicodeDecodeError, json.JSONDecodeError):
        # 如果是二进制文件或 JSON 解析失败，记录信息并跳过
        binary_files_info.append((segment_id, token))
        print(f"Binary or invalid JSON file detected: {json_name}, skipping...")
        return
    # if segment_id  == '00085'  and token == '315968331549927214':
    #     print('json file has something wrong!! not complete!!')
    #     return
    # if segment_id  == '00085'  and token == '315968339549927213':
    #     print('json file has something wrong!! not complete!!')
    #     return
    output_data = {
        'lane_centerline': [],
        'trajectory': []
    }
    x_min1, x_max1, y_min1, y_max1 = -50, 50, -25, 25
    resolution = 0.5
    bev_h = 100
    bev_w = 200
    min_len, num_sample_points = 10, 10
    processed_traj_dict = {}
    with open(json_name, 'r') as f:
            data_ori = json.load(f)
    #车道线数目判断
    map_elements=data_ori['lane_centerline']
    map_line_list=[]
    # map_line_list1=[]
    # map_line_list2=[]
    flag=0
    flag1=0
    for element in map_elements:
        flag1+=1
        your_points_data = [(x, y, z) for x, y, z in element["points"]]
        x_coords = [point[0] for point in your_points_data]
        y_coords = [point[1] for point in your_points_data]
        xy_points_list = [(x, y) for x, y in zip(x_coords, y_coords)]
        line_string = LineString(xy_points_list)
        if element['is_intersection_or_connector'] is True:
            flag+=1
            map_line_list.append(line_string)
        else:
            flag1+=1


    # if flag>=8:
        # from sklearn.cluster import DBSCAN
        # from shapely.geometry import MultiLineString
        # from shapely.ops import unary_union
        # points = [point for line in map_line_list for point in line.coords]
        # clustering = DBSCAN(eps=3, min_samples=2).fit(points)
        # clusters = []
        # for label in set(clustering.labels_):
        #     if label == -1:
        #         continue  # 忽略噪声点
        #     lines = [line for line, label_ in zip(map_line_list, clustering.labels_) if label_ == label]
        #     clusters.append(MultiLineString(lines))
        # # 对每个聚类计算凸包
        # convex_hulls = [unary_union(cluster).convex_hull for cluster in clusters]
    trajs = data_ori ['trajectory']
    flag3=0
    for traj in trajs:
        data = np.array(traj['data'])
        df = pd.DataFrame({'x': data[:, 0], 'y': data[:, 1]})
        df_smoothed = moving_average_filter(df)
        points_in_range = [Point(x, y) for x, y in df_smoothed.values if point_in_range(x, y)]
        if len(points_in_range) < 2:
            continue
        line_smoothed = LineString(points_in_range)
        if line_smoothed.length < 10:
            continue
        else:
            if traj['id'] == 'vehicle' or traj['id'] == 'bus':
                processed_traj_dict[traj['id']+str(flag3)] = line_smoothed
                flag3+=1
                
    if flag3>flag1*5:
        filter_list.append((segment_id, token))
    
    os.makedirs(f"/data/lza/olv2-traj-data/filter_train_new/{segment_id}",exist_ok=True)
    os.makedirs(destination_folder_path,exist_ok=True)
    os.makedirs(image_folder_path,exist_ok=True)
    image_path = os.path.join(image_folder_path,f"{token}.jpg")
    json_path = os.path.join(destination_folder_path,f"{token}.json")
    for element in map_elements:
        output_data['lane_centerline'].append({
            'id': element['id'],
            'points': element['points'],
            'is_intersection_or_connector': element['is_intersection_or_connector']
        })
    for traj_id, line_string in processed_traj_dict.items():
        trajectory_data = [[round(x,2), round(y,2), 0] for x, y in line_string.coords]
        output_data['trajectory'].append({
            'id': traj_id,
            'data': trajectory_data
        })
    with open(json_path, 'w') as f:
        json.dump(output_data, f)
    try:
        draw_heatmap(output_data,resolution,x_min1,x_max1,y_min1,y_max1,min_len,num_sample_points,bev_h,bev_w,segment_id,token)
        npy_path = os.path.join('/data/lza/olv2-traj-data/traj_train_Final_no_filter', segment_id,  f'{token}.npy')
        data = np.load(npy_path)
        density = data[:, :, 0] * 255

        density_map_normalized = cv2.normalize(density, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        density_map_colored = cv2.applyColorMap(density_map_normalized, cv2.COLORMAP_JET)

        target = '/data/lza/olv2-traj-data/vis_train_npy'
        target_path = os.path.join(target, segment_id)
        os.makedirs(target_path, exist_ok=True)
        target_final = os.path.join(target_path, f'{token}.jpg')

        cv2.imwrite(target_final, density)
        fig, ax = plt.subplots()
        for element in map_elements:
            map_data=element["points"]
            x = [point[0] for point in map_data]
            y = [point[1] for point in map_data]
            ax.plot(x, y,linewidth=0.5,color='black')
        for id,line in processed_traj_dict.items():
            x,y=line.coords.xy
            ax.plot(x, y, linewidth=0.1)
        plt.savefig(image_path, format='jpg',dpi=500)
        plt.close()
    except Exception as e:  # 捕获其他所有异常
        # 记录其他错误信息
        other_error_files_info.append((segment_id, token, str(e)))
        print(f"Error processing {json_name}: {e}, skipping...")
        return
    print("satisfy!")
    # except Exception as e:
    #     print(f"Error processing info: {e}")

def parallel_process(json_file_list):
    num_processes = os.cpu_count()
    pool =multiprocessing.Pool(processes=8)
    try:
        for _ in tqdm(pool.imap_unordered(process_info, json_file_list), total=len(json_file_list)):
            pass
    finally:
        pool.close()
        pool.join()

    print("所有任务已完成。")

binary_files_info = []
filter_list = []
other_error_files_info = []

for file in tqdm(json_file_list):
    process_info(file)

filter_json_path = '/data/lza/olv2-traj-data/filter_train.json'
with open(filter_json_path, 'w') as json_file:
    json.dump(filter_list, json_file,indent=4)

false_info = '/data/lza/olv2-traj-data/false_info.json'
with open(false_info, 'w') as json_file:
    json.dump(binary_files_info, json_file,indent=4)

other_error_files_info = '/data/lza/olv2-traj-data/other_error_files_info.json'
with open(other_error_files_info, 'w') as json_file:
    json.dump(other_error_files_info, json_file,indent=4)


































