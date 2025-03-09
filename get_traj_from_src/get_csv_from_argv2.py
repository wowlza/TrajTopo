# process Argoverse2 Motion-forcasting data to get trajectory data
# Finally get csv files

import os
import json
import pandas as pd

folder_a_path = '/data/argoverse2/motion-forecasting/train'
subfolders = [os.path.join(folder_a_path, folder) for folder in os.listdir(folder_a_path) if os.path.isdir(os.path.join(folder_a_path, folder))]
flag = 1

for folder in subfolders:
    
    files = os.listdir(folder)
    json_files = [os.path.join(folder, file) for file in files if file.endswith('.json')]
    parquet_files = [os.path.join(folder, file) for file in files if file.endswith('.parquet')]
    
    with open(json_files[0], "r") as file:
        data = json.load(file)
    
    z_values = []
    z_values1 = []

    for area_id, area_data in data["drivable_areas"].items():
        area_boundary = area_data.get("area_boundary", [])
        z_values1.extend([point["z"] for point in area_boundary])

    average_z1 = sum(z_values1) / len(z_values1) if z_values1 else 0

    for lane_id, lane_data in data["lane_segments"].items():
        centerline = lane_data.get("centerline", [])
        z_values.extend([point["z"] for point in centerline])

        left_boundary = lane_data.get("left_lane_boundary", [])
        z_values.extend([point["z"] for point in left_boundary])

        right_boundary = lane_data.get("right_lane_boundary", [])
        z_values.extend([point["z"] for point in right_boundary])

    average_z = sum(z_values) / len(z_values) if z_values else 0
    average_z = (average_z + average_z1) / 2

    df = pd.read_parquet(parquet_files[0])
    df['position_z'] = average_z
    df.to_csv(f"/data/lzy/csv_train2/csv{flag}.csv", index=False)
    print(flag)
    flag += 1
    