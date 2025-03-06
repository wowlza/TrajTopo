import os
import json
import shutil
import json
import os
from matplotlib import pyplot as plt
import time
import matplotlib.pyplot as plt
import glob
#  '/data/jpj/val/dearborn-csv'
paths = ['/data/jpj/val/dearborn-csv/']
merged_path = '/data/lzy/val_all_traj_fromtrain/'
subfolders = [f.name for f in os.scandir(paths[0]) if f.is_dir()]
#print(type(subfolders))
# state_folder='/home/jpj/lzy/output/state_output/pittsburgh-csv'
# os.makedirs(state_folder, exist_ok=True)
for subfolder in subfolders:
    json_files = glob.glob(os.path.join(paths[0], subfolder, '*.json'))
    png_files = glob.glob(os.path.join(paths[0], subfolder, '*.jpg'))
    json_names = [os.path.splitext(os.path.basename(file))[0] for file in json_files]           
    for json_name in json_names:
        os.makedirs(os.path.join(merged_path,f"{subfolder}","image"), exist_ok=True)
        os.makedirs(os.path.join(merged_path,f"{subfolder}", "info"), exist_ok=True)
        source_json = os.path.join(paths[0], subfolder, f"{json_name}.json")
        target_json = os.path.join(merged_path,f"{subfolder}","info",f"{json_name}.json")
        source_png = os.path.join(paths[0], subfolder, f"{json_name}.jpg")
        target_png = os.path.join(merged_path,f"{subfolder}","image",f"{json_name}.jpg")
        shutil.move(source_json, target_json)
        shutil.move(source_png, target_png)    


