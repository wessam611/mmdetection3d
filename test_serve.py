import requests
import pickle
import numpy as np
import time
import glob

url = 'http://127.0.0.1:8080/predictions/st3d_hv'
pcd = 'demo/data/kitti/000008.bin'
pkl_paths = glob.glob('data/waymo/waymo_format/records_shuffled/validation/pre_data/1020365635352417*.pkl')
# print(points.shape)
# with open(pcd, 'rb') as points:?
# points = np.random.random((11, 5)).astype(np.float32)
# print(points)
for i in range(100):
    pkl_path = pkl_paths[i]
    with open(pkl_path, 'rb') as pkl_file:
        pkl_dict = pickle.load(pkl_file)
    points = pkl_dict['points'][:, :5]
    b = time.time()
    response = requests.post(url, points.tobytes())
    a = time.time()
    # print(response.content.decode())
    print(a-b)
