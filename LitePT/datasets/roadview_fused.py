import copy

from torch.utils import data
import torch
import yaml
import numpy as np
import os
import pickle
from tqdm.auto import tqdm
import cv2

def exists(fs):
    for f in fs:
        if not os.path.exists(f):
            return False
    return True



class FusedDataSetLitePT(data.Dataset):  # Image U Thermal = IUT
    def __init__(self, data_root, data_set, transform, device=torch.device('cuda:0'), k=9, label_conf='roadview_snow.yaml',
                 pcd_mean=[-1.8294078,  -2.639477,    0.74836063,  4.191953  ],
                 pcd_std= [14.600004,  36.418285 ,  2.3450274 , 15.861519 ],
                 thermal_mean=[9544.443, ],
                 thermal_std=(11534.338,),
                 rgb_mean=[82.05462,   156.97069,    19.621336,   19.621336,   19.621336, 2511.535 ],
                 rgb_std=[216.24039,   566.9885,     50.119503,   52.831364,   70.873566, 6414.933],
                 recalculate=False, limit=None,
                 thermal_normalized=False,
                 rgb_trf=None,
                 thermal_trf=None,
                 drop_thermal=False,
                 drop_rgb=False,
                 falling_only=False):
        self.transform = transform
        self.drop_thermal = drop_thermal
        self.drop_rgb = drop_rgb
        self.data_root = data_root
        self.device = device
        assert not (drop_rgb and drop_thermal)
        assert data_set in ['train', 'test', 'val']
        self.k = k
        with open(label_conf, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml['learning_map']
        self.recalculate = recalculate
        self.data_set = data_set
        self.pcd_mean = np.array(pcd_mean)
        self.pcd_std = np.array(pcd_std)
        self.thermal_mean = np.array(thermal_mean)

        self.thermal_std = np.array(thermal_std)

        self.rgb_mean = np.array(rgb_mean)
        self.rgb_std = np.array(rgb_std)
        self.rgb_trf = rgb_trf
        self.thermal_trf = thermal_trf

        split_file = os.path.join(data_root, 'SplitTxt', f'{data_set}.txt')
        self.idx = [x.strip().split(' ')[0] for x in open(split_file).readlines()]
        if falling_only:
            tmp = [t for t in self.idx if 'falling' in t]
            self.idx = tmp
        # collect files
        self.file_list = list()
        # 1 pcd, 3 thermal, one rgb
        for i in self.idx:
            pcd_file = os.path.join(data_root, data_set, 'velodyne', f'{i}.bin')
            ther1_file = os.path.join(data_root, data_set, 'thermal_fused', f'{i}.tiff')

            rgb_file = os.path.join(data_root, data_set, 'images', f'{i}.png')
            fs = [pcd_file, ther1_file, rgb_file]
            if exists(fs):
                self.file_list.append(fs)
        if limit is not None:
            self.file_list = self.file_list[:limit]
        # read rgb pikels
        with open('/var/local/home/aburai/multimodal/data/rgb.pkl', 'rb') as f:
            rgb_dict = pickle.load(f)
        self.rgb_points = rgb_dict['img_points']
        self.rgb_dist = rgb_dict['img_dist']
        self.rgb_ind = rgb_dict['img_indices']


    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        pcd_file, ther_file, rgb_file = self.file_list[item]
        front_pcd_ind_file = pcd_file.replace('velodyne', 'front_pcd_ind')
        pkl_path = pcd_file.replace('velodyne', 'rgb_helper')[:-3] + 'pkl'

        # read the pcd
        pcd = np.fromfile(pcd_file, dtype=np.float32).reshape(-1, 4)
        pcd_label = np.fromfile(pcd_file.replace('velodyne', 'labels'),
                                dtype=np.int8).reshape((-1, 1))
        N = pcd.shape[0]
        if pcd_label.shape[0] > N:
            pcd_label = pcd_label[:N]

        inlier = np.fromfile(front_pcd_ind_file, dtype=bool)
        rgb, rgb_ind, rgb_dist, selected_ind = None, None, None, None
        with open(pkl_path, 'rb') as f:
            rgb = pickle.load(f)
            rgb_dist = pickle.load(f)
            rgb_ind = pickle.load(f)
            selected_ind = pickle.load(f)
        if rgb is None:
            raise ValueError('Error Cannot proceed')
        rgb_coord = np.copy(rgb[:, :2]).astype(np.int64)
        rgb = (rgb - self.rgb_mean) / self.rgb_std
        if self.drop_thermal:
            rgb = rgb[:, :5]
        elif self.drop_rgb:
            rgb = rgb[:, [0,1,5]]

        img = cv2.imread(rgb_file, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        thermal_img = cv2.imread(ther_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        rgbt = np.concatenate((img, np.expand_dims(thermal_img, -1)), -1)
        rgbt = (rgbt - self.rgb_mean[2:]) / self.rgb_std[2:]

        rgb = rgb[selected_ind, 2:]
        pcd = pcd[inlier]
        pcd_label = pcd_label[inlier]
        pcd_label_mapped = np.vectorize(self.learning_map.__getitem__)(pcd_label).reshape(-1)
        # pcd = (pcd - self.pcd_mean) / self.pcd_std
        pcd_coord = pcd[:, :3]
        pcd_intensity = pcd[:, 3].reshape(-1, 1)
        pcd_intensity = (pcd_intensity - self.pcd_mean[3]) / self.pcd_std[3]

        out_dict = dict(coord=pcd_coord, intensity=pcd_intensity, rgb=rgb, sem_label=pcd_label_mapped, selected_ind=selected_ind,
                        rgb_ind=rgb_coord, rgbt=rgbt)
        out_dict = self.transform(out_dict)
        return out_dict