from torch.utils import data
import torch
import pickle
import yaml
import numpy as np
import os
from tqdm.auto import tqdm 
import cv2 
from datasets.data_util import exists, get_circular_mask, image_to_points, multiply_matching_coordinates_rgb, \
    get_circular_mask_center, multiply_matching_coordinates_thermal


class FusedDataSetSimplified(data.Dataset):  # Image U Thermal = IUT
    def __init__(self, data_root, data_set, device=torch.device('cuda:0'), k=9, label_conf='roadview_snow.yaml',
                 pcd_mean=[-1.8294078,  -2.639477,    0.74836063,  4.191953  ],
                 pcd_std= [14.600004,  36.418285 ,  2.3450274 , 15.861519 ],
                 thermal_mean=[9544.443, ],
                 thermal_std=(11534.338,),
                 rgb_mean=[82.05462, 156.97069, 19.621336, 19.621336, 19.621336, 2511.535],
                 rgb_std=[216.24039, 566.9885, 50.119503, 52.831364, 70.873566, 6414.933],
                 recalculate=False, limit=None,
                 rgb_trf=None,
                 thermal_trf=None,
                 drop_thermal=False,
                 drop_rgb=False,
                 falling_only=False,
                 mask_rgb=None,
                 mask_thermal=None):
        # read rgb pikels
        with open('/var/local/home/aburai/roadview-snow-devkit/OutDet/dataset/rgb.pkl', 'rb') as f:
            rgb_dict = pickle.load(f)
        self.rgb_points = rgb_dict['img_points']
        self.rgb_dist = rgb_dict['img_dist']
        self.rgb_ind = rgb_dict['img_indices']
        rgb_ind_mean = rgb_dict['mean'].tolist()
        rgb_ind_std = rgb_dict['std'].tolist()
        self.drop_thermal = drop_thermal
        self.drop_rgb = drop_rgb
        self.data_root = data_root
        self.device = device
        self.mask_rgb = mask_rgb
        self.mask_thermal = mask_thermal
        # assert mask_rgb in [None, float]
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


    def calculate_pcd_mean_std(self):
        N = len(self.file_list)
        all_pcd = list()
        for i in range(N):
            pcd_file, ther_file, rgb_file = self.file_list[i]
            batch = self.__getitem__(i)
            inlier = batch['inlier']
            pcd = np.fromfile(pcd_file, dtype=np.float32).reshape(-1, 4)
            pcd = pcd[inlier]
            all_pcd.append(pcd)
        all_imgs = np.concatenate(all_pcd, axis=0)
        mean = np.mean(all_imgs, axis=0)
        std = np.std(all_imgs, axis=0)
        print(mean, std)

    def calculate_rgb_mean_std(self):
        img_list = list()
        pbar = tqdm(total=len(self.file_list))
        for i in range(len(self.file_list)):
            pcd_file, ther_file, rgb_file = self.file_list[i]
            img = cv2.imread(rgb_file, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_list.append(img)
            pbar.update()
        all_imgs = np.stack(img_list)
        mean = np.mean(all_imgs, axis=(0, 1, 2))
        std = np.std(all_imgs, axis=(0, 1, 2))
        print(mean, std)

    def save_front_view(self):
        bar = tqdm(total=len(self.file_list))
        for i in range(len(self.file_list)):
            batch = self.__getitem__(i)
            keys = batch.keys()
            index = batch['index']
            pcd_file, therC_file, rgb_file = self.file_list[i]
            pcd = np.fromfile(pcd_file, dtype=np.float32).reshape(-1, 4)
            inlier = batch['inlier']
            front_pcd = pcd[inlier]
            rgb = batch['rgb']
            rgb_dist = batch['rgb_dist']
            rgb_ind = batch['rgb_ind']
            rgb_to_pcd_ind = batch['selected_ind']
            front_pcd_file = pcd_file.replace('velodyne', 'front_pcd_ind')
            os.makedirs(os.path.dirname(front_pcd_file), exist_ok=True)
            inlier.tofile(front_pcd_file)
            # save rgb stuff
            pkl_path = pcd_file.replace('velodyne', 'rgb_helper')[:-3] + 'pkl'
            os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
            with open(pkl_path, 'wb') as f:
                pickle.dump(rgb, f)
                pickle.dump(rgb_dist, f)
                pickle.dump(rgb_ind, f)
                pickle.dump(rgb_to_pcd_ind, f)
            bar.update()
        bar.close()

    def calculate_thermal_mean_std(self):
        img_list_C = list()
        pbar = tqdm(total=len(self.file_list))
        for i in range(len(self.file_list)):
            pcd_file, therC_file, rgb_file = self.file_list[i]
            ther_c = cv2.imread(therC_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            img_list_C.append(ther_c)
            pbar.update()
        pbar.close()
        all_C = np.stack(img_list_C)
        mean_c = np.mean(all_C)
        std_c = np.std(all_C)

        print(mean_c)
        print(std_c)

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
        kd_path = pcd_file.replace('velodyne', 'knn')[:-3] + 'pkl'
        if os.path.exists(kd_path) and not self.recalculate:
            with open(kd_path, 'rb') as f:
                try:
                    ind = pickle.load(f)
                    dist = pickle.load(f)
                    if ind.shape[1] >= self.k:
                        ind = ind[:, :self.k]
                        dist = dist[:, :self.k]
                        err = False
                    else:
                        err = True
                except EOFError:
                    err = True
        else:
            err = True

        dist = dist + 1.0

        # read rgb file
        img = cv2.imread(rgb_file, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        thermal_img = cv2.imread(ther_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        rgbt = np.concatenate((img, np.expand_dims(thermal_img, -1)), -1)

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
        if self.drop_thermal:
            if self.data_set == 'train':
                if np.random.uniform() > 0.5:
                    rgb[:, 5] *= 0
            else:
                rgb[:, 5] *= 0
        elif self.drop_rgb:
            if self.data_set == 'train':
                if np.random.uniform() > 0.5:
                    rgb[:, 2:5] *= 0
            else:
                rgb[:, 2:5] *= 0

        if self.mask_rgb:
            if self.data_set == "train":
                h, w, _ = img.shape
                mask = get_circular_mask(h, w, self.mask_rgb)
                pts = image_to_points(mask)
                rgb[:, 2:5] = multiply_matching_coordinates_rgb(rgb[:, :5], pts )
            else:
                h, w, _ = img.shape
                mask = get_circular_mask_center(h, w, self.mask_rgb)
                pts = image_to_points(mask)
                rgb[:, 2:5] = multiply_matching_coordinates_rgb(rgb[:, :5], pts )
        if self.mask_thermal:
            if self.data_set == "train":
                h, w, _ = img.shape
                mask = get_circular_mask(h, w, self.mask_rgb)
                pts = image_to_points(mask)
                rgb[:, 5] = multiply_matching_coordinates_thermal(rgb[:, [0,1,5]], pts )
            else:
                h, w, _ = img.shape
                mask = get_circular_mask_center(h, w, self.mask_rgb)
                pts = image_to_points(mask)
                rgb[:, 5] =  multiply_matching_coordinates_thermal(rgb[:, [0,1,5]], pts )

        rgb = (rgb - self.rgb_mean) / self.rgb_std



        pcd_label_mapped = np.vectorize(self.learning_map.__getitem__)(pcd_label).reshape(-1)
        pcd = (pcd - self.pcd_mean) / self.pcd_std
        rgbt = (rgbt - self.rgb_mean[2:]) / self.rgb_std[2:]
        rgbt = rgbt.astype(np.float32)
        # rgbt += np.random.normal(0, 0.5, rgbt.shape)
        # pcd += np.random.normal(0,0.04, pcd.shape)
        out_dict = {'pcd': pcd.astype(np.float32), 'pcd_dist': dist.astype(np.float32),
                    'pcd_ind': ind.astype(np.int64), 'pcd_label': pcd_label.astype(np.int64),
                    'mapped_label': pcd_label_mapped.astype(np.int64),
                    # 'thermal': thermal_img,
                    # 'rgb_image': img.transpose(2, 0, 1),
                    'rgbt': rgbt.transpose(2,0,1),
                    'rgb_coord': rgb_coord,
                    'rgb': rgb.astype(np.float32),
                    'inlier': inlier.astype(np.bool), 'index': item,
                    'rgb_dist': rgb_dist.astype(np.float32), 'rgb_ind': rgb_ind.astype(np.int64),
                    'selected_ind': selected_ind.astype(np.int64).reshape(-1),
                    'fname': pcd_file}
        return out_dict