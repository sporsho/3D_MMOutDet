from functools import partial

import spconv
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
import numpy as np

from LitePT.datasets.transform import Compose
from LitePT.litept.model import PointSequential, Point
from models import LitePTEncoder, LitePTDecoder


class LiRA_PT(nn.Module):
    def __init__(self, conv_kernel, pcd_size, rgb_size, thermal_size):
        super().__init__()
        self.pcd_size = pcd_size
        self.rgb_size = rgb_size
        self.thermal_size = thermal_size
        in_ch = pcd_size + rgb_size + thermal_size
        self.pcd_conv = PointSequential(
            conv=spconv.SubMConv3d(in_channels=in_ch, out_channels=pcd_size, kernel_size=conv_kernel, bias=False,
                                   indice_key='lira0'))
        self.img_conv = PointSequential(
            conv=spconv.SubMConv3d(in_channels=in_ch, out_channels=rgb_size, kernel_size=conv_kernel, bias=False,
                                   indice_key='lira0'))
        if thermal_size != 0:
            self.ther_conv = PointSequential(
                conv=spconv.SubMConv3d(in_channels=in_ch, out_channels=thermal_size, kernel_size=conv_kernel,
                                       bias=False, indice_key='lira0'))

    def forward(self, pcd_dict, rgb_dict, ther_dict):
        selected_ind = rgb_dict['selected_ind']
        if "K" in rgb_dict.keys():
            K = rgb_dict['K']
        else:
            K = len(selected_ind)
        if K == 0:
            selected_k = selected_ind
        else:
            selected_k = selected_ind[:K]
        rgb_dict["K"] = K
        fsize = rgb_dict['feat'].size(1)
        if K != 0:
            all_feat_pcd = torch.concatenate((pcd_dict['feat'], rgb_dict['feat'][selected_ind[: K]],), dim=1)
        else:
            all_feat_pcd = torch.concatenate((pcd_dict['feat'], rgb_dict['feat'],), dim=1)

        all_feat_rgb = torch.zeros(size=(rgb_dict['feat'].size(0), fsize * 2), dtype=torch.float32,
                                   device=rgb_dict['feat'].device)
        if K != 0:
            all_feat_rgb[selected_k, : fsize] += pcd_dict['feat']
        else:
            all_feat_rgb[:, : fsize] += pcd_dict['feat']
        all_feat_rgb[:, fsize:] += rgb_dict['feat']

        # all_feat = all_feat.unsqueeze(-1)
        pcd_dict['feat'] = all_feat_pcd
        pcd_dict['sparse_conv_feat'] = pcd_dict['sparse_conv_feat'].replace_feature(all_feat_pcd)
        rgb_dict['feat'] = all_feat_rgb
        rgb_dict['sparse_conv_feat'] = rgb_dict['sparse_conv_feat'].replace_feature(all_feat_rgb)

        pcd_dict = self.pcd_conv(pcd_dict)
        rgb_dict = self.img_conv(rgb_dict)
        # if rgb_dict['feat'].size(0) > rgb_dict['K']:
        #     rgb_dict['feat'] = rgb_dict['feat'][selected_ind]
        # if rgb_dict['feat'].size(0) > rgb_dict['K']:
        #     rgb_dict['feat'] = rgb_dict['feat'][: rgb_dict['K']]
        rgb_dict['sparse_conv_feat'] = rgb_dict['sparse_conv_feat'].replace_feature(rgb_dict['feat'])

        pcd_weight = torch.sigmoid(pcd_dict['feat'])
        img_weight = torch.sigmoid(rgb_dict['feat'])
        pxw = pcd_dict['feat'] * pcd_weight
        ixw = rgb_dict['feat'] * img_weight

        # pcd_sparse_weight = torch.sigmoid(pcd_dict['sparse_conv_feat'].features)
        # img_sparse_weight = torch.sigmoid(rgb_dict['sparse_conv_feat'].features)
        # sp_pxw = pcd_dict['sparse_conv_feat'].features * pcd_sparse_weight
        # sp_ixw = rgb_dict['sparse_conv_feat'].features * img_sparse_weight

        if self.thermal_size != 0:
            ther_dict = self.ther_conv(ther_dict)
            ther_weight = torch.sigmoid(ther_dict['feat'])
            txw = ther_dict['feat'] * ther_weight
            out_feat = torch.concatenate((pxw, ixw, txw), dim=-1)
            ther_sparse_weight = torch.sigmoid(ther_dict['sparse_conv_feat'].features)
            sp_txw = ther_dict['sparse_conv_feat'].features * ther_sparse_weight
            # sp_out_feat = torch.concatenate((sp_pxw, sp_ixw, sp_txw), dim=-1)

        else:
            out_feat = torch.concatenate((pxw, ixw), dim=-1)
            # sp_out_feat = torch.concatenate((sp_pxw, sp_ixw), dim=-1)
        pcd_dict['feat'] = out_feat
        pcd_dict['sparse_conv_feat'] = pcd_dict['sparse_conv_feat'].replace_feature(out_feat)

        return pcd_dict



class MidFusedLitePT_LiRA(nn.Module):
    def __init__(self, num_classes=20, kernel_size=5, depth=6, pool=True, dilate=1, pcd_in_channels=4, rgb_in_channel=5,
                 gated=False,
                 attention=False, pcd_feat_size=504, img_feat_size=504, ther_feat_size=0, ther_in_channel=0):
        super().__init__()
        self.gated = gated
        self.pcd_feat_size = pcd_feat_size
        self.img_feat_size = img_feat_size
        self.ther_feat_size = ther_feat_size
        self.pcd_extractor = LitePTEncoder()
        self.rgb_extractor = LitePTEncoder(in_channels=4)
        if ther_feat_size != 0:
            self.ther_extractor = LitePTEncoder()

        if gated:
            self.gate = LiRA_PT(3, pcd_feat_size, img_feat_size, ther_feat_size)
            self.pred_in_ch = 3 + pcd_feat_size + img_feat_size + ther_feat_size
            # self.pred_in_ch = 3 + pcd_feat_size
        else:
            self.pred_in_ch = 3 + pcd_feat_size + img_feat_size
            # self.pred_in_ch = 3 + pcd_feat_size
        self.predictor = LitePTDecoder()

    def forward(self, data_dict):
        point_pcd = Point(data_dict)
        point_rgb = Point(data_dict)
        point_pcd['feat'] = point_pcd['feat_pcd']
        point_rgb['feat'] = point_rgb['feat_rgb']
        point_pcd = self.pcd_extractor(point_pcd)
        point_rgb = self.rgb_extractor(point_rgb)
        if self.gated:
            point_pcd = self.gate(point_pcd, point_rgb, None)
        else:
            point_pcd = torch.concat((point_pcd['feat'], point_rgb['feat']), dim=1)
        pred_dict = self.predictor(point_pcd)
        return pred_dict


class IndexSelector(nn.Module):
    def __init__(self, feat_size):
        super().__init__()
        layers = [
            nn.Linear(in_features=feat_size, out_features=64),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=1),
            nn.Sigmoid(),
        ]
        self.sel = nn.Sequential(*layers)

    def forward(self, x):
        out = self.sel(x)
        # out = torch.where(out > 0.0, 1.0, out)
        return out


class MidFusedLitePT_LiRA_OFA3(nn.Module):
    def __init__(self, num_classes=20, kernel_size=5, depth=6, pool=True, dilate=1, pcd_in_channels=4, rgb_in_channel=5,
                 gated=False,
                 attention=False, pcd_feat_size=504, img_feat_size=504, ther_feat_size=0, ther_in_channel=0):
        super().__init__()
        self.gated = gated
        self.pcd_feat_size = pcd_feat_size
        self.img_feat_size = img_feat_size
        self.ther_feat_size = ther_feat_size
        self.pcd_extractor = LitePTEncoder()
        self.rgb_extractor = LitePTEncoder(in_channels=6)
        if ther_feat_size != 0:
            self.ther_extractor = LitePTEncoder()

        if gated:
            self.gate = LiRA_PT(3, pcd_feat_size, img_feat_size, ther_feat_size)
            self.pred_in_ch = 3 + pcd_feat_size + img_feat_size + ther_feat_size
            # self.pred_in_ch = 3 + pcd_feat_size
        else:
            self.pred_in_ch = 3 + pcd_feat_size + img_feat_size
            # self.pred_in_ch = 3 + pcd_feat_size
        self.selector = IndexSelector(feat_size=img_feat_size)

        self.predictor = LitePTDecoder()

    def forward(self, data_dict):
        pcd_dict = dict()
        rgb_dict = dict()
        for k, v in data_dict.items():
            if k.startswith('rgb'):
                rgb_dict[k[4:]] = v
            else:
                pcd_dict[k] = v
        point_pcd = Point(pcd_dict)
        point_rgb = Point(rgb_dict)
        # point_pcd['feat'] = point_pcd['feat_pcd']
        # point_rgb['feat'] = point_rgb['feat_rgb']
        point_pcd = self.pcd_extractor(point_pcd)
        point_rgb = self.rgb_extractor(point_rgb)
        pcd_feat = point_pcd['feat']
        # pcd_sparse_feat = point_pcd['sparse_conv_feat']
        rgb_feat = point_rgb['feat']
        K = pcd_feat.size(0)
        ind_pred = self.selector(rgb_feat)
        # rgb_sum = torch.sum(torch.relu(rgb_feat), dim=1)
        # _, selected_ind = torch.topk(ind_pred.reshape(-1), K)
        selected_ind = torch.where(ind_pred >= 0.5)[0]
        # rgb_feat = rgb_feat[selected_ind]
        # point_rgb['coord'] = point_rgb['coord'][selected_ind]
        # point_rgb['grid_coord'] = point_rgb['grid_coord'][selected_ind]
        # point_rgb['batch'] = point_rgb['batch'][selected_ind]
        # point_rgb['serialized_code'] = point_rgb['serialized_code'][:, selected_ind]
        # point_rgb['serialized_order'] = point_rgb['serialized_order'][:, selected_ind]
        # point_rgb['serialized_inverse'] = point_rgb['serialized_inverse'][:, selected_ind]
        # unpad_mask = point_rgb['unpad'] < K
        # point_rgb['unpad'] = point_rgb['unpad'][unpad_mask]
        # pad_mask = point_rgb['pad'] < K
        # point_rgb['pad'] = point_rgb['pad'][pad_mask]
        point_rgb['selected_ind'] = selected_ind
        point_rgb['K'] = K
        # if rgb_feat.shape[0] > K:
        #     rgb_feat = rgb_feat[:K]
        #     point_rgb['coord'] = point_rgb['coord'][:K]
        #     point_rgb['grid_coord'] = point_rgb['grid_coord'][:K]
        #     point_rgb['batch'] = point_rgb['batch'][:K]
        #     point_rgb['serialized_code'] = point_rgb['serialized_code'][:, :K]
        #     point_rgb['serialized_order'] = point_rgb['serialized_order'][:, :K]
        #     point_rgb['serialized_inverse'] = point_rgb['serialized_inverse'][:, :K]
        #     point_rgb['unpad'] = point_rgb['unpad'][ :K]
        # rgb_sparse_feat = rgb_sparse_feat[:K]

        # point_rgb['feat'] = rgb_feat
        # point_rgb['sparse_conv_feat'] = point_rgb['sparse_conv_feat'].replace_feature(rgb_feat)
        if self.gated:
            point_pcd = self.gate(point_pcd, point_rgb, None)
        else:
            point_pcd = torch.concat((point_pcd['feat'], point_rgb['feat']), dim=1)
        pred_dict = self.predictor(point_pcd)
        if self.training:
            index_loss = torch.pow((K * torch.log(torch.tensor(0.99)) - torch.sum(torch.log(ind_pred + 1e-9))), 2)
            return pred_dict, index_loss
        else:
            return pred_dict


if __name__ == "__main__":
    model = MidFusedLitePT_LiRA()
    model.cuda()
    lidar_path = hf_hub_download(
        repo_id="prs-eth/LitePT_demo",
        filename=f"outdoor_sample1.bin",
        repo_type="dataset",
        revision="main",
    )
    points = np.fromfile(lidar_path, dtype=np.float32, count=-1).reshape(
        [-1, 5]
    )
    coord = points[:, :3]  # [N, 3]
    strength = points[:, 3].reshape([-1, 1]) / 255  # scale strength to [0, 1]
    point = dict(
        coord=coord,
        strength=strength,
    )
    print("Number of points: ", coord.shape[0])

    ### apply basic transforms
    data_config = [
        dict(
            type="GridSample",
            grid_size=0.05,
            hash_type="fnv",
            mode="train",
            return_grid_coord=True,
            return_inverse=True,
        ),
        dict(type="ToTensor"),
        dict(
            type="Collect",
            keys=("coord", "grid_coord", "inverse"),
            feat_keys=("coord", "strength"),
        ),
    ]
    transform = Compose(data_config)
    point = transform(point)
    for key in point.keys():
        if isinstance(point[key], torch.Tensor):
            point[key] = point[key].cuda(non_blocking=True)
    # forward
    point = model(point)  # [N_down, C]
    # point is downsampled by GridSample in transform
    print("Downsampled output feature: ", point.feat.shape)
    # obtain per-point feature
    dense_feat = point.feat[point.inverse]  # [N, C]
    print("Dense output feature: ", dense_feat.shape)