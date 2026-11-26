import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
import numpy as np

from LitePT.litept.model import LitePT, Point


class LitePTSeg(nn.Module):
    def __init__(self, num_classes=20,
                 in_channels=4,
                 order=("z", "z-trans", "hilbert", "hilbert-trans"),
                 stride=(2, 2, 2, 2),
                 enc_depths=(2, 2, 2, 6, 2),
                 enc_channels=(36, 72, 144, 252, 504),
                 enc_num_head=(2, 4, 8, 14, 28),
                 enc_patch_size=(1024, 1024, 1024, 1024, 1024),
                 enc_conv=(True, True, True, False, False),
                 enc_attn=(False, False, False, True, True),
                 enc_rope_freq=(100.0, 100.0, 100.0, 100.0, 100.0),
                 dec_depths=(0, 0, 0, 0),
                 dec_channels=(72, 72, 144, 252),
                 dec_num_head=(4, 4, 8, 14),
                 dec_patch_size=(1024, 1024, 1024, 1024),
                 dec_conv=(False, False, False, False),
                 dec_attn=(False, False, False, False),
                 dec_rope_freq=(100.0, 100.0, 100.0, 100.0),
                 mlp_ratio=4,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 drop_path=0.3,
                 pre_norm=True,
                 shuffle_orders=True,
                 enc_mode=False,
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.point_processor = LitePT(in_channels,
        order,
        stride,
        enc_depths,
        enc_channels,
        enc_num_head,
        enc_patch_size,
        enc_conv,
        enc_attn,
        enc_rope_freq,
        dec_depths,
        dec_channels,
        dec_num_head,
        dec_patch_size,
        dec_conv,
        dec_attn,
        dec_rope_freq,
        mlp_ratio,
        qkv_bias,
        qk_scale,
        attn_drop,
        proj_drop,
        drop_path,
        pre_norm,
        shuffle_orders,
        enc_mode,)
        self.fc = nn.Linear(in_features=self.point_processor.dec_channels[0], out_features=num_classes)

    def forward(self, data_dict):
        point_pcd = Point(data_dict)
        point_pcd = self.point_processor(point_pcd)
        point_pcd['logit'] = self.fc(point_pcd['feat'])
        return point_pcd


class LitePTEncoder(nn.Module):
    def __init__(
            self,
            in_channels=4,
            order=("z", "z-trans", "hilbert", "hilbert-trans"),
            stride=(2, 2, 2, 2),
            enc_depths=(2, 2, 2, 6, 2),
            enc_channels=(36, 72, 144, 252, 504),
            enc_num_head=(2, 4, 8, 14, 28),
            enc_patch_size=(1024, 1024, 1024, 1024, 1024),
            enc_conv=(True, True, True, False, False),
            enc_attn=(False, False, False, True, True),
            enc_rope_freq=(100.0, 100.0, 100.0, 100.0, 100.0),
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            drop_path=0.3,
            pre_norm=True,
            shuffle_orders=True,
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.shuffle_orders = shuffle_orders

        self.enc_conv = enc_conv
        self.enc_attn = enc_attn

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)

        # norm layers
        bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        ln_layer = nn.LayerNorm

        # activation layers
        act_layer = nn.GELU

        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
        )

        # encoder
        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[
                             sum(enc_depths[:s]): sum(enc_depths[: s + 1])
                             ]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    GridPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                        re_serialization=enc_attn[s],
                        serialization_order=self.order
                    ),

                    name="down",
                )
            for i in range(enc_depths[s]):
                enc.add(
                    Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_conv=enc_conv[s],
                        enable_attn=enc_attn[s],
                        rope_freq=enc_rope_freq[s]
                    ),
                    name=f"block{i}",
                )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

    def forward(self, point):
        """
        data_dict is the batched input point cloud, it should contain as least:
        1. feat [N, input_dim]: input feature for the point cloud
        2. grid_coord [N, 3]: voxelized coordinate after grid sampling
           or/and
           coord [N, 3]: original coordinate + grid_size: grid_size used for grid sampling
        3. offset [batch_size]: separator of point clouds in batched data
           or/and
           batch [N]: batch index of each point
        """
        if self.enc_attn[0]:
            point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        point = self.embedding(point)
        point = self.enc(point)
        return point



class LitePTDecoder(nn.Module):
    def __init__(
            self,
            num_classes=2,
            order=("z", "z-trans", "hilbert", "hilbert-trans"),
            stride=(2, 2, 2, 2),
            enc_depths=(2, 2, 2, 6, 2),
            enc_channels=(36, 72, 144, 252, 504 * 2),
            dec_depths=(0, 0, 0, 0),
            dec_channels=(72, 72, 144, 252),
            dec_num_head=(4, 4, 8, 14),
            dec_patch_size=(1024, 1024, 1024, 1024),
            dec_conv=(False, False, False, False),
            dec_attn=(False, False, False, False),
            dec_rope_freq=(100.0, 100.0, 100.0, 100.0),
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            drop_path=0.3,
            pre_norm=True,
            shuffle_orders=True,
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.shuffle_orders = shuffle_orders

        self.dec_conv = dec_conv
        self.dec_attn = dec_attn

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(dec_depths) + 1
        assert self.num_stages == len(dec_channels) + 1
        assert self.num_stages == len(dec_num_head) + 1
        assert self.num_stages == len(dec_patch_size) + 1

        # norm layers
        bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        ln_layer = nn.LayerNorm

        # activation layers
        act_layer = nn.GELU
        # decoder

        dec_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
        ]
        self.dec = PointSequential()
        dec_channels = list(dec_channels) + [enc_channels[-1]]
        for s in reversed(range(self.num_stages - 1)):
            dec_drop_path_ = dec_drop_path[
                             sum(dec_depths[:s]): sum(dec_depths[: s + 1])
                             ]
            dec_drop_path_.reverse()
            dec = PointSequential()
            dec.add(
                GridUnpooling(
                    in_channels=dec_channels[s + 1],
                    skip_channels=enc_channels[s],
                    out_channels=dec_channels[s],
                    norm_layer=bn_layer,
                    act_layer=act_layer,
                ),

                name="up",
            )
            for i in range(dec_depths[s]):
                dec.add(
                    Block(
                        channels=dec_channels[s],
                        num_heads=dec_num_head[s],
                        patch_size=dec_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=dec_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_conv=dec_conv[s],
                        enable_attn=dec_attn[s],
                        rope_freq=dec_rope_freq[s]
                    ),
                    name=f"block{i}",
                )
            self.dec.add(module=dec, name=f"dec{s}")
        self.fc = nn.Linear(in_features=dec_channels[0], out_features=num_classes)

    def forward(self, point):
        """
        data_dict is the batched input point cloud, it should contain as least:
        1. feat [N, input_dim]: input feature for the point cloud
        2. grid_coord [N, 3]: voxelized coordinate after grid sampling
           or/and
           coord [N, 3]: original coordinate + grid_size: grid_size used for grid sampling
        3. offset [batch_size]: separator of point clouds in batched data
           or/and
           batch [N]: batch index of each point
        """

        point = self.dec(point)
        point['logit'] = self.fc(point['feat'])

        return point


class LitePTExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = LitePTEncoder()
        # self.bn = nn.BatchNorm1d(num_features=feature_size)
        self.act = nn.LeakyReLU(negative_slope=0.2)
        # self.final_fc = nn.Linear(in_features=feature_size, out_features=num_classes, bias=True)

    def forward(self, points):
        points = Point(points)
        out = self.enc(points)
        out['logit'] = self.act(out['feat'])
        feat = points.feat[points.inverse]
        return points.inverse, feat


class LitePTSegV2(nn.Module):
    def __init__(self, num_classes=20, kernel_size=5, depth=6, pool=True, dilate=1, pcd_in_channels=4, rgb_in_channel=5,
                 gated=False,
                 attention=False, pcd_feat_size=504, img_feat_size=504, ther_feat_size=0, ther_in_channel=0):
        super().__init__()
        self.pcd_feat_size = pcd_feat_size
        self.img_feat_size = img_feat_size
        self.ther_feat_size = ther_feat_size
        self.pcd_extractor = LitePTEncoder()
        self.predictor = LitePTDecoder(num_classes=num_classes, enc_channels=(36, 72, 144, 252, 504))

    def forward(self, data_dict):
        point_pcd = Point(data_dict)
        # point_pcd['feat'] = point_pcd['feat_pcd']
        point_pcd = self.pcd_extractor(point_pcd)
        pred_dict = self.predictor(point_pcd)
        return pred_dict


if __name__ == "__main__":
    model = LitePTSegV2()
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