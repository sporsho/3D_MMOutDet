import torch.nn as nn
import yaml
import os
import torch
import numpy as np
import glob

from sklearn.neighbors import KDTree



class NHConvBlock(nn.Module):
    def __init__(self, kernel_size=9, in_channels=1, out_channels=8, init=False, dilate=1):
        super().__init__()
        self.ks = kernel_size
        self.dilate = dilate
        self.conv1 = NHConv(kernel_size, in_channels, out_channels, bias=True, init=init, dilate=dilate)
        self.bn1 = nn.BatchNorm1d(out_channels)
        # self.bn1 = kNNBatchNorm(k=kernel_size, num_feat=out_channels)
        self.act1 = nn.LeakyReLU(negative_slope=0.2)
        # self.pool1 = PoolAvgTree()

        self.conv2 = NHConv(kernel_size, out_channels, out_channels, bias=False, init=False, dilate=dilate)
        self.bn2 = nn.BatchNorm1d(out_channels)
        # self.bn2 = kNNBatchNorm(k=kernel_size, num_feat=out_channels)
        self.act2 = nn.LeakyReLU(negative_slope=0.2)
        # self.pool2 = PoolTree()
        if in_channels != out_channels:
            # self.downsample = ConvTree1D(kernel_size=1, in_channels=in_channels, out_channels=out_channels, bias=True, init=False)
            self.downsample = nn.Conv1d(kernel_size=1, in_channels=in_channels, out_channels=out_channels, bias=False)
            # nn.init.xavier_normal(self.downsample.weight)
            # nn.init.zeros_(self.downsample.bias)
        else:
            self.downsample = None

    def forward(self, data, ind, dist=None):
        inp = data[ind]
        out = self.conv1(inp, dist)
        out = self.bn1(out)
        # out = self.bn1(out, ind)
        out = self.act1(out)
        # out = self.pool1(out, ind)

        out = out[ind]
        out = self.conv2(out, dist)
        out = self.bn2(out)
        # out = self.bn2(out, ind)
        # out = self.pool2(out, ind)

        if self.downsample is not None:
            ds = self.downsample(data.unsqueeze(-1))
        else:
            ds = data
        out = out + ds.squeeze(-1)
        out = self.act2(out)
        return out


class NHConv(nn.Module):
    def __init__(self, kernel_size=9, in_channels=1, out_channels=8, bias=True, init=True, dilate=1):
        super().__init__()
        self.ks = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init = init
        self.bias = bias
        self.dilate = dilate
        if dilate == 2:
            a = [1, 3, 5]
            self.dilate_ind = [8, 10, 12, 22, 24, 26, 36, 38, 40]
        assert dilate <= 2, 'dilation bigger than 2 is not supported right now'
        if init:
            self.in_channels += 1
            if dilate == 2:
                new_size = (np.sqrt(kernel_size) * 2 + 1) ** 2
                self.dw = nn.Parameter(torch.randn(int(new_size)))
            else:
                self.dw = nn.Parameter(torch.randn(kernel_size))
            nn.init.ones_(self.dw)
        self.weight = nn.Parameter(torch.randn(self.in_channels, out_channels, kernel_size))
        nn.init.xavier_normal_(self.weight)
        if bias:
            self.b = nn.Parameter(torch.randn(out_channels))
            # nn.init.zeros_(self.b)

    def forward(self, selected_points, dist=None):
        if self.init:
            tmp = self.dw * dist
            data = torch.cat((selected_points, tmp.unsqueeze(-1)), dim=-1)
        else:
            data = selected_points
        if self.in_channels == 1:
            data = data.unsqueeze(-1)
        if self.dilate == 2:
            data = data[:, self.dilate_ind]
        out = torch.einsum("ijk,klj->il", data, self.weight)
        if self.bias:
            out = out + self.b
        return out


class PoolTree(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, points, indices):
        selected_points = points[indices]
        out, _ = torch.max(selected_points, dim=1)
        return out


class PoolAvgTree(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, points, indices):
        selected_points = points[indices]
        out = torch.mean(selected_points, dim=1)
        return out



class OutDet(nn.Module):

    def __init__(self, num_classes=20, kernel_size=5, depth=6, pool=True, dilate=1, in_channel=4, width=32):
        super().__init__()
        self.pool = pool
        self.num_classes = num_classes
        self.depth = depth
        self.dilate = dilate

        self.tree_kernel = int(np.round(kernel_size * kernel_size))
        self.convs = nn.ModuleList()
        if pool:
            self.pools = nn.ModuleList()
        # self.drops = nn.ModuleList()
        self.drop = nn.Dropout(p=0.5)
        self.in_channels = in_channel
        self.build_model(in_channels=in_channel, out_channels=width, dilate=dilate, pool=pool)

    def build_model(self, in_channels=4, out_channels=32, dilate=False, pool=False ):
        for i in range(self.depth):
            if i == 0:
                init = True
                suff = 0
            else:
                init = False
                suff = 0
            conv1 = NHConvBlock(kernel_size=self.tree_kernel, in_channels=in_channels + suff,
                                  out_channels=out_channels, init=init, dilate=dilate)
            in_channels = out_channels
            if i < 3:
                out_channels = in_channels * 2
            self.convs.append(conv1)
            if pool and i != self.depth - 1:
                pool1 = PoolTree()
                self.pools.append(pool1)
            # self.drops.append(nn.Dropout(p=0.5))
        # suff = 3
        self.fc = nn.Linear(in_channels + suff, self.num_classes, bias=True)
        # nn.init.xavier_normal(self.fc.weight)

    def forward(self, points, dist, indices):
        # xyz = points[:, :3].clone()
        out = points
        # out = self.drop(out)
        for i in range(self.depth):
            if i == 0:
                out = self.convs[i](out, indices, dist=dist)
            else:
                out = self.convs[i](out, indices, dist=dist)
            if self.pool and i != self.depth - 1:
                out = self.pools[i](out, indices)
            # out = self.drops[i](out)
            # out = torch.cat((xyz, out), dim=1)
        out = self.drop(out)
        out = self.fc(out)
        return out

class GlobalExtension(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=2, out_features=10)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc2 = nn.Linear(in_features=10, out_features=2)


    def forward(self, logit):
        out = self.fc1(logit)
        out = torch.t(out)
        out = self.pool(out)
        out = torch.t(out)
        out = self.fc2(out)
        return out
class DropChunk(nn.Module):
    def __init__(self, p, s):
        super().__init__()
        self.p = p
        self.s = s
    def forward(self, x):
        if self.training:
            if torch.rand(size=(1,))[0] > self.p:
                indices = torch.randint(size=(x.size(0), 1), low=0, high=x.size(1) - self.s)
                for a in range(self.s):
                    x[:, indices[:, 0] + a ] = 0
            return x
        else:
            return x
class LiRA(nn.Module):
    def __init__(self, conv_kernel, pcd_size, rgb_size, thermal_size, drop=False):
        super().__init__()
        self.pcd_size = pcd_size
        self.rgb_size = rgb_size
        self.thermal_size = thermal_size
        self.drop = drop
        in_ch = pcd_size + rgb_size + thermal_size
        self.pcd_conv = nn.Conv1d(in_channels=in_ch, out_channels=pcd_size, kernel_size=conv_kernel)
        self.img_conv = nn.Conv1d(in_channels=in_ch, out_channels=rgb_size, kernel_size=conv_kernel)
        if thermal_size != 0:
            self.ther_conv = nn.Conv1d(in_channels=in_ch, out_channels=thermal_size, kernel_size=conv_kernel)

    def forward(self, pcd_feat, rgb_feat, ther_feat):
        if self.thermal_size != 0:
            all_feat = torch.concatenate((pcd_feat, rgb_feat, ther_feat), dim=1)

        else:
            all_feat = torch.concatenate((pcd_feat, rgb_feat), dim=1)
        all_feat = all_feat.unsqueeze(-1)
        pcd_out = self.pcd_conv(all_feat).squeeze()
        img_out = self.img_conv(all_feat).squeeze()
        pcd_weight = torch.sigmoid(pcd_out)
        img_weight = torch.sigmoid(img_out)
        pxw = pcd_feat * pcd_weight
        ixw = rgb_feat * img_weight
        if self.thermal_size != 0:
            ther_out = self.ther_conv(all_feat).squeeze()
            ther_weight = torch.sigmoid(ther_out)
            txw = ther_feat * ther_weight
            out = torch.concatenate((pxw, ixw, txw), dim=-1)
        else:
            out = torch.concatenate((pxw, ixw), dim=-1)

        return out

class Gate2Unit(nn.Module):
    def __init__(self, feat_size):
        super().__init__()
        self.lin1 = nn.Linear(feat_size, feat_size)
        self.lin2 = nn.Linear(feat_size, feat_size)

    def forward(self, x):
        th = torch.tanh(self.lin1(x))
        sig = torch.sigmoid(self.lin2(x))
        out = th * sig
        return out
class Gate2(nn.Module):
    def __init__(self, pcd_size, rgb_size, thermal_size):
        super().__init__()
        self.pcd_size = pcd_size
        self.rgb_size = rgb_size
        self.thermal_size = thermal_size
        self.pcd_unit = Gate2Unit(pcd_size)
        self.img_unit = Gate2Unit(rgb_size)
        if thermal_size != 0:
            self.ther_unit = Gate2Unit(thermal_size)

    def forward(self, pcd_feat, rgb_feat, ther_feat):
        pcd_out = self.pcd_unit(pcd_feat)
        rgb_out = self.img_unit(rgb_feat)
        if self.thermal_size != 0:
            ther_out = self.ther_unit(ther_feat)
            return pcd_out + rgb_out + ther_out
        else:
            return pcd_out + rgb_out

class Gate0(nn.Module):
    def __init__(self, pcd_size, rgb_size, thermal_size):
        super().__init__()
        self.pcd_size = pcd_size
        self.rgb_size = rgb_size
        self.thermal_size = thermal_size
        self.mlp1 = nn.Linear(in_features=pcd_size, out_features=pcd_size)
        self.mlp2 = nn.Linear(in_features=rgb_size, out_features=rgb_size)
        if thermal_size != 0:
            self.mlp3 = nn.Linear(in_features=thermal_size, out_features=thermal_size)
            self.mlp4 = nn.Linear(in_features=pcd_size + rgb_size, out_features=pcd_size + rgb_size)
            self.mlp5 = nn.Linear(in_features=pcd_size + rgb_size+thermal_size, out_features= pcd_size+ rgb_size + thermal_size)

    def forward(self, pcd_feat, rgb_feat, thermal_feat):
        p1 = self.mlp1(pcd_feat)
        p2 = self.mlp2(rgb_feat)
        pr = torch.concat((p1, p2), dim=1)
        pr = torch.sigmoid(pr)
        pr = torch.softmax(pr, dim=1)
        pcd_w = pr[:, :pcd_feat.size(1)]
        rgb_w = pr[:, pcd_feat.size(1):]
        pcd_feat = pcd_feat * pcd_w
        rgb_feat = rgb_feat * rgb_w
        pr_feat = torch.concat((pcd_feat, rgb_feat), dim=1)
        if self.thermal_size != 0:
            p3 = self.mlp4(pr_feat)
            p3 = torch.relu(p3)
            p4 = torch.relu(self.mlp3(thermal_feat))
            prt = torch.concat((pr, thermal_feat), dim=1)
            prt = torch.sigmoid(self.mlp5(prt))
            pr_w = prt[:, :pr.size(1)]
            t_w = prt[:, pr.size(1):]
            p3 = p3 * pr_w
            p4 = p4 * t_w
            feat = torch.concat((p3, p4), dim=1)
            return feat
        else:
            return pr_feat


class GenericOutDet(nn.Module):

    def __init__(self, num_classes=20, kernel_size=5, depth=6, pool=True, dilate=1, in_channels=4, gated=False,
                 attention=False):
        super().__init__()
        self.pool = pool
        self.num_classes = num_classes
        self.depth = depth
        self.dilate = dilate
        self.gated = gated
        self.attention = attention

        self.tree_kernel = int(np.round(kernel_size * kernel_size))
        self.convs = nn.ModuleList()
        if pool:
            self.pools = nn.ModuleList()
        # self.drops = nn.ModuleList()
        self.drop = nn.Dropout(p=0.5)
        # if in_channels > 32:
        #     out_channels = 128
        # else:
        #     out_channels = 32
        out_channels = 256
        if gated:
            self.gate = Gate0(16,16,48)
        if attention:
            self.mha = nn.MultiheadAttention(embed_dim=10, num_heads=2)

        for i in range(depth):
            if i == 0:
                init = True
                suff = 0
            else:
                init = False
                suff = 0
            conv1 = NHConvBlock(kernel_size=self.tree_kernel, in_channels=in_channels + suff,
                                  out_channels=out_channels, init=init, dilate=dilate)
            in_channels = out_channels
            if i < 3:
                out_channels = in_channels * 2
            self.convs.append(conv1)
            if pool and i != depth - 1:
                pool1 = PoolTree()
                self.pools.append(pool1)
            # self.drops.append(nn.Dropout(p=0.5))
        # suff = 3
        self.fc = nn.Linear(in_channels + suff, num_classes, bias=True)
        self.final_bn = nn.BatchNorm1d(out_channels)
        # nn.init.xavier_normal(self.fc.weight)

    def forward(self, points, dist, indices):
        # xyz = points[:, :3].clone()

        if self.gated:
            xyzi = points[:, :4].clone().detach()
            rgb = points[:, 4: 7].clone().detach()
            thermal = points[: , 7: 10]

            lidar_feat = points[:, 10: 26].clone().detach()
            rgb_feat = points[:, 26: 42].clone().detach()
            thermal_feat = points[:, 42:].clone().detach()
            feat = self.gate(lidar_feat, rgb_feat, thermal_feat)
            out = torch.concat((xyzi, rgb, thermal, feat), dim=1)
        else:
            out = points

        if self.attention:
            attn, _ = self.mha(query=points[:, :10], key=points[:, :10], value=points[:, :10], need_weights=False)
            out = torch.concatenate([attn, points[:, 10:]], dim=1)
        # out = self.drop(out)
        for i in range(self.depth):

            out = self.convs[i](out, indices, dist=dist)

            if self.pool and i != self.depth - 1:
                out = self.pools[i](out, indices)
        out = self.drop(out)
        logit = self.fc(out)
        if self.training:
            return logit
        else:
            return logit, out
class OutDetExtractor(nn.Module):
    def __init__(self, feature_size=16, num_classes=20, kernel_size=5, depth=6, pool=True, dilate=1, in_channels=4):
        super().__init__()
        self.outdet = GenericOutDet(num_classes=feature_size, kernel_size=kernel_size, depth=depth, pool=pool,
                                    dilate=dilate, in_channels=in_channels)
        self.bn = nn.BatchNorm1d(num_features=feature_size)
        self.act = nn.LeakyReLU(negative_slope=0.2)
        # self.final_fc = nn.Linear(in_features=feature_size, out_features=num_classes, bias=True)

    def forward(self, points, dist, indices):
        if self.training:
            logit = self.outdet(points, dist, indices)
        else:
            logit, out = self.outdet(points, dist, indices)
        feat = self.bn(logit)
        out = self.act(feat)
        # out = self.final_fc(out)
        return out, feat


if __name__ == '__main__':
    conf_path = '../dataset/binary_desnow_rs.yaml'
    with open(conf_path, 'r') as stream:
        semkittiyaml = yaml.safe_load(stream)
    color_map = semkittiyaml['color_map']
    learning_map = semkittiyaml['learning_map']
    labels = semkittiyaml['labels']
    seq_root = '/var/local/home/aburai/DATA/roadview_dataset_v1/roadview_release'
    pc_list = sorted(glob.glob(os.path.join(seq_root, 'velodyne', "*.bin")))
    lab_list = sorted(glob.glob(os.path.join(seq_root,  'lidarset', "*.bin")))
    device = torch.device('cuda:0')
    # model = KDTree().to(device)

    for pc_file, lab_file in zip(pc_list, lab_list):
        labels = np.fromfile(lab_file, dtype=np.int32).reshape(-1)
        pc = np.fromfile(pc_file, dtype=np.float32).reshape(-1, 4)

        tree = KDTree(pc[:, :3], leaf_size=200)
        dist, ind = tree.query(pc[:, :3], k=9)
        dist = dist.reshape(pc.shape[0], -1)
        dist = dist + 1.0
        labels = labels & 0xFFFF
        labels = np.vectorize(learning_map.__getitem__)(labels)
        label_tensor = torch.from_numpy(labels).long().to(device)
        net = OutDet(num_classes=2, kernel_size=3, depth=1).to(device)
        ext = GlobalExtension().to(device)
        tensor_pt = torch.from_numpy(pc).float().to(device)
        dist = torch.from_numpy(dist).float().to(device)
        ind = torch.from_numpy(ind).to(device)
        logit = net(tensor_pt, dist, ind)
        logit2 = ext(logit)
        print(logit2.shape)
        break
