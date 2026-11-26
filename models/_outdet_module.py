import torch.nn  as nn
import torch
import numpy as np

from models import Gate0


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


class GenericOutDet(nn.Module):

    def __init__(self, num_classes=20, kernel_size=5, depth=6, pool=True, dilate=1, in_channels=4, gated=False,
                 attention=False, width=32):
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
        if gated:
            self.gate = Gate0(16,16,48)
        if attention:
            self.mha = nn.MultiheadAttention(embed_dim=10, num_heads=2)
        self.make_outdet_core(dilate=dilate, out_channels=width, depth=depth, pool=pool, in_channels=in_channels )

    def make_outdet_core(self, dilate=False, out_channels=32, depth=1, pool=True, in_channels=4):
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
            if pool and i != depth - 1:
                pool1 = PoolTree()
                self.pools.append(pool1)
            # self.drops.append(nn.Dropout(p=0.5))
        # suff = 3
        self.fc = nn.Linear(in_channels + suff, self.num_classes, bias=True)
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