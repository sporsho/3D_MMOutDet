import torch.nn as nn
import torch


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