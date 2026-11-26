import torch.nn as nn
import torch
import segmentation_models_pytorch as smp

from models import LiRA, OutDetExtractor, GenericOutDet


class MidFusedOutDet_LiRA_UNet(nn.Module):
    def __init__(self, num_classes=20, kernel_size=5, depth=6, pool=True, dilate=1, pcd_in_channels=4, rgb_in_channel=5,
                 gated=False,
                 attention=False, pcd_feat_size=16, img_feat_size=16, ther_feat_size=16, ther_in_channel=3):
        super().__init__()
        self.gated = gated
        self.pcd_feat_size = pcd_feat_size
        self.img_feat_size = img_feat_size
        self.ther_feat_size = ther_feat_size
        self.pcd_extractor = OutDetExtractor(feature_size=pcd_feat_size, num_classes=num_classes,
                                             kernel_size=kernel_size,
                                             depth=depth,
                                             pool=pool, dilate=dilate, in_channels=pcd_in_channels)
        self.rgb_extractor = smp.Unet(in_channels=rgb_in_channel, classes=img_feat_size, activation='identity')

        # self.selector = IndexSelector(img_feat_size)
        if gated:
            self.gate = LiRA(1, pcd_feat_size, img_feat_size, ther_feat_size, drop=False)
            self.pred_in_ch = 3 + pcd_feat_size + img_feat_size + ther_feat_size
            # self.pred_in_ch = 3 + pcd_feat_size
        else:
            self.pred_in_ch = 3 + pcd_feat_size + img_feat_size
            # self.pred_in_ch = 3 + pcd_feat_size
        self.predictor = GenericOutDet(num_classes=num_classes, kernel_size=kernel_size, depth=depth, dilate=dilate,
                                       in_channels=self.pred_in_ch)

    def forward(self, batch):
        device = next(self.pcd_extractor.parameters()).device
        data = batch['pcd'][0]
        ind = batch['pcd_ind'][0]
        dist = batch['pcd_dist'][0]
        selected_ind = batch['selected_ind'][0]
        inlier = batch['inlier'][0]
        rgbt = batch['rgbt']
        rgb_ind = batch['rgb_coord'][0]

        _, pcd_feat = self.pcd_extractor(data, dist, ind)


        rgb_feat = self.rgb_extractor(rgbt)[0]

        pcd_feat = pcd_feat[inlier]
        rgb_feat = rgb_feat[:, rgb_ind[selected_ind, 0], rgb_ind[selected_ind, 1]]
        rgb_feat = torch.transpose(rgb_feat, 0, 1)
        if self.gated:
            mid_fused = self.gate(pcd_feat, rgb_feat, None)
        else:

            mid_fused = torch.concatenate((pcd_feat, rgb_feat), dim=1)
        # tg = time.time_ns()
        pred_inp = torch.zeros(size=(data.shape[0], self.pred_in_ch), device=device)
        pred_inp[:, :3] = data[:, :3]
        pred_inp[inlier, 3:] = mid_fused
        out = self.predictor(pred_inp, dist, ind)

        return out, inlier


class MidFusedOutDet_PCD_Only(nn.Module):
    def __init__(self, num_classes=20, kernel_size=5, depth=6, pool=True, dilate=1, pcd_in_channels=4, rgb_in_channel=5, gated=False,
                 attention=False, width=256):
        super().__init__()
        self.predictor = GenericOutDet(num_classes=num_classes, kernel_size=kernel_size, depth=depth, dilate=dilate, in_channels=pcd_in_channels, width=width)

    def forward(self, batch):
        device = next(self.predictor.parameters()).device
        data = batch['pcd'][0].to(device)
        ind = batch['pcd_ind'][0]
        dist = batch['pcd_dist'][0].to(device)
        if 'inlier' in batch.keys():
            inlier = batch['inlier'][0].to(device)
        else:
            inlier = None

        out = self.predictor(data, dist, ind)
        return out, inlier