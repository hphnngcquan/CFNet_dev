import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .networks import backbone
from .networks.fusion_module import SpatialAttention_mtf
from . import common_utils
from datasets.utils import save_feature_map
from utils.config_parser import get_module
import time
import pdb

class CFNet_Shifted(nn.Module):
    def __init__(self, pModel):
        super(CFNet_Shifted, self).__init__()
        self.total_time = 0
        self.pModel = pModel

        self.bev_shape = list(self.pModel.Voxel.bev_shape)
        self.rv_shape = list(self.pModel.Voxel.rv_shape)
        self.bev_wl_shape = self.bev_shape[:2]

        self.dx = (self.pModel.Voxel.range_x[1] - self.pModel.Voxel.range_x[0]) / self.pModel.Voxel.bev_shape[0]
        self.dy = (self.pModel.Voxel.range_y[1] - self.pModel.Voxel.range_y[0]) / self.pModel.Voxel.bev_shape[1]

        self.phi_range_radian = (-np.pi, np.pi)
        self.theta_range_radian = (self.pModel.Voxel.RV_theta[0] * np.pi / 180.0, self.pModel.Voxel.RV_theta[1] * np.pi / 180.0)

        self.dphi = (self.phi_range_radian[1] - self.phi_range_radian[0]) / self.pModel.Voxel.rv_shape[1]
        self.dtheta = (self.theta_range_radian[1] - self.theta_range_radian[0]) / self.pModel.Voxel.rv_shape[0]

        self.point_feat_out_channels = self.pModel.point_feat_out_channels
        self.build_network()
    
    def build_network(self):
        # build network
        bev_net_cfg = self.pModel.BEVParam
        rv_net_cfg = self.pModel.RVParam
        bev_base_channels = bev_net_cfg.base_channels

        fusion_cfg = self.pModel.FusionParam
        
        attn_map = self.pModel.attn_map
        T = self.pModel.n_past_pcls + 1
        
        # base network
        self.point_pre = nn.Sequential(
            backbone.bn_conv1x1_bn_relu(7, bev_base_channels[0]),
            backbone.conv1x1_bn_relu(bev_base_channels[0], bev_base_channels[0])
        )
        if attn_map:
            attn_cfg = self.pModel.AttnParam
            self.point2bev_attn = get_module(attn_cfg.BEVParam.P2VParam)
            self.point2rv_attn = get_module(attn_cfg.RVParam.P2VParam)
            self.attn_bev = SpatialAttention_mtf()
            self.attn_rv = SpatialAttention_mtf()
            # self.fuse_bev_conv = nn.Sequential(
            #     backbone.bn_conv3x3_bn_relu(bev_base_channels[2] * 2, bev_base_channels[2]),               
            # )
            # self.fuse_rv_conv = nn.Sequential(
            #     backbone.bn_conv3x3_bn_relu(bev_base_channels[2] * 2, bev_base_channels[2]),               
            # )
            
        # BEV network
        self.point2bev = get_module(bev_net_cfg.P2VParam)
        self.bev_net = get_module(bev_net_cfg)
        self.bev2point = get_module(bev_net_cfg.V2PParam)

        # RV network
        self.point2rv = get_module(rv_net_cfg.P2VParam)
        self.rv_net = get_module(rv_net_cfg)
        self.rv2point = get_module(rv_net_cfg.V2PParam)

        # stage0
        # sem branch
        point_fusion_channels = (bev_base_channels[0], self.bev_net.out_channels, self.rv_net.out_channels)
        self.point_fusion_sem = get_module(fusion_cfg, in_channel_list=point_fusion_channels, out_channel=self.point_feat_out_channels)
        if self.pModel.auxiliary:
            self.pred_layer_sem = backbone.PredBranch(self.point_feat_out_channels, self.pModel.class_num)

            # ins branch
            self.point_fusion_ins = get_module(fusion_cfg, in_channel_list=point_fusion_channels, out_channel=self.point_feat_out_channels)
            self.pred_layer_offset = backbone.PredBranch(self.point_feat_out_channels, 3)
            self.pred_layer_hmap = nn.Sequential(
                backbone.PredBranch(self.point_feat_out_channels, 1),
                nn.Sigmoid()
            )

        # CFFE
        if hasattr(self.pModel, "CFFEParam") and self.pModel.cffe_used:
            cffe_cfg = self.pModel.CFFEParam
            # BEV network
            self.point2bev_cffe = get_module(cffe_cfg.BEVParam.P2VParam)
            self.bev_cffe = get_module(cffe_cfg.BEVParam, in_channel_list=(self.bev_net.out_channels, self.point_feat_out_channels), out_channel=self.bev_net.out_channels)
            self.bev2point_cffe = get_module(cffe_cfg.BEVParam.V2PParam)

            # RV network
            self.point2rv_cffe = get_module(cffe_cfg.RVParam.P2VParam)
            self.rv_cffe = get_module(cffe_cfg.RVParam, in_channel_list=(self.rv_net.out_channels, self.point_feat_out_channels), out_channel=self.rv_net.out_channels)
            self.rv2point_cffe = get_module(cffe_cfg.RVParam.V2PParam)

            # sem branch
            point_fusion_channels = (bev_base_channels[0], self.bev_net.out_channels, self.rv_net.out_channels)
            self.point_fusion_sem_cffe = get_module(fusion_cfg, in_channel_list=point_fusion_channels, out_channel=self.point_feat_out_channels)
            self.pred_layer_sem_cffe = backbone.PredBranch(self.point_feat_out_channels, self.pModel.class_num)

            # ins branch
            self.point_fusion_ins_cffe = get_module(fusion_cfg, in_channel_list=point_fusion_channels, out_channel=self.point_feat_out_channels)
            self.pred_layer_offset_cffe = backbone.PredBranch(self.point_feat_out_channels, 3)
            self.pred_layer_hmap_cffe = nn.Sequential(
                backbone.PredBranch(self.point_feat_out_channels, 1),
                nn.Sigmoid()
            )
    
    def forward(self, pcds_xyzi, pcds_coord, pcds_sphere_coord,
                shifted_pcds=None, mapping_mat=None):
        '''
        Input:
            pcds_xyzi (BS, 7, N, 1), 7 -> (x, y, z, intensity, dist, diff_x, diff_y)
            pcds_coord (BS, N, 3, 1), 3 -> (x_quan, y_quan, z_quan)
            pcds_sphere_coord (BS, N, 2, 1), 2 -> (vertical_quan, horizon_quan)
        Output:
            pred_sem (BS, C, N, 1)
            pred_offset (BS, N, 3)
            pred_hmap (BS, N, 1)
        '''
        # start = time.perf_counter()
        pcds_coord_wl = pcds_coord[:, :, :2].contiguous()
        pcds_coord_wl_0 = pcds_coord[:, :mapping_mat['n_0'][0], :2].contiguous()
        

        point_feat_tmp = self.point_pre(pcds_xyzi)

        if hasattr(self, 'attn_bev'):
            assert pcds_coord_wl_0.shape[1] == pcds_sphere_coord[:, :mapping_mat['n_0'][0], :, :].shape[1] == mapping_mat['n_0'][0]
            # mapping point_feat_tmp
            point_feat_temp_0 = point_feat_tmp[:,:,:mapping_mat['n_0'][0],:].clone()
            bev_input_0 = self.point2bev_attn(point_feat_temp_0, pcds_coord_wl_0) 
            rv_input_0 = self.point2rv_attn(point_feat_temp_0, pcds_sphere_coord[:, :mapping_mat['n_0'][0], :, :].contiguous())
        
        # BEV network
        bev_input = self.point2bev(point_feat_tmp, pcds_coord_wl)
        bev_input = torch.cat([bev_input, bev_input_0], dim=0)
        bev_feat_sem, bev_feat_ins = self.bev_net(bev_input)
        
        if hasattr(self, 'attn_bev'):
            prev, curr = torch.split(bev_feat_sem, bev_feat_sem.shape[0] // 2)
            bev_feat_sem = self.attn_bev(curr, prev)
            # bev_feat_sem = torch.cat([curr, bev_fused], dim=1)
            # bev_feat_sem = self.fuse_bev_conv(bev_feat_sem)

        point_bev_sem = self.bev2point(bev_feat_sem, pcds_coord_wl)
        

        # RV network
        rv_input = self.point2rv(point_feat_tmp, pcds_sphere_coord)
        rv_input = torch.cat([rv_input, rv_input_0], dim=0)
        rv_feat_sem, rv_feat_ins = self.rv_net(rv_input)

        if hasattr(self, 'attn_bev'):
            prev, curr = torch.split(rv_feat_sem, rv_feat_sem.shape[0] // 2)
            rv_feat_sem = self.attn_rv(curr, prev)
            # rv_feat_sem = torch.cat([prev, rv_fused], dim=1)
            # rv_feat_sem = self.fuse_rv_conv(rv_feat_sem)
        
        point_rv_sem = self.rv2point(rv_feat_sem, pcds_sphere_coord)
        

        # stage0
        point_feat_sem = self.point_fusion_sem(point_feat_tmp, point_bev_sem, point_rv_sem) #TODO try point_feat_tmp_0 if point_feat_tmp is not good enough
        

        if self.pModel.auxiliary:
            prev_bev_ins, curr_bev_ins = torch.split(bev_feat_ins, bev_feat_ins.shape[0] // 2)
            prev_rv_ins, curr_rv_ins = torch.split(rv_feat_ins, rv_feat_ins.shape[0] // 2)
            
            point_bev_ins = self.bev2point(prev_bev_ins, pcds_coord_wl)
            point_rv_ins = self.rv2point(prev_rv_ins, pcds_sphere_coord)
            pred_sem = self.pred_layer_sem(point_feat_sem).float()

            # ins branch
            point_feat_ins = self.point_fusion_ins(point_feat_tmp, point_bev_ins, point_rv_ins) #TODO try point_feat_tmp_0 if point_feat_tmp is not good enough
            pred_offset = self.pred_layer_offset(point_feat_ins).float().squeeze(-1).transpose(1, 2).contiguous()
            pred_hmap = self.pred_layer_hmap(point_feat_ins).float().squeeze(1)
            preds_list = [(pred_sem, pred_offset, pred_hmap)]

        if hasattr(self.pModel, "CFFEParam") and self.pModel.cffe_used:
            # mapping
            point_feat_sem_1 = point_feat_sem[:, :, -mapping_mat['n_1'][0]:,].clone() 

            
            # reprojection
            pcds_coord_wl_reproj_1, pcds_sphere_coord_reproj_1 = common_utils.reproj_with_shifted(pcds_xyzi, shifted_pcds,\
                self.pModel.Voxel, self.dx, self.dy, self.phi_range_radian, self.theta_range_radian, self.dphi, self.dtheta)
            
            # BEV network
            bev_cfg_feat_1 = self.point2bev_cffe(point_feat_sem_1, pcds_coord_wl_reproj_1)
            bev_feat_sem_final, bev_feat_ins_final = self.bev_cffe(bev_feat_sem, bev_cfg_feat_1)

            point_bev_sem_cffe = self.bev2point_cffe(bev_feat_sem_final, pcds_coord_wl)
            point_bev_ins_cffe = self.bev2point_cffe(bev_feat_ins_final, pcds_coord_wl)

            # RV network
            rv_cfg_feat_1 = self.point2rv_cffe(point_feat_sem_1, pcds_sphere_coord_reproj_1)
            rv_feat_sem_final, rv_feat_ins_final = self.rv_cffe(rv_feat_sem, rv_cfg_feat_1)

            point_rv_sem_cffe = self.rv2point_cffe(rv_feat_sem_final, pcds_sphere_coord)
            point_rv_ins_cffe = self.rv2point_cffe(rv_feat_ins_final, pcds_sphere_coord)

            # sem branch
            point_feat_sem_cffe = self.point_fusion_sem_cffe(point_feat_tmp, point_bev_sem_cffe, point_rv_sem_cffe)
            pred_sem_cffe = self.pred_layer_sem_cffe(point_feat_sem_cffe).float()

            # ins branch
            point_feat_ins_cffe = self.point_fusion_ins_cffe(point_feat_tmp, point_bev_ins_cffe, point_rv_ins_cffe)
            pred_offset_cffe = self.pred_layer_offset_cffe(point_feat_ins_cffe).float().squeeze(-1).transpose(1, 2).contiguous()
            pred_hmap_cffe = self.pred_layer_hmap_cffe(point_feat_ins_cffe).float().squeeze(1)

            if self.pModel.auxiliary:
                preds_list.append((pred_sem_cffe, pred_offset_cffe, pred_hmap_cffe))
            else:
                preds_list = [(pred_sem_cffe, pred_offset_cffe, pred_hmap_cffe)]
        # end = time.perf_counter()
        # self.total_time += end - start
        # print('total time:', self.total_time)
        return preds_list