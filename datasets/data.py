import torch

import PIL.Image as Im
from torch.utils.data import Dataset
import torch.nn.functional as F

import numpy as np
import numpy.linalg as lg

import yaml
import random
import json
from . import utils, copy_paste
import os

from .utils import draw_point_with


def make_point_feat(pcds_xyzi, pcds_coord, pcds_sphere_coord, Voxel):
    # make point feat
    x = pcds_xyzi[:, 0].copy()
    y = pcds_xyzi[:, 1].copy()
    z = pcds_xyzi[:, 2].copy()
    intensity = pcds_xyzi[:, 3].copy()

    dist = np.sqrt(x ** 2 + y ** 2 + z ** 2) + 1e-12

    # grid diff
    diff_x = pcds_coord[:, 0] - np.floor(pcds_coord[:, 0])
    diff_y = pcds_coord[:, 1] - np.floor(pcds_coord[:, 1])
    diff_z = pcds_coord[:, 2] - np.floor(pcds_coord[:, 2])

    # sphere diff
    phi_range_radian = (-np.pi, np.pi)
    theta_range_radian = (Voxel.RV_theta[0] * np.pi / 180.0, Voxel.RV_theta[1] * np.pi / 180.0)

    phi = phi_range_radian[1] - np.arctan2(x, y)
    theta = theta_range_radian[1] - np.arcsin(z / dist)

    diff_phi = pcds_sphere_coord[:, 0] - np.floor(pcds_sphere_coord[:, 0])
    diff_theta = pcds_sphere_coord[:, 1] - np.floor(pcds_sphere_coord[:, 1])

    point_feat = np.stack((x, y, z, intensity, dist, diff_x, diff_y), axis=-1)
    return point_feat


# define the class of dataloader
class DataloadTrain(Dataset):
    def __init__(self, config):
        self.flist = []
        self.config = config
        self.frame_point_num = config.frame_point_num
        self.align = config.align_used
        self.offset_dir = config.OffsetDir
        if self.align:
            self.pose = {}
            self.n_past_pcls = config.n_past_pcls
        self.Voxel = config.Voxel
        with open('datasets/nuscenes.yaml', 'r') as f:
            self.task_cfg = yaml.safe_load(f)
        
        self.cp_aug = None
        if config.CopyPasteAug.is_use:
            self.cp_aug = copy_paste.CutPaste(config.CopyPasteAug.ObjBackDir, config.CopyPasteAug.paste_max_obj_num, road_idx=[11, 13])
        
        self.aug = utils.DataAugment(noise_mean=config.AugParam.noise_mean,
                        noise_std=config.AugParam.noise_std,
                        theta_range=config.AugParam.theta_range,
                        shift_range=config.AugParam.shift_range,
                        size_range=config.AugParam.size_range)

        self.aug_raw = utils.DataAugment(noise_mean=0,
                        noise_std=0,
                        theta_range=(0, 0),
                        shift_range=((0, 0), (0, 0), (0, 0)),
                        size_range=(1, 1))
        
        # add training data
        seq_split = [str(i).rjust(4, '0') for i in self.task_cfg['split']['train']]
        for seq_id in seq_split:
            fpath = os.path.join(config.SeqDir, seq_id)
            fpath_pcds = os.path.join(fpath, 'velodyne')
            fpath_labels = os.path.join(fpath, 'labels')
            file_list_length = len([x for x in os.listdir(fpath_pcds) if x.endswith('.bin')])
            for fn_id in range(file_list_length):
                fname_pcds = os.path.join(fpath_pcds, f"{str(fn_id).rjust(6, '0')}.bin")
                fname_labels = os.path.join(fpath_labels, f"{str(fn_id).rjust(6, '0')}.label")
                self.flist.append((fname_pcds, fname_labels, seq_id, f"{str(fn_id).rjust(6, '0')}.bin"))
                
            # read poses
            if self.align:
                self.pose[seq_id] = read_poses(fpath)
                assert len(self.pose[seq_id]) == file_list_length
        
        assert len(self.flist) == 28130
        print('Training Samples: ', len(self.flist))

    def form_batch(self, pcds_total):
        #augment pcds
        # pcds_total = self.aug(pcds_total)

        #quantize
        pcds_xyzi = pcds_total[:, :4]
        pcds_sem_label = pcds_total[:, 4]
        pcds_ins_label = pcds_total[:, 5]
        pcds_offset = utils.gene_point_offset(pcds_total, center_type=self.config.center_type)
        pcds_coord = utils.Quantize(pcds_xyzi,
                                    range_x=self.Voxel.range_x,
                                    range_y=self.Voxel.range_y,
                                    range_z=self.Voxel.range_z,
                                    size=self.Voxel.bev_shape)

        pcds_sphere_coord = utils.SphereQuantize(pcds_xyzi,
                                            phi_range=(-180.0, 180.0),
                                            theta_range=self.Voxel.RV_theta,
                                            size=self.Voxel.rv_shape)

        #convert numpy matrix to pytorch tensor
        pcds_xyzi = make_point_feat(pcds_xyzi, pcds_coord, pcds_sphere_coord, self.Voxel)
        pcds_xyzi = torch.FloatTensor(pcds_xyzi.astype(np.float32))
        pcds_xyzi = pcds_xyzi.transpose(1, 0).contiguous()

        pcds_coord = torch.FloatTensor(pcds_coord.astype(np.float32))
        pcds_sphere_coord = torch.FloatTensor(pcds_sphere_coord.astype(np.float32))

        pcds_sem_label = torch.LongTensor(pcds_sem_label.astype(np.int64))
        pcds_ins_label = torch.LongTensor(pcds_ins_label.astype(np.int64))
        pcds_offset = torch.FloatTensor(pcds_offset.astype(np.float32))
        return pcds_xyzi.unsqueeze(-1), pcds_coord.unsqueeze(-1), pcds_sphere_coord.unsqueeze(-1), pcds_sem_label.unsqueeze(-1), pcds_ins_label.unsqueeze(-1), pcds_offset

    def form_batch_raw(self, pcds_total):
        #augment pcds
        # pcds_total = self.aug_raw(pcds_total)

        #quantize
        pcds_xyzi = pcds_total[:, :4]
        pcds_sem_label = pcds_total[:, 4]
        pcds_ins_label = pcds_total[:, 5]
        pcds_offset = utils.gene_point_offset(pcds_total, center_type=self.config.center_type)
        pcds_coord = utils.Quantize(pcds_xyzi,
                                    range_x=self.Voxel.range_x,
                                    range_y=self.Voxel.range_y,
                                    range_z=self.Voxel.range_z,
                                    size=self.Voxel.bev_shape)

        pcds_sphere_coord = utils.SphereQuantize(pcds_xyzi,
                                            phi_range=(-180.0, 180.0),
                                            theta_range=self.Voxel.RV_theta,
                                            size=self.Voxel.rv_shape)

        #convert numpy matrix to pytorch tensor
        pcds_xyzi = make_point_feat(pcds_xyzi, pcds_coord, pcds_sphere_coord, self.Voxel)
        pcds_xyzi = torch.FloatTensor(pcds_xyzi.astype(np.float32))
        pcds_xyzi = pcds_xyzi.transpose(1, 0).contiguous()

        pcds_coord = torch.FloatTensor(pcds_coord.astype(np.float32))
        pcds_sphere_coord = torch.FloatTensor(pcds_sphere_coord.astype(np.float32))

        pcds_sem_label = torch.LongTensor(pcds_sem_label.astype(np.int64))
        pcds_ins_label = torch.LongTensor(pcds_ins_label.astype(np.int64))
        pcds_offset = torch.FloatTensor(pcds_offset.astype(np.float32))
        return pcds_xyzi.unsqueeze(-1), pcds_coord.unsqueeze(-1), pcds_sphere_coord.unsqueeze(-1), pcds_sem_label.unsqueeze(-1), pcds_ins_label.unsqueeze(-1), pcds_offset

    def __getitem__(self, index):
        fname_pcds, fname_labels, seq_id, fn = self.flist[index]
        
        # init mapping matrix
        mapping_mat = {}
        
        #load point clouds and label file
        pcds = np.fromfile(fname_pcds, dtype=np.float32)
        pcds = pcds.reshape((-1, 4))
        
        # update mapping matrix
        mapping_mat['n_0'] = pcds.shape[0]
        mapping_mat['n_2'] = 0
        
        # read n_0 label
        pcds_label = np.fromfile(fname_labels, dtype=np.uint32)
        pcds_label = pcds_label.reshape((-1))

        # process labels
        sem_label = pcds_label & 0xFFFF
        inst_label = pcds_label >> 16
        
        # prev pcls and shifted
        if self.align and (int(fn[:-4]) > 0):
            
            # get index of prev scans
            scan_idx = index
            # from_idx = scan_idx - self.n_past_pcls  #TODO Changed here
            from_idx = max(0, index - self.n_past_pcls)

                
            # init lists
            prev_pcds_list = []
            prev_pcds_label_use_list = []
            prev_pcds_ins_label_list = []
            
            # enumerate prev scans
            for i, data_list in enumerate(self.flist[from_idx : scan_idx]):
                
                # read from previous index
                fname_pcds_prev, fname_labels_prev, seq_id_prev, fn_prev = data_list
                
                #  continue if not the same sequence
                if seq_id_prev != seq_id:
                    continue
                
                # read prev pcds
                pcds_prev = np.fromfile(fname_pcds_prev, dtype=np.float32).reshape((-1,4))
                
                # parse shifted prev pcds if fn_prev + 1 = fn
                if (int(fn_prev[:-4]) + 1) == int(fn[:-4]):
                    '''TODO Nuscenes'''
                    # offset_fname = os.path.join(self.offset_dir, seq_id_prev, 'velodyne_offsets', fn_prev)
                    # offset = np.fromfile(offset_fname, dtype=np.float32).reshape((-1,3))
                    offset = np.zeros((pcds_prev.shape[0], 3))
                    
                    # output shifted prev pcds
                    shifted_pcds = pcds_prev[:, :3] + offset[:, :3]
                    
                    # mapping mat
                    mapping_mat['n_1'] = pcds_prev.shape[0]
                    
                    # pose transformation of shifted prev pcds
                    shifted_pcds = transform_point_cloud(shifted_pcds, self.pose[seq_id][int(fn_prev[:-4])], self.pose[seq_id][int(fn[:-4])])
                    assert pcds_prev.shape[0] == shifted_pcds.shape[0]
                
                # # update mapping matrix
                # mapping_mat[f'n_{int(fn[:-4]) - int(fn_prev[:-4])}'] = pcds_prev.shape[0]
                mapping_mat['n_2'] += pcds_prev.shape[0]
                
                # pose transformation of prev pcds 
                pcds_prev[:, :3] = transform_point_cloud(pcds_prev[:, :3], self.pose[seq_id][int(fn_prev[:-4])], self.pose[seq_id][int(fn[:-4])])
                
                # read prev labels
                pcds_label_prev = np.fromfile(fname_labels_prev, dtype=np.uint32).reshape((-1))
                prev_sem_label = pcds_label_prev & 0xFFFF
                prev_inst_label = pcds_label_prev >> 16
                
                # append labels to lists
                prev_pcds_label_use_list.append(prev_sem_label)
                prev_pcds_ins_label_list.append(prev_inst_label)
                prev_pcds_list.append(pcds_prev)
                

        else:
            # if fn is 0 or align is not used
            prev_pcds_list = [pcds]
            prev_pcds_label_use_list = [sem_label]
            prev_pcds_ins_label_list = [inst_label]
            shifted_pcds = pcds
            
            mapping_mat['n_1'] = pcds.shape[0]
            mapping_mat['n_2'] = pcds.shape[0]
            
        
        final_pcds_label_use = utils.relabel(np.concatenate((sem_label, np.concatenate(prev_pcds_label_use_list)), axis=0), self.task_cfg['learning_map'])
        final_pcds_ins_label = utils.gene_ins_label(final_pcds_label_use ,np.concatenate((inst_label, np.concatenate(prev_pcds_ins_label_list)), axis=0))

        
        # when fn > 1
        if mapping_mat['n_2'] > mapping_mat['n_1']:
            prev_pcds_label_use_list = [final_pcds_label_use[mapping_mat['n_0']: -mapping_mat['n_1']], final_pcds_label_use[-mapping_mat['n_1']:]]
            prev_pcds_ins_label_list = [final_pcds_ins_label[mapping_mat['n_0']: -mapping_mat['n_1']], final_pcds_ins_label[-mapping_mat['n_1']:]]
            prev_pcds_list = [np.concatenate(prev_pcds_list)[: -mapping_mat['n_1']], np.concatenate(prev_pcds_list)[-mapping_mat['n_1']:]]
        
        #  when fn = 1 or fn = 0
        elif mapping_mat['n_2'] == mapping_mat['n_1']:
            prev_pcds_label_use_list = [final_pcds_label_use[mapping_mat['n_0']:]]
            prev_pcds_ins_label_list = [final_pcds_ins_label[mapping_mat['n_0']:]]
            assert len(prev_pcds_list) == 1
            assert prev_pcds_list[0].shape[0] == mapping_mat['n_1']
            
        
        # copy-paste augmentation
        if self.cp_aug is not None:
            pcds, pcds_label_use, pcds_ins_label = self.cp_aug(pcds, final_pcds_label_use[:mapping_mat['n_0']], final_pcds_ins_label[:mapping_mat['n_0']])
        else:
            pcds, pcds_label_use, pcds_ins_label = pcds, final_pcds_label_use[:mapping_mat['n_0']], final_pcds_ins_label[:mapping_mat['n_0']]
            
        
        # merge pcds and labels
        pcds_total = np.concatenate((pcds, pcds_label_use[:, np.newaxis], pcds_ins_label[:, np.newaxis]), axis=1)

        # resample
        choice = np.random.choice(pcds_total.shape[0], self.frame_point_num, replace=True)
        pcds_total = pcds_total[choice]
        
        mapping_mat['n_0'] = pcds_total.shape[0]
        
        prev_pcds_total = []
        for i, _ in enumerate(prev_pcds_list):
            prev_pcds_total_n = np.concatenate((prev_pcds_list[i], prev_pcds_label_use_list[i][:, np.newaxis], prev_pcds_ins_label_list[i][:, np.newaxis]), axis=1)
            if prev_pcds_total_n.shape[0] == (mapping_mat['n_2'] - mapping_mat['n_1']):
                choice = np.random.choice(prev_pcds_total_n.shape[0],  self.frame_point_num * (self.n_past_pcls - 1), replace=True)
                prev_pcds_total_n = prev_pcds_total_n[choice]
            else:
                assert shifted_pcds.shape[0] == prev_pcds_total_n.shape[0]
                choice = np.random.choice(prev_pcds_total_n.shape[0], self.frame_point_num, replace=True)
                prev_pcds_total_n = prev_pcds_total_n[choice]
                shifted_pcds = shifted_pcds[choice]
                mapping_mat['n_1'] = prev_pcds_total_n.shape[0]
            prev_pcds_total.append(prev_pcds_total_n)
        
        shifted_pcds = np.concatenate((shifted_pcds, np.zeros((shifted_pcds.shape[0], (pcds_total.shape[-1] - shifted_pcds.shape[-1])))), axis=1)   
        pcds_for_aug = np.concatenate((pcds_total.copy(), np.concatenate(prev_pcds_total.copy()), shifted_pcds), axis=0)
        
        pcds_for_aug = self.aug(pcds_for_aug.copy())
        pcds_for_aug_raw = self.aug_raw(pcds_for_aug.copy())
        shifted_pcds = torch.FloatTensor(pcds_for_aug[-mapping_mat['n_1']:][:, :3].astype(np.float32)).transpose(1, 0).contiguous().unsqueeze(-1)
        shifted_pcds_raw = torch.FloatTensor(pcds_for_aug_raw[-mapping_mat['n_1']:][:, :3].astype(np.float32)).transpose(1, 0).contiguous().unsqueeze(-1)
        
        mapping_mat["n_2"] = pcds_for_aug[mapping_mat['n_0']:-mapping_mat['n_1']].shape[0] - pcds_for_aug[-mapping_mat['n_1']:].shape[0]
        
        # preprocess
        if self.align and (int(fn[:-4]) > 0):
            pcds_xyzi, pcds_coord, pcds_sphere_coord, pcds_sem_label, pcds_ins_label, pcds_offset = self.form_batch(pcds_for_aug[:-mapping_mat['n_1']])
            pcds_xyzi_raw, pcds_coord_raw, pcds_sphere_coord_raw, pcds_sem_label_raw, pcds_ins_label_raw, pcds_offset_raw = self.form_batch_raw(pcds_for_aug_raw[:-mapping_mat['n_1']])
        else:
            pcds_xyzi, pcds_coord, pcds_sphere_coord, pcds_sem_label, pcds_ins_label, pcds_offset = self.form_batch(pcds_for_aug[:mapping_mat['n_0']])
            pcds_xyzi_raw, pcds_coord_raw, pcds_sphere_coord_raw, pcds_sem_label_raw, pcds_ins_label_raw, pcds_offset_raw = self.form_batch_raw(pcds_for_aug_raw[:mapping_mat['n_0']])
            
            # because fn is 0, there is not prev shifted pcds
            shifted_pcds = torch.zeros_like(shifted_pcds)
            shifted_pcds_raw = torch.zeros_like(shifted_pcds_raw)
            mapping_mat['n_1'] = 0
        
        

        return pcds_xyzi, pcds_coord, pcds_sphere_coord, pcds_sem_label, pcds_ins_label, pcds_offset, shifted_pcds,\
            pcds_xyzi_raw, pcds_coord_raw, pcds_sphere_coord_raw, pcds_sem_label_raw, pcds_ins_label_raw, pcds_offset_raw, shifted_pcds_raw, mapping_mat, seq_id, fn

    def __len__(self):
        return len(self.flist)


# define the class of dataloader
class DataloadVal(Dataset):
    def __init__(self, config):
        self.flist = []
        self.config = config
        self.frame_point_num = config.frame_point_num
        self.align = config.align_used
        self.offset_dir = config.OffsetDir
        if self.align:
            self.pose = {}
            self.n_past_pcls = config.n_past_pcls
        self.Voxel = config.Voxel
        with open('datasets/nuscenes.yaml', 'r') as f:
            self.task_cfg = yaml.safe_load(f)
        
        seq_split = [str(i).rjust(4, '0') for i in self.task_cfg['split']['valid']]
        for seq_id in seq_split:
            fpath = os.path.join(config.SeqDir, seq_id)
            fpath_pcds = os.path.join(fpath, 'velodyne')
            fpath_labels = os.path.join(fpath, 'labels')
            file_list_length = len([x for x in os.listdir(fpath_pcds) if x.endswith('.bin')])
            for fn_id in range(file_list_length):
                fname_pcds = os.path.join(fpath_pcds, f"{str(fn_id).rjust(6, '0')}.bin")
                fname_labels = os.path.join(fpath_labels, f"{str(fn_id).rjust(6, '0')}.label")
                self.flist.append((fname_pcds, fname_labels, seq_id, f"{str(fn_id).rjust(6, '0')}.bin"))
            
            # read poses
            if self.align:
                self.pose[seq_id] = read_poses(fpath)
                assert len(self.pose[seq_id]) == file_list_length
        assert len(self.flist) == 6019
        print('Validation Samples: ', len(self.flist))
    
    def form_batch(self, pcds_total):
        #quantize
        pcds_xyzi = pcds_total[:, :4]
        pcds_sem_label = pcds_total[:, 4]
        pcds_ins_label = pcds_total[:, 5]
        pcds_offset = utils.gene_point_offset(pcds_total, center_type=self.config.center_type)
        pcds_coord = utils.Quantize(pcds_xyzi,
                                    range_x=self.Voxel.range_x,
                                    range_y=self.Voxel.range_y,
                                    range_z=self.Voxel.range_z,
                                    size=self.Voxel.bev_shape)

        pcds_sphere_coord = utils.SphereQuantize(pcds_xyzi,
                                            phi_range=(-180.0, 180.0),
                                            theta_range=self.Voxel.RV_theta,
                                            size=self.Voxel.rv_shape)

        #convert numpy matrix to pytorch tensor
        pcds_xyzi = make_point_feat(pcds_xyzi, pcds_coord, pcds_sphere_coord, self.Voxel)
        pcds_xyzi = torch.FloatTensor(pcds_xyzi.astype(np.float32))
        pcds_xyzi = pcds_xyzi.transpose(1, 0).contiguous()

        pcds_coord = torch.FloatTensor(pcds_coord.astype(np.float32))
        pcds_sphere_coord = torch.FloatTensor(pcds_sphere_coord.astype(np.float32))

        pcds_sem_label = torch.LongTensor(pcds_sem_label.astype(np.int64))
        pcds_ins_label = torch.LongTensor(pcds_ins_label.astype(np.int64))
        pcds_offset = torch.FloatTensor(pcds_offset.astype(np.float32))
        return pcds_xyzi.unsqueeze(-1), pcds_coord.unsqueeze(-1), pcds_sphere_coord.unsqueeze(-1), pcds_sem_label.unsqueeze(-1), pcds_ins_label.unsqueeze(-1), pcds_offset
    
    def __getitem__(self, index):
        fname_pcds, fname_labels, seq_id, fn = self.flist[index]
        
        # init mapping matrix
        mapping_mat = {}

        #load point clouds and label file
        pcds = np.fromfile(fname_pcds, dtype=np.float32)
        pcds = pcds.reshape((-1, 4))
        
        # update mapping matrix
        mapping_mat['n_0'] = pcds.shape[0]
        mapping_mat['n_2'] = 0

        pcds_label = np.fromfile(fname_labels, dtype=np.uint32)
        pcds_label = pcds_label.reshape((-1))

        sem_label = pcds_label & 0xFFFF
        inst_label = pcds_label >> 16

        if self.align and (int(fn[:-4]) > 0):
            
            # get index of prev scans
            scan_idx = index
            from_idx = max(0, scan_idx - self.n_past_pcls)
            
            # init lists
            prev_pcds_list = []
            prev_pcds_label_use_list = []
            prev_pcds_ins_label_list = []
            
            # enumerate prev scans
            for i, data_list in enumerate(self.flist[from_idx : scan_idx]):
                fname_pcds_prev, fname_labels_prev, seq_id_prev, fn_prev = data_list
                
                #  continue if not the same sequence
                if seq_id_prev != seq_id:
                    continue
                
                # read prev pcds
                pcds_prev = np.fromfile(fname_pcds_prev, dtype=np.float32).reshape((-1,4))
                
                # parse shifted prev pcds if fn_prev + 1 = fn
                if (int(fn_prev[:-4]) + 1) == int(fn[:-4]):
                    '''TODO Nuscenes'''
                    # offset_fname = os.path.join(self.offset_dir, seq_id_prev, 'velodyne_offsets', fn_prev)
                    # offset = np.fromfile(offset_fname, dtype=np.float32).reshape((-1,3))
                    offset = np.zeros((pcds_prev.shape[0], 3))
                    
                    # output shifted prev pcds
                    shifted_pcds = pcds_prev[:, :3] + offset[:, :3]
                    
                    # mapping_mat
                    mapping_mat['n_1'] = pcds_prev.shape[0]
                    
                    # pose transformation of shifted prev pcds
                    shifted_pcds = transform_point_cloud(shifted_pcds, self.pose[seq_id][int(fn_prev[:-4])], self.pose[seq_id][int(fn[:-4])])
                    assert pcds_prev.shape[0] == shifted_pcds.shape[0]
                    
                # # update mapping matrix
                # mapping_mat[f'n_{int(fn[:-4]) - int(fn_prev[:-4])}'] = pcds_prev.shape[0]
                mapping_mat['n_2'] += pcds_prev.shape[0]
                
                # pose transformation of prev pcds
                pcds_prev[:, :3] = transform_point_cloud(pcds_prev[:, :3], self.pose[seq_id][int(fn_prev[:-4])], self.pose[seq_id][int(fn[:-4])])
                
                # read prev labels
                pcds_label_prev = np.fromfile(fname_labels_prev, dtype=np.uint32).reshape((-1))
                prev_sem_label = pcds_label_prev & 0xFFFF
                prev_inst_label = pcds_label_prev >> 16
                
                # append labels to lists
                prev_pcds_label_use_list.append(prev_sem_label)
                prev_pcds_ins_label_list.append(prev_inst_label)
                prev_pcds_list.append(pcds_prev)
                
        else:
            # if fn is 0 or align is not used
            prev_pcds_list = [pcds]
            prev_pcds_label_use_list = [sem_label]
            prev_pcds_ins_label_list = [inst_label]
            shifted_pcds = pcds
            
            mapping_mat['n_1'] = pcds.shape[0]
            mapping_mat['n_2'] = pcds.shape[0]
        
        final_pcds_label_use = utils.relabel(np.concatenate((sem_label, np.concatenate(prev_pcds_label_use_list)), axis=0), self.task_cfg['learning_map'])
        final_pcds_ins_label = utils.gene_ins_label(final_pcds_label_use ,np.concatenate((inst_label, np.concatenate(prev_pcds_ins_label_list)), axis=0))
        
        # when fn > 1
        if mapping_mat['n_2'] > mapping_mat['n_1']:
            prev_pcds_label_use_list = [final_pcds_label_use[mapping_mat['n_0']: -mapping_mat['n_1']], final_pcds_label_use[-mapping_mat['n_1']:]]
            prev_pcds_ins_label_list = [final_pcds_ins_label[mapping_mat['n_0']: -mapping_mat['n_1']], final_pcds_ins_label[-mapping_mat['n_1']:]]
            prev_pcds_list = [np.concatenate(prev_pcds_list)[: -mapping_mat['n_1']], np.concatenate(prev_pcds_list)[-mapping_mat['n_1']:]]
        
        # when fn = 1 or fn = 0
        elif mapping_mat['n_2'] == mapping_mat['n_1']:
            prev_pcds_label_use_list = [final_pcds_label_use[mapping_mat['n_0']:]]
            prev_pcds_ins_label_list = [final_pcds_ins_label[mapping_mat['n_0']:]]
            assert len(prev_pcds_list) == 1
            assert prev_pcds_list[0].shape[0] == mapping_mat['n_1']
        else:
            raise ValueError('Error in mapping matrix')
                    
                    
        # merge pcds and labels
        pcds_total = np.concatenate((pcds, final_pcds_label_use[:mapping_mat['n_0']][:, np.newaxis], final_pcds_ins_label[:mapping_mat['n_0']][:, np.newaxis]), axis=1)
        pano_label = (inst_label << 16) + final_pcds_label_use[:mapping_mat['n_0']]
        
        # prev pcds total
        prev_pcds_total = []
        for i, _ in enumerate(prev_pcds_list):
            prev_pcds_total_n = np.concatenate((prev_pcds_list[i], prev_pcds_label_use_list[i][:, np.newaxis], prev_pcds_ins_label_list[i][:, np.newaxis]), axis=1)
            if prev_pcds_total_n.shape[0] == (mapping_mat['n_2'] - mapping_mat['n_1']):
                choice = np.random.choice(prev_pcds_total_n.shape[0], self.frame_point_num, replace=True)
                prev_pcds_total_n = prev_pcds_total_n[choice]
            prev_pcds_total.append(prev_pcds_total_n)
        
        shifted_pcds = np.concatenate((shifted_pcds, np.zeros((shifted_pcds.shape[0], (pcds_total.shape[-1] - shifted_pcds.shape[-1])))), axis=1)
        pcds_for_aug = np.concatenate((pcds_total.copy(), np.concatenate(prev_pcds_total.copy()), shifted_pcds), axis=0)
        
        mapping_mat["n_2"] = pcds_for_aug[mapping_mat["n_0"]:-mapping_mat['n_1']].shape[0] - pcds_for_aug[-mapping_mat['n_1']:].shape[0]
        
        # data aug
        pcds_xyzi_list = []
        pcds_coord_list = []
        pcds_sphere_coord_list = []
        pcds_sem_label_list = []
        pcds_ins_label_list = []
        pcds_offset_list = []
        shifted_pcds_list = []
        # for x_sign in [1, -1]:
        #     for y_sign in [1, -1]:
        #         pcds_tmp = pcds_for_aug.copy()
        #         pcds_tmp[:, 0] *= x_sign
        #         pcds_tmp[:, 1] *= y_sign
        pcds_tmp = pcds_for_aug.copy()
        # shifted
        shifted_pcds = torch.FloatTensor(pcds_tmp[-mapping_mat['n_1']:][:, :3].astype(np.float32)).transpose(1, 0).contiguous().unsqueeze(-1)
        
        if self.align and (int(fn[:-4]) > 0):
            pcds_xyzi, pcds_coord, pcds_sphere_coord, pcds_sem_label, pcds_ins_label, pcds_offset = self.form_batch(pcds_tmp[:-mapping_mat['n_1']])
        else:
            pcds_xyzi, pcds_coord, pcds_sphere_coord, pcds_sem_label, pcds_ins_label, pcds_offset = self.form_batch(pcds_tmp[:mapping_mat['n_0']])
            shifted_pcds = torch.zeros_like(shifted_pcds)

        pcds_xyzi_list.append(pcds_xyzi)
        pcds_coord_list.append(pcds_coord)
        pcds_sphere_coord_list.append(pcds_sphere_coord)
        pcds_sem_label_list.append(pcds_sem_label)
        pcds_ins_label_list.append(pcds_ins_label)
        pcds_offset_list.append(pcds_offset)
        shifted_pcds_list.append(shifted_pcds)
                
        if self.align and (int(fn[:-4]) <= 0):
            mapping_mat['n_1'] = 0
        pcds_xyzi = torch.stack(pcds_xyzi_list, dim=0)
        pcds_coord = torch.stack(pcds_coord_list, dim=0)
        pcds_sphere_coord = torch.stack(pcds_sphere_coord_list, dim=0)
        pcds_sem_label = torch.stack(pcds_sem_label_list, dim=0)
        pcds_ins_label = torch.stack(pcds_ins_label_list, dim=0)
        pcds_offset = torch.stack(pcds_offset_list, dim=0)
        shifted_pcds = torch.stack(shifted_pcds_list, dim=0)
        pano_label = torch.LongTensor(pano_label.astype(np.int64))
        return pcds_xyzi, pcds_coord, pcds_sphere_coord, pcds_sem_label, pcds_ins_label, pcds_offset, pano_label, shifted_pcds, mapping_mat, seq_id, fn
    
    def __len__(self):
        return len(self.flist)
       
    
'''Utils'''
def load_poses(pose_path):
    """Load ground truth poses (T_w_cam0) from file.
    Args:
      pose_path: (Complete) filename for the pose file
    Returns:
      A numpy array of size nx4x4 with n poses as 4x4 transformation
      matrices
    """
    # Read and parse the poses
    poses = []
    try:
        if ".txt" in pose_path:
            with open(pose_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    T_w_cam0 = np.fromstring(line, dtype=float, sep=" ")

                    if len(T_w_cam0) == 12:
                        T_w_cam0 = T_w_cam0.reshape(3, 4)
                        T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
                    elif len(T_w_cam0) == 16:
                        T_w_cam0 = T_w_cam0.reshape(4, 4)
                    poses.append(T_w_cam0)
        else:
            poses = np.load(pose_path)["arr_0"]

    except FileNotFoundError:
        print("Ground truth poses are not avaialble.")

    return np.array(poses)

def load_calib(calib_path):
    """Load calibrations (T_cam_velo) from file."""
    # Read and parse the calibrations
    T_cam_velo = []
    try:
        with open(calib_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                if "Tr:" in line:
                    line = line.replace("Tr:", "")
                    T_cam_velo = np.fromstring(line, dtype=float, sep=" ")
                    T_cam_velo = T_cam_velo.reshape(3, 4)
                    T_cam_velo = np.vstack((T_cam_velo, [0, 0, 0, 1]))

    except FileNotFoundError:
        print("Calibrations are not avaialble.")

    return np.array(T_cam_velo)

def read_poses(path_to_seq):
    pose_file = os.path.join(path_to_seq, "poses.txt")
    calib_file = os.path.join(path_to_seq, "calib.txt")
    poses = np.array(load_poses(pose_file))
    inv_frame0 = np.linalg.inv(poses[0])

    # load calibrations
    T_cam_velo = load_calib(calib_file)
    T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
    T_velo_cam = np.linalg.inv(T_cam_velo)

    # convert kitti poses from camera coord to LiDAR coord
    new_poses = []
    for pose in poses:
        new_poses.append(T_velo_cam.dot(inv_frame0).dot(pose).dot(T_cam_velo))
    poses = np.array(new_poses)
    return poses

def transform_point_cloud(past_point_clouds, from_pose, to_pose):
        transformation = np.linalg.inv(to_pose) @ from_pose
        NP = past_point_clouds.shape[0]
        xyz1 = np.hstack([past_point_clouds, np.ones((NP, 1))]).T
        past_point_clouds = (transformation @ xyz1).T[:, :3]
        return past_point_clouds