import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pickle as pkl

from utils.config_parser import get_module, class2dic, class2dic_iterative
from .model_utils import builder, metric
from .model_utils.gen_shiftedkitti import gen_offsetkitti
from light_trainer import trainer
import pytorch_lib
from datasets import utils

import collections
import yaml
import os

from datasets.utils import draw_point_with

offset_loss_raw = []
offset_loss = []

if hasattr(torch.optim.lr_scheduler, "LRScheduler"):
    setattr(torch.optim.lr_scheduler, "_LRScheduler", torch.optim.lr_scheduler.LRScheduler)


def merge_offset_tta(pred_offset):
    '''
    Input:
        pred_offset, (4, N, 3)
    Output:
        pred_offset_result, (N, 3)
    '''
    assert pred_offset.ndim == 3
    assert (pred_offset.shape[0] == 4) or (pred_offset.shape[0] == 1)
    assert pred_offset.shape[2] == 3
    if pred_offset.shape[0] == 4:
        p = 0
        for x_sign in [1, -1]:
            for y_sign in [1, -1]:
                pred_offset[p, :, 0] *= x_sign
                pred_offset[p, :, 1] *= y_sign
                p += 1
    pred_offset_result = pred_offset.mean(dim=0)
    return pred_offset_result


class ModelRunnerSemKITTI(trainer.ADDistTrainer):
    def __init__(self,
        recorder = None,
        max_epochs = 100,
        precision = 32,
        log_every_n_steps = 1,
        check_val_every_n_epoch = 1,
        clip_grad_norm = None,
        clip_grad_norm_type = 2,
        detect_anomaly = False,
        sync_batchnorm = False,
        pModel = None,
        per_epoch_num_iters = None,
        **kwargs):
        kwargs.update(locals())
        kwargs.pop('kwargs')
        kwargs.pop('self')
        if '__class__' in kwargs:
            kwargs.pop('__class__')
        super().__init__(**kwargs)
        with open('datasets/nuscenes.yaml', 'r') as f:
            self.task_cfg = yaml.safe_load(f)
        # save model config
        save_dict = class2dic_iterative(self.pModel)
        self.save_hyperparameters(save_dict, "hparams.yaml")
        self.log(enable_sync_dist=False, ignore_log_step=True, **kwargs)
        self.build_loss()
        self.build_postprocess()
        self.pred_data_list = []
    
    def configure_optimizers(self):
        self.optimizer = builder.get_optimizer(self.pModel.optimizer, self.model)
        self.scheduler = builder.get_scheduler(self.optimizer, self.pModel.scheduler, self.per_epoch_num_iters)
    
    def build_loss(self):
        self.panoptic_loss = get_module(self.pModel.LossParam)
    
    def build_postprocess(self):
        self.pv_nms = pytorch_lib.PointVoteNMS(self.pModel.point_nms_dic)
    
    def consistency_loss_l1(self, pred_cls, pred_cls_raw):
        '''
        Input:
            pred_cls, pred_cls_raw (BS, C, N, 1)
        '''
        pred_cls_softmax = F.softmax(pred_cls, dim=1)
        pred_cls_raw_softmax = F.softmax(pred_cls_raw, dim=1)

        loss = (pred_cls_softmax - pred_cls_raw_softmax).abs().sum(dim=1).mean()
        return loss
    
    def training_step(self, batch_idx, batch):
        '''
        Input:
            pcds_xyzi, pcds_xyzi_raw (BS, 7, N, 1), 7 -> (x, y, z, intensity, dist, diff_x, diff_y)
            pcds_coord, pcds_coord_raw (BS, N, 3, 1), 3 -> (x_quan, y_quan, z_quan)
            pcds_sphere_coord, pcds_sphere_coord_raw (BS, N, 2, 1), 2 -> (vertical_quan, horizon_quan)
            pcds_sem_label, pcds_sem_label_raw, pcds_ins_label, pcds_ins_label_raw (BS, N, 1)
            pcds_offset, pcds_offset_raw (BS, N, 3)
        '''
        pcds_xyzi, pcds_coord, pcds_sphere_coord, pcds_sem_label, pcds_ins_label, pcds_offset, shifted_pcds,\
        pcds_xyzi_raw, pcds_coord_raw, pcds_sphere_coord_raw, pcds_sem_label_raw, pcds_ins_label_raw, pcds_offset_raw, shifted_pcds_raw, mapping_mat, seq_id, fn = batch
        
        batch_size = pcds_xyzi.shape[0]
        # forward
        pcds_xyzi_total = torch.cat((pcds_xyzi, pcds_xyzi_raw), dim=0)
        pcds_coord_total = torch.cat((pcds_coord, pcds_coord_raw), dim=0)
        pcds_sphere_coord_total = torch.cat((pcds_sphere_coord, pcds_sphere_coord_raw), dim=0)
        shifted_pcds = torch.cat((shifted_pcds, shifted_pcds_raw), dim=0)
        pred_list_total = self.model(pcds_xyzi_total, pcds_coord_total, pcds_sphere_coord_total, 
                                     shifted_pcds=shifted_pcds, mapping_mat=mapping_mat)

        pred_list = []
        pred_list_raw = []
        for i in range(len(pred_list_total)):
            pred_list.append([x[:batch_size].contiguous() for x in pred_list_total[i]])
            pred_list_raw.append([x[batch_size:].contiguous() for x in pred_list_total[i]])

        # total loss
        loss_dic = collections.OrderedDict()
        loss = 0
        for i in range(len(pred_list)):
            # if pred_list[i][0].shape[2] == mapping_mat['n_0'][0]:
            #     pcds_offset = pcds_offset[:, :mapping_mat['n_0'][0]]
            #     pcds_offset_raw = pcds_offset_raw[:, :mapping_mat['n_0'][0]]
            #     pcds_ins_label = pcds_ins_label[:, :mapping_mat['n_0'][0]]
            #     pcds_sem_label = pcds_sem_label[:, :mapping_mat['n_0'][0]]
            if len(pred_list[i]) <= 1:
                loss_pano = self.panoptic_loss(pred_list[i][0], None, None, pcds_offset, pcds_ins_label, pcds_sem_label)
                loss_pano_raw = self.panoptic_loss(pred_list_raw[i][0], None, None, pcds_offset_raw, pcds_ins_label, pcds_sem_label)
            else:
                  
                loss_pano = self.panoptic_loss(pred_list[i][0], pred_list[i][1], pred_list[i][2], pcds_offset, pcds_ins_label, pcds_sem_label)
                loss_pano_raw = self.panoptic_loss(pred_list_raw[i][0], pred_list_raw[i][1], pred_list_raw[i][2], pcds_offset_raw, pcds_ins_label, pcds_sem_label)

            loss_consist = self.consistency_loss_l1(pred_list[i][0], pred_list_raw[i][0])

            loss += 0.5 * (loss_pano + loss_pano_raw) + loss_consist

            loss_dic["loss_pano{}".format(i)] = loss_pano
            loss_dic["loss_pano_raw{}".format(i)] = loss_pano_raw
            loss_dic["loss_consist{}".format(i)] = loss_consist
        
        loss_dic["learning_rate"] = self.optimizer.state_dict()['param_groups'][0]['lr']
        self.log(enable_sync_dist = True, ignore_log_step = False, **loss_dic)
        return loss
    
    @torch.no_grad()
    def validation_step(self, batch_idx, batch):
        pcds_xyzi, pcds_coord, pcds_sphere_coord, pcds_sem_label, pcds_ins_label, pcds_offset, pano_label, shifted_pcds, mapping_mat, seq_id, fn = batch

        pano_label = pano_label.cpu().numpy().astype(np.uint32)[0]
        pred_list = self.model(pcds_xyzi.squeeze(0), pcds_coord.squeeze(0), pcds_sphere_coord.squeeze(0),
                               shifted_pcds=shifted_pcds.squeeze(0), mapping_mat=mapping_mat)

        pred_panoptic_list = []
        for i, (pred_sem, pred_offset, pred_hmap) in enumerate(pred_list):
            pred_sem = pred_sem[:, :, :mapping_mat['n_0'][0]]
            pred_offset = pred_offset[:, :mapping_mat['n_0'][0]]
            pred_hmap = pred_hmap[:, :mapping_mat['n_0'][0]]
            # calculate offset loss
            loss_point = (pred_offset - pcds_offset[0][:, :mapping_mat['n_0'][0]]).pow(2).sum(dim=2, keepdim=True).sqrt() #(BS, N, 1)
            loss_offset = (loss_point).mean()
            if i == 0:
                offset_loss.append(loss_offset)
            if i == 1:
                offset_loss_raw.append(loss_offset)
            pred_sem = F.softmax(pred_sem, dim=1).mean(dim=0).permute(2, 1, 0).contiguous()[0]
            pred_offset = merge_offset_tta(pred_offset)
            pred_hmap = pred_hmap.mean(dim=0).squeeze(1)

            # make result
            pred_obj_center, pred_panoptic = self.pv_nms(pcds_xyzi[0, 0, :3, :mapping_mat['n_0'][0], 0].T.contiguous(), pred_sem, pred_offset, pred_hmap)
            pred_panoptic = pred_panoptic.cpu().numpy().astype(np.uint32)

            pred_panoptic_list.append(pred_panoptic)
        
        pred_panoptic_list = np.stack(pred_panoptic_list, axis=0)
        self.pred_data_list.append((os.path.basename(fn[0]).replace('.bin', '.npz'), pano_label, pred_panoptic_list))
    
    def validation_epoch_end(self):
        fpath_record = os.path.join(self.recorder.log_dir, "val_record")
        
        if not os.path.exists(fpath_record):
            os.system("mkdir -p {}".format(fpath_record))
        
        with open(os.path.join(fpath_record, f"rank_{self.global_rank}.pkl"), "wb") as f:
            pkl.dump(self.pred_data_list, f)
        self.pred_data_list.clear()
        self.sync_all_process()
        
        monitor_metric = [None]
        # compute metric
        if self.global_rank == 0:
            criterion_pano_list = []
            fname_set = set()
            for rank in range(self.world_size):
                fname_pkl = os.path.join(fpath_record, f"rank_{rank}.pkl")
                with open(fname_pkl, "rb") as f:
                    data_list = pkl.load(f)
                
                for (key, pano_label, pred_panoptic) in data_list:
                    if key not in fname_set:
                        fname_set.add(key)
                        if len(criterion_pano_list) == 0:
                            criterion_pano_list = [metric.PanopticEval(self.pModel.category_list, None, [0], min_points=50) for i in range(pred_panoptic.shape[0])]
                        
                        for i in range(pred_panoptic.shape[0]):
                            criterion_pano_list[i].addBatch(pred_panoptic[i] & 0xFFFF, pred_panoptic[i], pano_label & 0xFFFF, pano_label)
            
            # record metric
            record_dic = collections.OrderedDict()
            record_dic['validation_sample_number'] = len(fname_set)
            for i in range(len(criterion_pano_list)):
                metric_pano = criterion_pano_list[i].get_metric()
                for key in metric_pano:
                    record_dic["{}_{}".format(key, i)] = metric_pano[key]
            # Save result metrics
            record_dic["offset_loss"] = (sum(offset_loss) / len(offset_loss)).item()
            if offset_loss_raw != []:
                record_dic["offset_loss_raw"] = (sum(offset_loss_raw) / len(offset_loss_raw)).item()
            self.log(enable_sync_dist=False, ignore_log_step=True, **record_dic)
            monitor_metric[0] = float(metric_pano["pq_mean"])
        
        torch.distributed.broadcast_object_list(monitor_metric, src=0)
        return monitor_metric[0]
    
    @torch.no_grad()
    def test_step(self, batch_idx, batch):
        pcds_xyzi, pcds_coord, pcds_sphere_coord, pcds_sem_label, pcds_ins_label, pcds_offset, pano_label, shifted_pcds, mapping_mat, seq_id, fn = batch

        pano_label = pano_label.cpu().numpy().astype(np.uint32)[0]
        pred_list = self.model(pcds_xyzi.squeeze(0), pcds_coord.squeeze(0), pcds_sphere_coord.squeeze(0),
                               shifted_pcds=shifted_pcds.squeeze(0), mapping_mat=mapping_mat)
        
        pc_offset_weight = get_fg_pc_offset_weight(pcds_ins_label[0])

        pred_panoptic_list = []
        
        for i, (pred_sem, pred_offset, pred_hmap) in enumerate(pred_list):
            pred_sem = pred_sem[:, :, :mapping_mat['n_0'][0]]
            pred_offset = pred_offset[:, :mapping_mat['n_0'][0]]
            pred_hmap = pred_hmap[:, :mapping_mat['n_0'][0]]
            # calculate offset loss
            loss_point = (pred_offset - pcds_offset[0][:, :mapping_mat['n_0'][0]]).pow(2).sum(dim=2, keepdim=True).sqrt() #(BS, N, 1)
            loss_offset = (loss_point).mean()
            if i == 0:
                offset_loss.append(loss_offset)
            if i == 1:
                offset_loss_raw.append(loss_offset)
            
            pred_sem = F.softmax(pred_sem, dim=1).mean(dim=0).permute(2, 1, 0).contiguous()[0]
            pred_offset = merge_offset_tta(pred_offset)
            pred_hmap = pred_hmap.mean(dim=0).squeeze(1)

            # make result
            pred_obj_center, pred_panoptic = self.pv_nms(pcds_xyzi[0, 0, :3, :mapping_mat['n_0'][0], 0].T.contiguous(), pred_sem, pred_offset, pred_hmap)
            pred_panoptic = pred_panoptic.cpu().numpy().astype(np.uint32)
            
            pred_panoptic_list.append(pred_panoptic)
            
        
        pred_panoptic_list = np.stack(pred_panoptic_list, axis=0)
        self.pred_data_list.append((os.path.basename(fn[0]).replace('.bin', '.npz'), pano_label, pred_panoptic_list))
    
    def test_epoch_end(self):
        fpath_record = os.path.join(self.test_save_path, "val_record")
        if not os.path.exists(fpath_record):
            os.system("mkdir -p {}".format(fpath_record))
        
        with open(os.path.join(fpath_record, f"rank_{self.global_rank}.pkl"), "wb") as f:
            pkl.dump(self.pred_data_list, f)
        self.pred_data_list.clear()
        self.sync_all_process()
        
        # compute metric
        if self.global_rank == 0:
            criterion_pano_list = []
            fname_set = set()
            for rank in range(self.world_size):
                fname_pkl = os.path.join(fpath_record, f"rank_{rank}.pkl")
                with open(fname_pkl, "rb") as f:
                    data_list = pkl.load(f)
                
                for (key, pano_label, pred_panoptic) in data_list:
                    if key not in fname_set:
                        fname_set.add(key)
                        if len(criterion_pano_list) == 0:
                            criterion_pano_list = [metric.PanopticEval(self.pModel.category_list, None, [0], min_points=50) for i in range(pred_panoptic.shape[0])]
                        
                        for i in range(pred_panoptic.shape[0]):
                            criterion_pano_list[i].addBatch(pred_panoptic[i] & 0xFFFF, pred_panoptic[i], pano_label & 0xFFFF, pano_label)
            
            # record metric
            record_dic = collections.OrderedDict()
            record_dic['validation_sample_number'] = len(fname_set)
            for i in range(len(criterion_pano_list)):
                metric_pano = criterion_pano_list[i].get_metric()
                for key in metric_pano:
                    record_dic["{}_{}".format(key, i)] = metric_pano[key]
            record_dic["offset_loss"] = (sum(offset_loss) / len(offset_loss)).item()
            # record_dic["offset_loss_raw"] = (sum(offset_loss_raw) / len(offset_loss_raw)).item()            
            with open(os.path.join(self.test_save_path, "metric.yaml"), "w") as f:
                yaml.dump(record_dic, f)
                
    @torch.no_grad()
    def predict_step(self, batch_idx, batch):
        generate_offset = False
        test = False
        if test:
            pcds_xyzi, pcds_coord, pcds_sphere_coord, shifted_pcds, mapping_mat, seq_id, fn = batch
        else:
            pcds_xyzi, pcds_coord, pcds_sphere_coord, pcds_sem_label, pcds_ins_label, pcds_offset, pano_label, shifted_pcds, mapping_mat, seq_id, fn = batch

        pred_list = self.model(pcds_xyzi.squeeze(0), pcds_coord.squeeze(0), pcds_sphere_coord.squeeze(0),
                               shifted_pcds=shifted_pcds.squeeze(0), mapping_mat=mapping_mat)

        pred_panoptic_list = []
        for i, (pred_sem, pred_offset, pred_hmap) in enumerate(pred_list):
            pred_sem = pred_sem[:, :, :mapping_mat['n_0'][0]]
            pred_offset = pred_offset[:, :mapping_mat['n_0'][0]]
            pred_hmap = pred_hmap[:, :mapping_mat['n_0'][0]]
            
            pred_sem = F.softmax(pred_sem, dim=1).mean(dim=0).permute(2, 1, 0).contiguous()[0]
            pred_offset = merge_offset_tta(pred_offset)
            pred_hmap = pred_hmap.mean(dim=0).squeeze(1)

            # make result
            pred_obj_center, pred_panoptic = self.pv_nms(pcds_xyzi[0, 0, :3, :mapping_mat['n_0'][0], 0].T.contiguous(), pred_sem, pred_offset, pred_hmap)
            if i == 0 and  generate_offset:
                assert pred_offset.shape[0] == pcds_xyzi.shape[3]
                pred_offset_save = pred_offset * (pred_hmap.unsqueeze(1) > self.pModel.score_thresh).float()
                gen_offsetkitti(pred_offset_save, seq_id[0], fn)
            pred_panoptic = pred_panoptic.cpu().numpy().astype(np.uint32)
            
            pred_panoptic_list.append(pred_panoptic)
            sem_pred_panoptic = pred_panoptic & 0xFFFF
            ins_pred_panoptic = pred_panoptic >> 16
            
            pred_panoptic_use = utils.relabel(sem_pred_panoptic, self.task_cfg['learning_map_inv'])
            pred_panoptic_save = (ins_pred_panoptic << 16) + pred_panoptic_use
            
            output_path = os.path.join("output", "sequences", seq_id[0], "predictions")
            if not os.path.exists(output_path):
                os.system("mkdir -p {}".format(output_path))
            
            pcd_name = os.path.join(output_path, os.path.basename(fn[0]).replace('.bin', '.label'))
            pred_panoptic_save.reshape(-1).astype(np.uint32).tofile(pcd_name)
        


def get_fg_pc_offset_weight(gt_ins_label):
        BS = gt_ins_label.shape[0]
        N = gt_ins_label.shape[1]

        if gt_ins_label.max() >= 0:
            per_obj_num = pytorch_lib.VoxelSum(pcds_ind=gt_ins_label.view(BS, N, 1, 1), output_size=(int(gt_ins_label.max()) + 1,)).float().unsqueeze(1) #(BS, 1, K)
            obj_sum = int((per_obj_num > 0).sum())
            per_obj_weight = 1 / (per_obj_num * obj_sum + 1e-12) #(BS, 1, K)

            pc_offset_weight = pytorch_lib.VoxelQuery(voxel_in=per_obj_weight, pcds_ind=gt_ins_label.view(BS, N, 1, 1)).squeeze(1) #(BS, N, 1)
            return pc_offset_weight
        else:
            pc_offset_weight = torch.zeros_like(gt_ins_label).float()
            return pc_offset_weight