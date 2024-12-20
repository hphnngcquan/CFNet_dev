import numpy as np
import random

from scipy.spatial.transform import Rotation as R
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt


def gene_ins_label(pcds_label_use, inst_label):
    pcds_ins_label = np.full((pcds_label_use.shape[0],), -1, dtype=np.int64)
    ins_num = 0
    for cate_id in range(1, 9):
        valid_mask_cate_id = (pcds_label_use == cate_id)
        # indicate the categoty exists
        if valid_mask_cate_id.sum() > 0:
            inst_label_cate_id = inst_label[valid_mask_cate_id]
            valid_inst_id = np.unique(inst_label_cate_id)
            for obj_id in valid_inst_id:
                valid_obj_mask = valid_mask_cate_id * (inst_label == obj_id)
                pcds_ins_label[valid_obj_mask] = ins_num
                ins_num = ins_num + 1
    return pcds_ins_label


def gene_point_offset(pcds_total, center_type='mass'):
    assert center_type in ['mass', 'axis']
    pcds_xyz = pcds_total[:, :3]
    pcds_ins_label = pcds_total[:, 5].astype(np.int64)

    pcds_offset = np.zeros((pcds_xyz.shape[0], 3), dtype=np.float32)
    if center_type == 'mass':
        for obj_id in range(pcds_ins_label.max() + 1):
            valid_mask_obj = pcds_ins_label == obj_id
            pcds_xyz_obj = pcds_xyz[valid_mask_obj]
            if pcds_xyz_obj.shape[0] > 0:
                obj_center = pcds_xyz_obj.mean(axis=0, keepdims=True)
                pcds_offset[valid_mask_obj] = obj_center - pcds_xyz_obj
    else:
        for obj_id in range(pcds_ins_label.max() + 1):
            valid_mask_obj = pcds_ins_label == obj_id
            pcds_xyz_obj = pcds_xyz[valid_mask_obj]
            if pcds_xyz_obj.shape[0] > 0:
                obj_center = (pcds_xyz_obj.max(axis=0, keepdims=True) + pcds_xyz_obj.min(axis=0, keepdims=True)) / 2.0
                pcds_offset[valid_mask_obj] = obj_center - pcds_xyz_obj
    return pcds_offset


def in_hull(p, hull):
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p) >= 0


def compute_box_3d(center, size, yaw):
    c = np.cos(yaw)
    s = np.sin(yaw)
    R = np.array([[c, -s, 0],
                  [s, c, 0],
                  [0, 0, 1]])
    
    # 3d bounding box dimensions
    l = size[0]
    w = size[1]
    h = size[2]
    
    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    z_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    
    corners_3d[0, :] = corners_3d[0, :] + center[0]
    corners_3d[1, :] = corners_3d[1, :] + center[1]
    corners_3d[2, :] = corners_3d[2, :] + center[2]
    return corners_3d.T


def random_float(v_range):
    v = random.random()
    v = v * (v_range[1] - v_range[0]) + v_range[0]
    return v


def in_range(v, r):
    return (v >= r[0]) * (v < r[1])


def filter_pcds(pcds, range_x=(-40, 60), range_y=(-40, 40), range_z=(-3, 5)):
    valid_x = (pcds[:, 0] >= range_x[0]) * (pcds[:, 0] < range_x[1])
    valid_y = (pcds[:, 1] >= range_y[0]) * (pcds[:, 1] < range_y[1])
    valid_z = (pcds[:, 2] >= range_z[0]) * (pcds[:, 2] < range_z[1])
    
    pcds_filter = pcds[valid_x * valid_y * valid_z]
    return pcds_filter


def filter_pcds_mask(pcds, range_x=(-40, 60), range_y=(-40, 40), range_z=(-3, 5)):
    valid_x = (pcds[:, 0] >= range_x[0]) * (pcds[:, 0] < range_x[1])
    valid_y = (pcds[:, 1] >= range_y[0]) * (pcds[:, 1] < range_y[1])
    valid_z = (pcds[:, 2] >= range_z[0]) * (pcds[:, 2] < range_z[1])
    
    valid_mask = valid_x * valid_y * valid_z
    return valid_mask


def Trans(pcds, mat):
    pcds_out = pcds.copy()
    
    pcds_tmp = pcds_out[:, :4].T
    pcds_tmp[-1] = 1
    pcds_tmp = mat.dot(pcds_tmp)
    pcds_tmp = pcds_tmp.T
    
    pcds_out[..., :3] = pcds_tmp[..., :3]
    pcds_out[..., 3:] = pcds[..., 3:]
    return pcds_out


def relabel(pcds_labels, label_map):
    result_labels = np.zeros((pcds_labels.shape[0],), dtype=pcds_labels.dtype)
    for key in label_map:
        value = label_map[key]
        mask = (pcds_labels == key)
        result_labels[mask] = value
    
    return result_labels


def recolor(pcds_labels, color_map):
    result_color = np.zeros((pcds_labels.shape[0], 3), dtype=np.uint8)
    for key in color_map:
        value = color_map[key]
        mask = pcds_labels == key
        result_color[mask, 0] = value[0]
        result_color[mask, 1] = value[1]
        result_color[mask, 2] = value[2]
    
    return result_color


def Quantize(pcds, range_x=(-40, 62.4), range_y=(-40, 40), range_z=(-3, 5), size=(512, 512, 20)):
    x = pcds[:, 0].copy()
    y = pcds[:, 1].copy()
    z = pcds[:, 2].copy()
    
    size_x = size[0]
    size_y = size[1]
    size_z = size[2]
    
    dx = (range_x[1] - range_x[0]) / size_x
    dy = (range_y[1] - range_y[0]) / size_y
    dz = (range_z[1] - range_z[0]) / size_z
    
    x_quan = ((x - range_x[0]) / dx)
    y_quan = ((y - range_y[0]) / dy)
    z_quan = ((z - range_z[0]) / dz)
    
    pcds_quan = np.stack((x_quan, y_quan, z_quan), axis=-1)
    return pcds_quan


def SphereQuantize(pcds, phi_range=(-180.0, 180.0), theta_range=(-16.0, 10.0), size=(64, 2048)):
    H = size[0]
    W = size[1]
    
    phi_range_radian = (phi_range[0] * np.pi / 180.0, phi_range[1] * np.pi / 180.0)
    theta_range_radian = (theta_range[0] * np.pi / 180.0, theta_range[1] * np.pi / 180.0)
    
    dphi = (phi_range_radian[1] - phi_range_radian[0]) / W
    dtheta = (theta_range_radian[1] - theta_range_radian[0]) / H
    
    x, y, z = pcds[:, 0], pcds[:, 1], pcds[:, 2]
    d = np.sqrt(x ** 2 + y ** 2 + z ** 2) + 1e-12
    
    phi = phi_range_radian[1] - np.arctan2(x, y)
    phi_quan = (phi / dphi)
    
    theta = theta_range_radian[1] - np.arcsin(z / d)
    theta_quan = (theta / dtheta)
    
    sphere_coords = np.stack((theta_quan, phi_quan), axis=-1)
    return sphere_coords


def CylinderQuantize(pcds, phi_range=(-180.0, 180.0), range_z=(-3, 5), size=(64, 2048)):
    H = size[0]
    W = size[1]
    
    phi_range_radian = (phi_range[0] * np.pi / 180.0, phi_range[1] * np.pi / 180.0)
    dphi = (phi_range_radian[1] - phi_range_radian[0]) / W
    
    dz = (range_z[1] - range_z[0]) / H
    
    x, y, z = pcds[:, 0], pcds[:, 1], pcds[:, 2]
    
    phi = phi_range_radian[1] - np.arctan2(x, y)
    phi_quan = (phi / dphi)
    
    z_quan = ((z - range_z[0]) / dz)
    
    cylinder_coords = np.stack((z_quan, phi_quan), axis=-1)
    return cylinder_coords


class DataAugment:
    def __init__(self, noise_mean=0, noise_std=0.01, theta_range=(-45, 45), shift_range=(0, 0), size_range=(0.95, 1.05)):
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.theta_range = theta_range
        self.shift_range = shift_range
        self.size_range = size_range
        assert len(self.shift_range) == 3
    
    def __call__(self, pcds):
        """Inputs:
            pcds: (N, C) N demotes the number of point clouds; C contains (x, y, z, i, ...)
           Output:
            pcds: (N, C)
        """
        #random noise
        xyz_noise = np.random.normal(self.noise_mean, self.noise_std, size=(pcds.shape[0], 3))
        pcds[:, :3] = pcds[:, :3] + xyz_noise
        
        #random shift
        shift_xyz = [random_float(self.shift_range[i]) for i in range(3)]
        pcds[:, 0] = pcds[:, 0] + shift_xyz[0]
        pcds[:, 1] = pcds[:, 1] + shift_xyz[1]
        pcds[:, 2] = pcds[:, 2] + shift_xyz[2]
        
        #random scale
        scale = random_float(self.size_range)
        pcds[:, :3] = pcds[:, :3] * scale
        
        #random flip on xy plane
        h_flip = 0
        v_flip = 0
        if random.random() < 0.5:
            h_flip = 1
        else:
            h_flip = 0
        
        if random.random() < 0.5:
            v_flip = 1
        else:
            v_flip = 0
        
        if(v_flip == 1):
            pcds[:, 0] = pcds[:, 0] * -1
        
        if(h_flip == 1):
            pcds[:, 1] = pcds[:, 1] * -1
        
        #random rotate on xy plane
        theta_z = random_float(self.theta_range)
        rotateMatrix = R.from_euler('z', theta_z, degrees=True).as_matrix()[:2, :2].T
        pcds[:, :2] = pcds[:, :2].dot(rotateMatrix)
        return pcds
    
    
def draw_point_with(pcl, indices=None, name='pcl_img.png', s=1, pcl2=None, s2=1):
    # Assuming `embedding` is your point cloud and `indices` are the indices of the centroids
    embedding_np = pcl  # Convert to numpy array for plotting
    
    # Create a new color array
    colors = np.full(embedding_np.shape[0], 'r')  # All points are blue
    
    # Create a new size array
    sizes = np.full(embedding_np.shape[0], s)  # All points are size 1
    
    if indices is not None:
        indices_np = indices.flatten()  # Flatten the indices array for indexing
        colors[indices_np] = 'w'  # Centroid points are red
        sizes[indices_np] = 10  # Centroid points are size 20

    # Create a 3D plot
    fig = plt.figure(figsize=(30, 30))
    ax = fig.add_subplot()

    

    # Plot the point cloud in 3D
    scatter = ax.scatter(embedding_np[:, 0], embedding_np[:, 1], c=colors, s=sizes)
    if pcl2 is not None:
        scatter = ax.scatter(pcl2[:, 0], pcl2[:, 1], c='r', s=s2)

    # Set background color to black
    ax.set_facecolor('black')
    ax.set_aspect('auto')

    plt.savefig(name)
    
import matplotlib.pyplot as plt
import torch
import os

def save_feature_map(feature_map, base_filename='feat'):
    # Ensure the input is a tensor
    if not isinstance(feature_map, torch.Tensor):
        raise ValueError("The feature map must be a PyTorch tensor.")

    # Move tensor to CPU if necessary and convert to numpy
    feature_map = feature_map.detach().cpu().numpy()

    # Check if feature map has more than one channel
    if feature_map.ndim == 4:  # Batch size x Channels x Height x Width
        num_channels = feature_map.shape[1]
        for i in range(num_channels):
            plt.imsave(os.path.join(f"{base_filename}_channel_{i+1}.png"), feature_map[0, i], cmap='viridis')

    elif feature_map.ndim == 3:  # Channels x Height x Width
        num_channels = feature_map.shape[0]
        for i in range(num_channels):
            plt.imsave(os.path.join(f"{base_filename}_channel_{i+1}.png"), feature_map[i], cmap='viridis')

