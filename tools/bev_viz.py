import os
import json
import copy
import tempfile
import pickle
from typing import Dict, List
from mmcv.fileio.io import dump,load
import numpy as np
from mmcv.datasets import NuScenesDataset
import pyquaternion
import mmcv
from os import path as osp
import torch
import numpy as np
from mmcv.nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from mmcv.nuscenes.eval.common.utils import center_distance
from mmcv.utils.visual import save_tensor
from mmcv.parallel import DataContainer as DC
import random
from mmcv.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from mmcv.nuscenes.utils.data_classes import Box as NuScenesBox
from mmcv.core.bbox.structures.nuscenes_box import CustomNuscenesBox
from shapely import affinity, ops
from shapely.geometry import LineString, box, MultiPolygon, MultiLineString
from mmcv.datasets.pipelines import to_tensor
from mmcv.nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from mmcv.nuscenes.eval.detection.constants import DETECTION_NAMES

import numpy as np
import torch
from shapely.geometry import LineString
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm

import copy

import matplotlib.animation as animation
import imageio
from tqdm import tqdm
import argparse
from multiprocessing import Pool


class LiDARInstanceLines(object):
    """Line instance in LIDAR coordinates

    """
    def __init__(self, 
                 instance_line_list, 
                 sample_dist=1,
                 num_samples=250,
                 padding=False,
                 fixed_num=-1,
                 padding_value=-10000,
                 patch_size=None):
        assert isinstance(instance_line_list, list)
        assert patch_size is not None
        if len(instance_line_list) != 0:
            assert isinstance(instance_line_list[0], LineString)
        self.patch_size = patch_size
        self.max_x = self.patch_size[1] / 2
        self.max_y = self.patch_size[0] / 2
        self.sample_dist = sample_dist
        self.num_samples = num_samples
        self.padding = padding
        self.fixed_num = fixed_num
        self.padding_value = padding_value

        self.instance_list = instance_line_list

    @property
    def start_end_points(self):
        """
        return torch.Tensor([N,4]), in xstart, ystart, xend, yend form
        """
        assert len(self.instance_list) != 0
        instance_se_points_list = []
        for instance in self.instance_list:
            se_points = []
            se_points.extend(instance.coords[0])
            se_points.extend(instance.coords[-1])
            instance_se_points_list.append(se_points)
        instance_se_points_array = np.array(instance_se_points_list)
        instance_se_points_tensor = to_tensor(instance_se_points_array)
        instance_se_points_tensor = instance_se_points_tensor.to(
                                dtype=torch.float32)
        instance_se_points_tensor[:,0] = torch.clamp(instance_se_points_tensor[:,0], min=-self.max_x,max=self.max_x)
        instance_se_points_tensor[:,1] = torch.clamp(instance_se_points_tensor[:,1], min=-self.max_y,max=self.max_y)
        instance_se_points_tensor[:,2] = torch.clamp(instance_se_points_tensor[:,2], min=-self.max_x,max=self.max_x)
        instance_se_points_tensor[:,3] = torch.clamp(instance_se_points_tensor[:,3], min=-self.max_y,max=self.max_y)
        return instance_se_points_tensor

    @property
    def bbox(self):
        """
        return torch.Tensor([N,4]), in xmin, ymin, xmax, ymax form
        """
        assert len(self.instance_list) != 0
        instance_bbox_list = []
        for instance in self.instance_list:
            # bounds is bbox: [xmin, ymin, xmax, ymax]
            instance_bbox_list.append(instance.bounds)
        instance_bbox_array = np.array(instance_bbox_list)
        instance_bbox_tensor = to_tensor(instance_bbox_array)
        instance_bbox_tensor = instance_bbox_tensor.to(
                            dtype=torch.float32)
        instance_bbox_tensor[:,0] = torch.clamp(instance_bbox_tensor[:,0], min=-self.max_x,max=self.max_x)
        instance_bbox_tensor[:,1] = torch.clamp(instance_bbox_tensor[:,1], min=-self.max_y,max=self.max_y)
        instance_bbox_tensor[:,2] = torch.clamp(instance_bbox_tensor[:,2], min=-self.max_x,max=self.max_x)
        instance_bbox_tensor[:,3] = torch.clamp(instance_bbox_tensor[:,3], min=-self.max_y,max=self.max_y)
        return instance_bbox_tensor

    @property
    def fixed_num_sampled_points(self):
        """
        return torch.Tensor([N,fixed_num,2]), in xmin, ymin, xmax, ymax form
            N means the num of instances
        """
        assert len(self.instance_list) != 0
        instance_points_list = []
        for instance in self.instance_list:
            distances = np.linspace(0, instance.length, self.fixed_num)
            sampled_points = np.array([list(instance.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
            instance_points_list.append(sampled_points)
        instance_points_array = np.array(instance_points_list)
        instance_points_tensor = to_tensor(instance_points_array)
        instance_points_tensor = instance_points_tensor.to(
                            dtype=torch.float32)
        instance_points_tensor[:,:,0] = torch.clamp(instance_points_tensor[:,:,0], min=-self.max_x,max=self.max_x)
        instance_points_tensor[:,:,1] = torch.clamp(instance_points_tensor[:,:,1], min=-self.max_y,max=self.max_y)
        return instance_points_tensor

    @property
    def fixed_num_sampled_points_ambiguity(self):
        """
        return torch.Tensor([N,fixed_num,2]), in xmin, ymin, xmax, ymax form
            N means the num of instances
        """
        assert len(self.instance_list) != 0
        instance_points_list = []
        for instance in self.instance_list:
            distances = np.linspace(0, instance.length, self.fixed_num)
            sampled_points = np.array([list(instance.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
            instance_points_list.append(sampled_points)
        instance_points_array = np.array(instance_points_list)
        instance_points_tensor = to_tensor(instance_points_array)
        instance_points_tensor = instance_points_tensor.to(
                            dtype=torch.float32)
        instance_points_tensor[:,:,0] = torch.clamp(instance_points_tensor[:,:,0], min=-self.max_x,max=self.max_x)
        instance_points_tensor[:,:,1] = torch.clamp(instance_points_tensor[:,:,1], min=-self.max_y,max=self.max_y)
        instance_points_tensor = instance_points_tensor.unsqueeze(1)
        return instance_points_tensor

    @property
    def fixed_num_sampled_points_torch(self):
        """
        return torch.Tensor([N,fixed_num,2]), in xmin, ymin, xmax, ymax form
            N means the num of instances
        """
        assert len(self.instance_list) != 0
        instance_points_list = []
        for instance in self.instance_list:
            # distances = np.linspace(0, instance.length, self.fixed_num)
            # sampled_points = np.array([list(instance.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
            poly_pts = to_tensor(np.array(list(instance.coords)))
            poly_pts = poly_pts.unsqueeze(0).permute(0,2,1)
            sampled_pts = torch.nn.functional.interpolate(poly_pts,size=(self.fixed_num),mode='linear',align_corners=True)
            sampled_pts = sampled_pts.permute(0,2,1).squeeze(0)
            instance_points_list.append(sampled_pts)
        # instance_points_array = np.array(instance_points_list)
        # instance_points_tensor = to_tensor(instance_points_array)
        instance_points_tensor = torch.stack(instance_points_list,dim=0)
        instance_points_tensor = instance_points_tensor.to(
                            dtype=torch.float32)
        instance_points_tensor[:,:,0] = torch.clamp(instance_points_tensor[:,:,0], min=-self.max_x,max=self.max_x)
        instance_points_tensor[:,:,1] = torch.clamp(instance_points_tensor[:,:,1], min=-self.max_y,max=self.max_y)
        return instance_points_tensor

    @property
    def shift_fixed_num_sampled_points(self):
        """
        return  [instances_num, num_shifts, fixed_num, 2]
        """
        fixed_num_sampled_points = self.fixed_num_sampled_points
        instances_list = []
        is_poly = False
        # is_line = False
        # import pdb;pdb.set_trace()
        for fixed_num_pts in fixed_num_sampled_points:
            # [fixed_num, 2]
            is_poly = fixed_num_pts[0].equal(fixed_num_pts[-1])
            fixed_num = fixed_num_pts.shape[0]
            shift_pts_list = []
            if is_poly:
                # import pdb;pdb.set_trace()
                for shift_right_i in range(fixed_num):
                    shift_pts_list.append(fixed_num_pts.roll(shift_right_i,0))
            else:
                shift_pts_list.append(fixed_num_pts)
                shift_pts_list.append(fixed_num_pts.flip(0))
            shift_pts = torch.stack(shift_pts_list,dim=0)

            shift_pts[:,:,0] = torch.clamp(shift_pts[:,:,0], min=-self.max_x,max=self.max_x)
            shift_pts[:,:,1] = torch.clamp(shift_pts[:,:,1], min=-self.max_y,max=self.max_y)

            if not is_poly:
                padding = torch.full([fixed_num-shift_pts.shape[0],fixed_num,2], self.padding_value)
                shift_pts = torch.cat([shift_pts,padding],dim=0)
                # padding = np.zeros((self.num_samples - len(sampled_points), 2))
                # sampled_points = np.concatenate([sampled_points, padding], axis=0)
            instances_list.append(shift_pts)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(
                            dtype=torch.float32)
        return instances_tensor

    @property
    def shift_fixed_num_sampled_points_v1(self):
        """
        return  [instances_num, num_shifts, fixed_num, 2]
        """
        fixed_num_sampled_points = self.fixed_num_sampled_points
        instances_list = []
        is_poly = False
        # is_line = False
        # import pdb;pdb.set_trace()
        for fixed_num_pts in fixed_num_sampled_points:
            # [fixed_num, 2]
            is_poly = fixed_num_pts[0].equal(fixed_num_pts[-1])
            pts_num = fixed_num_pts.shape[0]
            shift_num = pts_num - 1
            if is_poly:
                pts_to_shift = fixed_num_pts[:-1,:]
            shift_pts_list = []
            if is_poly:
                for shift_right_i in range(shift_num):
                    shift_pts_list.append(pts_to_shift.roll(shift_right_i,0))
            else:
                shift_pts_list.append(fixed_num_pts)
                shift_pts_list.append(fixed_num_pts.flip(0))
            shift_pts = torch.stack(shift_pts_list,dim=0)

            if is_poly:
                _, _, num_coords = shift_pts.shape
                tmp_shift_pts = shift_pts.new_zeros((shift_num, pts_num, num_coords))
                tmp_shift_pts[:,:-1,:] = shift_pts
                tmp_shift_pts[:,-1,:] = shift_pts[:,0,:]
                shift_pts = tmp_shift_pts

            shift_pts[:,:,0] = torch.clamp(shift_pts[:,:,0], min=-self.max_x,max=self.max_x)
            shift_pts[:,:,1] = torch.clamp(shift_pts[:,:,1], min=-self.max_y,max=self.max_y)

            if not is_poly:
                padding = torch.full([shift_num-shift_pts.shape[0],pts_num,2], self.padding_value)
                shift_pts = torch.cat([shift_pts,padding],dim=0)
                # padding = np.zeros((self.num_samples - len(sampled_points), 2))
                # sampled_points = np.concatenate([sampled_points, padding], axis=0)
            instances_list.append(shift_pts)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(
                            dtype=torch.float32)
        return instances_tensor

    @property
    def shift_fixed_num_sampled_points_v2(self):
        """
        return  [instances_num, num_shifts, fixed_num, 2]
        """
        assert len(self.instance_list) != 0
        instances_list = []
        for instance in self.instance_list:
            distances = np.linspace(0, instance.length, self.fixed_num)
            poly_pts = np.array(list(instance.coords))
            start_pts = poly_pts[0]
            end_pts = poly_pts[-1]
            is_poly = np.equal(start_pts, end_pts)
            is_poly = is_poly.all()
            shift_pts_list = []
            pts_num, coords_num = poly_pts.shape
            shift_num = pts_num - 1
            final_shift_num = self.fixed_num - 1
            if is_poly:
                pts_to_shift = poly_pts[:-1,:]
                for shift_right_i in range(shift_num):
                    shift_pts = np.roll(pts_to_shift,shift_right_i,axis=0)
                    pts_to_concat = shift_pts[0]
                    pts_to_concat = np.expand_dims(pts_to_concat,axis=0)
                    shift_pts = np.concatenate((shift_pts,pts_to_concat),axis=0)
                    shift_instance = LineString(shift_pts)
                    shift_sampled_points = np.array([list(shift_instance.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
                    shift_pts_list.append(shift_sampled_points)
                # import pdb;pdb.set_trace()
            else:
                sampled_points = np.array([list(instance.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
                flip_sampled_points = np.flip(sampled_points, axis=0)
                shift_pts_list.append(sampled_points)
                shift_pts_list.append(flip_sampled_points)
            
            multi_shifts_pts = np.stack(shift_pts_list,axis=0)
            shifts_num,_,_ = multi_shifts_pts.shape

            if shifts_num > final_shift_num:
                index = np.random.choice(multi_shifts_pts.shape[0], final_shift_num, replace=False)
                multi_shifts_pts = multi_shifts_pts[index]
            
            multi_shifts_pts_tensor = to_tensor(multi_shifts_pts)
            multi_shifts_pts_tensor = multi_shifts_pts_tensor.to(
                            dtype=torch.float32)
            
            multi_shifts_pts_tensor[:,:,0] = torch.clamp(multi_shifts_pts_tensor[:,:,0], min=-self.max_x,max=self.max_x)
            multi_shifts_pts_tensor[:,:,1] = torch.clamp(multi_shifts_pts_tensor[:,:,1], min=-self.max_y,max=self.max_y)
            # if not is_poly:
            if multi_shifts_pts_tensor.shape[0] < final_shift_num:
                padding = torch.full([final_shift_num-multi_shifts_pts_tensor.shape[0],self.fixed_num,2], self.padding_value)
                multi_shifts_pts_tensor = torch.cat([multi_shifts_pts_tensor,padding],dim=0)
            instances_list.append(multi_shifts_pts_tensor)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(
                            dtype=torch.float32)
        return instances_tensor

    @property
    def shift_fixed_num_sampled_points_v3(self):
        """
        return  [instances_num, num_shifts, fixed_num, 2]
        """
        assert len(self.instance_list) != 0
        instances_list = []
        for instance in self.instance_list:
            distances = np.linspace(0, instance.length, self.fixed_num)
            poly_pts = np.array(list(instance.coords))
            start_pts = poly_pts[0]
            end_pts = poly_pts[-1]
            is_poly = np.equal(start_pts, end_pts)
            is_poly = is_poly.all()
            shift_pts_list = []
            pts_num, coords_num = poly_pts.shape
            shift_num = pts_num - 1
            final_shift_num = self.fixed_num - 1
            if is_poly:
                pts_to_shift = poly_pts[:-1,:]
                for shift_right_i in range(shift_num):
                    shift_pts = np.roll(pts_to_shift,shift_right_i,axis=0)
                    pts_to_concat = shift_pts[0]
                    pts_to_concat = np.expand_dims(pts_to_concat,axis=0)
                    shift_pts = np.concatenate((shift_pts,pts_to_concat),axis=0)
                    shift_instance = LineString(shift_pts)
                    shift_sampled_points = np.array([list(shift_instance.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
                    shift_pts_list.append(shift_sampled_points)
                flip_pts_to_shift = np.flip(pts_to_shift, axis=0)
                for shift_right_i in range(shift_num):
                    shift_pts = np.roll(flip_pts_to_shift,shift_right_i,axis=0)
                    pts_to_concat = shift_pts[0]
                    pts_to_concat = np.expand_dims(pts_to_concat,axis=0)
                    shift_pts = np.concatenate((shift_pts,pts_to_concat),axis=0)
                    shift_instance = LineString(shift_pts)
                    shift_sampled_points = np.array([list(shift_instance.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
                    shift_pts_list.append(shift_sampled_points)
                # import pdb;pdb.set_trace()
            else:
                sampled_points = np.array([list(instance.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
                flip_sampled_points = np.flip(sampled_points, axis=0)
                shift_pts_list.append(sampled_points)
                shift_pts_list.append(flip_sampled_points)
            
            multi_shifts_pts = np.stack(shift_pts_list,axis=0)
            shifts_num,_,_ = multi_shifts_pts.shape
            # import pdb;pdb.set_trace()
            if shifts_num > 2*final_shift_num:
                index = np.random.choice(shift_num, final_shift_num, replace=False)
                flip0_shifts_pts = multi_shifts_pts[index]
                flip1_shifts_pts = multi_shifts_pts[index+shift_num]
                multi_shifts_pts = np.concatenate((flip0_shifts_pts,flip1_shifts_pts),axis=0)
            
            multi_shifts_pts_tensor = to_tensor(multi_shifts_pts)
            multi_shifts_pts_tensor = multi_shifts_pts_tensor.to(
                            dtype=torch.float32)
            
            multi_shifts_pts_tensor[:,:,0] = torch.clamp(multi_shifts_pts_tensor[:,:,0], min=-self.max_x,max=self.max_x)
            multi_shifts_pts_tensor[:,:,1] = torch.clamp(multi_shifts_pts_tensor[:,:,1], min=-self.max_y,max=self.max_y)
            # if not is_poly:
            if multi_shifts_pts_tensor.shape[0] < 2*final_shift_num:
                padding = torch.full([final_shift_num*2-multi_shifts_pts_tensor.shape[0],self.fixed_num,2], self.padding_value)
                multi_shifts_pts_tensor = torch.cat([multi_shifts_pts_tensor,padding],dim=0)
            instances_list.append(multi_shifts_pts_tensor)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(
                            dtype=torch.float32)
        return instances_tensor

    @property
    def shift_fixed_num_sampled_points_v4(self):
        """
        return  [instances_num, num_shifts, fixed_num, 2]
        """
        fixed_num_sampled_points = self.fixed_num_sampled_points
        instances_list = []
        is_poly = False
        # is_line = False
        # import pdb;pdb.set_trace()
        for fixed_num_pts in fixed_num_sampled_points:
            # [fixed_num, 2]
            is_poly = fixed_num_pts[0].equal(fixed_num_pts[-1])
            pts_num = fixed_num_pts.shape[0]
            shift_num = pts_num - 1
            shift_pts_list = []
            if is_poly:
                pts_to_shift = fixed_num_pts[:-1,:]
                for shift_right_i in range(shift_num):
                    shift_pts_list.append(pts_to_shift.roll(shift_right_i,0))
                flip_pts_to_shift = pts_to_shift.flip(0)
                for shift_right_i in range(shift_num):
                    shift_pts_list.append(flip_pts_to_shift.roll(shift_right_i,0))
            else:
                shift_pts_list.append(fixed_num_pts)
                shift_pts_list.append(fixed_num_pts.flip(0))
            shift_pts = torch.stack(shift_pts_list,dim=0)

            if is_poly:
                _, _, num_coords = shift_pts.shape
                tmp_shift_pts = shift_pts.new_zeros((shift_num*2, pts_num, num_coords))
                tmp_shift_pts[:,:-1,:] = shift_pts
                tmp_shift_pts[:,-1,:] = shift_pts[:,0,:]
                shift_pts = tmp_shift_pts

            shift_pts[:,:,0] = torch.clamp(shift_pts[:,:,0], min=-self.max_x,max=self.max_x)
            shift_pts[:,:,1] = torch.clamp(shift_pts[:,:,1], min=-self.max_y,max=self.max_y)

            if not is_poly:
                padding = torch.full([shift_num*2-shift_pts.shape[0],pts_num,2], self.padding_value)
                shift_pts = torch.cat([shift_pts,padding],dim=0)
                # padding = np.zeros((self.num_samples - len(sampled_points), 2))
                # sampled_points = np.concatenate([sampled_points, padding], axis=0)
            instances_list.append(shift_pts)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(
                            dtype=torch.float32)
        return instances_tensor

    @property
    def shift_fixed_num_sampled_points_torch(self):
        """
        return  [instances_num, num_shifts, fixed_num, 2]
        """
        fixed_num_sampled_points = self.fixed_num_sampled_points_torch
        instances_list = []
        is_poly = False
        # is_line = False
        # import pdb;pdb.set_trace()
        for fixed_num_pts in fixed_num_sampled_points:
            # [fixed_num, 2]
            is_poly = fixed_num_pts[0].equal(fixed_num_pts[-1])
            fixed_num = fixed_num_pts.shape[0]
            shift_pts_list = []
            if is_poly:
                # import pdb;pdb.set_trace()
                for shift_right_i in range(fixed_num):
                    shift_pts_list.append(fixed_num_pts.roll(shift_right_i,0))
            else:
                shift_pts_list.append(fixed_num_pts)
                shift_pts_list.append(fixed_num_pts.flip(0))
            shift_pts = torch.stack(shift_pts_list,dim=0)

            shift_pts[:,:,0] = torch.clamp(shift_pts[:,:,0], min=-self.max_x,max=self.max_x)
            shift_pts[:,:,1] = torch.clamp(shift_pts[:,:,1], min=-self.max_y,max=self.max_y)

            if not is_poly:
                padding = torch.full([fixed_num-shift_pts.shape[0],fixed_num,2], self.padding_value)
                shift_pts = torch.cat([shift_pts,padding],dim=0)
                # padding = np.zeros((self.num_samples - len(sampled_points), 2))
                # sampled_points = np.concatenate([sampled_points, padding], axis=0)
            instances_list.append(shift_pts)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(
                            dtype=torch.float32)
        return instances_tensor

    # @property
    # def polyline_points(self):
    #     """
    #     return [[x0,y0],[x1,y1],...]
    #     """
    #     assert len(self.instance_list) != 0
    #     for instance in self.instance_list:

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import numpy as np
from shapely.geometry import LineString, box, MultiPolygon
from shapely import affinity, ops
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R

class MapVisualizer:
    def __init__(self, vector_maps):
        self.vector_maps = vector_maps
        self.MAPCLASSES = ['divider', 'ped_crossing', 'boundary']
        
    def visualize_map(self, frame, ax):
        location = frame['map_location']
        vector_map = self.vector_maps[location]
        
        lidar2global = self._get_lidar2global_matrix(frame)
        lidar2global_translation = list(lidar2global[:3,3])
        lidar2global_rotation = list(Quaternion(matrix=lidar2global).q)
        
        anns_results = vector_map.gen_vectorized_samples(location, lidar2global_translation, lidar2global_rotation)
        
        self._plot_map_elements(ax, anns_results)
        
        return ax
    
    def _get_lidar2global_matrix(self, frame):
        lidar2ego = np.eye(4)
        lidar2ego[:3,:3] = Quaternion(frame['lidar2ego_rotation']).rotation_matrix
        lidar2ego[:3, 3] = frame['lidar2ego_translation']
        ego2global = np.eye(4)
        ego2global[:3,:3] = Quaternion(frame['ego2global_rotation']).rotation_matrix
        ego2global[:3, 3] = frame['ego2global_translation']
        return ego2global @ lidar2ego
    
    def _plot_map_elements(self, ax, anns_results):
        gt_vecs_label = anns_results['gt_vecs_label']
        gt_vecs_pts_loc = anns_results['gt_vecs_pts_loc']
        
        points_tensor = gt_vecs_pts_loc.fixed_num_sampled_points
        points_numpy = points_tensor.numpy()
        
        labels_seen = set()
        unique_labels = np.unique(gt_vecs_label)
        colors = cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        color_dict = dict(zip(unique_labels, colors))
        
        for i in range(points_numpy.shape[0]):
            x = points_numpy[i, :, 0]
            y = points_numpy[i, :, 1]
            label = self.MAPCLASSES[gt_vecs_label[i].item()]
            color = color_dict[gt_vecs_label[i].item()]
            ax.plot(x, y, marker='o', markersize=2, color=color, label=label if label not in labels_seen else "")
            labels_seen.add(label)

class VectorizedLocalMap(object):
    CLASS2LABEL = {
        'road_divider': 0,
        'lane_divider': 0,
        'ped_crossing': 1,
        'contours': 2,
        'others': -1
    }
    def __init__(self,
                 dataroot,
                 patch_size,
                 map_classes=['divider','ped_crossing','boundary'],
                 line_classes=['road_divider', 'lane_divider'],
                 ped_crossing_classes=['ped_crossing'],
                 contour_classes=['road_segment', 'lane'],
                 sample_dist=1,
                 num_samples=250,
                 padding=False,
                 fixed_ptsnum_per_line=-1,
                 padding_value=-10000,
                 MAPS = ['singapore-onenorth', 'singapore-hollandvillage', 'singapore-queenstown', 'boston-seaport', 
                        'us-ma-boston', 'sg-one-north', 'us-nv-las-vegas-strip', 'us-pa-pittsburgh-hazelwood']
                ):
        '''
        Args:
            fixed_ptsnum_per_line = -1 : no fixed num
        '''
        super().__init__()
        self.data_root = dataroot
        self.MAPS = MAPS
        self.vec_classes = map_classes
        self.line_classes = line_classes
        self.ped_crossing_classes = ped_crossing_classes
        self.polygon_classes = contour_classes
        self.nusc_maps = {}
        self.map_explorer = {}
        for loc in self.MAPS:
            self.nusc_maps[loc] = NuScenesMap(dataroot=self.data_root, map_name=loc)
            self.map_explorer[loc] = NuScenesMapExplorer(self.nusc_maps[loc])

        self.patch_size = patch_size
        self.sample_dist = sample_dist
        self.num_samples = num_samples
        self.padding = padding
        self.fixed_num = fixed_ptsnum_per_line
        self.padding_value = padding_value

    def gen_vectorized_samples(self, location, lidar2global_translation, lidar2global_rotation):
        '''
        use lidar2global to get gt map layers
        '''
        
        map_pose = lidar2global_translation[:2]
        rotation = Quaternion(lidar2global_rotation)

        patch_box = (map_pose[0], map_pose[1], self.patch_size[0], self.patch_size[1])
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        # import pdb;pdb.set_trace()
        vectors = []
        for vec_class in self.vec_classes:
            if vec_class == 'divider':
                line_geom = self.get_map_geom(patch_box, patch_angle, self.line_classes, location)
                line_instances_dict = self.line_geoms_to_instances(line_geom)     
                for line_type, instances in line_instances_dict.items():
                    for instance in instances:
                        vectors.append((instance, self.CLASS2LABEL.get(line_type, -1)))
            elif vec_class == 'ped_crossing':
                ped_geom = self.get_map_geom(patch_box, patch_angle, self.ped_crossing_classes, location)
                # ped_vector_list = self.ped_geoms_to_vectors(ped_geom)
                ped_instance_list = self.ped_poly_geoms_to_instances(ped_geom)
                # import pdb;pdb.set_trace()
                for instance in ped_instance_list:
                    vectors.append((instance, self.CLASS2LABEL.get('ped_crossing', -1)))
            elif vec_class == 'boundary':
                polygon_geom = self.get_map_geom(patch_box, patch_angle, self.polygon_classes, location)
                # import pdb;pdb.set_trace()
                poly_bound_list = self.poly_geoms_to_instances(polygon_geom)
                # import pdb;pdb.set_trace()
                for contour in poly_bound_list:
                    vectors.append((contour, self.CLASS2LABEL.get('contours', -1)))
            else:
                raise ValueError(f'WRONG vec_class: {vec_class}')

        # filter out -1
        filtered_vectors = []
        gt_pts_loc_3d = []
        gt_pts_num_3d = []
        gt_labels = []
        gt_instance = []
        for instance, type in vectors:
            if type != -1:
                gt_instance.append(instance)
                gt_labels.append(type)
        
        gt_instance = LiDARInstanceLines(gt_instance,self.sample_dist,
                        self.num_samples, self.padding, self.fixed_num, self.padding_value, patch_size=self.patch_size)

        anns_results = dict(
            gt_vecs_pts_loc=gt_instance,
            gt_vecs_label=gt_labels,

        )
        return anns_results

    def get_map_geom(self, patch_box, patch_angle, layer_names, location):
        map_geom = []
        for layer_name in layer_names:
            if layer_name in self.line_classes:
                # import pdb;pdb.set_trace()
                geoms = self.get_divider_line(patch_box, patch_angle, layer_name, location)
                # import pdb;pdb.set_trace()
                # geoms = self.map_explorer[location]._get_layer_line(patch_box, patch_angle, layer_name)
                map_geom.append((layer_name, geoms))
            elif layer_name in self.polygon_classes:
                geoms = self.get_contour_line(patch_box, patch_angle, layer_name, location)
                # geoms = self.map_explorer[location]._get_layer_polygon(patch_box, patch_angle, layer_name)
                map_geom.append((layer_name, geoms))
            elif layer_name in self.ped_crossing_classes:
                geoms = self.get_ped_crossing_line(patch_box, patch_angle, location)
                # geoms = self.map_explorer[location]._get_layer_polygon(patch_box, patch_angle, layer_name)
                map_geom.append((layer_name, geoms))
        return map_geom

    def _one_type_line_geom_to_vectors(self, line_geom):
        line_vectors = []
        
        for line in line_geom:
            if not line.is_empty:
                if line.geom_type == 'MultiLineString':
                    for single_line in line.geoms:
                        line_vectors.append(self.sample_pts_from_line(single_line))
                elif line.geom_type == 'LineString':
                    line_vectors.append(self.sample_pts_from_line(line))
                else:
                    raise NotImplementedError
        return line_vectors

    def _one_type_line_geom_to_instances(self, line_geom):
        line_instances = []
        
        for line in line_geom:
            if not line.is_empty:
                if line.geom_type == 'MultiLineString':
                    for single_line in line.geoms:
                        line_instances.append(single_line)
                elif line.geom_type == 'LineString':
                    line_instances.append(line)
                else:
                    raise NotImplementedError
        return line_instances

    def poly_geoms_to_vectors(self, polygon_geom):
        roads = polygon_geom[0][1]
        lanes = polygon_geom[1][1]
        union_roads = ops.unary_union(roads)
        union_lanes = ops.unary_union(lanes)
        union_segments = ops.unary_union([union_roads, union_lanes])
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        exteriors = []
        interiors = []
        if union_segments.geom_type != 'MultiPolygon':
            union_segments = MultiPolygon([union_segments])
        for poly in union_segments.geoms:
            exteriors.append(poly.exterior)
            for inter in poly.interiors:
                interiors.append(inter)

        results = []
        for ext in exteriors:
            if ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        for inter in interiors:
            if not inter.is_ccw:
                inter.coords = list(inter.coords)[::-1]
            lines = inter.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        return self._one_type_line_geom_to_vectors(results)

    def ped_poly_geoms_to_instances(self, ped_geom):
        # import pdb;pdb.set_trace()
        ped = ped_geom[0][1]
        union_segments = ops.unary_union(ped)
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        # local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        local_patch = box(-max_x - 0.2, -max_y - 0.2, max_x + 0.2, max_y + 0.2)
        exteriors = []
        interiors = []
        if union_segments.geom_type != 'MultiPolygon':
            union_segments = MultiPolygon([union_segments])
        for poly in union_segments.geoms:
            exteriors.append(poly.exterior)
            for inter in poly.interiors:
                interiors.append(inter)

        results = []
        for ext in exteriors:
            if ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        for inter in interiors:
            if not inter.is_ccw:
                inter.coords = list(inter.coords)[::-1]
            lines = inter.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        return self._one_type_line_geom_to_instances(results)


    def poly_geoms_to_instances(self, polygon_geom):
        roads = polygon_geom[0][1]
        lanes = polygon_geom[1][1]
        union_roads = ops.unary_union(roads)
        union_lanes = ops.unary_union(lanes)
        union_segments = ops.unary_union([union_roads, union_lanes])
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        exteriors = []
        interiors = []
        if union_segments.geom_type != 'MultiPolygon':
            union_segments = MultiPolygon([union_segments])
        for poly in union_segments.geoms:
            exteriors.append(poly.exterior)
            for inter in poly.interiors:
                interiors.append(inter)

        results = []
        for ext in exteriors:
            if ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        for inter in interiors:
            if not inter.is_ccw:
                inter.coords = list(inter.coords)[::-1]
            lines = inter.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        return self._one_type_line_geom_to_instances(results)

    def line_geoms_to_vectors(self, line_geom):
        line_vectors_dict = dict()
        for line_type, a_type_of_lines in line_geom:
            one_type_vectors = self._one_type_line_geom_to_vectors(a_type_of_lines)
            line_vectors_dict[line_type] = one_type_vectors

        return line_vectors_dict
    def line_geoms_to_instances(self, line_geom):
        line_instances_dict = dict()
        for line_type, a_type_of_lines in line_geom:
            one_type_instances = self._one_type_line_geom_to_instances(a_type_of_lines)
            line_instances_dict[line_type] = one_type_instances

        return line_instances_dict

    def ped_geoms_to_vectors(self, ped_geom):
        ped_geom = ped_geom[0][1]
        union_ped = ops.unary_union(ped_geom)
        if union_ped.geom_type != 'MultiPolygon':
            union_ped = MultiPolygon([union_ped])

        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        results = []
        for ped_poly in union_ped:
            # rect = ped_poly.minimum_rotated_rectangle
            ext = ped_poly.exterior
            if not ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            results.append(lines)

        return self._one_type_line_geom_to_vectors(results)

    def get_contour_line(self,patch_box,patch_angle,layer_name,location):
        # if layer_name not in self.map_explorer[location].map_api.non_geometric_polygon_layers:
        #     raise ValueError('{} is not a polygonal layer'.format(layer_name))

        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = self.map_explorer[location].get_patch_coord(patch_box, patch_angle)

        records = getattr(self.map_explorer[location].map_api, layer_name)

        polygon_list = []
        if layer_name == 'drivable_area':
            for record in records:
                polygons = [self.map_explorer[location].map_api.extract_polygon(polygon_token) for polygon_token in record['polygon_tokens']]

                for polygon in polygons:
                    new_polygon = polygon.intersection(patch)
                    if not new_polygon.is_empty:
                        new_polygon = affinity.rotate(new_polygon, -patch_angle,
                                                      origin=(patch_x, patch_y), use_radians=False)
                        new_polygon = affinity.affine_transform(new_polygon,
                                                                [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                        if new_polygon.geom_type == 'Polygon':
                            new_polygon = MultiPolygon([new_polygon])
                        polygon_list.append(new_polygon)

        else:
            for record in records:
                polygon = self.map_explorer[location].map_api.extract_polygon(record['polygon_token'])

                if polygon.is_valid:
                    new_polygon = polygon.intersection(patch)
                    if not new_polygon.is_empty:
                        new_polygon = affinity.rotate(new_polygon, -patch_angle,
                                                      origin=(patch_x, patch_y), use_radians=False)
                        new_polygon = affinity.affine_transform(new_polygon,
                                                                [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                        if new_polygon.geom_type == 'Polygon':
                            new_polygon = MultiPolygon([new_polygon])
                        polygon_list.append(new_polygon)

        return polygon_list

    def get_divider_line(self,patch_box,patch_angle,layer_name,location):
        # if layer_name not in self.map_explorer[location].map_api.non_geometric_line_layers:
        #     raise ValueError("{} is not a line layer".format(layer_name))

        if layer_name == 'traffic_light':
            return None

        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = self.map_explorer[location].get_patch_coord(patch_box, patch_angle)

        line_list = []
        records = getattr(self.map_explorer[location].map_api, layer_name)
        for record in records:
            line = self.map_explorer[location].map_api.extract_line(record['line_token'])
            if line.is_empty:  # Skip lines without nodes.
                continue

            new_line = line.intersection(patch)
            if not new_line.is_empty:
                new_line = affinity.rotate(new_line, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
                new_line = affinity.affine_transform(new_line,
                                                     [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                line_list.append(new_line)

        return line_list

    def get_ped_crossing_line(self, patch_box, patch_angle, location):
        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = self.map_explorer[location].get_patch_coord(patch_box, patch_angle)
        polygon_list = []
        records = getattr(self.map_explorer[location].map_api, 'ped_crossing')
        # records = getattr(self.nusc_maps[location], 'ped_crossing')
        for record in records:
            polygon = self.map_explorer[location].map_api.extract_polygon(record['polygon_token'])
            if polygon.is_valid:
                new_polygon = polygon.intersection(patch)
                if not new_polygon.is_empty:
                    new_polygon = affinity.rotate(new_polygon, -patch_angle,
                                                      origin=(patch_x, patch_y), use_radians=False)
                    new_polygon = affinity.affine_transform(new_polygon,
                                                            [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                    if new_polygon.geom_type == 'Polygon':
                        new_polygon = MultiPolygon([new_polygon])
                    polygon_list.append(new_polygon)

        return polygon_list

    def sample_pts_from_line(self, line):
        if self.fixed_num < 0:
            distances = np.arange(0, line.length, self.sample_dist)
            sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
        else:
            # fixed number of points, so distance is line.length / self.fixed_num
            distances = np.linspace(0, line.length, self.fixed_num)
            sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)

            # tmpdistances = np.linspace(0, line.length, 2)
            # tmpsampled_points = np.array([list(line.interpolate(tmpdistance).coords) for tmpdistance in tmpdistances]).reshape(-1, 2)
        # import pdb;pdb.set_trace()
        # if self.normalize:
        #     sampled_points = sampled_points / np.array([self.patch_size[1], self.patch_size[0]])

        num_valid = len(sampled_points)

        if not self.padding or self.fixed_num > 0:
            # fixed num sample can return now!
            return sampled_points, num_valid

        # fixed distance sampling need padding!
        num_valid = len(sampled_points)

        if self.fixed_num < 0:
            if num_valid < self.num_samples:
                padding = np.zeros((self.num_samples - len(sampled_points), 2))
                sampled_points = np.concatenate([sampled_points, padding], axis=0)
            else:
                sampled_points = sampled_points[:self.num_samples, :]
                num_valid = self.num_samples

            # if self.normalize:
            #     sampled_points = sampled_points / np.array([self.patch_size[1], self.patch_size[0]])
            #     num_valid = len(sampled_points)

        return sampled_points, num_valid

def map_viz(frame, ax, vector_maps):

    location = frame['map_location']

    vector_map = vector_maps[location]

    lidar2ego = np.eye(4)
    lidar2ego[:3,:3] = Quaternion(frame['lidar2ego_rotation']).rotation_matrix
    lidar2ego[:3, 3] = frame['lidar2ego_translation']
    ego2global = np.eye(4)
    ego2global[:3,:3] = Quaternion(frame['ego2global_rotation']).rotation_matrix
    ego2global[:3, 3] = frame['ego2global_translation']
    lidar2global = ego2global @ lidar2ego
    lidar2global_translation = list(lidar2global[:3,3])
    lidar2global_rotation = list(Quaternion(matrix=lidar2global).q)
    location = frame['map_location']
    ego2global_translation = frame['ego2global_translation']
    ego2global_rotation = frame['ego2global_rotation']
    anns_results = vector_map.gen_vectorized_samples(location, lidar2global_translation, lidar2global_rotation)
    gt_vecs_label = to_tensor(anns_results['gt_vecs_label'])
    
    if isinstance(anns_results['gt_vecs_pts_loc'], LiDARInstanceLines):
        gt_vecs_pts_loc = anns_results['gt_vecs_pts_loc']
    else:
        gt_vecs_pts_loc = to_tensor(anns_results['gt_vecs_pts_loc'])
        try:
            gt_vecs_pts_loc = gt_vecs_pts_loc.flatten(1).to(dtype=torch.float32)
        except:
            # empty tensor, will be passed in train, 
            # but we preserve it for test
            gt_vecs_pts_loc = gt_vecs_pts_loc
    
    lidar_instance_lines: LiDARInstanceLines = gt_vecs_pts_loc
    lidar_instance_labels = gt_vecs_label
    points_tensor = lidar_instance_lines.fixed_num_sampled_points  # Shape: [N, fixed_num, 2]
    points_numpy = points_tensor.numpy()
    

    labels_seen = set()
    unique_labels = np.unique(lidar_instance_labels)
    colors = cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    color_dict = dict(zip(unique_labels, colors))
    
    for i in range(points_numpy.shape[0]):
        x = points_numpy[i, :, 0]
        y = points_numpy[i, :, 1]
        label = MAPCLASSES[lidar_instance_labels[i].item()]
        # Plot the line
        color = color_dict[lidar_instance_labels[i].item()]
        ax.plot(x, y, marker='o', markersize=2, color=color, label=label if label not in labels_seen else "")
        labels_seen.add(label)
    
    return ax
        
import matplotlib.patches as patches
from scipy.spatial.transform import Rotation as R
import math

def create_bev_bbox(center, length, width):
    x_c, y_c = center
    half_l = length / 2
    half_w = width / 2
    vertices = [
        (x_c - half_l, y_c - half_w),  # 左下角
        (x_c + half_l, y_c - half_w),  # 右下角
        (x_c + half_l, y_c + half_w),  # 右上角
        (x_c - half_l, y_c + half_w),  # 左上角
        (x_c - half_l, y_c - half_w)   # 闭合多边形
    ]
    return vertices

def rotate_bbox(vertices, angle_rad, center):
    """
    Rotate bounding box vertices around a center point by a given angle,
    adjusting for yaw and coordinate system conventions.
    """
    # Adjust angle for clockwise rotation if necessary
    angle_rad = -angle_rad  # Uncomment if positive yaw is clockwise

    # Convert inputs to numpy arrays
    vertices = np.array(vertices)
    center = np.array(center)

    # Create rotation matrix (adjust if necessary)
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    rotation_matrix = np.array([
        [cos_theta, -sin_theta],
        [sin_theta,  cos_theta]
    ])

    # Translate vertices to origin (center the rotation)
    translated_vertices = vertices - center

    # Rotate vertices
    rotated_vertices = np.dot(translated_vertices, rotation_matrix.T)

    # Translate vertices back to original center
    rotated_vertices += center

    return rotated_vertices


def bev_viz(frame, class_to_vis, ax):
    token = frame['token']
    next_token = frame['token']
    prev_token = frame['token']
    timestamp = frame['timestamp']
    ego2global_translation = frame['ego2global_translation']
    ego2global_rotation = frame['ego2global_rotation']
    obj_list = frame['gt_boxes']    # x,y,z,w,l,h,yaw
    obj_class_list = frame['gt_names']
    gt_agent_fut_trajs = frame['gt_agent_fut_trajs']
    gt_ego_fut_trajs = frame['gt_ego_fut_trajs']
    ego_l = 4.6
    ego_w = 1.8
    ego2global_rpy = R.from_quat([ego2global_rotation[1], 
                        ego2global_rotation[2],
                        ego2global_rotation[3],
                        ego2global_rotation[0]]).as_euler('xyz') # ego to global rotation quat -> roll pitch yaw
    
    ego2global_yaw = ego2global_rpy[-1]
    ego_bbox = create_bev_bbox((0,0), ego_l, ego_w)  # ego car size: l=4.6 w=1.8
    ego_polygon = patches.Polygon(ego_bbox, closed=True, edgecolor='red', facecolor='none', linewidth=2)
    ax.add_patch(ego_polygon)
    # gt_ego_fut_trajs = rotate_traj(gt_ego_fut_trajs, ego2global_rpy[2], (0,0))
    gt_ego_fut_trajs = copy.deepcopy(gt_ego_fut_trajs)
    for i in range(1, len(gt_ego_fut_trajs)):
        gt_ego_fut_trajs[i, :] += gt_ego_fut_trajs[i-1, :]

    gt_ego_fut_trajs = np.vstack(([0, 0], gt_ego_fut_trajs))
    ax.plot(gt_ego_fut_trajs[:, 0], gt_ego_fut_trajs[:, 1], 'r-', linewidth=1) # plot ego trajectories
    ax.plot(gt_ego_fut_trajs[:, 0], gt_ego_fut_trajs[:, 1], 'ro', markersize=3) # plot ego trajectories


    for i, obj in enumerate(obj_list):
        # if obj_class_list[i] == "vehicle":
        # plot obj bbox
        obj_x = obj[0] # - ego2global_translation[0]
        obj_y = obj[1] # - ego2global_translation[1]
        obj_w = obj[3]
        obj_l = obj[4]
        obj_yaw = obj[-1]
        vertices = create_bev_bbox((obj_x, obj_y), obj_l, obj_w)
        bbox = rotate_bbox(vertices, obj_yaw, (obj_x, obj_y))
        obj_polygon = patches.Polygon(bbox, closed=True, edgecolor='blue', facecolor='none', linewidth=2)
        ax.add_patch(obj_polygon)
        # plot obj traj
        obj_fut_traj = copy.deepcopy(gt_agent_fut_trajs[i].reshape(-1, 2))
        # obj_fut_traj += (obj_x, obj_y)
        obj_fut_traj = np.vstack((frame['gt_boxes'][i][:2], obj_fut_traj))
        for i in range(1, len(obj_fut_traj)):
            obj_fut_traj[i, :] += obj_fut_traj[i - 1, :]

        
        ax.plot(obj_fut_traj[:, 0], obj_fut_traj[:, 1], 'b-', markersize=1)
        ax.plot(obj_fut_traj[:, 0], obj_fut_traj[:, 1], 'bo', markersize=3)
        
    return ax
    
def quaternion_to_yaw(q):
    """
    Convert a quaternion to a yaw angle (rotation around Z-axis).
    q: array-like of shape (4,), representing [w, x, y, z]
    """
    w, x, y, z = q
    # Compute yaw angle
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y**2 + z**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return yaw

kwargs = {
    "data_root":'/home/taoran.lu/E2E/VAD/data/nuplan/', #'/home/shaoze.yang/code/Bench2DriveZoo/data/nuscenes/', 
}
MAPCLASSES = ['divider', 'ped_crossing', 'boundary']
DETCLASS = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone', 
    'generic_object', 'czone_sign', 'vehicle', 
]

# pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
pc_range = [-28, -21, -5, 28, 21, 3]
patch_h = pc_range[4]-pc_range[1]
patch_w = pc_range[3]-pc_range[0]
patch_size = (patch_h, patch_w)
map_fixed_ptsnum_per_line = 20 # now only support fixed_pts > 0
padding_value = -10000

class MapVisualizer:
    def __init__(self, vector_maps):
        self.vector_maps = vector_maps
        self.MAPCLASSES = ['divider', 'ped_crossing', 'boundary']
        
    def visualize_map(self, frame, ax, alpha=1.0):
        location = frame['map_location']
        vector_map = self.vector_maps[location]
        
        lidar2global = self._get_lidar2global_matrix(frame)
        lidar2global_translation = list(lidar2global[:3,3])
        lidar2global_rotation = list(Quaternion(matrix=lidar2global).q)
        
        anns_results = vector_map.gen_vectorized_samples(location, lidar2global_translation, lidar2global_rotation)
        
        self._plot_map_elements(ax, anns_results, alpha)
        
        return ax
    
    def _get_lidar2global_matrix(self, frame):
        lidar2ego = np.eye(4)
        lidar2ego[:3,:3] = Quaternion(frame['lidar2ego_rotation']).rotation_matrix
        lidar2ego[:3, 3] = frame['lidar2ego_translation']
        ego2global = np.eye(4)
        ego2global[:3,:3] = Quaternion(frame['ego2global_rotation']).rotation_matrix
        ego2global[:3, 3] = frame['ego2global_translation']
        return ego2global @ lidar2ego
    
    def _plot_map_elements(self, ax, anns_results, alpha):
        gt_vecs_label = anns_results['gt_vecs_label']
        gt_vecs_pts_loc = anns_results['gt_vecs_pts_loc']
        
        points_tensor = gt_vecs_pts_loc.fixed_num_sampled_points
        points_numpy = points_tensor.numpy()
        
        labels_seen = set()
        unique_labels = np.unique(gt_vecs_label)
        colors = cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        color_dict = dict(zip(unique_labels, colors))
        
        for i in range(points_numpy.shape[0]):
            x = points_numpy[i, :, 0]
            y = points_numpy[i, :, 1]
            label = self.MAPCLASSES[gt_vecs_label[i]]
            color = color_dict[gt_vecs_label[i]]
            ax.plot(x, y, marker='o', markersize=2, color=color, label=label if label not in labels_seen else "", alpha=alpha)
            labels_seen.add(label)

class BEVVisualizer:
    def __init__(self):
        self.DETCLASS = [
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
            'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone', 
            'generic_object', 'czone_sign', 'vehicle'
        ]
    
    def visualize_bev(self, frame, ax, alpha=1.0):
        self._plot_ego_vehicle(frame, ax, alpha=1.0)
        self._plot_objects(frame, ax, alpha)
        return ax
    
    def _plot_ego_vehicle(self, frame, ax, alpha=1.0):
        ego_l, ego_w = 4.6, 1.8
        ego_bbox = create_bev_bbox((0,0), ego_l, ego_w)
        ego_polygon = patches.Polygon(ego_bbox, closed=True, edgecolor='red', facecolor='none', linewidth=2, alpha=alpha)
        ax.add_patch(ego_polygon)
        
        gt_ego_fut_trajs = self._process_ego_trajectory(frame)
        ax.plot(gt_ego_fut_trajs[:, 0], gt_ego_fut_trajs[:, 1], 'r-', linewidth=1, alpha=alpha)
        ax.plot(gt_ego_fut_trajs[:, 0], gt_ego_fut_trajs[:, 1], 'ro', markersize=3, alpha=alpha)
    
    def _plot_objects(self, frame, ax, alpha=1.0):
        for i, obj in enumerate(frame['gt_boxes']):
            obj_x, obj_y = obj[0], obj[1]
            obj_w, obj_l = obj[3], obj[4]
            obj_yaw = obj[-1]
            
            vertices = create_bev_bbox((obj_x, obj_y), obj_l, obj_w)
            bbox = rotate_bbox(vertices, obj_yaw, (obj_x, obj_y))
            obj_polygon = patches.Polygon(bbox, closed=True, edgecolor='blue', facecolor='none', linewidth=2, alpha=alpha)
            ax.add_patch(obj_polygon)

            obj_fut_traj = self._process_object_trajectory(frame, i)
            ax.plot(obj_fut_traj[:, 0], obj_fut_traj[:, 1], 'b-', markersize=1, alpha=alpha)
            ax.plot(obj_fut_traj[:, 0], obj_fut_traj[:, 1], 'bo', markersize=3, alpha=alpha)
    
    def _process_ego_trajectory(self, frame):
        gt_ego_fut_trajs = copy.deepcopy(frame['gt_ego_fut_trajs'])
        for i in range(1, len(gt_ego_fut_trajs)):
            gt_ego_fut_trajs[i, :] += gt_ego_fut_trajs[i-1, :]
        return np.vstack(([0, 0], gt_ego_fut_trajs))
    
    def _process_object_trajectory(self, frame, obj_index):
        obj_fut_traj = copy.deepcopy(frame['gt_agent_fut_trajs'][obj_index].reshape(-1, 2))
        obj_fut_traj = np.vstack((frame['gt_boxes'][obj_index][:2], obj_fut_traj))
        for i in range(1, len(obj_fut_traj)):
            obj_fut_traj[i, :] += obj_fut_traj[i - 1, :]
        return obj_fut_traj

class VisualizationManager:
    def __init__(self, data_root):
        self.vector_maps = self._initialize_vector_maps(data_root)
        self.map_visualizer = MapVisualizer(self.vector_maps)
        self.bev_visualizer = BEVVisualizer()
    
    def _initialize_vector_maps(self, data_root):
        locations = ['us-ma-boston', 'us-nv-las-vegas-strip', 'sg-one-north', 'us-pa-pittsburgh-hazelwood']
        vector_maps = {}
        for loc in locations:
            vector_map = VectorizedLocalMap(data_root, 
                            patch_size=patch_size, map_classes=MAPCLASSES, 
                            fixed_ptsnum_per_line=map_fixed_ptsnum_per_line,
                            padding_value=padding_value, 
                                MAPS=[loc])
            vector_maps[loc] = vector_map
        return vector_maps
    
    def visualize_frame(self, frame, pred_result=None, pred_plan_result=None, pred_map_result=None):
        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_axes([0, 0, 1, 1])  # 创建一个占满整个图形的坐标轴
        
        ax = self.map_visualizer.visualize_map(frame, ax, alpha=0.3)
        ax = self.bev_visualizer.visualize_bev(frame, ax, alpha=0.3)
        
        if pred_result is not None:
            ax = self._visualize_predictions(frame, pred_result, pred_plan_result, pred_map_result, ax)
        
        ax.set_xlim(-40, 40)
        ax.set_ylim(-20, 25)
        ax.set_aspect('equal')
        plt.grid(True)
        # plt.legend()

        # 移除坐标轴
        ax.axis('off')
        plt.margins(0, 0)
        
        return fig, ax
    
    def _visualize_predictions(self, frame, pred_result, pred_plan_result, pred_map_result, ax):
        self._visualize_predicted_map(pred_map_result, ax)
        self._visualize_predicted_objects(pred_result, ax)
        self._visualize_predicted_plan(pred_plan_result, ax)
        return ax

    def _visualize_predicted_map(self, pred_map_result, ax):
        color = {'divider': 'purple', 'ped_crossing': 'green', 'boundary': 'red'}
        labels_seen = set()

        if pred_map_result is not None:
            for i in range(len(pred_map_result)):
                pts = pred_map_result[i]['pts']
                cls_name = pred_map_result[i]['cls_name']
                confidence_level = pred_map_result[i]['confidence_level']
                if confidence_level > 0.4:
                    x, y = zip(*pts)
                    ax.plot(x, y, color=color[cls_name], linewidth=1, 
                            label=f"{cls_name}_pred" if cls_name not in labels_seen else "")
                    ax.plot(x, y, color=color[cls_name], marker='x', markersize=2, linestyle='')
                    labels_seen.add(cls_name)

    def _visualize_predicted_objects(self, pred_result, ax):
        for pred in pred_result:
            obj_x = pred['translation'][0]
            obj_y = pred['translation'][1]
            obj_w = pred['size'][0]
            obj_l = pred['size'][1]
            obj_yaw = quaternion_to_yaw(pred['rotation'])
            vertices = create_bev_bbox((obj_x, obj_y), obj_w, obj_l)
            bbox = rotate_bbox(vertices, obj_yaw, (obj_x, obj_y))
            obj_polygon = patches.Polygon(bbox, closed=True, edgecolor='green', facecolor='none', linewidth=2)
            ax.add_patch(obj_polygon)

            pred_obj_fut_traj = copy.deepcopy(np.array(pred['fut_traj'][0]).reshape(6, 2))
            pred_obj_fut_traj = np.vstack((pred['translation'][:2], pred_obj_fut_traj))
            for i in range(1, len(pred_obj_fut_traj)):
                pred_obj_fut_traj[i, :] += pred_obj_fut_traj[i - 1, :]
            ax.plot(pred_obj_fut_traj[:, 0], pred_obj_fut_traj[:, 1], 'g-', linewidth=1)
            ax.plot(pred_obj_fut_traj[:, 0], pred_obj_fut_traj[:, 1], 'go', markersize=1)

    # def _visualize_predicted_plan(self, pred_plan_result, ax):
    #     # for VADv1
    #     plan_cmd = np.argmax(pred_plan_result[1][0, 0, 0])
    #     plan_traj = copy.deepcopy(pred_plan_result[0][plan_cmd])
    #     plan_traj = np.vstack(([0, 0], plan_traj))
    #     plan_traj[abs(plan_traj) < 0.01] = 0.0
    #     plan_traj = plan_traj.cumsum(axis=0)

    #     ax.plot(plan_traj[:, 0], plan_traj[:, 1], 'g-', linewidth=1)
    #     ax.plot(plan_traj[:, 0], plan_traj[:, 1], 'go', markersize=3)

    def _visualize_predicted_plan(self, pred_plan_result, ax):
        """
        For VADv2!!!
        """
        ego_fut_preds, ego_cls_expert_preds, ego_fut_preds_all, ego_fut_cmd = pred_plan_result
        
        # Get scores and sort them
        scores = ego_cls_expert_preds.squeeze()  # Shape: [4096]
        top_k = 80  # Number of trajectories to visualize
        down_sample_rate = 2
        
        # Get indices of top k scores
        top_k_indices = torch.argsort(scores, descending=True)[::down_sample_rate][:top_k]
        top_k_scores = scores[top_k_indices]
        
        # Normalize scores for color mapping
        normalized_scores = (top_k_scores - top_k_scores.min()) / (top_k_scores.max() - top_k_scores.min())
        
        # Create colormap
        # cmap = plt.cm.get_cmap('YlOrRd')
        # cmap = plt.colormaps['YlOrRd'].reversed()
        cmap = plt.colormaps['YlOrRd']
        
        # Plot trajectories
        # reverse:
        top_k_indices_rev = reversed(top_k_indices)
        normalized_scores_rev = reversed(normalized_scores)
        for idx, (traj_idx, norm_score) in enumerate(zip(top_k_indices_rev, normalized_scores_rev)):
            # Get trajectory and convert to numpy
            traj = ego_fut_preds_all[traj_idx].numpy()
            
            # Accumulate positions
            plan_traj = np.vstack(([0, 0], traj))
            # plan_traj = plan_traj.cumsum(axis=0) # no need to cumsum for VADv2
            
            # Plot with color based on score
            color = cmap(float(norm_score))
            # alpha = 0.1 + 0.9 * float(norm_score)  # Higher scores get higher opacity
            alpha = 0.8

            # Plot line
            ax.plot(plan_traj[:, 0], plan_traj[:, 1], '-', 
                color=color, linewidth=2, alpha=alpha)
            
            # # Plot points
            # ax.plot(plan_traj[:, 0], plan_traj[:, 1], 'o', 
            #     color=color, markersize=2, alpha=alpha)
        
        # Plot the highest scoring trajectory with a different style
        best_traj = ego_fut_preds_all[top_k_indices[0]].numpy()
        best_plan_traj = np.vstack(([0, 0], best_traj))
        # best_plan_traj = best_plan_traj.cumsum(axis=0)
        
        # Plot best trajectory with distinct style
        ax.plot(best_plan_traj[:, 0], best_plan_traj[:, 1], 'r-', 
            linewidth=2, label='Best Trajectory')
        ax.plot(best_plan_traj[:, 0], best_plan_traj[:, 1], 'ro', 
            markersize=3)


    def visualize_frame_sequence(self, frames, pred_data, num_frames=10):
        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_axes([0, 0, 1, 1])
        
        def update(frame_idx):
            ax.clear()
            frame = frames[frame_idx]
            gt_token = frame['token']
            pred_result = pred_data['results'][gt_token]
            pred_plan_result = pred_data['plan_results'][gt_token]
            pred_map_result = pred_data['map_results'][gt_token]['vectors']
            
            self.map_visualizer.visualize_map(frame, ax, alpha=0.3)
            self.bev_visualizer.visualize_bev(frame, ax, alpha=0.3)
            self._visualize_predictions(frame, pred_result, pred_plan_result, pred_map_result, ax)
            
            ax.set_xlim(-40, 40)
            ax.set_ylim(-20, 25)
            ax.set_aspect('equal')
            ax.grid(True)
            # ax.legend()
            ax.axis('off')
        
        anim = animation.FuncAnimation(fig, update, frames=tqdm(range(num_frames), desc="生成动画帧"), interval=500)
        return anim


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize BEV data')
    parser.add_argument('--data_root', type=str, default='/data/ceph/data/nuplan/dataset/', help='Root directory of the dataset')
    parser.add_argument('--ann_data_path', type=str, default='/workspace/lwad/tools/vis_tools/demo/gt_sampled_300.pkl', help='Path to annotation data')
    parser.add_argument('--pred_pickle_path', type=str, default='/workspace/lwad/test/VADv2_config_voca4096_h800_1127/Sat_Dec_14_02_41_41_2024/pts_bbox/results_nusc.pkl', help='Path to prediction data')
    parser.add_argument('--output_type', type=str, choices=['image', 'gif'], default='gif', help='Output type: image or gif')
    parser.add_argument('--output_dir', type=str, default='demo/', help='Output directory for visualizations')
    parser.add_argument('--num_frames', type=int, default=300, help='Number of frames to visualize')
    parser.add_argument('--gif_fps', type=int, default=2, help='Frames per second for GIF output')
    return parser.parse_args()

def main(args):
    # 初始化
    visualization_manager = VisualizationManager(args.data_root)
    
    # 加载数据
    with open(args.ann_data_path, 'rb') as f:
        data = pickle.load(f)
    infos, metadata = data['infos'], data['metadata']
    
    with open(args.pred_pickle_path, 'rb') as f:
        pred_data = pickle.load(f)
    
    from tqdm import tqdm
    if args.output_type == 'image':
        # 保存为图片
        for idx, frame in enumerate(tqdm(infos[:args.num_frames])):
            gt_token = frame['token']
            pred_result = pred_data['results'][gt_token]
            pred_plan_result = pred_data['plan_results'][gt_token]
            pred_map_result = pred_data['map_results'][gt_token]['vectors']

            fig, ax = visualization_manager.visualize_frame(frame, pred_result, pred_plan_result, pred_map_result)
            
            save_path = os.path.join(args.output_dir, gt_token)
            os.makedirs(save_path, exist_ok=True)
            fig.savefig(os.path.join(save_path, 'bev.png'))
            plt.close(fig)
    
    elif args.output_type == 'gif':
        # 保存为GIF
        frames_viz = infos[:args.num_frames]
        anim = visualization_manager.visualize_frame_sequence(frames_viz, pred_data, len(frames_viz))
        
        os.makedirs(args.output_dir, exist_ok=True)
        gif_path = os.path.join(args.output_dir, 'vis_bev.gif')
        anim.save(gif_path, writer='pillow', fps=args.gif_fps)
        print(f"GIF动画已保存为 '{gif_path}'")



if __name__ == "__main__":
    args = parse_args()
    main(args)