from functools import partial
import time
import numba
import torch
import numpy as np
from skimage import transform
import os, sys
import cv2
from ...utils import box_utils, common_utils
from ...ops import pypatchworkpp, pccluster


tv = None
try:
    import cumm.tensorview as tv
except:
    pass




class VoxelGeneratorWrapper():
    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels):
        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            self.spconv_ver = 1
        except:
            try:
                from spconv.utils import VoxelGenerator
                self.spconv_ver = 1
            except:
                from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
                self.spconv_ver = 2

        if self.spconv_ver == 1:
            self._voxel_generator = VoxelGenerator(
                voxel_size=vsize_xyz,
                point_cloud_range=coors_range_xyz,
                max_num_points=max_num_points_per_voxel,
                max_voxels=max_num_voxels
            )
        else:
            self._voxel_generator = VoxelGenerator(
                vsize_xyz=vsize_xyz,
                coors_range_xyz=coors_range_xyz,
                num_point_features=num_point_features,
                max_num_points_per_voxel=max_num_points_per_voxel,
                max_num_voxels=max_num_voxels
            )

    def generate(self, points):
        if self.spconv_ver == 1:
            voxel_output = self._voxel_generator.generate(points)
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output
        else:
            assert tv is not None, f"Unexpected error, library: 'cumm' wasn't imported properly."
            voxel_output = self._voxel_generator.point_to_voxel(tv.from_numpy(points))
            tv_voxels, tv_coordinates, tv_num_points = voxel_output
            # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
            voxels = tv_voxels.numpy()
            coordinates = tv_coordinates.numpy()
            num_points = tv_num_points.numpy()
        return voxels, coordinates, num_points


class DataProcessor(object):
    def __init__(self, processor_configs, point_cloud_range, training, num_point_features, local_rank=0):
        self.point_cloud_range = point_cloud_range
        self.training = training
        self.num_point_features = num_point_features
        self.mode = 'train' if training else 'test'
        self.grid_size = self.voxel_size = None
        self.restored_pillar_size = self.cluster_grid_size = None
        self.data_processor_queue = []
        self.data_processor_remove = []
        self.scale_xy = self.scale_y = None

        self.voxel_generator = None
        self.remove_ground_worker1 = None
        self.remove_ground_worker2 = None
        # self.remove_ground_worker3 = None
        self.cluster = None
        self.pc_dealer = None

        for cur_cfg in processor_configs:
            cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            if cur_cfg.NAME == 'remove_ground':
                self.data_processor_remove.append(cur_processor)
            else:
                self.data_processor_queue.append(cur_processor)

    def mask_points_and_boxes_outside_range(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.mask_points_and_boxes_outside_range, config=config)

        if data_dict.get('points', None) is not None:
            mask = common_utils.mask_points_by_range(data_dict['points'], self.point_cloud_range)
            data_dict['points'] = data_dict['points'][mask]

        if data_dict.get('gt_boxes', None) is not None and config.REMOVE_OUTSIDE_BOXES and self.training:
            mask = box_utils.mask_boxes_outside_range_numpy(
                data_dict['gt_boxes'], self.point_cloud_range, min_num_corners=config.get('min_num_corners', 1)
            )
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
        return data_dict

    def shuffle_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.shuffle_points, config=config)

        if config.SHUFFLE_ENABLED[self.mode]:
            points = data_dict['points']
            shuffle_idx = np.random.permutation(points.shape[0])
            points = points[shuffle_idx]
            data_dict['points'] = points

        return data_dict

    def transform_points_to_voxels_placeholder(self, data_dict=None, config=None):
        # just calculate grid size
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.transform_points_to_voxels_placeholder, config=config)
        
        return data_dict

    def transform_points_to_voxel(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            # just bind the config, we will create the VoxelGeneratorWrapper later,
            # to avoid pickling issues in multiprocess spawn
            return partial(self.transform_points_to_voxel, config=config)
        return data_dict

    def get_points_fenzu_mask(self, actual_num, max_num, axis=0):
        actual_num = np.expand_dims(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num_matrix = np.arange(max_num, dtype=np.int32).reshape(max_num_shape)
        paddings_indicator = actual_num > max_num_matrix
        zero_point_mask = np.logical_not(paddings_indicator)
        zero_point_mask = zero_point_mask.astype(np.int64) * (-np.sum(actual_num,axis=0)*10)
        actual_num = np.concatenate((np.array([[0]]),actual_num),axis=0)
        actual_num_cumsum = np.cumsum(actual_num,axis=0)#.view(len(actual_num))
        actual_num_cumsum = actual_num_cumsum[:-1,:]
        # actual_num_cumsum[2:,:] -= 1
        paddings_indicator = paddings_indicator.astype(np.int64)
        # paddings_indicator += zero_point_mask
        paddings_indicator = paddings_indicator  * actual_num_cumsum
        add_mask = np.arange(max_num, dtype=np.int32).reshape(1,-1).repeat(paddings_indicator.shape[0],axis=0)
        paddings_indicator += add_mask + zero_point_mask
        return paddings_indicator

    def get_points_coords_info(self,points,voxel_size,is_sorted=False):
        # points = torch.from_numpy(points)
        point_cloud_range = np.array(self.point_cloud_range)
        # voxel_size = torch.from_numpy(voxel_size)
        points_coords = np.floor((points[:, [0,1]] -  point_cloud_range[[0,1]]) / voxel_size[[0,1]]).astype(np.int32)
        merge_coords =  points_coords[:, 0] * self.scale_y + \
                        points_coords[:, 1]
        if is_sorted:
            sorted_coords = np.sort(merge_coords)                 #让点按顺序排好
            coords_indices = np.argsort(merge_coords)
            sorted_points = points[coords_indices]
            points_coords = points_coords[coords_indices]
        else: 
            sorted_points = points
            sorted_coords = merge_coords
        unq_coords, unq_inv, unq_cnt = np.unique(sorted_coords, return_index=False, return_inverse =True, return_counts=True,axis=0)
        return unq_coords, unq_inv, unq_cnt,sorted_points

    def resume_nog_points(self, nonground,unq_coords,voxel_size,points_fenzu):
        point_cloud_range = np.array(self.point_cloud_range)
        # voxel_size = torch.from_numpy(voxel_size)
        # nonground = torch.from_numpy(nonground)
        # print(nonground.shape)
        nog_coords = np.floor((nonground[:, [0,1]] - point_cloud_range[[0,1]]) / voxel_size[[0,1]]).astype(np.int32)
        nog_merge_coords =  nog_coords[:, 0] * self.scale_y + \
                        nog_coords[:, 1]
        nog_unq_coords, nog_unq_inv, nog_unq_cnt = np.unique(nog_merge_coords, return_index=False, return_inverse =True, return_counts=True,axis=0)
        has_points_mask = np.isin(unq_coords,nog_unq_coords)
        nog_points_in_src_inds = points_fenzu[has_points_mask]
        nog_points_in_src_inds = nog_points_in_src_inds.flatten()
        inds_positive_mask = nog_points_in_src_inds >= 0.0
        resume_src_points_inds = nog_points_in_src_inds[inds_positive_mask]
        return resume_src_points_inds

    def get_radius_points(self, points, ridius_range):
        pts_radius = np.linalg.norm(points[:, 0:2], axis=1)   # shape = (N,)
        radius_flag = np.logical_and(ridius_range[0] < pts_radius , pts_radius <= ridius_range[1])
        return points[radius_flag]


    def remove_ground(self, data_dict=None, config=None):
        if data_dict is None:
            self.is_remove_ground = config.ROMOVE
            cluster_grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.RESTORED_PILLAR_SIZE)
            self.cluster_grid_size = np.round(cluster_grid_size).astype(np.int64)
            self.restored_pillar_size = np.array(config.RESTORED_PILLAR_SIZE)
            self.scale_xy = cluster_grid_size[0] * cluster_grid_size[1]
            self.scale_y = cluster_grid_size[1]
            return partial(self.remove_ground, config=config)
        if self.remove_ground_worker1 is None:
            PatchworkParams1 = pypatchworkpp.Parameters()
            PatchworkParams1.RNR_intensity_thr = 0.0
            PatchworkParams1.max_range = 1.0
            PatchworkParams1.max_range = 60
            PatchworkParams1.enable_RNR = False
            self.remove_ground_worker1 = pypatchworkpp.patchworkpp(PatchworkParams1)

        if self.cluster is None:
            cluster_params = pccluster.Parameters()
            cluster_params.verbose = False
            self.cluster  =  pccluster.clusteringprocessor(cluster_params)
        
        if self.is_remove_ground:
            points = data_dict['points']
            pts_30 = self.get_radius_points(points,[0,30])
            pts_80 = self.get_radius_points(points,[30,80])
            point_inds = np.arange(
            (points.shape[0]),
            dtype=points.dtype).reshape(-1,1)
            points = np.concatenate((points,point_inds),axis=1)
            
            self.remove_ground_worker1.estimateGround(points)
            nonground   = self.remove_ground_worker1.getNonground()

            points = points[:,:-1]
            nonground = nonground[:,:-1]
            points_1 = points[points[:,2]>-1.0]
            nonground_1 = nonground[nonground[:,2]<=-1.0]
            nonground = np.concatenate((points_1,nonground_1),axis=0)
            data_dict['resume_points'] = points
            data_dict['nog_points'] = nonground

            
            self.cluster.process_cluster(nonground[:,0:3])
            clusted_cluster_all      = self.cluster.getCluster()
            nocluster_all   = self.cluster.getNoncluster()
            
            
            nonground_30 = self.get_radius_points(nonground,[0,30])
            unq_coords4, unq_inv4, unq_cnt4,pts_30 = self.get_points_coords_info(pts_30, self.restored_pillar_size ,True)
            max_num_points_in_pillar4 = np.max(unq_cnt4,axis=0)
            points_fenzu4 = self.get_points_fenzu_mask(unq_cnt4, max_num_points_in_pillar4, axis=0)
            resume_src_points_inds = self.resume_nog_points(nonground_30,unq_coords4,self.restored_pillar_size , points_fenzu4)
            nonground_30 = pts_30[resume_src_points_inds]
            nonground = np.concatenate((nonground_30,pts_80),axis=0)
            
            cluster_unq_coords = None
            if len(clusted_cluster_all) != 0:
                cluster_unq_coords, _, _ ,_ = self.get_points_coords_info(clusted_cluster_all,self.restored_pillar_size /4.0)
                data_dict['cluster_points'] = clusted_cluster_all
                data_dict['nocluster_points'] = nocluster_all
            else:
                data_dict['cluster_points'] = np.zeros((2,3))
                data_dict['nocluster_points'] = np.zeros((2,3))

            unq_coords, unq_inv, unq_cnt,points = self.get_points_coords_info(points, self.restored_pillar_size /4.0,True)
            max_num_points_in_pillar = np.max(unq_cnt,axis=0)
            points_fenzu = self.get_points_fenzu_mask(unq_cnt, max_num_points_in_pillar, axis=0)
            if cluster_unq_coords is not None:
                nog_unq_coords, _, _, _ = self.get_points_coords_info(nonground, self.restored_pillar_size /4.0)
                cluster_points_mask = np.isin(nog_unq_coords,cluster_unq_coords)   #length:nog_unq_coords
                no_cluster_mask = np.logical_not(cluster_points_mask)      #length:nog_unq_coords
                nog_nocluster_unq_coords = nog_unq_coords[no_cluster_mask]
                nog_nocluster_points_mask = np.isin(unq_coords,nog_nocluster_unq_coords)     #length:unq_coords
                nog_nocluster_inds = points_fenzu[nog_nocluster_points_mask]
                nog_nocluster_inds = nog_nocluster_inds.flatten()
                nog_nocluster_inds_positive_mask = nog_nocluster_inds >= 0.0
                nog_nocluster_src_points_inds = nog_nocluster_inds[nog_nocluster_inds_positive_mask]
                nonground = points[nog_nocluster_src_points_inds]      
            
            data_dict['points'] = nonground
        return data_dict


    def transform_points_to_voxels(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            # just bind the config, we will create the VoxelGeneratorWrapper later,
            # to avoid pickling issues in multiprocess spawn
            return partial(self.transform_points_to_voxels, config=config)

        if self.voxel_generator is None:
            self.voxel_generator = VoxelGeneratorWrapper(
                vsize_xyz=config.VOXEL_SIZE,
                coors_range_xyz=self.point_cloud_range,
                num_point_features=self.num_point_features,
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
            )

        points = data_dict['points']
        voxel_output = self.voxel_generator.generate(points)
        voxels, coordinates, num_points = voxel_output

        if not data_dict['use_lead_xyz']:
        # if data_dict.get('use_lead_xyz', False):
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)

        data_dict['voxels'] = voxels
        data_dict['voxel_coords'] = coordinates
        data_dict['voxel_num_points'] = num_points
        return data_dict

    def sample_points_by_voxels(self, data_dict=None, config=None, voxel_generator=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE

            if self.voxel_generator is None:
                voxel_generator = VoxelGeneratorWrapper(
                    vsize_xyz=config.VOXEL_SIZE,
                    coors_range_xyz=self.point_cloud_range,
                    num_point_features=self.num_point_features,
                    max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                    max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
                )

            return partial(self.sample_points_by_voxels, config=config)

        num_points = config.NUM_POINTS[self.mode]
        if num_points == -1:  # dynamic voxelization !
            return data_dict

        # voxelization
        data_dict = self.transform_points_to_voxels(data_dict, config)
        if config.get('SAMPLE_TYPE', 'raw') == 'mean_vfe':
            voxels = data_dict['voxels']
            voxel_num_points = data_dict['voxel_num_points']
            a = voxels.sum(axis=1)
            b = np.expand_dims(voxel_num_points, axis=1).repeat(voxels.shape[-1], axis=-1)
            points = a / b

        else: # defalt: 'raw'
            points = data_dict['voxels'][:,0] # remain only one point per voxel

        data_dict['points'] = points
        # sampling
        data_dict = self.sample_points(data_dict, config)
        data_dict.pop('voxels')
        data_dict.pop('voxel_coords')
        data_dict.pop('voxel_num_points')

        return data_dict

    def sample_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.sample_points, config=config)

        num_points = config.NUM_POINTS[self.mode]
        if num_points == -1:
            return data_dict

        points = data_dict['points']
        if num_points < len(points):
            pts_depth = np.linalg.norm(points[:, 0:3], axis=1)
            pts_near_flag = pts_depth < 40.0
            far_idxs_choice = np.where(pts_near_flag == 0)[0]
            near_idxs = np.where(pts_near_flag == 1)[0]
            choice = []
            if num_points > len(far_idxs_choice):
                near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
                choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                    if len(far_idxs_choice) > 0 else near_idxs_choice
            else: 
                choice = np.arange(0, len(points), dtype=np.int32)
                choice = np.random.choice(choice, num_points, replace=False)
            np.random.shuffle(choice)
        else:
            choice = np.arange(0, len(points), dtype=np.int32)
            if num_points > len(points):
                extra_choice = np.random.choice(choice, num_points - len(points))
                choice = np.concatenate((choice, extra_choice), axis=0)
            np.random.shuffle(choice)
        data_dict['points'] = points[choice]
        return data_dict

    def calculate_grid_size(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.calculate_grid_size, config=config)
        return data_dict

    def downsample_depth_map(self, data_dict=None, config=None):
        if data_dict is None:
            self.depth_downsample_factor = config.DOWNSAMPLE_FACTOR
            return partial(self.downsample_depth_map, config=config)

        data_dict['depth_maps'] = transform.downscale_local_mean(
            image=data_dict['depth_maps'],
            factors=(self.depth_downsample_factor, self.depth_downsample_factor)
        )
        return data_dict

    def forward(self, data_dict,is_remove=False):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        if is_remove and len(self.data_processor_remove) != 0:
            data_dict = self.data_processor_remove[0](data_dict=data_dict)
        elif is_remove and len(self.data_processor_remove) == 0:
            pass
        else:
            for cur_processor in self.data_processor_queue:
                data_dict = cur_processor(data_dict=data_dict)

        return data_dict
