import os
import torch
import torch.optim as optim
import numpy as np
import random
import torch.nn.functional as F
import argparse
import shutil
import json
import cv2
import open3d as o3d

from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from model.scene_rep import JointEncoding
from model.keyframe import KeyFrameDatabase
from datasets.dataset import get_dataset
from utils import coordinates, extract_mesh, colormap_image
from tools.eval_ate import pose_evaluation
from optimization.utils import at_to_transform_matrix, qt_to_transform_matrix, matrix_to_axis_angle, matrix_to_quaternion

# 注意: 导入 torchvision.models.detection 以使用 maskrcnn_resnet50_fpn
from torchvision import models, transforms as T

class CoSLAM():
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = get_dataset(config)
        self.intrinsics = [self.config['cam']['fx'], self.config['cam']['fy'],
                           self.config['cam']['cx'], self.config['cam']['cy']]

        self.create_bounds()
        self.create_pose_data()
        self.get_pose_representation()
        self.keyframeDatabase = self.create_kf_database(config)
        self.model = JointEncoding(config, self.bounding_box).to(self.device)

        # 初始化 Mask R-CNN 模型（用于检测人、猫、狗）
        self.mask_rcnn_model = models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(self.device)
        self.mask_rcnn_model.eval()

        # 为了让结果可复现（可选）
        self.seed_everything(config['seed'] if 'seed' in config else 42)

    def seed_everything(self, seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def get_pose_representation(self):
        if self.config['training']['rot_rep'] == 'axis_angle':
            self.matrix_to_tensor = matrix_to_axis_angle
            self.matrix_from_tensor = at_to_transform_matrix
            print('使用轴角表示旋转')
        elif self.config['training']['rot_rep'] == "quat":
            print("使用四元数表示旋转")
            self.matrix_to_tensor = matrix_to_quaternion
            self.matrix_from_tensor = qt_to_transform_matrix
        else:
            raise NotImplementedError

    def create_pose_data(self):
        self.est_c2w_data = {}
        self.est_c2w_data_rel = {}
        self.load_gt_pose()

    def create_bounds(self):
        self.bounding_box = torch.from_numpy(np.array(self.config['mapping']['bound'])).to(torch.float32).to(self.device)
        self.marching_cube_bound = torch.from_numpy(np.array(self.config['mapping']['marching_cubes_bound'])).to(torch.float32).to(self.device)

    def create_kf_database(self, config):
        num_kf = int(self.dataset.num_frames // self.config['mapping']['keyframe_every'] + 1)
        print('#关键帧数量:', num_kf)
        print('#保存的像素数量:', self.dataset.num_rays_to_save)
        return KeyFrameDatabase(config,
                                self.dataset.H,
                                self.dataset.W,
                                num_kf,
                                self.dataset.num_rays_to_save,
                                self.device)

    def load_gt_pose(self):
        self.pose_gt = {}
        for i, pose in enumerate(self.dataset.poses):
            self.pose_gt[i] = pose

    def save_state_dict(self, save_path):
        torch.save(self.model.state_dict(), save_path)

    def load(self, load_path):
        self.model.load_state_dict(torch.load(load_path))

    def save_ckpt(self, save_path):
        save_dict = {'pose': self.est_c2w_data,
                     'pose_rel': self.est_c2w_data_rel,
                     'model': self.model.state_dict()}
        torch.save(save_dict, save_path)
        print('已保存检查点')

    def load_ckpt(self, load_path):
        dict_ckpt = torch.load(load_path)
        self.model.load_state_dict(dict_ckpt['model'])
        self.est_c2w_data = dict_ckpt['pose']
        self.est_c2w_data_rel = dict_ckpt['pose_rel']

    # =========== 下面是新的使用 Mask R-CNN 来获取动态掩码的逻辑 ===========
    def compute_maskrcnn_dynamic_mask(self, rgb_frame_np):
        """
        使用 Mask R-CNN 检测图片中的 person/cat/dog，对这些区域生成掩码。
        - rgb_frame_np: numpy array, shape [H, W, 3], 0~255 (BGR 转 RGB 后也可)
        返回：bool 型数组 (H, W)，True 表示“动态物体区域”。
        """
        # torchvision detection 模型默认使用 [C,H,W] 归一化到 [0,1] 范围
        transform = T.ToTensor()
        # 如果传进来的 rgb_frame_np 是 BGR，需要转换为 RGB，也要确保 dtype=float 或 uint8
        # 这里假设已经是 RGB，如果是 BGR 可用: rgb_frame_np = cv2.cvtColor(rgb_frame_np, cv2.COLOR_BGR2RGB)
        image_tensor = transform(rgb_frame_np).to(self.device)  # shape=[3, H, W], range=[0,1]

        with torch.no_grad():
            predictions = self.mask_rcnn_model([image_tensor])[0]  # 只输入单张图

        labels = predictions['labels'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()
        masks = predictions['masks'].cpu().numpy()  # shape=[N, 1, H, W]

        # COCO: person=1, cat=17, dog=18
        dynamic_classes = [1, 17, 18]
        dynamic_mask = np.zeros((rgb_frame_np.shape[0], rgb_frame_np.shape[1]), dtype=np.bool_)

        for i in range(len(labels)):
            label_i = labels[i]
            score_i = scores[i]
            if (label_i in dynamic_classes) and (score_i > 0.5):
                # 当前预测的 mask
                mask_i = masks[i, 0] > 0.5  # shape=[H, W]
                dynamic_mask = np.logical_or(dynamic_mask, mask_i)

        return dynamic_mask

    def select_samples(self, batch, prev_batch, samples, frame_id):
        """
        在这里，我们只根据 Mask R-CNN 的检测结果来判断动态物体。
        任何属于动态物体（人、猫、狗）的像素都不采样。
        """
        # 不用 prev_batch 做 ICP 或光流比较，彻底改为 Mask R-CNN
        dynamic_mask = self.compute_dynamic_mask(batch)
        static_mask = ~dynamic_mask
        static_indices = np.where(static_mask.flatten())[0]

        if len(static_indices) == 0:
            # 如果整帧都检测成动态（极端情况），那就随机采一些，以免全部跳过
            indice = random.sample(range(self.dataset.H * self.dataset.W), int(samples))
        else:
            if len(static_indices) < samples:
                samples = len(static_indices)
            indice = np.random.choice(static_indices, samples, replace=False)

        indice = torch.tensor(indice)
        return indice

    def compute_dynamic_mask(self, batch):
        """
        对当前帧使用 Mask R-CNN 获取包含人/猫/狗的动态区域。
        """
        # batch['rgb'] 形状: [1, H, W, 3], 这里先取出 numpy
        rgb_frame = batch['rgb'].squeeze(0).cpu().numpy().astype(np.uint8)

        # 如果原先是 [0,1] 浮点或其他情况，需要先处理到 [0,255] 并确保通道顺序
        # 具体根据你数据加载方式而定，假设此时已是 0~255
        # 如果你的 batch['rgb'] 是 float [0,1]，请做:
        # rgb_frame = (rgb_frame * 255).astype(np.uint8)

        # 调用 mask r-cnn 生成动态区域掩码
        dynamic_mask = self.compute_maskrcnn_dynamic_mask(rgb_frame)
        return dynamic_mask

    def filter_batch_with_dynamic_mask(self, batch, dynamic_mask):
        """
        过滤掉动态物体处的像素，不让其进入 KeyFrame Database。
        """
        if dynamic_mask is not None:
            mask = torch.from_numpy(~dynamic_mask.flatten()).to(batch['rgb'].device)
            mask = mask.bool()
            filtered_batch = {}
            for key in ['rgb', 'depth', 'direction']:
                if key in ['rgb', 'direction']:
                    # [1, H, W, 3] -> [H*W, 3]
                    data = batch[key].squeeze(0).reshape(-1, 3)
                    filtered_data = data[mask]
                    filtered_batch[key] = filtered_data.unsqueeze(0)
                elif key == 'depth':
                    data = batch[key].squeeze(0).reshape(-1)
                    filtered_data = data[mask]
                    filtered_batch[key] = filtered_data.unsqueeze(0)
            filtered_batch['c2w'] = batch['c2w']
            filtered_batch['frame_id'] = batch['frame_id']
        else:
            filtered_batch = batch
        return filtered_batch

    # =========== 以上是与 Mask R-CNN 相关的主要改动部分 ===========

    def get_loss_from_ret(self, ret, rgb=True, sdf=True, depth=True, fs=True, smooth=False):
        loss = 0
        if rgb:
            loss += self.config['training']['rgb_weight'] * ret['rgb_loss']
        if depth:
            loss += self.config['training']['depth_weight'] * ret['depth_loss']
        if sdf:
            loss += self.config['training']['sdf_weight'] * ret["sdf_loss"]
        if fs:
            loss += self.config['training']['fs_weight'] * ret["fs_loss"]

        if smooth and self.config['training']['smooth_weight'] > 0:
            loss += self.config['training']['smooth_weight'] * self.smoothness(self.config['training']['smooth_pts'],
                                                                               self.config['training']['smooth_vox'],
                                                                               margin=self.config['training']['smooth_margin'])

        return loss

    def first_frame_mapping(self, batch, n_iters=100):
        print('第一帧映射...')
        c2w = batch['c2w'][0].to(self.device)
        self.est_c2w_data[0] = c2w
        self.est_c2w_data_rel[0] = c2w

        self.model.train()

        # 优化器需要先创建
        if not hasattr(self, 'map_optimizer'):
            self.create_optimizer()

        for i in range(n_iters):
            self.map_optimizer.zero_grad()
            indice = self.select_samples(batch, None, self.config['mapping']['sample'], 0)

            indice_h = torch.div(indice, self.dataset.W, rounding_mode='trunc')
            indice_w = indice % self.dataset.W

            rays_d_cam = batch['direction'].squeeze(0)[indice_h, indice_w, :].to(self.device)
            target_s = batch['rgb'].squeeze(0)[indice_h, indice_w, :].to(self.device)
            target_d = batch['depth'].squeeze(0)[indice_h, indice_w].to(self.device).unsqueeze(-1)

            rays_o = c2w[None, :3, -1].repeat(self.config['mapping']['sample'], 1)
            rays_d = torch.sum(rays_d_cam[..., None, :] * c2w[:3, :3], -1)

            ret = self.model.forward(rays_o, rays_d, target_s, target_d)
            loss = self.get_loss_from_ret(ret)
            loss.backward()
            self.map_optimizer.step()

        # 第一帧无需动态mask，这里可直接传 None，也可实际调用
        dynamic_mask = None
        filtered_batch = self.filter_batch_with_dynamic_mask(batch, dynamic_mask)
        self.keyframeDatabase.add_keyframe(filtered_batch, filter_depth=self.config['mapping']['filter_depth'])
        if self.config['mapping']['first_mesh']:
            self.save_mesh(0)

        print('第一帧映射完成')
        return ret, loss

    def current_frame_mapping(self, batch, cur_frame_id, prev_batch):
        if self.config['mapping']['cur_frame_iters'] <= 0:
            return
        print('当前帧映射...')

        c2w = self.est_c2w_data[cur_frame_id].to(self.device)

        self.model.train()

        for i in range(self.config['mapping']['cur_frame_iters']):
            self.cur_map_optimizer.zero_grad()
            indice = self.select_samples(batch, prev_batch, self.config['mapping']['sample'], cur_frame_id)

            indice_h = torch.div(indice, self.dataset.W, rounding_mode='trunc')
            indice_w = indice % self.dataset.W

            rays_d_cam = batch['direction'].squeeze(0)[indice_h, indice_w, :].to(self.device)
            target_s = batch['rgb'].squeeze(0)[indice_h, indice_w, :].to(self.device)
            target_d = batch['depth'].squeeze(0)[indice_h, indice_w].to(self.device).unsqueeze(-1)

            rays_o = c2w[None, :3, -1].repeat(self.config['mapping']['sample'], 1)
            rays_d = torch.sum(rays_d_cam[..., None, :] * c2w[:3, :3], -1)

            ret = self.model.forward(rays_o, rays_d, target_s, target_d)
            loss = self.get_loss_from_ret(ret)
            loss.backward()
            self.cur_map_optimizer.step()

        return ret, loss

    def smoothness(self, sample_points=256, voxel_size=0.1, margin=0.05, color=False):
        volume = self.bounding_box[:, 1] - self.bounding_box[:, 0]

        grid_size = (sample_points - 1) * voxel_size
        offset_max = self.bounding_box[:, 1] - self.bounding_box[:, 0] - grid_size - 2 * margin

        offset = torch.rand(3).to(offset_max) * offset_max + margin
        coords = coordinates(sample_points - 1, 'cpu', flatten=False).float().to(volume)
        pts = (coords + torch.rand((1, 1, 1, 3)).to(volume)) * voxel_size + self.bounding_box[:, 0] + offset

        if self.config['grid']['tcnn_encoding']:
            pts_tcnn = (pts - self.bounding_box[:, 0]) / (self.bounding_box[:, 1] - self.bounding_box[:, 0])

        sdf = self.model.query_sdf(pts_tcnn, embed=True)
        tv_x = torch.pow(sdf[1:, ...] - sdf[:-1, ...], 2).sum()
        tv_y = torch.pow(sdf[:, 1:, ...] - sdf[:, :-1, ...], 2).sum()
        tv_z = torch.pow(sdf[:, :, 1:, ...] - sdf[:, :, :-1, ...], 2).sum()

        loss = (tv_x + tv_y + tv_z) / (sample_points ** 3)

        return loss

    def get_pose_param_optim(self, poses, mapping=True):
        task = 'mapping' if mapping else 'tracking'
        cur_trans = torch.nn.parameter.Parameter(poses[:, :3, 3])
        cur_rot = torch.nn.parameter.Parameter(self.matrix_to_tensor(poses[:, :3, :3]))
        pose_optimizer = torch.optim.Adam([{"params": cur_rot, "lr": self.config[task]['lr_rot']},
                                           {"params": cur_trans, "lr": self.config[task]['lr_trans']}])

        return cur_rot, cur_trans, pose_optimizer

    def global_BA(self, batch, cur_frame_id, prev_batch):
        pose_optimizer = None

        poses = torch.stack([self.est_c2w_data[i] for i in range(0, cur_frame_id + 1, self.config['mapping']['keyframe_every'])])
        frame_ids_all = torch.tensor(list(range(0, cur_frame_id + 1, self.config['mapping']['keyframe_every'])))

        if len(self.keyframeDatabase.frame_ids) < 2:
            poses_fixed = torch.nn.parameter.Parameter(poses).to(self.device)
            poses_all = poses_fixed
        else:
            poses_fixed = torch.nn.parameter.Parameter(poses[:1]).to(self.device)

            cur_rot, cur_trans, pose_optimizer = self.get_pose_param_optim(poses[1:], mapping=True)
            pose_optim = self.matrix_from_tensor(cur_rot, cur_trans).to(self.device)
            poses_all = torch.cat([poses_fixed, pose_optim], dim=0)

        self.map_optimizer.zero_grad()
        if pose_optimizer is not None:
            pose_optimizer.zero_grad()

        for i in range(self.config['mapping']['iters']):
            rays, ids = self.keyframeDatabase.sample_global_rays(self.config['mapping']['sample'])

            if rays.shape[0] == 0:
                continue

            ids_all = torch.div(ids, self.config['mapping']['keyframe_every'], rounding_mode='trunc')

            rays_d_cam = rays[..., :3].to(self.device)
            target_s = rays[..., 3:6].to(self.device)
            target_d = rays[..., 6:7].to(self.device)

            rays_d = torch.sum(rays_d_cam[..., None, :] * poses_all[ids_all, :3, :3], -1)
            rays_o = poses_all[ids_all, :3, -1]

            ret = self.model.forward(rays_o, rays_d, target_s, target_d)

            loss = self.get_loss_from_ret(ret, smooth=True)

            loss.backward(retain_graph=True)

            if (i + 1) % self.config["mapping"]["map_accum_step"] == 0:
                if (i + 1) > self.config["mapping"]["map_wait_step"]:
                    self.map_optimizer.step()
                else:
                    print('等待更新')
                self.map_optimizer.zero_grad()

            if pose_optimizer is not None and (i + 1) % self.config["mapping"]["pose_accum_step"] == 0:
                pose_optimizer.step()
                pose_optim = self.matrix_from_tensor(cur_rot, cur_trans)
                pose_optim = pose_optim.to(self.device)
                poses_all = torch.cat([poses_fixed, pose_optim], dim=0)

                pose_optimizer.zero_grad()

        if pose_optimizer is not None and len(frame_ids_all) > 1:
            for idx, frame_id_ in enumerate(frame_ids_all[1:]):
                self.est_c2w_data[int(frame_id_.item())] = self.matrix_from_tensor(
                    cur_rot[idx:idx + 1], cur_trans[idx:idx + 1]
                ).detach().clone()[0]

    def predict_current_pose(self, frame_id, constant_speed=True):
        if frame_id == 1 or (not constant_speed):
            c2w_est_prev = self.est_c2w_data[frame_id - 1].to(self.device)
            self.est_c2w_data[frame_id] = c2w_est_prev
        else:
            c2w_est_prev_prev = self.est_c2w_data[frame_id - 2].to(self.device)
            c2w_est_prev = self.est_c2w_data[frame_id - 1].to(self.device)
            delta = c2w_est_prev @ c2w_est_prev_prev.float().inverse()
            self.est_c2w_data[frame_id] = delta @ c2w_est_prev

        return self.est_c2w_data[frame_id]

    def tracking_pc(self, batch, frame_id):
        """
        这里保留原先的 tracking_pc 函数，但把原先的 depth_diff / ICP 等逻辑去掉，
        并且改为只对静态像素进行采样（使用 Mask R-CNN 掩码）。
        """
        c2w_gt = batch['c2w'][0].to(self.device)

        cur_c2w = self.predict_current_pose(frame_id, self.config['tracking']['const_speed'])

        cur_trans = torch.nn.parameter.Parameter(cur_c2w[..., :3, 3].unsqueeze(0))
        cur_rot = torch.nn.parameter.Parameter(self.matrix_to_tensor(cur_c2w[..., :3, :3]).unsqueeze(0))
        pose_optimizer = torch.optim.Adam([
            {"params": cur_rot, "lr": self.config['tracking']['lr_rot']},
            {"params": cur_trans, "lr": self.config['tracking']['lr_trans']}
        ])
        best_sdf_loss = None

        iW = self.config['tracking']['ignore_edge_W']
        iH = self.config['tracking']['ignore_edge_H']

        thresh = 0

        if self.config['tracking']['iter_point'] > 0:
            # 根据 Mask R-CNN 检测，将动态区域跳过
            dynamic_mask = self.compute_dynamic_mask(batch)
            static_mask = ~dynamic_mask

            # 只在静态区域内采样
            static_indices = np.where(static_mask.flatten())[0]
            # 再考虑 ignore_edge 裁剪
            new_H = self.dataset.H - iH * 2
            new_W = self.dataset.W - iW * 2
            # 将中心区域（iH:-iH, iW:-iW）映射到 flatten 后的 index
            # 先取中心子图 flatten 后对应的起始 idx
            # 简单方法：对中心区域做 flatten，再做布尔索引
            center_indices = []
            for h in range(iH, self.dataset.H - iH):
                for w in range(iW, self.dataset.W - iW):
                    center_indices.append(h * self.dataset.W + w)
            center_indices = np.array(center_indices, dtype=np.int64)
            # 和 static_indices 取交集
            final_candidates = np.intersect1d(center_indices, static_indices)
            if len(final_candidates) < self.config['tracking']['pc_samples']:
                indice_pc = final_candidates
            else:
                indice_pc = np.random.choice(final_candidates, self.config['tracking']['pc_samples'], replace=False)

            # 准备射线
            rays_d_cam = batch['direction'].reshape(-1, 3)[indice_pc].to(self.device)
            target_s = batch['rgb'].reshape(-1, 3)[indice_pc].to(self.device)
            target_d = batch['depth'].reshape(-1, 1)[indice_pc].to(self.device)

            valid_depth_mask = ((target_d > 0.) * (target_d < 5.))[:, 0]

            rays_d_cam = rays_d_cam[valid_depth_mask]
            target_s = target_s[valid_depth_mask]
            target_d = target_d[valid_depth_mask]

            for i in range(self.config['tracking']['iter_point']):
                pose_optimizer.zero_grad()
                c2w_est = self.matrix_from_tensor(cur_rot, cur_trans)

                rays_o = c2w_est[..., :3, -1].repeat(len(rays_d_cam), 1)
                rays_d = torch.sum(rays_d_cam[..., None, :] * c2w_est[:, :3, :3], -1)
                pts = rays_o + target_d * rays_d

                pts_flat = (pts - self.bounding_box[:, 0]) / (self.bounding_box[:, 1] - self.bounding_box[:, 0])

                out = self.model.query_color_sdf(pts_flat)

                sdf = out[:, -1]
                rgb = torch.sigmoid(out[:, :3])

                loss = 5 * torch.mean(torch.square(rgb - target_s)) + 1000 * torch.mean(torch.square(sdf))

                if best_sdf_loss is None:
                    best_sdf_loss = loss.cpu().item()
                    best_c2w_est = c2w_est.detach()

                with torch.no_grad():
                    if loss.cpu().item() < best_sdf_loss:
                        best_sdf_loss = loss.cpu().item()
                        best_c2w_est = c2w_est.detach()
                        thresh = 0
                    else:
                        thresh += 1
                if thresh > self.config['tracking']['wait_iters']:
                    break

                loss.backward()
                pose_optimizer.step()

        if self.config['tracking']['best']:
            self.est_c2w_data[frame_id] = best_c2w_est.detach().clone()[0]
        else:
            self.est_c2w_data[frame_id] = c2w_est.detach().clone()[0]

        if frame_id % self.config['mapping']['keyframe_every'] != 0:
            kf_id = frame_id // self.config['mapping']['keyframe_every']
            kf_frame_id = kf_id * self.config['mapping']['keyframe_every']
            c2w_key = self.est_c2w_data[kf_frame_id]
            delta = self.est_c2w_data[frame_id] @ c2w_key.float().inverse()
            self.est_c2w_data_rel[frame_id] = delta

        print('最佳损失: {}, 相机损失 {}'.format(
            F.l1_loss(best_c2w_est.to(self.device)[0, :3], c2w_gt[:3]).cpu().item() if best_sdf_loss else None,
            F.l1_loss(self.est_c2w_data[frame_id][:3], c2w_gt[:3]).cpu().item()
        ))

    def tracking_render(self, batch, frame_id, prev_batch):
        """
        保持与原先相同的 render-based tracking 流程，仅在采样时跳过动态像素。
        """
        c2w_gt = batch['c2w'][0].to(self.device)

        if self.config['tracking']['iter_point'] > 0:
            cur_c2w = self.est_c2w_data[frame_id]
        else:
            cur_c2w = self.predict_current_pose(frame_id, self.config['tracking']['const_speed'])

        indice = None
        best_sdf_loss = None
        thresh = 0

        iW = self.config['tracking']['ignore_edge_W']
        iH = self.config['tracking']['ignore_edge_H']

        cur_rot, cur_trans, pose_optimizer = self.get_pose_param_optim(cur_c2w[None, ...], mapping=False)

        new_H = self.dataset.H - iH * 2
        new_W = self.dataset.W - iW * 2

        # 动态检测
        dynamic_mask = self.compute_dynamic_mask(batch)
        static_mask = ~dynamic_mask
        # 只取中心区域
        center_mask = np.zeros((self.dataset.H, self.dataset.W), dtype=np.bool_)
        center_mask[iH:-iH, iW:-iW] = True
        valid_mask = center_mask & static_mask
        static_indices = np.where(valid_mask.flatten())[0]

        for i in range(self.config['tracking']['iter']):
            pose_optimizer.zero_grad()
            c2w_est = self.matrix_from_tensor(cur_rot, cur_trans)

            if indice is None:
                total_pixels = len(static_indices)
                if total_pixels < self.config['tracking']['sample']:
                    idx_sample = static_indices
                else:
                    idx_sample = np.random.choice(static_indices, self.config['tracking']['sample'], replace=False)

                indice_h = idx_sample // self.dataset.W
                indice_w = idx_sample % self.dataset.W

                rays_d_cam = batch['direction'].squeeze(0)[indice_h, indice_w, :].to(self.device)
                target_s = batch['rgb'].squeeze(0)[indice_h, indice_w, :].to(self.device)
                target_d = batch['depth'].squeeze(0)[indice_h, indice_w].to(self.device).unsqueeze(-1)

            rays_o = c2w_est[..., :3, -1].repeat(len(idx_sample), 1)
            rays_d = torch.sum(rays_d_cam[..., None, :] * c2w_est[:, :3, :3], -1)

            ret = self.model.forward(rays_o, rays_d, target_s, target_d)
            loss = self.get_loss_from_ret(ret)

            if best_sdf_loss is None:
                best_sdf_loss = loss.cpu().item()
                best_c2w_est = c2w_est.detach()

            with torch.no_grad():
                if loss.cpu().item() < best_sdf_loss:
                    best_sdf_loss = loss.cpu().item()
                    best_c2w_est = c2w_est.detach()
                    thresh = 0
                else:
                    thresh += 1

            if thresh > self.config['tracking']['wait_iters']:
                break

            loss.backward()
            pose_optimizer.step()

        if self.config['tracking']['best']:
            self.est_c2w_data[frame_id] = best_c2w_est.detach().clone()[0]
        else:
            self.est_c2w_data[frame_id] = c2w_est.detach().clone()[0]

        if frame_id % self.config['mapping']['keyframe_every'] != 0:
            kf_id = frame_id // self.config['mapping']['keyframe_every']
            kf_frame_id = kf_id * self.config['mapping']['keyframe_every']
            c2w_key = self.est_c2w_data[kf_frame_id]
            delta = self.est_c2w_data[frame_id] @ c2w_key.float().inverse()
            self.est_c2w_data_rel[frame_id] = delta

        print('最佳损失: {}, 最后损失 {}'.format(
            F.l1_loss(best_c2w_est.to(self.device)[0, :3], c2w_gt[:3]).cpu().item() if best_sdf_loss else None,
            F.l1_loss(self.est_c2w_data[frame_id][:3], c2w_gt[:3]).cpu().item()
        ))

    def convert_relative_pose(self):
        poses = {}
        for i in range(len(self.est_c2w_data)):
            if i % self.config['mapping']['keyframe_every'] == 0:
                poses[i] = self.est_c2w_data[i]
            else:
                kf_id = i // self.config['mapping']['keyframe_every']
                kf_frame_id = kf_id * self.config['mapping']['keyframe_every']
                c2w_key = self.est_c2w_data[kf_frame_id]
                delta = self.est_c2w_data_rel[i]
                poses[i] = delta @ c2w_key

        return poses

    def create_optimizer(self):
        trainable_parameters = [
            {
                'params': self.model.decoder.parameters(),
                'weight_decay': 1e-6,
                'lr': self.config['mapping']['lr_decoder']
            },
            {
                'params': self.model.embed_fn.parameters(),
                'eps': 1e-15,
                'lr': self.config['mapping']['lr_embed']
            }
        ]

        if not self.config['grid']['oneGrid']:
            trainable_parameters.append({
                'params': self.model.embed_fn_color.parameters(),
                'eps': 1e-15,
                'lr': self.config['mapping']['lr_embed_color']
            })

        self.map_optimizer = optim.Adam(trainable_parameters, betas=(0.9, 0.99))

        if self.config['mapping']['cur_frame_iters'] > 0:
            params_cur_mapping = [
                {
                    'params': self.model.embed_fn.parameters(),
                    'eps': 1e-15,
                    'lr': self.config['mapping']['lr_embed']
                }
            ]
            if not self.config['grid']['oneGrid']:
                params_cur_mapping.append({
                    'params': self.model.embed_fn_color.parameters(),
                    'eps': 1e-15,
                    'lr': self.config['mapping']['lr_embed_color']
                })

            self.cur_map_optimizer = optim.Adam(params_cur_mapping, betas=(0.9, 0.99))

    def save_mesh(self, i, voxel_size=0.05):
        mesh_savepath = os.path.join(
            self.config['data']['output'],
            self.config['data']['exp_name'],
            'mesh_track{}.ply'.format(i)
        )
        if self.config['mesh']['render_color']:
            color_func = self.model.render_surface_color
        else:
            color_func = self.model.query_color

        extract_mesh(
            self.model.query_sdf,
            self.config,
            self.bounding_box,
            color_func=color_func,
            marching_cube_bound=self.marching_cube_bound,
            voxel_size=voxel_size,
            mesh_savepath=mesh_savepath
        )

    def run(self):
        self.create_optimizer()
        data_loader = DataLoader(self.dataset, num_workers=self.config['data']['num_workers'])

        prev_batch = None

        for i, batch in tqdm(enumerate(data_loader)):
            if self.config['mesh']['visualisation']:
                rgb_vis = batch["rgb"].squeeze().cpu().numpy().astype(np.uint8)
                # 如果你原先是 BGR，需要转换
                # rgb_vis = cv2.cvtColor(rgb_vis, cv2.COLOR_BGR2RGB)
                raw_depth = batch["depth"]
                mask = (raw_depth >= self.config["cam"]["depth_trunc"]).squeeze(0)
                depth_colormap = colormap_image(batch["depth"])
                depth_colormap[:, mask] = 255.
                depth_colormap = depth_colormap.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                image = np.hstack((rgb_vis, depth_colormap))
                cv2.namedWindow('RGB-D'.format(i), cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RGB-D'.format(i), image)
                key = cv2.waitKey(1)

            if i == 0:
                self.first_frame_mapping(batch, self.config['mapping']['first_iters'])
            else:
                if self.config['tracking']['iter_point'] > 0:
                    self.tracking_pc(batch, i)
                self.tracking_render(batch, i, prev_batch)

                if i % self.config['mapping']['map_every'] == 0:
                    self.current_frame_mapping(batch, i, prev_batch)
                    self.global_BA(batch, i, prev_batch)

                if i % self.config['mapping']['keyframe_every'] == 0:
                    # 对当前帧做一次 mask r-cnn
                    dynamic_mask = self.compute_dynamic_mask(batch)
                    filtered_batch = self.filter_batch_with_dynamic_mask(batch, dynamic_mask)
                    self.keyframeDatabase.add_keyframe(filtered_batch, filter_depth=self.config['mapping']['filter_depth'])
                    print('添加关键帧:', i)

            prev_batch = batch

        model_savepath = os.path.join(
            self.config['data']['output'],
            self.config['data']['exp_name'],
            'checkpoint{}.pt'.format(i)
        )

        self.save_ckpt(model_savepath)
        self.save_mesh(i, voxel_size=self.config['mesh']['voxel_final'])

        pose_relative = self.convert_relative_pose()
        pose_evaluation(
            self.pose_gt,
            self.est_c2w_data,
            1,
            os.path.join(self.config['data']['output'], self.config['data']['exp_name']),
            i
        )
        pose_evaluation(
            self.pose_gt,
            pose_relative,
            1,
            os.path.join(self.config['data']['output'], self.config['data']['exp_name']),
            i,
            img='pose_r',
            name='output_relative.txt'
        )


if __name__ == '__main__':

    print('开始...')
    parser = argparse.ArgumentParser(
        description='运行CoSLAM的参数。'
    )
    parser.add_argument('--config', type=str, help='配置文件的路径。')
    parser.add_argument('--input_folder', type=str,
                        help='输入文件夹，如果提供，将覆盖配置文件中的设置')
    parser.add_argument('--output', type=str,
                        help='输出文件夹，如果提供，将覆盖配置文件中的设置')

    args = parser.parse_args()

    cfg = config.load_config(args.config)
    if args.output is not None:
        cfg['data']['output'] = args.output

    # 动态物体部分参数可留空或自定义，这里仅作为占位
    # 如果不再需要，可删除。
    cfg['dynamic'] = {
        # 不再使用 depth_tau, flow_tau 等
        'depth_tau': 150,
        'flow_tau': 11.0
    }

    print("保存配置和脚本...")
    save_path = os.path.join(cfg["data"]["output"], cfg['data']['exp_name'])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    shutil.copy("coslam.py", os.path.join(save_path, 'coslam.py'))

    with open(os.path.join(save_path, 'config.json'), "w", encoding='utf-8') as f:
        f.write(json.dumps(cfg, indent=4))

    slam = CoSLAM(cfg)
    slam.run()
