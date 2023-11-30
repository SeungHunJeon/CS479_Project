import math

import numpy as np
import torch
import json
import os
import glob
def load_intrinsic_params(opt):
    with open(os.path.join(opt.path, 'transforms.json'), 'r') as f:
        transform = json.load(f)

    # load image size
    if 'h' in transform and 'w' in transform:
        H = int(transform['h'])
        W = int(transform['w'])
    else:
        # we have to actually read an image to get H and W later.
        H = W = None

    # load intrinsics
    if 'fl_x' in transform or 'fl_y' in transform:
        fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y'])
        fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x'])
    elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
        # blender, assert in radians. already downscaled since we use H/W
        fl_x = W / (2 * np.tan(transform['camera_angle_x'] / 2)) if 'camera_angle_x' in transform else None
        fl_y = H / (2 * np.tan(transform['camera_angle_y'] / 2)) if 'camera_angle_y' in transform else None
        if fl_x is None: fl_x = fl_y
        if fl_y is None: fl_y = fl_x
    else:
        raise RuntimeError('Failed to load focal length, please check the transforms.json!')

    cx = (transform['cx']) if 'cx' in transform else (W / 2)
    cy = (transform['cy']) if 'cy' in transform else (H / 2)

    intrinsics = np.array([fl_x, fl_y, cx, cy])

    return intrinsics, W, H

def load_param(name, opt, nerf_model, device='cuda'):
    global checkpoint_list
    model = nerf_model.to(device)
    checkpoint = None
    ckpt_path = os.path.join(opt.workspace, 'checkpoints')

    if checkpoint is None:
        checkpoint_list = sorted(glob.glob(f'{ckpt_path}/{name}_ep*.pth'))
    if checkpoint_list:
        checkpoint = checkpoint_list[-1]
        print(f"[INFO] Latest checkpoint is {checkpoint}")
    else:
        print("[WARN] No checkpoint found, model randomly initialized.")
        return

    checkpoint_dict = torch.load(checkpoint, map_location=device)

    if 'model' not in checkpoint_dict:
        model.load_state_dict(checkpoint_dict)
        print("[INFO] loaded model.")
        return

    missing_keys, unexpected_keys = model.load_state_dict(checkpoint_dict['model'], strict=False)
    print("[INFO] loaded model.")
    if len(missing_keys) > 0:
        print(f"[WARN] missing keys: {missing_keys}")
    if len(unexpected_keys) > 0:
        print(f"[WARN] unexpected keys: {unexpected_keys}")

    # if self.ema is not None and 'ema' in checkpoint_dict:
    #     self.ema.load_state_dict(checkpoint_dict['ema'])

    # if self.model.cuda_ray:
    #     if 'mean_count' in checkpoint_dict:
    #         self.model.mean_count = checkpoint_dict['mean_count']
    #     if 'mean_density' in checkpoint_dict:
    #         self.model.mean_density = checkpoint_dict['mean_density']

    # if model_only:
    #     return

    # self.stats = checkpoint_dict['stats']
    # self.epoch = checkpoint_dict['epoch']
    # self.global_step = checkpoint_dict['global_step']
    # self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")
    #
    # if self.optimizer and 'optimizer' in checkpoint_dict:
    #     try:
    #         self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
    #         self.log("[INFO] loaded optimizer.")
    #     except:
    #         self.log("[WARN] Failed to load optimizer.")
    #
    # if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
    #     try:
    #         self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
    #         self.log("[INFO] loaded scheduler.")
    #     except:
    #         self.log("[WARN] Failed to load scheduler.")
    #
    # if self.scaler and 'scaler' in checkpoint_dict:
    #     try:
    #         self.scaler.load_state_dict(checkpoint_dict['scaler'])
    #         self.log("[INFO] loaded scaler.")
    #     except:
    #         self.log("[WARN] Failed to load scaler.")