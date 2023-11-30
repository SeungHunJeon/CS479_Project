import os
import numpy as np
import torch
import json
import cv2
import time
import imageio
from raisimGymTorch.nav.math_utils import vec_to_rot_matrix, rot_matrix_to_vec, rot_x, skew_matrix_torch
import subprocess

# #Helper functions
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# rot_x = lambda phi: torch.tensor([
#         [1., 0., 0.],
#         [0., torch.cos(phi), -torch.sin(phi)],
#         [0., torch.sin(phi), torch.cos(phi)]], dtype=torch.float32, device=device)

# def skew_matrix_torch(vector):  # vector to skewsym. matrix

#     ss_matrix = torch.zeros((3,3))
#     ss_matrix[0, 1] = -vector[2]
#     ss_matrix[0, 2] = vector[1]
#     ss_matrix[1, 0] = vector[2]
#     ss_matrix[1, 2] = -vector[0]
#     ss_matrix[2, 0] = -vector[1]
#     ss_matrix[2, 1] = vector[0]

#     return ss_matrix

def add_noise_to_state(state, noise):
    return state + noise

class Agent():
    def __init__(self , camera_cfg, blender_cfg) -> None:

        #Initialize camera params
        self.path = camera_cfg['path']
        self.half_res = camera_cfg['half_res']
        self.white_bg = camera_cfg['white_bg']

        self.data = {
        'pose': None,
        'res_x': camera_cfg['res_x'],           # x resolution
        'res_y': camera_cfg['res_y'],           # y resolution
        'trans': camera_cfg['trans'],     # Boolean
        'mode': camera_cfg['mode']             # Must be either 'RGB' or 'RGBA'
        }   

        self.blend = blender_cfg['blend_path']
        self.blend_script = blender_cfg['script_path']

        self.iter = 0








