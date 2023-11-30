import math

import numpy as np
import torch
import json
import time
import cv2
import matplotlib.pyplot as plt
import imageio
from nav.math_utils import vec_to_rot_matrix, mahalanobis, rot_x, nerf_matrix_to_ngp_torch, nearestPD, calcSE3Err
import os



def get_img_process(img, white_bg=True):
    img = (np.array(img) / 255.0).astype(np.float32)
    if white_bg is True:
        img = img[..., :3] * img[..., -1:] + (1.0 - img[..., -1:])
    img = (np.array(img) * 255.).astype(np.uint8)
    print('Received updated image')
    img = torch.from_numpy(img)
    return img


def find_POI(img_rgb, render=False): # img - RGB image in range 0...255
    img = np.copy(img_rgb)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    sift = cv2.SIFT_create()
    keypoints = sift.detect(img, None)

    # Initiate ORB detector
    # orb = cv2.ORB_create()
    # find the keypoints with ORB
    # keypoints2 = orb.detect(img_gray,None)

    if render:
        feat_img = cv2.drawKeypoints(img_gray.copy(), keypoints, img.copy())
    else:
        feat_img = None

    #keypoints = keypoints + keypoints2
    #keypoints = keypoints2

    xy = [keypoint.pt for keypoint in keypoints]
    xy = np.array(xy).astype(int)

    # Remove duplicate points
    xy_set = set(tuple(point) for point in xy)
    xy = np.array([list(point) for point in xy_set]).astype(int)

    extras = {
        'features': feat_img
    }

    return xy, extras # pixel coordinates

class state_estimator():
    def __init__(self, filter_cfg, get_rays_fn=None, render_fn=None, is_filter=True, device='cpu') -> None:

        # Parameters
        self.batch_size = filter_cfg['batch_size']
        self.kernel_size = filter_cfg['kernel_size']
        self.dil_iter = filter_cfg['dil_iter']
        self.is_filter = is_filter
        self.lrate = filter_cfg['lrate']


        self.render_viz = filter_cfg['render_viz']
        if self.render_viz:
            self.f, self.axarr = plt.subplots(1, 3, figsize=(15, 50))

        self.show_rate = filter_cfg['show_rate']
        self.error_print_rate, self.render_rate = self.show_rate

        #State initial estimate at time t=0

        self.iter = filter_cfg['N_iter']

        #NERF SPECIFIC CONFIGS
        self.get_rays = get_rays_fn
        self.render_fn = render_fn

        #Storage for plots
        self.losses = None
        self.covariance = None
        self.state_estimate = None
        self.states = None
        self.action = None

        self.iteration = 0
        self.device = device




    def estimate_relative_pose_fusion(self, sensor_image, init_anchor, sig, obs_img_pose=None):
    #start-state is 12-vector

        obs_img_noised = sensor_image.cpu()
        W_obs = sensor_image.shape[1]
        H_obs = sensor_image.shape[0]

    # find points of interest of the observed image
        POI, extras = find_POI(obs_img_noised, render=self.render_viz)  # xy pixel coordinates of points of interest (N x 2)

        print(f'Found {POI.shape[0]} features')
    ### IF FEATURE DETECTION CANT FIND POINTS, RETURN INITIAL
        if len(POI.shape) == 1:
            self.losses = []
            self.states = []
            error_text = 'Feature Detection Failed.'
            print(f'{error_text:.^20}')
            return init_anchor.clone().detach(), False

        obs_img_noised = (np.array(obs_img_noised) / 255.).astype(np.float32)
        obs_img_noised = torch.tensor(obs_img_noised).to(self.device)

        # create meshgrid from the observed image
        coords = np.asarray(np.stack(np.meshgrid(np.linspace(0, H_obs - 1, H_obs), np.linspace(0, W_obs - 1, W_obs)), -1), dtype=int)

        # create sampling mask for interest region sampling strategy
        # interest_regions = np.zeros((H_obs, W_obs, ), dtype=np.uint8)
        interest_regions = np.zeros((W_obs, H_obs, ), dtype=np.uint8)
        interest_regions[POI[:,0], POI[:,1]] = 1
        I = self.dil_iter
        interest_regions = cv2.dilate(interest_regions, np.ones((self.kernel_size, self.kernel_size), np.uint8), iterations=I)
        interest_regions = np.array(interest_regions, dtype=bool)
        interest_regions = coords[interest_regions]

        #Optimzied state is 12 vector initialized as the starting state to be optimized. Add small epsilon to avoid singularities
        optimized_anchor = init_anchor.clone().detach() + 1e-6
        optimized_anchor.requires_grad_(True)

        # Add velocities, omegas, and pose object to optimizer
        if self.is_filter is True:
            optimizer = torch.optim.Adam(params=[optimized_anchor], lr=self.lrate, betas=(0.9, 0.999), capturable=True)
        else:
            raise('Not implemented')

        # calculate initial angles and translation error from observed image's pose
        if obs_img_pose is not None:
            pos, rot = self.anchor_to_SE3(optimized_anchor)
            pose = torch.eye(4)
            pose[:3, :3] = rot
            pose[:3, 3] = pos
            print('initial error', calcSE3Err(pose.detach().cpu().numpy(), obs_img_pose.cpu().numpy()))

        #Store data
        losses = []
        states = []

        for k in range(self.iter):
            optimizer.zero_grad()
            rand_inds = np.random.choice(interest_regions.shape[0], size=self.batch_size, replace=False)
            batch = interest_regions[rand_inds]

            #pix_losses.append(loss.clone().cpu().detach().numpy().tolist())
            #Add dynamics loss

            loss = self.measurement_fn_fusion(optimized_anchor, init_anchor, sig, obs_img_noised, batch)

            losses.append(loss.item())
            states.append(optimized_anchor.clone().cpu().detach().numpy().tolist())

            loss.backward()
            optimizer.step()

            # NOT IMPLEMENTED: EXPONENTIAL DECAY OF LEARNING RATE
            #new_lrate = self.lrate * (0.8 ** ((k + 1) / 100))
            #new_lrate = extra_arg_dict['lrate'] * np.exp(-(k)/1000)
            #for param_group in optimizer.param_groups:
            #    param_group['lr'] = new_lrate

            # print results periodically
            if obs_img_pose is not None and ((k + 1) % self.error_print_rate == 0 or k == 0):
                print('Step: ', k)
                print('Loss: ', loss)
                pos, rot = self.anchor_to_SE3(optimized_anchor, include_offset=False)
                print('State', pos, rot)

                with torch.no_grad():
                    pos, rot = self.anchor_to_SE3(optimized_anchor)
                    pose = torch.eye(4)
                    pose[:3, :3] = rot
                    pose[:3, 3] = pos
                    pose_error = calcSE3Err(pose.detach().cpu().numpy(), obs_img_pose.detach().cpu().numpy())
                    print('error', pose_error)
                    print('-----------------------------------')

                    if (k+1) % self.render_rate == 0 and self.render_viz:
                        rgb = self.render_from_pose(pose)
                        rgb = torch.squeeze(rgb).cpu().detach().numpy()

                        #Add keypoint visualization
                        render = rgb.reshape((obs_img_noised.shape[0], obs_img_noised.shape[1], -1))
                        gt_img = obs_img_noised.cpu().numpy()
                        render[batch[:, 0], batch[:, 1]] = np.array([0., 1., 0.])
                        gt_img[batch[:, 0], batch[:, 1]] = np.array([0., 1., 0.])

                        self.f.suptitle(f'Time step: {self.iteration}. Grad step: {k+1}. Trans. error: {pose_error[0]} m. Rotate. error: {pose_error[1]} deg.')
                        self.axarr[0].imshow(gt_img)
                        self.axarr[0].set_title('Ground Truth')

                        self.axarr[1].imshow(extras['features'])
                        self.axarr[1].set_title('Features')

                        self.axarr[2].imshow(render)
                        self.axarr[2].set_title('NeRF Render')

                        plt.pause(1)

        print("Done with main relative_pose_estimation loop")

        return optimized_anchor.clone().detach(), True


    def measurement_fn_fusion(self, anchor, start_anchor, sig, target, batch):
        #Process loss.
        # start_anchor.requires_grad_(True)
        # sig.requires_grad_(True)
        loss_dyn = (((anchor[...,:2]-start_anchor[...,:2])**2) / sig[...,:2]).mean()

        H, W, _ = target.shape

        pos, rot = self.anchor_to_SE3(anchor)

        new_pose = torch.eye(4)
        new_pose[:3, :3] = rot
        new_pose[:3, 3] = pos
        new_pose = new_pose.to(self.device)

        rays = self.get_rays(new_pose.reshape((1, 4, 4)))

        rays_o = rays["rays_o"].reshape((H, W, -1))[batch[:, 0], batch[:, 1]]
        rays_d = rays["rays_d"].reshape((H, W, -1))[batch[:, 0], batch[:, 1]]

        output = self.render_fn(rays_o.reshape((1, -1, 3)), rays_d.reshape((1, -1, 3)))
        #output also contains a depth channel for use with depth data if one chooses

        rgb = output['image'].reshape((-1, 3))

        target = target[batch[:, 0], batch[:, 1]]

        loss_rgb = torch.nn.functional.mse_loss(rgb, target)

        loss = loss_rgb + loss_dyn

        return loss
    # def measurement_fn(self, state, start_state, sig, target, batch):
    #     #Process loss.
    #
    #     loss_dyn = mahalanobis(state, start_state, sig)
    #
    #     H, W, _ = target.shape
    #
    #     #Assuming the camera frustrum is oriented in the body y-axis. The camera frustrum is in the -z axis
    #     # in its own frame, so we need a 90 degree rotation about the x-axis to transform
    #
    #     R = vec_to_rot_matrix(state[6:9])
    #     rot = rot_x(torch.tensor(np.pi/2)) @ R[:3, :3]
    #
    #     pose, trans = nerf_matrix_to_ngp_torch(rot, state[:3])
    #
    #     new_pose = torch.eye(4)
    #     new_pose[:3, :3] = pose
    #     new_pose[:3, 3] = trans
    #
    #     rays = self.get_rays(new_pose.reshape((1, 4, 4)))
    #
    #     rays_o = rays["rays_o"].reshape((H, W, -1))[batch[:, 0], batch[:, 1]]
    #     rays_d = rays["rays_d"].reshape((H, W, -1))[batch[:, 0], batch[:, 1]]
    #
    #     output = self.render_fn(rays_o.reshape((1, -1, 3)), rays_d.reshape((1, -1, 3)))
    #     #output also contains a depth channel for use with depth data if one chooses
    #
    #     rgb = output['image'].reshape((-1, 3))
    #
    #     target = target[batch[:, 0], batch[:, 1]]
    #
    #     loss_rgb = torch.nn.functional.mse_loss(rgb, target)
    #
    #     loss = loss_rgb + loss_dyn
    #
    #     return loss

    def render_from_pose(self, pose):
        rot = pose[:3, :3]
        trans = pose[:3, 3]
        # pose, trans = nerf_matrix_to_ngp_torch(rot, trans)

        new_pose = torch.eye(4).to(self.device)
        new_pose[:3, :3] = rot
        new_pose[:3, 3] = trans

        rays = self.get_rays(new_pose.reshape((1, 4, 4)))

        output = self.render_fn(rays["rays_o"], rays["rays_d"])
        #output also contains a depth channel for use with depth data if one chooses

        rgb = torch.squeeze(output['image'])

        return rgb

    def process_anchors(self, anchor):
        # anchor to cartesian pos and 3*3 rotmat
        if (isinstance(anchor[0], np.ndarray)):
            anchor = torch.from_numpy(anchor)
        anchor=anchor.reshape(-1,3).to(self.device)
        trans = anchor.mean(dim=0)
        x_axis = torch.zeros(3).to(self.device)
        for i in range(4):
            x_axis += (anchor[4+i]-anchor[i])/4
        yaw_angle = math.atan2(x_axis[1],x_axis[0])
        rot = torch.eye(3,3).to(self.device)
        rot[0][0] = math.cos(yaw_angle)
        rot[0][1] = -math.sin(yaw_angle)
        rot[1][0] = math.sin(yaw_angle)
        rot[1][1] = math.cos(yaw_angle)
        return trans, rot

    def anchor_to_SE3(self, anchor, include_offset=True):
        # anchor to cartesian pos and 3*3 rotmat
        anchor = anchor.reshape(-1, 3).to(self.device)
        trans = anchor.mean(dim=0)
        x_axis = torch.zeros(3).to(self.device)
        for i in range(4):
            x_axis += (anchor[4+i]-anchor[i])/4
        yaw_angle = math.atan2(x_axis[1], x_axis[0])
        rot = torch.eye(3).to(self.device)
        rot[0][0] = math.cos(yaw_angle)
        rot[0][1] = -math.sin(yaw_angle)
        rot[1][0] = math.sin(yaw_angle)
        rot[1][1] = math.cos(yaw_angle)
        if(include_offset):
            offset = torch.tensor([0.12476, 0, 0.16252]).to(self.device)
            trans = trans + rot@offset
        return trans, rot
    # def get_rotation_from_abs_anchor(self, anchor):
    #     # anchor to cartesian pos and 3*3 rotmat
    #     anchor = anchor.reshape(-1,3)
    #     x_axis = torch.zeros(3)
    #     for i in range(4):
    #         x_axis += (anchor[4+i]-anchor[i])/4
    #     yaw_angle = math.atan2(x_axis[1], x_axis[0])
    #     rot = torch.eye(3)
    #     rot[0][0] = math.cos(yaw_angle)
    #     rot[0][1] = -math.sin(yaw_angle)
    #     rot[1][0] = math.sin(yaw_angle)
    #     rot[1][1] = math.cos(yaw_angle)
    #     return rot

    def estimate_state_fusion(self, sensor_img, network_anchors, network_cov, prev_anchor, obs_img_pose):

        _, yaw_rot = self.process_anchors(prev_anchor) # world frame
        prev_anchor = torch.from_numpy(prev_anchor.reshape(-1, 3)).to(self.device)
        network_anchors = network_anchors.reshape(-1, 3)
        anchor_init = torch.zeros(8, 3).to(self.device)
        for i in range(8):
            anchor_init[i] = prev_anchor[i]+(yaw_rot@network_anchors[i].unsqueeze(-1)).squeeze(-1) # prev_anchor : World frame

        network_cov = network_cov.reshape(-1,3)
        #Argmin of total cost. Encapsulate this argmin optimization as a function call
        then = time.time()
        #xt is 12-vector
        xt, success_flag = self.estimate_relative_pose_fusion(sensor_img, anchor_init.clone().detach(), network_cov, obs_img_pose=obs_img_pose)

        print('Optimization step for filter', time.time()-then)

        #Hessian to get updated covariance

        self.iteration += 1

        return xt.clone().detach()



    def save_data(self, filename):
        data = {}

        data['loss'] = self.losses
        data['covariance'] = self.covariance
        data['state_estimate'] = self.state_estimate
        data['grad_states'] = self.states
        data['action'] = self.action

        with open(filename,"w+") as f:
            json.dump(data, f, indent=4)
        return
