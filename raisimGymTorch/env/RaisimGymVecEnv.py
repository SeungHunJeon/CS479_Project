# //----------------------------//
# // This file is part of RaiSim//
# // Copyright 2020, RaiSim Tech//
# //----------------------------//

import numpy as np
import platform
import os


class RaisimGymVecEnv:

    def __init__(self, impl, cfg, normalize_ob=True, seed=0, normalize_rew=True, clip_obs=10.):
        if platform.system() == "Darwin":
            os.environ['KMP_DUPLICATE_LIB_OK']='True'
        self.normalize_ob = normalize_ob
        self.normalize_rew = normalize_rew
        self.clip_obs = clip_obs
        self.wrapper = impl
        self.wrapper.init()
        self.num_obs = self.wrapper.getObDim()
        self.num_acts = self.wrapper.getActionDim()
        self.anchor_history = np.zeros([self.num_envs, 20*24], dtype=np.float32)
        self._observation = np.zeros([self.num_envs, self.num_obs], dtype=np.float32)
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self._done = np.zeros(self.num_envs, dtype=bool)
        self._success = np.zeros(self.num_envs, dtype=bool)
        self._switch = np.zeros(self.num_envs, dtype=bool)
        self._contact = np.zeros(self.num_envs, dtype=bool)
        self._env_val = np.zeros([self.num_envs, 22], dtype=np.float32)
        self._cam_pos = np.zeros(3, dtype=np.float32)
        self._cam_rot = np.zeros([3,3], dtype=np.float32)
        # 0-2 : H W D / 3-11 : Inertia row1+row2+row3 / 12-14 : COM / 15 : Mass / 16 : friction
        self.rewards = [[] for _ in range(self.num_envs)]
        self.wrapper.setSeed(seed)
        self.count = 0.0
        self.mean = np.zeros(self.num_obs, dtype=np.float32)
        self.var = np.zeros(self.num_obs, dtype=np.float32)
        self.obj_pos = np.zeros([cfg['nSamples_'], 3], dtype=np.float32)
        self._observation_Rollout = np.zeros([cfg['nSamples_'], self.num_obs], dtype=np.float32)
        self.height = 720
        self.width = 1080
    def seed(self, seed=None):
        self.wrapper.setSeed(seed)

    def turn_on_visualization(self):
        self.wrapper.turnOnVisualization()

    def turn_off_visualization(self):
        self.wrapper.turnOffVisualization()

    def start_video_recording(self, file_name):
        self.wrapper.startRecordingVideo(file_name)

    def stop_video_recording(self):
        self.wrapper.stopRecordingVideo()

    def step(self, action):
        self.wrapper.step(action, self._reward, self._done)
        return self._reward.copy(), self._done.copy()

    def step_visualize(self, action):
        self.wrapper.step_visualize(action, self._reward, self._done)
        return self._reward.copy(), self._done.copy()

    def get_camera_pose(self):
        self.wrapper.getCameraPose(self._cam_pos, self._cam_rot)
        return self._cam_pos, self._cam_rot

    def step_evaluate(self, action, anchors):
        self.wrapper.step_evaluate(action, anchors, self._reward, self._done)
        return self._reward.copy(), self._done.copy()

    def predict_obj_update(self, predict_state_batch):
        self.wrapper.predict_obj_update(predict_state_batch)

    def step_visualize_success(self, action, success):
        self.wrapper.step_visualize_success(action, self._reward, self._done, self._success)
        return self._reward.copy(), self._done.copy()

    def step_rollout(self, action):
        self.wrapper.step_Rollout(action)

    def get_depth_image(self): ## only for one env
        return np.array(self.wrapper.getDepthImage(), dtype=np.float32).reshape(-1, self.height, self.width)
    def get_color_image(self): ## only for one env
        return np.array(self.wrapper.getColorImage()).reshape(-1, self.height, self.width,4)[..., [2,1,0]].astype(np.uint8)

    def getAnchorHistory(self, robotFrame = True):
        self.wrapper.getAnchorHistory(self.anchor_history, robotFrame)
        return self.anchor_history

    def get_error(self, get, anchors):

        return self.wrapper.get_error(get, anchors)


    def load_scaling(self, dir_name, iteration, count=1e5):
        mean_file_name = dir_name + "/mean" + str(iteration) + ".csv"
        var_file_name = dir_name + "/var" + str(iteration) + ".csv"
        self.count = count
        self.mean = np.loadtxt(mean_file_name, dtype=np.float32)
        self.var = np.loadtxt(var_file_name, dtype=np.float32)
        self.wrapper.setObStatistics(self.mean, self.var, self.count)

    def save_scaling(self, dir_name, iteration):
        mean_file_name = dir_name + "/mean" + iteration + ".csv"
        var_file_name = dir_name + "/var" + iteration + ".csv"
        self.wrapper.getObStatistics(self.mean, self.var, self.count)
        np.savetxt(mean_file_name, self.mean)
        np.savetxt(var_file_name, self.var)


    def observe_Rollout(self, update_statistics=False):
        self.wrapper.observe_Rollout(self._observation_Rollout, update_statistics)
        return self._observation_Rollout.copy()

    def observe_denormalize(self, observe):
        self.wrapper.observationDeNormalize(observe)
        return observe

    def observe(self, update_statistics=True):
        self.wrapper.observe(self._observation, update_statistics)
        return self._observation

    # def synchronize(self):
    #     self.wrapper.synchronize()

    def reset(self):
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self.wrapper.reset()

    def close(self):
        self.wrapper.close()

    def set_command(self, id):
        self.wrapper.setCommand(id)

    def move_controller_cursor(self, id, pos):
        self.wrapper.moveControllerCursor(id, pos)

    def curriculum_callback(self):
        self.wrapper.curriculumUpdate()

    def get_step_data_tag(self):
        return self.wrapper.getStepDataTag()

    def get_success_state(self):
        self.wrapper.getSuccess(self._success)

        return self._success

    def get_intrinsic_switch(self):
        self.wrapper.getIntrinsicSwitch(self._switch)

        return self._switch



    def get_contact(self):
        self.wrapper.getContact(self._contact)

        return self._contact

    def get_privileged_info(self):
        self.wrapper.getPrivilegedInformation(self._env_val)
        return self._env_val

    def get_step_data(self, data_size, data_mean, data_var, data_min, data_max):
        return self.wrapper.getStepData(data_size, data_mean, data_var, data_min, data_max)

    def get_state(self, gc, gv):
        self.wrapper.getState(gc, gv)

    def get_target_pos(self):
        return self.wrapper.get_target_pos()

    def get_obj_pos(self):
        self.wrapper.get_obj_pos(self.obj_pos)
        return self.obj_pos.copy()
    def get_state_rollout(self, gc, gv):
        self.wrapper.getState_Rollout(gc, gv)

    def synchronize(self):
        self.wrapper.synchronize()


    @property
    def num_envs(self):
        return self.wrapper.getNumOfEnvs()
