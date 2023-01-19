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
        self._observation = np.zeros([self.num_envs, self.num_obs], dtype=np.float32)
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self._done = np.zeros(self.num_envs, dtype=np.bool)
        self._success = np.zeros(self.num_envs, dtype=np.bool)
        self.rewards = [[] for _ in range(self.num_envs)]
        self.wrapper.setSeed(seed)
        self.count = 0.0
        self.mean = np.zeros(self.num_obs, dtype=np.float32)
        self.var = np.zeros(self.num_obs, dtype=np.float32)

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

    def step_visualize_success(self, action, success):
        self.wrapper.step_visualize_success(action, self._reward, self._done, self._success)
        return self._reward.copy(), self._done.copy()

    def step_rollout(self, action):
        self.wrapper.step_Rollout(action)

    def get_depth_image(self): ## only for one env
        return np.array(self.wrapper.getDepthImage(), dtype=np.float32).reshape(-1, self.height, self.width)
    def get_color_image(self): ## only for one env
        return np.array(self.wrapper.getColorImage()).reshape(-1, self.height, self.width,4)[..., [2,1,0]].astype(np.uint8)

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

    def get_step_data(self, data_size, data_mean, data_var, data_min, data_max):
        return self.wrapper.getStepData(data_size, data_mean, data_var, data_min, data_max)

    def get_state(self, gc, gv):
        self.wrapper.getState(gc, gv)

    def get_rollout_state(self, gc, gv):
        self.wrapper.getRolloutState(gc, gv)

    def synchronize(self):
        self.wrapper.synchronize()


    @property
    def num_envs(self):
        return self.wrapper.getNumOfEnvs()
