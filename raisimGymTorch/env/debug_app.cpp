//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#include "Environment.hpp"
#include "Environment_Rollout.hpp"
#include "VectorizedEnvironment.hpp"
#include <chrono>

int THREAD_COUNT = 1;

using namespace raisim;

void print_timediff(const char *prefix, int loopCount,
                    const std::chrono::steady_clock::time_point &start,
                    const std::chrono::steady_clock::time_point &end) {
  double milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  std::cout<<milliseconds<<" microseconds"<<std::endl;
  printf("The simulation of %s is running at: %lf MHz\n", prefix, loopCount / milliseconds);
}

inline std::string loadResource (const std::string& file) {
  std::string urdfPath(__FILE__);
  while (urdfPath.back() != raisim::Path::separator()[0])
    urdfPath.erase(urdfPath.size() - 1, 1);
  urdfPath += "../res/" + file;
  return urdfPath;
};

int main(int argc, char *argv[]) {
  RSFATAL_IF(argc != 3, "got "<<argc<<" arguments. "<<"This executable takes three arguments: 1. resource directory, 2. configuration file")

  std::string resourceDir(argv[1]), cfgFile(argv[2]);
  std::ifstream myfile (cfgFile);
  std::string config_str, line;
  bool escape = false;

  while (std::getline(myfile, line)) {
    if(line == "environment:") {
      escape = true;
      while (std::getline(myfile, line)) {
        if(line.substr(0, 2) == "  ")
          config_str += line.substr(2) + "\n";
        else if (line[0] == '#')
          continue;
        else
          break;
      }
    }
    if(escape)
      break;
  }
  config_str.pop_back();



  Yaml::Node config;
  Yaml::Parse(config, config_str);

  bool isRollout = config["Rollout"].template As<bool>();
  if(isRollout)
  {
    VectorizedEnvironment<ENVIRONMENT_ROLLOUT> vecEnvRollout(resourceDir, config_str);
    vecEnvRollout.init();

    EigenRowMajorMat observation(config["num_envs"].template As<int>(), vecEnvRollout.getObDim());
    EigenRowMajorMat action(config["num_envs"].template As<int>(), vecEnvRollout.getActionDim());
    EigenVec reward(config["num_envs"].template As<int>(), 1);
    EigenBoolVec dones(config["num_envs"].template As<int>(), 1);
    action.setZero();

    Eigen::Ref<EigenRowMajorMat> ob_ref(observation), action_ref(action);
    Eigen::Ref<EigenVec> reward_ref(reward);
    Eigen::Ref<EigenBoolVec> dones_ref(dones);

    vecEnvRollout.reset();
    vecEnvRollout.step(action_ref, reward_ref, dones_ref);
    vecEnvRollout.observe(ob_ref);

    vecEnvRollout.reset();
    vecEnvRollout.step(action_ref, reward_ref, dones_ref);
    vecEnvRollout.observe(ob_ref);

    vecEnvRollout.curriculumUpdate();
    vecEnvRollout.reset();
    vecEnvRollout.step(action_ref, reward_ref, dones_ref);
    vecEnvRollout.observe(ob_ref);
  }

  else
  {
    VectorizedEnvironment<ENVIRONMENT> vecEnv(resourceDir, config_str);
    vecEnv.init();

    EigenRowMajorMat observation(config["num_envs"].template As<int>(), vecEnv.getObDim());
    EigenRowMajorMat action(config["num_envs"].template As<int>(), vecEnv.getActionDim());
    EigenVec reward(config["num_envs"].template As<int>(), 1);
    EigenBoolVec dones(config["num_envs"].template As<int>(), 1);
    action.setZero();

    Eigen::Ref<EigenRowMajorMat> ob_ref(observation), action_ref(action);
    Eigen::Ref<EigenVec> reward_ref(reward);
    Eigen::Ref<EigenBoolVec> dones_ref(dones);

    vecEnv.reset();
    vecEnv.step(action_ref, reward_ref, dones_ref);
    vecEnv.observe(ob_ref);

    vecEnv.reset();
    vecEnv.step(action_ref, reward_ref, dones_ref);
    vecEnv.observe(ob_ref);

    vecEnv.curriculumUpdate();
    vecEnv.reset();
    vecEnv.step(action_ref, reward_ref, dones_ref);
    vecEnv.observe(ob_ref);

    vecEnv.curriculumUpdate();
    vecEnv.reset();
    vecEnv.step(action_ref, reward_ref, dones_ref);
    vecEnv.observe(ob_ref);

    vecEnv.curriculumUpdate();
    vecEnv.reset();
    vecEnv.step(action_ref, reward_ref, dones_ref);
    vecEnv.observe(ob_ref);

    vecEnv.curriculumUpdate();
    vecEnv.reset();
    vecEnv.step(action_ref, reward_ref, dones_ref);
    vecEnv.observe(ob_ref);

  }










  return 0;
}