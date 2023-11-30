//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#ifndef SRC_RAISIMGYMVECENV_HPP
#define SRC_RAISIMGYMVECENV_HPP
#include "omp.h"
#include "Yaml.hpp"
#include <Eigen/Core>
#include "BasicEigenTypes.hpp"
#include "../../default_controller_demo/include/neuralNet.hpp"

extern int THREAD_COUNT;

namespace raisim {

template<class ChildEnvironment>
class VectorizedEnvironment {

 public:

  explicit VectorizedEnvironment(std::string resourceDir, std::string cfg)
      : resourceDir_(resourceDir) {
    Yaml::Parse(cfg_, cfg);
    if(&cfg_["render"])
      render_ = cfg_["render"].template As<bool>();
  }

  ~VectorizedEnvironment() {
    for (auto *ptr: environments_)
      delete ptr;
  }

  void init() {
    THREAD_COUNT = cfg_["num_threads"].template As<int>();
    omp_set_num_threads(THREAD_COUNT);
    num_envs_ = cfg_["num_envs"].template As<int>();
    double simDt, conDt, low_conDt;
    RSINFO(1)
    READ_YAML(bool, is_position_goal_, cfg_["position_goal"])
    READ_YAML(double, conDt, cfg_["control_dt"])
    READ_YAML(double, simDt, cfg_["simulation_dt"])
    READ_YAML(double, low_conDt, cfg_["low_level_control_dt"])
    READ_YAML(bool, is_rollout_, cfg_["Rollout"])


    if(is_rollout_)
      num_envs_ = 1;

    if (cfg_["hierarchical"].template As<bool>()) {
      environments_.push_back(new ChildEnvironment(resourceDir_, cfg_, render_ && 0 == 0, 0));
      environments_.back()->setSimulationTimeStep(simDt);
      environments_.back()->setControlTimeStep(conDt, low_conDt);
      if(!is_rollout_)
        environments_[0]->Low_controller_create(is_position_goal_);

      for (int i = 1; i < num_envs_; i++) {
        environments_.push_back(new ChildEnvironment(resourceDir_, cfg_, render_ && i == 0, i));
        if(is_position_goal_)
          environments_[i]->adapt_Low_position_controller(environments_[0]->get_Low_position_controller());
        else
          environments_[i]->adapt_Low_velocity_controller(environments_[0]->get_Low_velocity_controller());
        environments_.back()->setSimulationTimeStep(simDt);
        environments_.back()->setControlTimeStep(conDt, low_conDt);
      }
    }

    else {
      for (int i = 0; i < num_envs_; i++) {
        environments_.push_back(new ChildEnvironment(resourceDir_, cfg_, render_ && i == 0, i));
        environments_.back()->setSimulationTimeStep(simDt);
        environments_.back()->setControlTimeStep(conDt, low_conDt);
      }
    }

    int startSeed;
    READ_YAML(int, startSeed, cfg_["seed"])
    for (int i = 0; i < num_envs_; i++) {
      environments_[i]->setSeed(startSeed + i);
      environments_[i]->init();
      environments_[i]->reset();
    }

    /// ob scaling
    if (normalizeObservation_) {
      obMean_.setZero(getObDim());
      obVar_.setOnes(getObDim());
      recentMean_.setZero(getObDim());
      recentVar_.setZero(getObDim());
      delta_.setZero(getObDim());
      epsilon.setZero(getObDim());
      epsilon.setConstant(1e-8);
    }
  }

  void synchronize() {
    for (int i = 0; i< num_envs_; i++)
      environments_[i]->synchronize();
  }

  // resets all environments and returns observation
  void reset() {
#pragma omp parallel for schedule(auto)
    for (int i = 0; i < num_envs_; i++)
      environments_[i]->reset();
  }

//  void synchronize() {
//#pragma omp parallel for schedule(auto)
//    for (int i = 1; i < num_envs_; i++){
////      environments_[i] = nullptr;
////      environments_[i] = environments_[0];
//    }
//
//  }

  void getAnchorHistory(Eigen::Ref<EigenRowMajorMat> &anchor, bool is_robotFrame=true) {
#pragma omp parallel for schedule(auto)
    for (int i = 0; i < num_envs_; i++)
      environments_[i]->get_anchor_history(anchor.row(i), is_robotFrame);
  }

  void observe_Rollout(Eigen::Ref<EigenRowMajorMat> &ob, bool updateStatistics=false) {
    environments_[0]->observe_Rollout(ob);
  }

  Eigen::VectorXd get_target_pos() {
    return environments_[0]->get_target_pos();
  }

  void getPrivilegedInformation(Eigen::Ref<EigenRowMajorMat> &value){
#pragma omp parallel for schedule(auto)
    for (int i = 0; i < num_envs_; i++)
      environments_[i]->get_env_value(value.row(i));
  }

  void observe(Eigen::Ref<EigenRowMajorMat> &ob, bool updateStatistics=false) {
#pragma omp parallel for schedule(auto)
    for (int i = 0; i < num_envs_; i++)
      environments_[i]->observe(ob.row(i));
//
//    (ob)
    if (normalizeObservation_)
      updateObservationStatisticsAndNormalize(ob, updateStatistics);
  }

  std::vector<std::string> getStepDataTag() {
    return environments_[0]->getStepDataTag();
  }

  int getStepData(int sample_size,
                  Eigen::Ref<EigenDoubleVec> &mean,
                  Eigen::Ref<EigenDoubleVec> &squareSum,
                  Eigen::Ref<EigenDoubleVec> &min,
                  Eigen::Ref<EigenDoubleVec> &max) {
    size_t data_size = getStepDataTag().size();
    if( data_size == 0 ) return sample_size;

    RSFATAL_IF(mean.size() != data_size ||
        squareSum.size() != data_size ||
        min.size() != data_size ||
        max.size() != data_size, "vector size mismatch")

    mean *= sample_size;

    for (int i = 0; i < num_envs_; i++) {
      mean += environments_[i]->getStepData();
      for (int j = 0; j < data_size; j++) {
        min(j) = std::min(min(j), environments_[i]->getStepData()[j]);
        max(j) = std::max(max(j), environments_[i]->getStepData()[j]);
      }
    }

    sample_size += num_envs_;
    mean /= sample_size;
    for (int i = 0; i < num_envs_; i++) {
      for (int j = 0; j < data_size; j++) {
        double temp = environments_[i]->getStepData()[j];
        squareSum[j] += temp * temp;
      }
    }

    return sample_size;
  }

  void getSuccess(Eigen::Ref<EigenBoolVec> &success) {
#pragma omp parallel for schedule(auto)
    for (int i = 0; i< num_envs_; i++) {
      environments_[i]->check_success(success[i]);
    }
  }

  void getIntrinsicSwitch(Eigen::Ref<EigenBoolVec> &intrinsic_switch) {
#pragma omp parallel for schedule(auto)
    for (int i = 0; i< num_envs_; i++) {
       environments_[i]->check_intrinsic_switch(intrinsic_switch[i]);
    }
  }

  void getState(Eigen::Ref<EigenVec> gc, Eigen::Ref<EigenVec> gv) {
    environments_[0]->getState(gc, gv);
  }

  double get_error(bool get, Eigen::Ref<EigenVec> anchor ){
      return environments_[0]->get_estimation_error(get, anchor);
  }

  void getContact(Eigen::Ref<EigenBoolVec> &contact) {
#pragma omp parallel for schedule(auto)
    for (int i = 0; i < num_envs_; i++) {
      contact[i] = environments_[i]->get_contact();
    }
  }

  void getState_Rollout(Eigen::Ref<EigenRowMajorMat> gc, Eigen::Ref<EigenRowMajorMat> gv) {
    environments_[0]->getState_Rollout(gc, gv);
  }

  void predict_obj_update(Eigen::Ref<EigenRowMajorMat> predict_state_batch) {
    environments_[0]->predict_obj_update(predict_state_batch);
  }

  void step_Rollout(Eigen::Ref<EigenRowMajorMat> &action) {
    perAgentStep_Rollout(0, action, false);
  }

  void step(Eigen::Ref<EigenRowMajorMat> &action,
            Eigen::Ref<EigenVec> &reward,
            Eigen::Ref<EigenBoolVec> &done) {
#pragma omp parallel for schedule(auto)
    for (int i = 0; i < num_envs_; i++)
      perAgentStep(i, action, reward, done, false);
  }

  void step_visualize(Eigen::Ref<EigenRowMajorMat> &action,
                      Eigen::Ref<EigenVec> &reward,
                      Eigen::Ref<EigenBoolVec> &done) {
#pragma omp parallel for schedule(auto)
    for (int i = 0; i < num_envs_; i++)
      perAgentStep(i, action, reward, done, true);
  }


  void step_evaluate(Eigen::Ref<EigenRowMajorMat> &action,
                     Eigen::Ref<EigenRowMajorMat> &anchors,
                        Eigen::Ref<EigenVec> &reward,
                        Eigen::Ref<EigenBoolVec> &done) {
#pragma omp parallel for schedule(auto)
        for (int i = 0; i < num_envs_; i++)
            perAgentStep_eval(i, action,anchors, reward, done, true);
    }


    void step_visualize_success(Eigen::Ref<EigenRowMajorMat> &action,
                              Eigen::Ref<EigenVec> &reward,
                              Eigen::Ref<EigenBoolVec> &done,
                              Eigen::Ref<EigenBoolVec> &success) {
#pragma omp parallel for schedule(auto)
    for (int i = 0; i< num_envs_; i++)
    {
      if(success[i] == false)
        perAgentStep(i, action, reward, done, true);
    }
  }

  std::vector<std::vector<float>> getDepthImage() {
    return environments_[0]->getDepthImage();
  }
  std::vector<std::vector<int>> getColorImage() {
    return environments_[0]->getColorImage();
  }

  void turnOnVisualization() { if(render_) environments_[0]->turnOnVisualization(); }
  void turnOffVisualization() { if(render_) environments_[0]->turnOffVisualization(); }
  void startRecordingVideo(const std::string& videoName) { if(render_) environments_[0]->startRecordingVideo(videoName); }
  void stopRecordingVideo() { if(render_) environments_[0]->stopRecordingVideo(); }
  void getObStatistics(Eigen::Ref<EigenVec> &mean, Eigen::Ref<EigenVec> &var, float &count) {
    mean = obMean_; var = obVar_; count = obCount_; }
  void setObStatistics(Eigen::Ref<EigenVec> &mean, Eigen::Ref<EigenVec> &var, float count) {
    obMean_ = mean; obVar_ = var; obCount_ = count; }

  void setSeed(int seed) {
    int seed_inc = seed;
#pragma omp parallel for schedule(auto)
    for (auto *env: environments_)
      env->setSeed(seed_inc++);
  }

  void moveControllerCursor(int id, Eigen::Ref<EigenVec> pos) {
    if (environments_.size() > id)
      environments_[id]->moveControllerCursor(pos);
  }

  void setCommand(int id) {
    if (environments_.size() > id)
      environments_[id]->setCommand();
  }

  void close() {
    for (auto *env: environments_)
      env->close();
  }

  void isTerminalState(Eigen::Ref<EigenBoolVec>& terminalState) {
    for (int i = 0; i < num_envs_; i++) {
      float terminalReward;
      terminalState[i] = environments_[i]->isTerminalState(terminalReward);
    }
  }

  void get_obj_pos (Eigen::Ref<EigenRowMajorMat> &obj_pos) {
    environments_[0]->get_obj_pos(obj_pos);
  }

  void setSimulationTimeStep(double dt) {
    for (auto *env: environments_)
      env->setSimulationTimeStep(dt);
  }

  void setControlTimeStep(double dt, double low_dt) {
    for (auto *env: environments_)
      env->setControlTimeStep(dt, low_dt);
  }

  int getObDim() { return ChildEnvironment::getObDim(); }
  int getActionDim() { return ChildEnvironment::getActionDim(); }
  int getNumOfEnvs() { return num_envs_; }

  ////// optional methods //////
  void curriculumUpdate() {
    for (auto *env: environments_)
      env->curriculumUpdate();
  };

  void ObservationDeNormalize(Eigen::Ref<EigenRowMajorMat> &ob) {
#pragma omp parallel for schedule(auto)
    for(int i=0; i<num_envs_; i++)
      ob.row(i) = ob.row(i).template cwiseProduct((obVar_ + epsilon).cwiseSqrt().transpose()) + obMean_.transpose();
  }

  void getCameraPose(Eigen::Ref<EigenVec> &pos, Eigen::Ref<EigenRowMajorMat> &rot) {
    environments_[0]->getCameraPose(pos, rot);
  }

 private:

  void updateObservationStatisticsAndNormalize(Eigen::Ref<EigenRowMajorMat> &ob, bool updateStatistics) {
    if (updateStatistics) {
      recentMean_ = ob.colwise().mean();
      recentVar_ = (ob.rowwise() - recentMean_.transpose()).colwise().squaredNorm() / num_envs_;

      delta_ = obMean_ - recentMean_;
      for (int i = 0; i < getObDim(); i++)
        delta_[i] = delta_[i] * delta_[i];

      float totCount = obCount_ + num_envs_;

      obMean_ = obMean_ * (obCount_ / totCount) + recentMean_ * (num_envs_ / totCount);
      obVar_ = (obVar_ * obCount_ + recentVar_ * num_envs_ + delta_ * (obCount_ * num_envs_ / totCount)) / (totCount);
      obCount_ = totCount;
    }

#pragma omp parallel for schedule(auto)
    for(int i=0; i<num_envs_; i++)
      ob.row(i) = (ob.row(i) - obMean_.transpose()).template cwiseQuotient((obVar_ + epsilon).cwiseSqrt().transpose());
  }



  inline void perAgentStep_Rollout(int agentId,
                           Eigen::Ref<EigenRowMajorMat> &action,
                           bool visualize) {
    environments_[agentId]->rollout_step(action, visualize);
  }


  inline void perAgentStep(int agentId,
                           Eigen::Ref<EigenRowMajorMat> &action,
                           Eigen::Ref<EigenVec> &reward,
                           Eigen::Ref<EigenBoolVec> &done,
                           bool visualize) {
    reward[agentId] = environments_[agentId]->step(action.row(agentId), visualize);

    float terminalReward = 0;
    done[agentId] = environments_[agentId]->isTerminalState(terminalReward);

    if (done[agentId]) {
      environments_[agentId]->reset();
//      std::cout << "agentID : " << agentId << "Done !" << std::endl;
      reward[agentId] += terminalReward;
    }
  }

    inline void perAgentStep_eval(int agentId,
                             Eigen::Ref<EigenRowMajorMat> &action,
                             Eigen::Ref<EigenRowMajorMat> &anchors,
                             Eigen::Ref<EigenVec> &reward,
                             Eigen::Ref<EigenBoolVec> &done,
                             bool visualize) {
        reward[agentId] = environments_[agentId]->step_evaluate(action.row(agentId), visualize, anchors.row(agentId));

        float terminalReward = 0;
        done[agentId] = environments_[agentId]->isTerminalState(terminalReward);

        if (done[agentId]) {
            environments_[agentId]->reset();
//      std::cout << "agentID : " << agentId << "Done !" << std::endl;
            reward[agentId] += terminalReward;
        }
    }

  std::vector<ChildEnvironment *> environments_;

  int num_envs_ = 1;
  bool render_=false;
  std::string resourceDir_;
  Yaml::Node cfg_;

  /// observation running mean
  bool normalizeObservation_ = true;
  EigenVec obMean_;
  EigenVec obVar_;
  float obCount_ = 1e-4;
  EigenVec recentMean_, recentVar_, delta_;
  EigenVec epsilon;
  bool is_rollout_;
  bool is_position_goal_;

};

class NormalDistribution {
 public:
  NormalDistribution() : normDist_(0.f, 1.f) {}

  float sample() { return normDist_(gen_); }
  void seed(int i) { gen_.seed(i); }

 private:
  std::normal_distribution<float> normDist_;
  static thread_local std::mt19937 gen_;
};

class DiscreteDistribution {
 public:
  DiscreteDistribution(): distDist_{1,2,3} {}

  void reinit(const Eigen::Ref<EigenVec> &prob) {
    distDist_.param({prob.begin(), prob.end()});
  }

  const Eigen::Ref<EigenVec> sample(int dim, int &idx){
    idx = distDist_(gen_);
    EigenVec sample_vector(dim, 1);
    sample_vector.setZero();
    Eigen::Ref<EigenVec> sample_vec(sample_vector);
    sample_vec[idx] = 1;
    return sample_vec;
    }

  void seed(int i) {gen_.seed(i);}
  double logprob(int idx) {
    return std::log(distDist_.probabilities()[idx]);
  }

 private:
  std::discrete_distribution<int> distDist_;
  static thread_local std::mt19937 gen_;
};

thread_local std::mt19937 raisim::NormalDistribution::gen_;
thread_local std::mt19937 raisim::DiscreteDistribution::gen_;

class DiscreteSampler {
 public:
  DiscreteSampler(int dim) {
    dim_ = dim;
    discrete_.resize(THREAD_COUNT);
    seed(0);
  }

  void seed(int seed) {
#pragma omp parallel for schedule(static, 1)
    for (int i = 0; i < THREAD_COUNT; i++)
      discrete_[0].seed(i + seed);
  }

  inline void sample(Eigen::Ref<EigenRowMajorMat> &prob,
                     Eigen::Ref<EigenRowMajorMat> &samples,
                     Eigen::Ref<EigenVec> &log_prob) {
    int agentNumber = log_prob.rows();
    int idx = 0;
#pragma omp parallel for schedule(auto)
    for (int agentId = 0; agentId < agentNumber; agentId++) {
      discrete_[omp_get_thread_num()].reinit(prob.row(agentId));
      samples.row(agentId) = discrete_[omp_get_thread_num()].sample(dim_, idx);
      log_prob(agentId) = discrete_[omp_get_thread_num()].logprob(idx);
    }
  }

  int dim_;
  std::vector<DiscreteDistribution> discrete_;
};

class NormalSampler {
 public:
  NormalSampler(int dim) {
    dim_ = dim;
    normal_.resize(THREAD_COUNT);
    seed(0);
  }

  void seed(int seed) {
    // this ensures that every thread gets a different seed
#pragma omp parallel for schedule(static, 1)
    for (int i = 0; i < THREAD_COUNT; i++)
      normal_[0].seed(i + seed);
  }

  inline void sample(Eigen::Ref<EigenRowMajorMat> &mean,
                     Eigen::Ref<EigenVec> &std,
                     Eigen::Ref<EigenRowMajorMat> &samples,
                     Eigen::Ref<EigenVec> &log_prob) {
    int agentNumber = log_prob.rows();

#pragma omp parallel for schedule(auto)
    for (int agentId = 0; agentId < agentNumber; agentId++) {
      log_prob(agentId) = 0;
      for (int i = 0; i < dim_; i++) {
        const float noise = normal_[omp_get_thread_num()].sample();
        samples(agentId, i) = mean(agentId, i) + noise * std(i);
        log_prob(agentId) -= noise * noise * 0.5 + std::log(std(i));
      }
      log_prob(agentId) -= float(dim_) * 0.9189385332f;
    }
  }

  int dim_;
  std::vector<NormalDistribution> normal_;
};

}

#endif //SRC_RAISIMGYMVECENV_HPP