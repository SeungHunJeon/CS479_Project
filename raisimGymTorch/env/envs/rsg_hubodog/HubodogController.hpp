// Copyright (c) 2020 Robotics and Artificial Intelligence Lab, KAIST
//
// Any unauthorized copying, alteration, distribution, transmission,
// performance, display or use of this material is prohibited.
//
// All rights reserved.

#pragma once

#include <set>
#include <map>
#include <cstdint>
#include "../../BasicEigenTypes.hpp"
#include "raisim/World.hpp"

namespace raisim {

class HubodogController {

 public:
  enum class RewardType : int {
    VELOCITY = 1,
    SLIP,
    JOINT_VELOCITY,
    JOINT_ACCELERATION,
    AIRTIME,
    SMOOTHNESS1,
    SMOOTHNESS2,
    TORQUE
  };

  void create(raisim::World *world) {
    auto* hubodog = reinterpret_cast<raisim::ArticulatedSystem*>(world->getObject("robot"));

    /// get robot data
    gcDim_ = hubodog->getGeneralizedCoordinateDim();
    gvDim_ = hubodog->getDOF();
    nJoints_ = gvDim_ - 6;

    /// initialize containers
    gc_.setZero(gcDim_); gc_init_.setZero(gcDim_);
    gv_.setZero(gvDim_); gv_init_.setZero(gvDim_);
    pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_);
    pTarget12_.setZero(nJoints_); targetT_1_.setZero(nJoints_); targetT_2_.setZero(nJoints_);

    /// this is nominal configuration of hubodog
    gc_init_ << 0, 0, 0.43, 1.0, 0.0, 0.0, 0.0, 0.0, 0.4, -0.8, -0.0, 0.4, -0.8, 0.0, 0.4, -0.8, -0.0, 0.4, -0.8;

    /// set pd gains
    Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
    jointPgain.setZero(); jointPgain.tail(nJoints_).setConstant(50.0);
    jointDgain.setZero(); jointDgain.tail(nJoints_).setConstant(0.3);
    hubodog->setPdGains(jointPgain, jointDgain);
    hubodog->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    actionDim_ = nJoints_;
    preJointVel_.setZero(nJoints_);
    actionMean_.setZero(actionDim_); actionStd_.setZero(actionDim_);
    jointVelHist_.setZero(nJoints_ * historyLength_);
    jointErrorHist_.setZero(nJoints_ * historyLength_);
    historyTempMem_.setZero(nJoints_ * historyLength_);
    airTime_.setZero(4);

    actionMean_.setZero(actionDim_); actionStd_.setZero(actionDim_);

    /// action scaling
    actionMean_ = gc_init_.tail(nJoints_);
    actionStd_.setConstant(0.1);

    /// indices of links that should not make contact with ground
    footIndices_.push_back(hubodog->getBodyIdx("FR_calf"));
    footIndices_.push_back(hubodog->getBodyIdx("FL_calf"));
    footIndices_.push_back(hubodog->getBodyIdx("RR_calf"));
    footIndices_.push_back(hubodog->getBodyIdx("RL_calf"));

    footFrameIndices_.push_back(hubodog->getFrameIdxByName("FR_foot_fixed"));
    footFrameIndices_.push_back(hubodog->getFrameIdxByName("FL_foot_fixed"));
    footFrameIndices_.push_back(hubodog->getFrameIdxByName("RR_foot_fixed"));
    footFrameIndices_.push_back(hubodog->getFrameIdxByName("RL_foot_fixed"));

    updateObservation(world);

    stepDataTag_ = {"vel_rew", "slip", "joint_acc_rew", "joint_vel_rew", "smooth_rew1", "smooth_rew2", "airtime", "torque", "total"};
    stepData_.resize(stepDataTag_.size());

    obDim_ = 157;
    obDouble_.setZero(obDim_);
  }

  void init(raisim::World *world) { }

  void advance(raisim::World *world, const Eigen::Ref<EigenVec>& action) {
    auto* hubodog = reinterpret_cast<raisim::ArticulatedSystem*>(world->getObject("robot"));

    targetT_2_ = targetT_1_;
    targetT_1_ = pTarget12_;

    /// action scaling
    pTarget12_ = action.cast<double>();
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    pTarget12_ += actionMean_;
    pTarget_.tail(nJoints_) = pTarget12_;

    hubodog->setPdTarget(pTarget_, vTarget_);
  }

  void setCommand(const Eigen::Vector3d& command) {
    command_ = command;
  }

  void reset(raisim::World *world, double curriculumFactor) {
    auto* hubodog = reinterpret_cast<raisim::ArticulatedSystem*>(world->getObject("robot"));
    command_ << uniDist_(gen_) * 2.0, uniDist_(gen_) * 0.7, uniDist_(gen_) * 1.0;
    if (command_.norm() > 2.5 * (1. - 0.7 * curriculumFactor)) command_ *= 2.5 * (1. - 0.7 * curriculumFactor) / command_.norm();

    Eigen::VectorXd gv_init_noisy(hubodog->getDOF());
    gv_init_noisy = gv_init_;
    gv_init_noisy[0] += uniDist_(gen_) * .5;
    gv_init_noisy[1] += uniDist_(gen_) * .5;
    gv_init_noisy[2] += uniDist_(gen_) * 0.3;
    gv_init_noisy[5] += uniDist_(gen_) * 0.8;

    hubodog->setState(gc_init_, gv_init_noisy);
    updateObservation(world);
    pTarget12_.setZero(); targetT_1_.setZero(); targetT_2_.setZero(); airTime_.setZero(); preJointVel_.setZero();
  }

  void updateHistory() {
    historyTempMem_ = jointVelHist_;
    jointVelHist_.head((historyLength_-1) * nJoints_) = historyTempMem_.tail((historyLength_-1) * nJoints_);
    jointVelHist_.tail(nJoints_) = gv_.tail(nJoints_);

    historyTempMem_ = jointErrorHist_;
    jointErrorHist_.head((historyLength_-1) * nJoints_) = historyTempMem_.tail((historyLength_-1) * nJoints_);
    jointErrorHist_.tail(nJoints_) = pTarget12_ - gc_.tail(nJoints_);
  }

  float getReward(raisim::World *world, const std::map<RewardType, float>& rewardCoeff, double simulation_dt, double curriculumFactor) {
    auto* hubodog = reinterpret_cast<raisim::ArticulatedSystem*>(world->getObject("robot"));
    updateObservation(world);
    preJointVel_ = gv_.tail(nJoints_);
    float velReward = 0;
    velReward += command_[0] > 0 ? std::min(bodyLinearVel_[0], command_[0]) : -std::max(bodyLinearVel_[0], command_[0]);
    velReward += command_[1] > 0 ? std::min(bodyLinearVel_[1], command_[1]) : -std::max(bodyLinearVel_[1], command_[1]);
    velReward += command_[2] > 0 ? std::min(bodyAngularVel_[2], command_[2]) : -std::max(bodyAngularVel_[2], command_[2]);
    velReward *= rewardCoeff.at(RewardType::VELOCITY);

    float slipExp = 0.f;

    for (int i=0; i < hubodog->getContacts().size(); i++) {
      if (hubodog->getContacts()[i].skip()) continue;
      raisim::Vec<3> vel;
      hubodog->getContactPointVel(i, vel);
      slipExp += std::sqrt(vel[0] * vel[0] + vel[1] * vel[1]);
    }
    slipExp *= rewardCoeff.at(RewardType::SLIP);

    float jointTorqueExp = rewardCoeff.at(RewardType::TORQUE) * (1.0-curriculumFactor*0.85)
        * hubodog->getGeneralizedForce().squaredNorm();
    float jointVelocityExp = rewardCoeff.at(RewardType::JOINT_VELOCITY) * (1.0-curriculumFactor*0.85)
        * gv_.tail(nJoints_).squaredNorm();
    float jointAccelerationExp = rewardCoeff.at(RewardType::JOINT_ACCELERATION) * (1.0-curriculumFactor*0.85)
        * (preJointVel_-gv_.tail(nJoints_)).squaredNorm();
    float smoothness1Exp = rewardCoeff.at(RewardType::SMOOTHNESS1) * (1.0-curriculumFactor*0.85)
        * (pTarget12_-targetT_1_).norm();
    float smoothness2Exp = rewardCoeff.at(RewardType::SMOOTHNESS2) * (1.0-curriculumFactor*0.85)
        * (pTarget12_- 2 * targetT_1_ + targetT_2_).norm();


    float airtimeRew = 0;

    for(size_t i=0; i<4; i++) {
      if (footContactState_[i])
        airTime_[i] = std::min(0., airTime_[i]) - simulation_dt;
      else
        airTime_[i] = std::max(0., airTime_[i]) + simulation_dt;

      if (airTime_[i] < 0.4 && airTime_[i] > 0.)
        airtimeRew += std::min(airTime_[i], 0.2);
      else if (airTime_[i] > -0.4 && airTime_[i] < 0.)
        airtimeRew += std::min(-airTime_[i], 0.2);
    }
    airtimeRew *= rewardCoeff.at(RewardType::AIRTIME) * (0.05 + curriculumFactor);

//      stepDataTag_ = {"vel_rew", "slip", "joint_acc_rew", "joint_vel_rew", "smooth_rew1", "smooth_rew2", "airtime", "torque", "total"};
    stepData_[0] = velReward;
    stepData_[1] = slipExp;
    stepData_[2] = jointAccelerationExp;
    stepData_[3] = jointVelocityExp;
    stepData_[4] = smoothness1Exp;
    stepData_[5] = smoothness2Exp;
    stepData_[6] = airtimeRew;
    stepData_[7] = jointTorqueExp;
    double total_reward = (airtimeRew + velReward) * exp(jointVelocityExp + jointAccelerationExp + slipExp + smoothness1Exp + smoothness2Exp + jointTorqueExp);
    stepData_[8] = total_reward;
    return total_reward;
  }

  void updateObservation(raisim::World *world) {
    auto* hubodog = reinterpret_cast<raisim::ArticulatedSystem*>(world->getObject("robot"));

    for(size_t i=0; i<4; i++)
      footContactState_[i] = false;

    for(auto& contact: hubodog->getContacts())
      for(size_t i=0; i<4; i++)
        if(contact.getlocalBodyIndex() == footIndices_[i])
          footContactState_[i] = true;


    hubodog->getState(gc_, gv_);
    quat_[0] = gc_[3]; quat_[1] = gc_[4]; quat_[2] = gc_[5]; quat_[3] = gc_[6];
    raisim::quatToRotMat(quat_, rot_);
    bodyLinearVel_ = rot_.e().transpose() * gv_.segment(0, 3);
    bodyAngularVel_ = rot_.e().transpose() * gv_.segment(3, 3);
  }

  const Eigen::VectorXd& getObservation() {
    obDouble_ << gc_[2],                 /// body height 1
        rot_.e().row(2).transpose(),   /// body orientation 3
        gc_.tail(12),                 /// joint angles 12
        bodyLinearVel_, bodyAngularVel_, /// body linear&angular velocity 6
        gv_.tail(12), /// joint velocity 12
        jointVelHist_, /// history 48
        jointErrorHist_, /// pos error 48
        targetT_1_, /// old target 12
        targetT_2_, /// old target 12
        command_; // goal position 3
    return obDouble_;
  }

  bool isTerminalState(raisim::World *world) {
    auto* hubodog = reinterpret_cast<raisim::ArticulatedSystem*>(world->getObject("robot"));
    /// if the contact body is not feet
    for(auto& contact: hubodog->getContacts())
      if(std::find(footIndices_.begin(), footIndices_.end(), contact.getlocalBodyIndex()) == footIndices_.end())
        return true;

    return false;
  }

  int getObDim() {
    return obDim_;
  }

  int getActionDim() {
    return actionDim_;
  }

  void setSeed(int seed) {
    gen_.seed(seed);
//    terrainGenerator_.setSeed(seed);
  }

  const std::vector<std::string>& getStepDataTag() {
    return stepDataTag_;
  }

  const Eigen::VectorXd& getStepData() {
    return stepData_;
  }

 private:
  int gcDim_, gvDim_, nJoints_;
  raisim::Vec<4> quat_;
  raisim::Mat<3,3> rot_;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, targetT_1_, targetT_2_, vTarget_;
  Eigen::Vector3d command_, bodyPos_;
  std::array<bool, 4> footContactState_;
  Eigen::VectorXd airTime_;
  Eigen::VectorXd actionMean_, actionStd_;
  Eigen::VectorXd obDouble_;
  Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;
  Eigen::VectorXd jointVelHist_, jointErrorHist_, historyTempMem_, preJointVel_;

//  raisim::HeightMap* heightMap_;
//  RandomHeightMapGenerator terrainGenerator_;
  std::vector<size_t> footIndices_;
  std::vector<size_t> footFrameIndices_;

  constexpr static size_t historyLength_ = 4;
  int obDim_=0, actionDim_=0;

  Eigen::VectorXd stepData_;
  std::vector<std::string> stepDataTag_;

  thread_local static std::mt19937 gen_;
  thread_local static std::normal_distribution<double> normDist_;
  thread_local static std::uniform_real_distribution<double> uniDist_;
};

thread_local std::mt19937 raisim::HubodogController::gen_;
thread_local std::normal_distribution<double> raisim::HubodogController::normDist_(0., 1.);
thread_local std::uniform_real_distribution<double> raisim::HubodogController::uniDist_(-1., 1.);

}