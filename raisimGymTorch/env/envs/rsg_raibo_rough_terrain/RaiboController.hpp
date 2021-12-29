//
// Created by jemin on 2/25/20.
//

#ifndef _RAISIM_GYM_RAIBO_CONTROLLER_HPP
#define _RAISIM_GYM_RAIBO_CONTROLLER_HPP

namespace raisim {

class RaiboController {
 public:
  inline bool create(raisim::World *world) {
    raibo_ = reinterpret_cast<raisim::ArticulatedSystem *>(world->getObject("robot"));
    gc_.resize(raibo_->getGeneralizedCoordinateDim());
    gv_.resize(raibo_->getDOF());
    jointVelocity_.resize(12);

    /// foot scan config
    scanConfig_.setZero(4);
    scanConfig_ << 6, 8, 10, 12;
    scanPoint_.resize(4, std::vector<raisim::Vec<2>>(scanConfig_.sum()));
    heightScan_.resize(4, raisim::VecDyn(scanConfig_.sum()));

    /// Observation
    jointPositionHistory_.setZero(nJoints_ * historyLength_);
    jointVelocityHistory_.setZero(nJoints_ * historyLength_);
    historyTempMemory_.setZero(nJoints_ * historyLength_);
    nominalJointConfig_.setZero(nJoints_);
    nominalJointConfig_ << 0.03, 0.4, -0.8, -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8;
    jointTarget_.setZero(nJoints_);

    /// action
    actionMean_.setZero(actionDim_);
    actionStd_.setZero(actionDim_);
    actionScaled_.setZero(actionDim_);
    previousAction_.setZero(actionDim_);

    actionMean_ << nominalJointConfig_; /// joint target
    actionStd_ << Eigen::VectorXd::Constant(12, 0.35); /// joint target

    obMean_.setZero(obDim_);
    obStd_.setZero(obDim_);
    obDouble_.setZero(obDim_);
    pTarget_.setZero(gcDim_);
    vTarget_.setZero(gvDim_);

    obMean_ << 0.5, /// average height
        0.0, 0.0, 1.4, /// gravity axis 3
        Eigen::VectorXd::Constant(6, 0.0), /// body lin/ang vel 6
        nominalJointConfig_, /// joint pos
        Eigen::VectorXd::Constant(nJoints_ * (3 - 1), 0.0), /// joint position error history
        Eigen::VectorXd::Constant(nJoints_ * 3, 0.0), /// joint vel history
        Eigen::VectorXd::Constant(scanConfig_.sum() * 4, -0.03), /// height scan
        Eigen::VectorXd::Constant(4, 0.), /// airtime
        nominalJointConfig_, /// previous action
        Eigen::VectorXd::Constant(3, 0.0); /// command

    obStd_ << 0.05, /// height
        Eigen::VectorXd::Constant(3, 0.3), /// gravity axes
        Eigen::VectorXd::Constant(3, 0.6), /// linear velocity
        Eigen::VectorXd::Constant(3, 1.0), /// angular velocities
        Eigen::VectorXd::Constant(nJoints_, 1.), /// joint angles
        Eigen::VectorXd::Constant(nJoints_ * (3 - 1), 0.6), /// joint position error history
        Eigen::VectorXd::Constant(nJoints_ * 3, 10.0), /// joint velocities
        Eigen::VectorXd::Constant(scanConfig_.sum() * 4, 0.1),
        Eigen::VectorXd::Constant(4, 0.3),
        actionStd_ * 1.5, /// previous action
        .5, 0.3, 0.6; /// command

    airTime_.setZero();
    stanceTime_.setZero();

    /// indices of links that should not make contact with ground
    footIndices_.push_back(raibo_->getBodyIdx("LF_SHANK"));
    footIndices_.push_back(raibo_->getBodyIdx("RF_SHANK"));
    footIndices_.push_back(raibo_->getBodyIdx("LH_SHANK"));
    footIndices_.push_back(raibo_->getBodyIdx("RH_SHANK"));
    RSFATAL_IF(std::any_of(footIndices_.begin(), footIndices_.end(), [](int i){return i < 0;}), "footIndices_ not found")

    /// indicies of the foot frame
    footFrameIndicies_.push_back(raibo_->getFrameIdxByName("LF_S2F"));
    footFrameIndicies_.push_back(raibo_->getFrameIdxByName("RF_S2F"));
    footFrameIndicies_.push_back(raibo_->getFrameIdxByName("LH_S2F"));
    footFrameIndicies_.push_back(raibo_->getFrameIdxByName("RH_S2F"));
    RSFATAL_IF(std::any_of(footFrameIndicies_.begin(), footFrameIndicies_.end(), [](int i){return i < 0;}), "footFrameIndicies_ not found")

    /// exported data
    stepDataTag_ = {"command_rew",
                    "height_rew",
                    "torque_rew",
                    "smooth_rew",
                    "ori_rew",
                    "joint_vel_rew",
                    "slip_rew",
                    "airtime_rew"};
    stepData_.resize(stepDataTag_.size());
    return true;
  };

  void updateHistory() {
    /// joint angles
    historyTempMemory_ = jointPositionHistory_;
    jointPositionHistory_.head((historyLength_ - 1) * nJoints_) =
        historyTempMemory_.tail((historyLength_ - 1) * nJoints_);
    jointPositionHistory_.tail(nJoints_) = jointTarget_ - gc_.tail(nJoints_);

    /// joint velocities
    historyTempMemory_ = jointVelocityHistory_;
    jointVelocityHistory_.head((historyLength_ - 1) * nJoints_) =
        historyTempMemory_.tail((historyLength_ - 1) * nJoints_);
    jointVelocityHistory_.tail(nJoints_) = gv_.tail(nJoints_);
  }

  void updateStateVariables() {
    raibo_->getState(gc_, gv_);
    jointVelocity_ = gv_.tail(nJoints_);

    raisim::Vec<4> quat;
    quat[0] = gc_[3];
    quat[1] = gc_[4];
    quat[2] = gc_[5];
    quat[3] = gc_[6];
    raisim::quatToRotMat(quat, baseRot_);
    bodyLinVel_ = baseRot_.e().transpose() * gv_.segment(0, 3);
    bodyAngVel_ = baseRot_.e().transpose() * gv_.segment(3, 3);

    /// foot info
    for (size_t i = 0; i < 4; i++) {
      raibo_->getFramePosition(footFrameIndicies_[i], footPos_[i]);
      raibo_->getFrameVelocity(footFrameIndicies_[i], footVel_[i]);
    }

    /// height map
    controlFrameX_ =
        {baseRot_[0], baseRot_[1], 0.}; /// body x axis projected on the world x-y plane, expressed in the world frame
    controlFrameX_ /= controlFrameX_.norm();
    raisim::cross(zAxis_, controlFrameX_, controlFrameY_);

    /// check if the feet are in contact with the ground
    for (auto &fs: footContactState_) fs = false;
    for (auto &contact: raibo_->getContacts()) {
      auto it = std::find(footIndices_.begin(), footIndices_.end(), contact.getlocalBodyIndex());
      size_t index = it - footIndices_.begin();
      if (index < 4)
        footContactState_[index] = true;
    }

    /// airtime & standtime
    for (int i = 0; i < 4; i++) {
      if (footContactState_[i]) {
        airTime_[i] = 0;
        stanceTime_[i] += simDt_;
      } else {
        airTime_[i] += simDt_;
        stanceTime_[i] = 0;
      }
    }
  }

  void getObservation(Eigen::VectorXd &observation) {
    observation = (obDouble_ - obMean_).cwiseQuotient(obStd_);
  }

  bool advance(raisim::World *world, const Eigen::Ref<EigenVec> &action, double curriculumFactor) {
    /// action scaling
    previousAction_ = jointTarget_;
    jointTarget_ = action.cast<double>();
    jointTarget_ = jointTarget_.cwiseProduct(actionStd_);
    jointTarget_ += actionMean_;
    pTarget_.tail(nJoints_) = jointTarget_;
    raibo_->setPdTarget(pTarget_, vTarget_);

    smoothReward_ = curriculumFactor * smoothRewardCoeff_ * (jointTarget_ - previousAction_).squaredNorm();
    return true;
  }

  void reset(std::mt19937 &gen_,
             std::normal_distribution<double> &normDist_) {
    previousAction_ = nominalJointConfig_;
    airTime_.setZero();
    stanceTime_.setZero();
    // history
    for (size_t i = 0; i < nJoints_ * historyLength_; i++)
      jointPositionHistory_[i] = normDist_(gen_) * .1;

    for (size_t i = 0; i < nJoints_ * historyLength_; i++)
      jointVelocityHistory_[i] = normDist_(gen_) * 1.0;
  }

  float getRewardSum() {
    stepData_[0] = commandTrackingReward_;
    stepData_[1] = heightReward_;
    stepData_[2] = torqueReward_;
    stepData_[3] = smoothReward_;
    stepData_[4] = orientationReward_;
    stepData_[5] = jointVelocityReward_;
    stepData_[6] = slipReward_;
    stepData_[7] = airtimeReward_;

    commandTrackingReward_ = 0.;
    heightReward_ = 0.;
    torqueReward_ = 0.;
    smoothReward_ = 0.;
    orientationReward_ = 0.;
    jointVelocityReward_ = 0.;
    slipReward_ = 0.;
    airtimeReward_ = 0.;

    return stepData_.sum();
  }

  bool isTerminalState(float &terminalReward) {
    terminalReward = float(terminalRewardCoeff_);

    /// if the contact body is not feet
    for (auto &contact: raibo_->getContacts())
      if (std::find(footIndices_.begin(), footIndices_.end(), contact.getlocalBodyIndex()) == footIndices_.end())
        return true;

    terminalReward = 0.f;
    return false;
  }

  void updateObservation(bool nosify,
                         const Eigen::Vector3d &command_,
                         const raisim::HeightMap *map,
                         std::mt19937 &gen_,
                         std::normal_distribution<double> &normDist_) {
    updateHeightScan(map);

    /// height of the origin of the body frame
    obDouble_[0] = 0.5 + nosify * (normDist_(gen_) * 0.01);

    /// body orientation
    obDouble_.segment(1, 3) = baseRot_.e().row(2);

    /// body velocities
    obDouble_.segment(4, 3) = bodyLinVel_;
    obDouble_.segment(7, 3) = bodyAngVel_;

    /// except the first joints, the joint history stores target-position
    obDouble_.segment(10, nJoints_) = gc_.tail(12);
    obDouble_.segment(22, 12) = jointPositionHistory_.segment((historyLength_ - 1 - 5) * 12, 12);
    obDouble_.segment(34, 12) = jointPositionHistory_.segment((historyLength_ - 1 - 3) * 12, 12);

    obDouble_.segment(46, 12) = jointVelocityHistory_.segment((historyLength_ - 1 - 5) * 12, 12);
    obDouble_.segment(58, 12) = jointVelocityHistory_.segment((historyLength_ - 1 - 3) * 12, 12);
    obDouble_.segment(70, 12) = jointVelocityHistory_.segment((historyLength_ - 1) * 12, 12);

    for (int i = 0; i < 4; i++)
      for (int j = 0; j < scanConfig_.sum(); j++)
        obDouble_[10 + 2 * nJoints_ * 3 + i * scanConfig_.sum() + j] = heightScan_[i][j];

    for (int i = 0; i < 4; i++)
      obDouble_[10 + 2 * nJoints_ * 3 + 4 * scanConfig_.sum() + i] =
          airTime_[i] > 0 ? airTime_[i] : -stanceTime_[i];

    /// previous action
    obDouble_.segment(14 + 2 * nJoints_ * 3 + 4 * scanConfig_.sum(), 12) = previousAction_;

    /// command
    obDouble_.segment(26 + 2 * nJoints_ * 3 + 4 * scanConfig_.sum(), 3) = command_;
  }

  inline void setRewardConfig(const Yaml::Node &cfg) {
    READ_YAML(double, commandTrackingRewardCoeff, cfg["reward"]["command_tracking_reward_coeff"])
    READ_YAML(double, torqueRewardCoeff_, cfg["reward"]["torque_reward_coeff"])
    READ_YAML(double, smoothRewardCoeff_, cfg["reward"]["smooth_reward_coeff"])
    READ_YAML(double, orientationRewardCoeff_, cfg["reward"]["orientation_reward_coeff"])
    READ_YAML(double, jointVelocityRewardCoeff_, cfg["reward"]["joint_velocity_reward_coeff"])
    READ_YAML(double, slipRewardCoeff_, cfg["reward"]["slip_reward_coeff"])
    READ_YAML(double, airtimeRewardCoeff_, cfg["reward"]["airtime_reward_coeff"])
    READ_YAML(double, heightRewardCoeff_, cfg["reward"]["height_reward_coeff"])
  }

  inline void accumulateRewards(double curriculumFactor, const Eigen::Vector3d &command) {
    const double cf = curriculumFactor;
    torqueReward_ += cf * torqueRewardCoeff_ * (raibo_->getGeneralizedForce().e().tail(12).squaredNorm()) * simDt_;
    commandTrackingReward_ +=
        command[0] > 0 ? std::min(bodyLinVel_[0], command[0]) : -std::max(bodyLinVel_[0], command[0]);
    commandTrackingReward_ +=
        command[1] > 0 ? std::min(bodyLinVel_[1], command[1]) : -std::max(bodyLinVel_[1], command[1]);
    commandTrackingReward_ +=
        command[2] > 0 ? std::min(bodyAngVel_[2], command[2]) : -std::max(bodyAngVel_[2], command[2]);
    commandTrackingReward_ -= 0.65 * bodyLinVel_[2] * bodyLinVel_[2];
    commandTrackingReward_ -= 0.2 * fabs(bodyAngVel_[0]);
    commandTrackingReward_ -= 0.2 * fabs(bodyAngVel_[1]);
    commandTrackingReward_ *= commandTrackingRewardCoeff * simDt_;
    orientationReward_ += cf * orientationRewardCoeff_ * simDt_ * std::acos(baseRot_[8]) * std::acos(baseRot_[8]);
    jointVelocityReward_ += cf * jointVelocityRewardCoeff_ * simDt_ * jointVelocity_.squaredNorm();

    for (size_t i = 0; i < 4; i++)
      if (footContactState_[i])
        slipReward_ += cf * slipRewardCoeff_ * footVel_[i].e().head(2).squaredNorm();

    if (standingMode_) {
      for (int i = 0; i < 4; i++) {
        airtimeReward_ += std::min(stanceTime_[i], 0.40) * airtimeRewardCoeff_;
      }
    } else {
      for (int i = 0; i < 4; i++)
        if (airTime_[i] < 0.41)
          airtimeReward_ += std::min(airTime_[i], 0.3) * airtimeRewardCoeff_;

      for (int i = 0; i < 4; i++)
        if (stanceTime_[i] < 0.55)
          airtimeReward_ += std::min(stanceTime_[i], 0.40) * airtimeRewardCoeff_;
    }

    heightReward_ += cf * heightRewardCoeff_ * gv_[2] * simDt_;
  }

  void updateHeightScan(const raisim::HeightMap *map) {
    /// heightmap
    for (int i = 0; i < 4; i++) {
      for (int k = 0; k < scanConfig_.size(); k++) {
        for (int j = 0; j < scanConfig_[k]; j++) {
          const double distance = 0.07 * (k + 1);
          const double angle = 2.0 * M_PI * double(j) / scanConfig_[k];
          scanPoint_[i][scanConfig_.head(k).sum() + j][0] =
              footPos_[i][0] + controlFrameX_[0] * distance * cos(angle) + controlFrameY_[0] * distance * sin(angle);
          scanPoint_[i][scanConfig_.head(k).sum() + j][1] =
              footPos_[i][1] + controlFrameX_[1] * distance * cos(angle) + controlFrameY_[1] * distance * sin(angle);
          heightScan_[i][scanConfig_.head(k).sum() + j] =
              map->getHeight(scanPoint_[i][scanConfig_.head(k).sum() + j][0],
                             scanPoint_[i][scanConfig_.head(k).sum() + j][1]) - footPos_[i][2];
        }
      }
    }
  }

  inline void setStandingMode(bool mode) { standingMode_ = mode; }

  const Eigen::VectorXd &getJointPositionHistory() const { return jointPositionHistory_; }
  const Eigen::VectorXd &getJointVelocityHistory() const { return jointVelocityHistory_; }
  const Eigen::Vector4d &getAirTime() const { return airTime_; }
  const Eigen::Vector4d &getStanceTime() const { return stanceTime_; }

  int getObDim() { return obDim_; }
  int getActionDim() { return actionDim_; }

  static double getSimDt() { return simDt_; }
  static double getConDt() { return conDt_; }

  void setSimDt(double dt) { RSFATAL_IF(fabs(dt - simDt_) > 1e-12, "sim dt is fixed to " << simDt_)};
  void setConDt(double dt) { RSFATAL_IF(fabs(dt - conDt_) > 1e-12, "con dt is fixed to " << conDt_)};

  inline const std::vector<std::string> &getStepDataTag() { return stepDataTag_; }
  inline const Eigen::VectorXd &getStepData() { return stepData_; }

  // robot configuration variables
  raisim::ArticulatedSystem *raibo_;
  std::vector<size_t> footIndices_, footFrameIndicies_;
  Eigen::VectorXd nominalJointConfig_;
  static constexpr int nJoints_ = 12;
  static constexpr int actionDim_ = 12;
  static constexpr size_t historyLength_ = 14;
  static constexpr size_t obDim_ = 245;
  static constexpr double simDt_ = .0025;
  static constexpr int gcDim_ = 19;
  static constexpr int gvDim_ = 18;

  // robot state variables
  Eigen::VectorXd gc_, gv_;
  Eigen::Vector3d bodyLinVel_, bodyAngVel_; /// body velocities are expressed in the body frame
  Eigen::VectorXd jointVelocity_;
  std::array<raisim::Vec<3>, 4> footPos_, footVel_;
  raisim::Vec<3> zAxis_ = {0., 0., 1.}, controlFrameX_, controlFrameY_;
  Eigen::VectorXd jointPositionHistory_;
  Eigen::VectorXd jointVelocityHistory_;
  Eigen::VectorXd historyTempMemory_;
  Eigen::Vector4d airTime_, stanceTime_;
  std::array<bool, 4> footContactState_;
  raisim::Mat<3, 3> baseRot_;

  // robot observation variables
  std::vector<raisim::VecDyn> heightScan_;
  Eigen::VectorXi scanConfig_;
  Eigen::VectorXd obDouble_, obMean_, obStd_;
  std::vector<std::vector<raisim::Vec<2>>> scanPoint_;

  // control variables
  static constexpr double conDt_ = 0.02;
  bool standingMode_ = false;
  Eigen::VectorXd actionMean_, actionStd_, actionScaled_, previousAction_;
  Eigen::VectorXd pTarget_, vTarget_; // full robot gc dim
  Eigen::VectorXd jointTarget_;

  // reward variables
  double commandTrackingRewardCoeff = 0., commandTrackingReward_ = 0.;
  double heightRewardCoeff_ = 0., heightReward_ = 0.;
  double torqueRewardCoeff_ = 0., torqueReward_ = 0.;
  double smoothRewardCoeff_ = 0., smoothReward_ = 0.;
  double orientationRewardCoeff_ = 0., orientationReward_ = 0.;
  double jointVelocityRewardCoeff_ = 0., jointVelocityReward_ = 0.;
  double slipRewardCoeff_ = 0., slipReward_ = 0.;
  double airtimeRewardCoeff_ = 0., airtimeReward_ = 0.;
  double terminalRewardCoeff_ = 0.0;

  // exported data
  Eigen::VectorXd stepData_;
  std::vector<std::string> stepDataTag_;
};

}

#endif //_RAISIM_GYM_RAIBO_CONTROLLER_HPP
