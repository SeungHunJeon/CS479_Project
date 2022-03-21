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
    scanConfig_.setZero(5);
    scanConfig_ << 6, 8, 10, 12, 14;
    scanPoint_.resize(4, std::vector<raisim::Vec<2>>(scanConfig_.sum()));
    heightScan_.resize(4, raisim::VecDyn(scanConfig_.sum()));

    /// Observation
    jointPositionHistory_.setZero(nJoints_ * historyLength_);
    jointVelocityHistory_.setZero(nJoints_ * historyLength_);
    historyTempMemory_.setZero(nJoints_ * historyLength_);
    nominalJointConfig_.setZero(nJoints_);
    nominalJointConfig_ << 0, 0.56, -1.12, 0, 0.56, -1.12, 0, 0.56, -1.12, 0, 0.56, -1.12;
    jointTarget_.setZero(nJoints_);
    jointTargetDelta_.setZero(nJoints_);

    /// action
    actionMean_.setZero(actionDim_);
    actionStd_.setZero(actionDim_);
    actionScaled_.setZero(actionDim_);
    previousAction_.setZero(actionDim_);
    prevprevAction_.setZero(actionDim_);

    actionMean_ << nominalJointConfig_; /// joint target
    actionStd_ << Eigen::VectorXd::Constant(12, 0.1); /// joint target

    obMean_.setZero(obDim_);
    obStd_.setZero(obDim_);
    obDouble_.setZero(obDim_);

    /// pd controller
    jointPgain_.setZero(gvDim_); jointPgain_.tail(nJoints_).setConstant(60.0);
    jointDgain_.setZero(gvDim_); jointDgain_.tail(nJoints_).setConstant(0.5);
    raibo_->setPdGains(jointPgain_, jointDgain_);
    pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_);

    /// observation
    obMean_ << 0.5, /// average height
        0.0, 0.0, 1.4, /// gravity axis 3
        Eigen::VectorXd::Constant(6, 0.0), /// body lin/ang vel 6
        nominalJointConfig_, /// joint pos
        Eigen::VectorXd::Constant(nJoints_ * (4 - 1), 0.0), /// joint position error history
        Eigen::VectorXd::Constant(nJoints_ * 4, 0.0), /// joint vel history
        Eigen::VectorXd::Constant(scanConfig_.sum() * 4, -0.03), /// height scan
        nominalJointConfig_, /// previous action
        nominalJointConfig_, /// preprev action
        Eigen::VectorXd::Constant(3, 0.0); /// command

    obStd_ << 0.05, /// height
        Eigen::VectorXd::Constant(3, 0.3), /// gravity axes
        Eigen::VectorXd::Constant(3, 0.6), /// linear velocity
        Eigen::VectorXd::Constant(3, 1.0), /// angular velocities
        Eigen::VectorXd::Constant(nJoints_, 1.), /// joint angles
        Eigen::VectorXd::Constant(nJoints_ * (4 - 1), 0.6), /// joint position error history
        Eigen::VectorXd::Constant(nJoints_ * 4, 10.0), /// joint velocities
        Eigen::VectorXd::Constant(scanConfig_.sum() * 4, 0.1),
        actionStd_ * 1.5, /// previous action
        actionStd_ * 1.5, /// previous action
        .5, 0.3, 0.6; /// command

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
                    "con_switch_rew",
                    "torque_rew",
                    "smooth_rew",
                    "ori_rew",
                    "joint_vel_rew",
                    "slip_rew",
                    "airtime_rew"};
    stepData_.resize(stepDataTag_.size());

    /// heightmap
    scanCos_.resize(scanConfig_.size(), scanConfig_.maxCoeff());
    scanSin_.resize(scanConfig_.size(), scanConfig_.maxCoeff());
    // precompute sin and cos because they take very long time
    for (int k = 0; k < scanConfig_.size(); k++) {
      for (int j = 0; j < scanConfig_[k]; j++) {
        const double angle = 2.0 * M_PI * double(j) / scanConfig_[k];
        scanCos_(k,j) = cos(angle);
        scanSin_(k,j) = sin(angle);
      }
    }

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
      if (index < 4 && !contact.isSelfCollision())
        footContactState_[index] = true;
    }
  }

  void getObservation(Eigen::VectorXd &observation) {
    observation = (obDouble_ - obMean_).cwiseQuotient(obStd_);
  }

  bool advance(raisim::World *world, const Eigen::Ref<EigenVec> &action, double curriculumFactor) {
    /// action scaling
    prevprevAction_ = previousAction_;
    previousAction_ = jointTarget_;

//    jointTargetDelta_ = action.cast<double>();
//    jointTarget_ += jointTargetDelta_;

    jointTarget_ = action.cast<double>();

    jointTarget_ = jointTarget_.cwiseProduct(actionStd_);
    jointTarget_ += actionMean_;
    pTarget_.tail(nJoints_) = jointTarget_;
    raibo_->setPdTarget(pTarget_, vTarget_);

    smoothReward_ = curriculumFactor * smoothRewardCoeff_ * (prevprevAction_ + jointTarget_ - 2 * previousAction_).squaredNorm();
    return true;
  }

  void reset(std::mt19937 &gen_,
             std::normal_distribution<double> &normDist_) {
    raibo_->getState(gc_, gv_);
    jointTarget_ = gc_.tail(12);
    previousAction_.setZero();
    prevprevAction_.setZero();

    // history
    for (int i = 0; i < nJoints_ * historyLength_; i++)
      jointPositionHistory_[i] = normDist_(gen_) * .1;

    for (int i = 0; i < nJoints_ * historyLength_; i++)
      jointVelocityHistory_[i] = normDist_(gen_) * 1.0;
  }

  [[nodiscard]] float getRewardSum(bool visualize) {
    stepData_[0] = commandTrackingReward_;
    stepData_[1] = contactSwitchReward_;
    stepData_[2] = torqueReward_;
    stepData_[3] = smoothReward_;
    stepData_[4] = orientationReward_;
    stepData_[5] = jointVelocityReward_;
    stepData_[6] = slipReward_;
    stepData_[7] = airtimeReward_;

    commandTrackingReward_ = 0.;
    contactSwitchReward_ = 0.;
    torqueReward_ = 0.;
    smoothReward_ = 0.;
    orientationReward_ = 0.;
    jointVelocityReward_ = 0.;
    slipReward_ = 0.;
    airtimeReward_ = 0.;

    return float(stepData_.sum());
  }

  [[nodiscard]] bool isTerminalState(float &terminalReward) {
    terminalReward = float(terminalRewardCoeff_);

    /// if the contact body is not feet
    for (auto &contact: raibo_->getContacts())
      if (std::find(footIndices_.begin(), footIndices_.end(), contact.getlocalBodyIndex()) == footIndices_.end() || contact.isSelfCollision())
        return true;

    terminalReward = 0.f;
    return false;
  }

  void updateObservation(bool nosify,
                         const Eigen::Vector3d &command,
                         const raisim::HeightMap *map,
                         std::mt19937 &gen_,
                         std::normal_distribution<double> &normDist_) {
    updateHeightScan(map, gen_, normDist_);

    /// height of the origin of the body frame
    obDouble_[0] = gc_[2] - map->getHeight(gc_[0], gc_[1]);

    /// body orientation
    obDouble_.segment(1, 3) = baseRot_.e().row(2);

    /// body velocities
    obDouble_.segment(4, 3) = bodyLinVel_;
    obDouble_.segment(7, 3) = bodyAngVel_;

    /// except the first joints, the joint history stores target-position
    obDouble_.segment(10, nJoints_) = gc_.tail(12);
    obDouble_.segment(22, 12) = jointPositionHistory_.segment((historyLength_ - 1 - 6) * 12, 12);
    obDouble_.segment(34, 12) = jointPositionHistory_.segment((historyLength_ - 1 - 4) * 12, 12);
    obDouble_.segment(46, 12) = jointPositionHistory_.segment((historyLength_ - 1 - 2) * 12, 12);

    obDouble_.segment(58, 12) = jointVelocityHistory_.segment((historyLength_ - 1 - 6) * 12, 12);
    obDouble_.segment(70, 12) = jointVelocityHistory_.segment((historyLength_ - 1 - 4) * 12, 12);
    obDouble_.segment(82, 12) = jointVelocityHistory_.segment((historyLength_ - 1 - 2) * 12, 12);
    obDouble_.segment(94, 12) = jointVelocityHistory_.segment((historyLength_ - 1) * 12, 12);

    /// height scan
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < scanConfig_.sum(); j++)
        obDouble_[10 + 2 * nJoints_ * 4 + i * scanConfig_.sum() + j] = heightScan_[i][j];

    /// previous action
    obDouble_.segment(10 + 2 * nJoints_ * 4 + 4 * scanConfig_.sum(), 12) = previousAction_;
    obDouble_.segment(22 + 2 * nJoints_ * 4 + 4 * scanConfig_.sum(), 12) = prevprevAction_;

    Eigen::Vector3d posXyz; posXyz << gc_[0], gc_[1], gc_[2];
    Eigen::Vector3d target; target << command[0], command[1], map->getHeight(command[0], command[1])+0.56;
    Eigen::Vector3d targetRel = target - posXyz;
    Eigen::Vector3d targetRelBody = baseRot_.e().transpose() * targetRel;
    const double dist = targetRelBody.norm();
    targetRelBody *= 1./targetRelBody.head<2>().norm();

    /// command
    obDouble_.segment(34 + 2 * nJoints_ * 4 + 4 * scanConfig_.sum(), 2) << targetRelBody[0], targetRelBody[1];
    obDouble_[34 + 2 * nJoints_ * 4 + 4 * scanConfig_.sum() + 2] = std::min(3., dist);
  }

  inline void setRewardConfig(const Yaml::Node &cfg) {
    READ_YAML(double, commandTrackingRewardCoeff, cfg["reward"]["command_tracking_reward_coeff"])
    READ_YAML(double, torqueRewardCoeff_, cfg["reward"]["torque_reward_coeff"])
    READ_YAML(double, smoothRewardCoeff_, cfg["reward"]["smooth_reward_coeff"])
    READ_YAML(double, orientationRewardCoeff_, cfg["reward"]["orientation_reward_coeff"])
    READ_YAML(double, jointVelocityRewardCoeff_, cfg["reward"]["joint_velocity_reward_coeff"])
    READ_YAML(double, slipRewardCoeff_, cfg["reward"]["slip_reward_coeff"])
    READ_YAML(double, airtimeRewardCoeff_, cfg["reward"]["airtime_reward_coeff"])
    READ_YAML(double, contactSwitchRewardCoeff_, cfg["reward"]["con_switch_rew_coeff"])
  }

  inline void accumulateRewards(double cf, const Eigen::Vector3d &cm) {
    torqueReward_ += cf * torqueRewardCoeff_ * (raibo_->getGeneralizedForce().e().tail(12).squaredNorm()) * simDt_;
//    commandTrackingReward_ += cm[0] > 0 ? std::min(bodyLinVel_[0], cm[0]) : -std::max(bodyLinVel_[0], cm[0]);
//    commandTrackingReward_ += cm[1] > 0 ? std::min(bodyLinVel_[1], cm[1]) : -std::max(bodyLinVel_[1], cm[1]);
//    commandTrackingReward_ -= 2.0 * fabs(bodyLinVel_[2]);
//    commandTrackingReward_ += 0.5 * (cm[2] > 0 ? std::min(bodyAngVel_[2], cm[2]) : -std::max(bodyAngVel_[2], cm[2]));

    Eigen::Vector2d posXy; posXy << gc_[0], gc_[1];
    Eigen::Vector2d targetRel; targetRel = cm.head(2) - posXy;
    Eigen::Vector2d heading; heading << baseRot_[0], baseRot_[1];
    commandTrackingReward_ += 6. - targetRel.norm();
    commandTrackingReward_ += 0.3 * heading.dot(targetRel) / (targetRel.norm() * heading.norm());
    commandTrackingReward_ *= commandTrackingRewardCoeff * simDt_;

//    orientationReward_ += cf * orientationRewardCoeff_ * simDt_ * std::asin(baseRot_[7]) * std::asin(baseRot_[7]);
    orientationReward_ += cf * orientationRewardCoeff_ * simDt_ * (gc_[7]*gc_[7] + gc_[10]*gc_[10] + gc_[13]*gc_[13] + gc_[16]*gc_[16]);

    jointVelocityReward_ += cf * jointVelocityRewardCoeff_ * simDt_ * jointVelocity_.squaredNorm();
//    for(int i=0; i<12; i++)
//      jointVelocityReward_ += cf * jointVelocityRewardCoeff_ * simDt_ * std::abs(jointVelocity_[i]*jointVelocity_[i]*jointVelocity_[i]);

    raisim::Vec<3> conVel;
    for (int i=0; i< raibo_->getContacts().size(); i++) {
      if (raibo_->getContacts()[i].isSelfCollision() ) continue;
      raibo_->getContactPointVel(i, conVel);
      slipReward_ += cf * slipRewardCoeff_ * conVel.e().head(2).squaredNorm();
    }
//    for (size_t i = 0; i < 4; i++)
//      if (footContactState_[i])
//        slipReward_ += cf * slipRewardCoeff_ * footVel_[i].e().head(2).squaredNorm();

    if (!footContactState_[0] &&
        !footContactState_[1] &&
        !footContactState_[2] &&
        !footContactState_[3])
      contactSwitchReward_ += contactSwitchRewardCoeff_ * simDt_;
  }

  void updateHeightScan(const raisim::HeightMap *map,
                        std::mt19937 &gen_,
                        std::normal_distribution<double> &normDist_) {
    /// heightmap
    for (int k = 0; k < scanConfig_.size(); k++) {
      for (int j = 0; j < scanConfig_[k]; j++) {
        const double distance = 0.07 * (k + 1);
        for (int i = 0; i < 4; i++) {
          scanPoint_[i][scanConfig_.head(k).sum() + j][0] =
              footPos_[i][0] + controlFrameX_[0] * distance * scanCos_(k,j) + controlFrameY_[0] * distance * scanSin_(k,j);
          scanPoint_[i][scanConfig_.head(k).sum() + j][1] =
              footPos_[i][1] + controlFrameX_[1] * distance * scanCos_(k,j) + controlFrameY_[1] * distance * scanSin_(k,j);
          heightScan_[i][scanConfig_.head(k).sum() + j] =
              map->getHeight(scanPoint_[i][scanConfig_.head(k).sum() + j][0],
                             scanPoint_[i][scanConfig_.head(k).sum() + j][1]) - footPos_[i][2] + normDist_(gen_) * 0.025;
        }
      }
    }
  }

  inline void setStandingMode(bool mode) { standingMode_ = mode; }

  [[nodiscard]] const Eigen::VectorXd &getJointPositionHistory() const { return jointPositionHistory_; }
  [[nodiscard]] const Eigen::VectorXd &getJointVelocityHistory() const { return jointVelocityHistory_; }

  [[nodiscard]] static constexpr int getObDim() { return obDim_; }
  [[nodiscard]] static constexpr int getActionDim() { return actionDim_; }
  [[nodiscard]] static constexpr double getSimDt() { return simDt_; }
  [[nodiscard]] static constexpr double getConDt() { return conDt_; }
  void getState(Eigen::Ref<EigenVec> gc, Eigen::Ref<EigenVec> gv) { gc = gc_.cast<float>(); gv = gv_.cast<float>(); }

  static void setSimDt(double dt) { RSFATAL_IF(fabs(dt - simDt_) > 1e-12, "sim dt is fixed to " << simDt_)};
  static void setConDt(double dt) { RSFATAL_IF(fabs(dt - conDt_) > 1e-12, "con dt is fixed to " << conDt_)};

  [[nodiscard]] inline const std::vector<std::string> &getStepDataTag() const { return stepDataTag_; }
  [[nodiscard]] inline const Eigen::VectorXd &getStepData() const { return stepData_; }

  // robot configuration variables
  raisim::ArticulatedSystem *raibo_;
  std::vector<size_t> footIndices_, footFrameIndicies_;
  Eigen::VectorXd nominalJointConfig_;
  static constexpr int nJoints_ = 12;
  static constexpr int actionDim_ = 12;
  static constexpr size_t historyLength_ = 14;
  static constexpr size_t obDim_ = 333;
  static constexpr double simDt_ = .001;
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
  std::array<bool, 4> footContactState_;
  raisim::Mat<3, 3> baseRot_;

  // robot observation variables
  std::vector<raisim::VecDyn> heightScan_;
  Eigen::VectorXi scanConfig_;
  Eigen::VectorXd obDouble_, obMean_, obStd_;
  std::vector<std::vector<raisim::Vec<2>>> scanPoint_;
  Eigen::MatrixXd scanSin_;
  Eigen::MatrixXd scanCos_;

  // control variables
  static constexpr double conDt_ = 0.005;
  bool standingMode_ = false;
  Eigen::VectorXd actionMean_, actionStd_, actionScaled_, previousAction_, prevprevAction_;
  Eigen::VectorXd pTarget_, vTarget_; // full robot gc dim
  Eigen::VectorXd jointTarget_, jointTargetDelta_;
  Eigen::VectorXd jointPgain_, jointDgain_;

  // reward variables
  double commandTrackingRewardCoeff = 0., commandTrackingReward_ = 0.;
  double contactSwitchRewardCoeff_ = 0., contactSwitchReward_ = 0.;
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
