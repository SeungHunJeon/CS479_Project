//
// Created by jemin on 2/25/20.
//

#ifndef _RAISIM_GYM_RAIBO_CONTROLLER_HPP
#define _RAISIM_GYM_RAIBO_CONTROLLER_HPP

#include "unsupported/Eigen/MatrixFunctions"

Eigen::Matrix3d hat(const Eigen::Vector3d R3) {
  Eigen::Matrix3d so3;
  so3 << 0, -R3(2), R3(1),
      R3(2), 0, -R3(0),
      -R3(1), R3(0), 0;

  return so3;
}

Eigen::Vector3d vee(const Eigen::Matrix3d so3) {
  Eigen::Vector3d R3;
  R3 << so3(2,1), -so3(2,0), so3(1,0);

  return R3;
}

Eigen::Matrix3d log(const Eigen::Matrix3d SO3) {
  Eigen::Matrix3d so3;
  so3 = SO3.log();

  return so3;
}

Eigen::Vector3d LOG(const Eigen::MatrixXd SO3) {
  Eigen::Vector3d R3;
  R3 = vee(SO3.log());

  return R3;
}

Eigen::Matrix3d exp(const Eigen::Matrix3d so3) {
  Eigen::Matrix3d SO3;
  SO3 = so3.exp();

  return SO3;
}

Eigen::Matrix3d EXP(const Eigen::Vector3d R3) {
  Eigen::Matrix3d SO3;
  SO3 = hat(R3).exp();

  return SO3;
}

namespace raisim {

class RaiboController {
 public:
  inline bool create(raisim::World *world, raisim::Box *box) {
    raibo_ = reinterpret_cast<raisim::ArticulatedSystem *>(world->getObject("robot"));
    gc_.resize(raibo_->getGeneralizedCoordinateDim());
    gv_.resize(raibo_->getDOF());
    jointVelocity_.resize(nJoints_);

    Obj_ = box;
    /// foot scan config
//    scanConfig_.setZero(5);
//    scanConfig_ << 6, 8, 10, 12, 14;
//    scanPoint_.resize(4, std::vector<raisim::Vec<2>>(scanConfig_.sum()));
//    heightScan_.resize(4, raisim::VecDyn(scanConfig_.sum()));

    /// Observation
    jointPositionHistory_.setZero(nJoints_ * historyLength_);
    jointVelocityHistory_.setZero(nJoints_ * historyLength_);
    historyTempMemory_.setZero(nJoints_ * historyLength_);
    nominalConfig_.setZero(nJoints_);
    // 12 base joint + 6 arm pos, ori residuals
    nominalConfig_ << 0, 0.56, -1.12, 0, 0.56, -1.12, 0, 0.56, -1.12, 0, 0.56, -1.12, 0, 0, 0, 0, 0, 0;
    jointTarget_.setZero(nJoints_);
    jointTargetDelta_.setZero(nJoints_);
    actionTarget_.setZero(actionDim_);

    /// action
    actionMean_.setZero(actionDim_);
    actionStd_.setZero(actionDim_);
    actionScaled_.setZero(actionDim_);
    previousAction_.setZero(nJoints_);
    prevprevAction_.setZero(nJoints_);

    actionMean_.head(nlegJoints_) << nominalConfig_.head(nlegJoints_); /// joint target
    actionMean_.segment(nlegJoints_, 3) << 0.0, 0.0, 0.0; /// task space position & orientation residuals
//    actionMean_.segment(nJoints_ - 6, 6) << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0; /// task space position & orientation residuals
//    actionMean_.tail(nVargain_) << 50, 0.5, 50, 0.5; /// positional jacobian p, d gain && orientation jacobian p, d gain
    actionStd_.head(nlegJoints_) << Eigen::VectorXd::Constant(nlegJoints_, 0.1); /// joint target
    actionStd_.segment(nlegJoints_, 3) << Eigen::VectorXd::Constant(3, 0.1); /// pos residual
//    actionStd_.segment(nJoints_ - 3, 3) << Eigen::VectorXd::Constant(3, 0.1); /// orientation residual
//    actionStd_.tail(nVargain_) << 0.25, 0.25, 0.25, 0.25; /// positional jacobian p, d gain && orientation jacobian p, d gain

    obMean_.setZero(obDim_);
    obStd_.setZero(obDim_);
    obDouble_.setZero(obDim_);

    /// PD gain ee
    posPgain_ = 50; posDgain_ = 4;
    oriPgain_ = 50; oriDgain_ = 4;

    /// pd controller
    jointPgain_.setZero(gvDim_);
//    jointPgain_.tail(nJoints_).setConstant(60.0);
    jointDgain_.setZero(gvDim_);
//    jointDgain_.tail(nJoints_).setConstant(0.5);

    jointPgain_.segment(6, nlegJoints_).setConstant(60.0);
    jointDgain_.segment(6, nlegJoints_).setConstant(0.5);

    /// set Manipulator joint PD gain as 0 -> torque control
//    jointPgain_.tail(6).setConstant(1.0);
//    jointDgain_.tail(6).setConstant(0.1);
    raibo_->setPdGains(jointPgain_, jointDgain_);
    pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_);

    /// observation
    obMean_ << 0.5, /// average height
        0.0, 0.0, 1.4, /// gravity axis 3
        Eigen::VectorXd::Constant(6, 0.0), /// body lin/ang vel 6
        nominalConfig_.head(nlegJoints_), /// joint pos
        Eigen::VectorXd::Constant(nlegJoints_, 0.0), /// joint velocity
        Eigen::VectorXd::Constant(3, 0.0), /// end-effector position
        Eigen::VectorXd::Constant(3, 0.0), /// end-effector orientation
        Eigen::VectorXd::Constant(3, 0.0), /// end-effector velocity
        Eigen::VectorXd::Constant(3, 0.0), /// end-effector angular-velocity
        Eigen::VectorXd::Constant(3, 0.0), /// end-effector to object distance
        Eigen::VectorXd::Constant(3, 0.0), /// object to target distance
//        Eigen::VectorXd::Constant(3, 0.0), /// object to target orientation
        Eigen::VectorXd::Constant(3, 0.0); /// object velocity

//        Eigen::VectorXd::Constant(nJoints_ * (4 - 1), 0.0), /// joint position error history
//        Eigen::VectorXd::Constant(nJoints_ * 4, 0.0), /// joint vel history
//        nominalConfig_, /// previous action
//        nominalConfig_; /// preprev action
//        Eigen::VectorXd::Constant(3, 0.0); /// command

    obStd_ << 0.05, /// height
        Eigen::VectorXd::Constant(3, 0.3), /// gravity axes
        Eigen::VectorXd::Constant(3, 0.6), /// linear velocity
        Eigen::VectorXd::Constant(3, 1.0), /// angular velocities
        Eigen::VectorXd::Constant(nlegJoints_, 1.), /// joint angles
        Eigen::VectorXd::Constant(nlegJoints_, 10.0), /// joint velocities
        Eigen::VectorXd::Constant(3, 0.1), /// end-effector position
        Eigen::VectorXd::Constant(3, 0.1), /// end-effector orientation
        Eigen::VectorXd::Constant(3, 0.1), /// end-effector velocity
        Eigen::VectorXd::Constant(3, 0.1), /// end-effector angular-velocity
        Eigen::VectorXd::Constant(3, 0.2), /// end-effector to object distance
        Eigen::VectorXd::Constant(3, 0.2), /// object to target distance
//        Eigen::VectorXd::Constant(3, 0.2), /// object to target orientation
        Eigen::VectorXd::Constant(3, 0.2); /// object velocity
//        Eigen::VectorXd::Constant(nJoints_ * (4 - 1), 0.6), /// joint position error history
//        Eigen::VectorXd::Constant(nJoints_ * 4, 10.0), /// joint velocities
//        actionStd_ * 1.5, /// previous action
//        actionStd_ * 1.5; /// previous action
//        .5, 0.3, 0.6; /// command

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

    /// indices of link that should not make contact with ground
    armIndices_.push_back(raibo_->getBodyIdx("link6"));

    /// indices of link that should not make contact with ground

    /// exported data
    stepDataTag_ = {"command_rew",
                    "con_switch_rew",
                    "torque_rew",
                    "smooth_rew",
                    "ori_rew",
                    "joint_vel_rew",
                    "slip_rew",
                    "airtime_rew",
                    "towardObject_rew",
                    "stayObject_rew",
                    "towardTarget_rew",
                    "stayTarget_rew"};
    stepData_.resize(stepDataTag_.size());

    /// Object info
//    Obj_ = dynamic_cast<Box *>(world->getObject("Obj_"));
//    std::cout << Obj_->getPosition() << std::endl;
    /// Jacobian from EE

    /// heightmap
//    scanCos_.resize(scanConfig_.size(), scanConfig_.maxCoeff());
//    scanSin_.resize(scanConfig_.size(), scanConfig_.maxCoeff());
    /// precompute sin and cos because they take very long time
//    for (int k = 0; k < scanConfig_.size(); k++) {
//      for (int j = 0; j < scanConfig_[k]; j++) {
//        const double angle = 2.0 * M_PI * double(j) / scanConfig_[k];
//        scanCos_(k,j) = cos(angle);
//        scanSin_(k,j) = sin(angle);
//      }
//    }


    /// Initialize
    Gen_Force_.setZero(raibo_->getDOF());
    Jb_a_.setZero(3, raibo_->getDOF());
    Jr_a_.setZero(3, raibo_->getDOF());
    dJb_a_.setZero(3, raibo_->getDOF());
    dJr_a_.setZero(3, raibo_->getDOF());

    pos_des_ << 0.7, 0, 0.2;

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
    /// End effector info

    // Get EE position
    raibo_->getBodyPosition(armIndices_[0], ee_Pos_w_);
    ee_Pos_ = baseRot_.e().transpose() * (ee_Pos_w_.e() - gc_.head(3)); // World frame into robot frame

    // Get EE orientation
    raibo_->getBodyOrientation(armIndices_[0], eeRot_w_);
    eeRot_ = baseRot_.e().transpose() * eeRot_w_.e(); // World frame into robot frame

    // Get EE velocity
    raibo_->getVelocity(armIndices_[0], ee_Vel_w_);
    ee_Vel_ = baseRot_.e().transpose() * ee_Vel_w_.e();

    // Get EE angular velocity
    raibo_->getAngularVelocity(armIndices_[0], ee_Avel_w_);
    ee_Avel_ = baseRot_.e().transpose() * ee_Avel_w_.e();

    /// Object info
    Obj_Info_.setZero(9);
    Obj_->getPosition(Obj_Pos_);

    Obj_->getVelocity(Obj_->getIndexInWorld(), Obj_Vel_);
//    Obj_->getAngularVelocity(Obj_AVel_);
//    Obj_->getOrientation(Obj_->getIndexInWorld(), Obj_Rot_);

    Obj_Info_ << baseRot_.e().transpose() * (Obj_Pos_.e()-ee_Pos_w_.e()),
        baseRot_.e().transpose() * (command_Obj_Pos_ - Obj_Pos_.e()),
//        LOG(Obj_Rot_.e().transpose() * Tar_Rot_.e()),
        baseRot_.e().transpose() * Obj_Vel_.e();
//        baseRot_.e().transpose() * Obj_AVel_.e();

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

    actionTarget_ = action.cast<double>();

    jointTarget_.head(nlegJoints_) << actionTarget_.head(nlegJoints_).cwiseProduct(actionStd_.head(nlegJoints_));
    jointTarget_.head(nlegJoints_) += actionMean_.head(nlegJoints_);

    pTarget_.segment(7,nlegJoints_) = jointTarget_.head(nlegJoints_);
    raibo_->setPdTarget(pTarget_, vTarget_);

    /// Variable impedance control
//    Eigen::VectorXd PDgainTarget_, PDgainTarget_exp_;
//    PDgainTarget_.setZero(nVargain_);
//    PDgainTarget_exp_.setZero(nVargain_);
//    PDgainTarget_ = actionTarget_.tail(nVargain_).cwiseProduct(actionStd_.tail(nVargain_));
//    PDgainTarget_exp_ = PDgainTarget_.array().exp();
//    PDgainTarget_ = actionMean_.tail(nVargain_).cwiseProduct(PDgainTarget_exp_);

//    posPgain_ = PDgainTarget_[0];
//    posDgain_ = PDgainTarget_[1];
//    oriPgain_ = PDgainTarget_[2];
//    oriDgain_ = PDgainTarget_[3];





    raibo_->getBodyPosition(armIndices_[0], point_);
    raibo_->getDenseJacobian(armIndices_[0], point_, Jb_a_);
    raibo_->getDenseRotationalJacobian(armIndices_[0], Jr_a_);
    raibo_->getMassMatrix();

    raibo_->getTimeDerivativeOfSparseJacobian(armIndices_[0], raisim::ArticulatedSystem::WORLD_FRAME, point_, dJb_a_s_);
    raibo_->getTimeDerivativeOfSparseRotationalJacobian(armIndices_[0], dJr_a_s_);

    int j = 0;
    for (auto i : dJb_a_s_.idx) {
      dJb_a_.col(i) << dJb_a_s_.v.e().col(j);
      j += 1;
    }

    j = 0;
    for (auto i : dJr_a_s_.idx) {
      dJr_a_.col(i) << dJr_a_s_.v.e().col(j);
      j += 1;
    }


    Gen_Force_ <<
               Jb_a_.transpose() * ((Jb_a_ * raibo_->getInverseMassMatrix().e() * Jb_a_.transpose()).inverse() *(
                   (posPgain_ * (baseRot_.e().transpose() * (pos_des_) - ee_Pos_)) - posDgain_ * ee_Vel_
                   )
               ) +
                   Jr_a_.transpose() * ((Jr_a_ * raibo_->getInverseMassMatrix().e() * Jr_a_.transpose()).inverse() * (
                       (- oriDgain_ * ee_Avel_)
                       )) + raibo_->getNonlinearities(world->getGravity()).e();

//    std::cout << "Pos residual : " << (baseRot_.e().transpose() * (pos_des_ + gc_.head(3)) - ee_Pos_) << std::endl;
//    std::cout << "action : " << action.segment(nJoints_, 6).cast<double>() << std::endl;
//    std::cout << "Gains : " << posPgain_ << " " << posDgain_  << " " << oriPgain_ << " " << oriDgain_ << std::endl;
//    std::cout << "ee_Vel_ : " << ee_Vel_ << std::endl;
//    std::cout << "ee_Avel_ : " << ee_Avel_ << std::endl;
    Gen_Force_.head(18).setZero();


    raibo_->setGeneralizedForce(Gen_Force_);

//    smoothReward_ = curriculumFactor * smoothRewardCoeff_ * (prevprevAction_ + jointTarget_ - 2 * previousAction_).squaredNorm();
    return true;
  }

  void reset(std::mt19937 &gen_,
             std::normal_distribution<double> &normDist_, Eigen::Vector3d command_obj_pos_) {
    raibo_->getState(gc_, gv_);
//    jointTarget_ = gc_.segment(7, nJoints_);
    jointTarget_.head(nlegJoints_) << gc_.segment(7, nlegJoints_);
    previousAction_.setZero();
    prevprevAction_.setZero();
    command_Obj_Pos_ = command_obj_pos_;

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
    stepData_[8] = towardObjectReward_;
    stepData_[9] = stayObjectReward_;
    stepData_[10] = towardTargetReward_;
    stepData_[11] = stayTargetReward_;

    commandTrackingReward_ = 0.;
    contactSwitchReward_ = 0.;
    torqueReward_ = 0.;
    smoothReward_ = 0.;
    orientationReward_ = 0.;
    jointVelocityReward_ = 0.;
    slipReward_ = 0.;
    airtimeReward_ = 0.;
    towardObjectReward_ = 0.;
    stayObjectReward_ = 0.;
    towardTargetReward_ = 0.;
    stayTargetReward_ = 0.;

    return float(stepData_.sum());
  }

  [[nodiscard]] bool isTerminalState(float &terminalReward) {
    terminalReward = float(terminalRewardCoeff_);

    /// if the contact body is not feet
    for (auto &contact: raibo_->getContacts())
      if (std::find(footIndices_.begin(), footIndices_.end(), contact.getlocalBodyIndex())
      == footIndices_.end()
      || contact.isSelfCollision()
      || std::find(armIndices_.begin(), armIndices_.end(), contact.getlocalBodyIndex())
      == armIndices_.end())
        return true;

    terminalReward = 0.f;
    return false;
  }

  void updateObservation(bool nosify,
                         const Eigen::Vector3d &command,
                         const raisim::HeightMap *map,
                         std::mt19937 &gen_,
                         std::normal_distribution<double> &normDist_) {
//    updateHeightScan(map, gen_, normDist_);

    /// height of the origin of the body frame
//    obDouble_[0] = gc_[2] - map->getHeight(gc_[0], gc_[1]);
    obDouble_[0] = gc_[2];

    /// body orientation
    obDouble_.segment(1, 3) = baseRot_.e().row(2);

    /// body velocities
    obDouble_.segment(4, 3) = bodyLinVel_;
    obDouble_.segment(7, 3) = bodyAngVel_;

    /// except the first joints, the joint history stores target-position
    obDouble_.segment(10, nlegJoints_) = gc_.tail(nJoints_).head(nlegJoints_);

    /// base joint velocity
    obDouble_.segment(22, nlegJoints_) = gv_.tail(nJoints_).head(nlegJoints_);

    /// ee position
    obDouble_.segment(34, 3) = ee_Pos_;

    /// ee orientation
    obDouble_.segment(37, 3) = eeRot_.row(2);

    /// ee velocity
    obDouble_.segment(40, 3) = ee_Vel_;

    /// ee angular velocity
    obDouble_.segment(43, 3) = ee_Avel_;

    /// Object information
    obDouble_.segment(46, 9) = Obj_Info_;

//    for (int i=0; i < nPosHist_; i++)
//      obDouble_.segment(10+nJoints_*(i+1), nJoints_) = jointPositionHistory_.segment((historyLength_ - 1 - (6-2*i)) * nJoints_, nJoints_);
//
//    for (int i=0; i < nVelHist_; i++)
//      obDouble_.segment(10+nJoints_*(nPosHist_) + nJoints_*(i+1), nJoints_) = jointVelocityHistory_.segment((historyLength_ - 1 - (6-2*i)) * nJoints_, nJoints_);

//    obDouble_.segment(22, nJoints_) = jointPositionHistory_.segment((historyLength_ - 1 - 6) * nJoints_, nJoints_);
//    obDouble_.segment(34, nJoints_) = jointPositionHistory_.segment((historyLength_ - 1 - 4) * nJoints_, nJoints_);
//    obDouble_.segment(46, nJoints_) = jointPositionHistory_.segment((historyLength_ - 1 - 2) * nJoints_, nJoints_);
//
//    obDouble_.segment(58, nJoints_) = jointVelocityHistory_.segment((historyLength_ - 1 - 6) * nJoints_, nJoints_);
//    obDouble_.segment(70, nJoints_) = jointVelocityHistory_.segment((historyLength_ - 1 - 4) * nJoints_, nJoints_);
//    obDouble_.segment(82, nJoints_) = jointVelocityHistory_.segment((historyLength_ - 1 - 2) * nJoints_, nJoints_);
//    obDouble_.segment(94, nJoints_) = jointVelocityHistory_.segment((historyLength_ - 1) * nJoints_, nJoints_);

    /// height scan
//    for (int i = 0; i < 4; i++)
//      for (int j = 0; j < scanConfig_.sum(); j++)
//        obDouble_[10 + 2 * nJoints_ * 4 + i * scanConfig_.sum() + j] = heightScan_[i][j];

    /// previous action
//    obDouble_.segment(10 + nJoints_*nPosHist_ + nJoints_*nVelHist_ + nJoints_, nJoints_) = previousAction_;
//    obDouble_.segment(10 + nJoints_*nPosHist_ + nJoints_*nVelHist_ + nJoints_*2, nJoints_) = prevprevAction_;

//    Eigen::Vector3d posXyz; posXyz << gc_[0], gc_[1], gc_[2];
//    Eigen::Vector3d target; target << command[0], command[1], map->getHeight(command[0], command[1])+0.56;
//    Eigen::Vector3d targetRel = target - posXyz;
//    Eigen::Vector3d targetRelBody = baseRot_.e().transpose() * targetRel;
//    const double dist = targetRelBody.norm();
//    targetRelBody *= 1./targetRelBody.head<2>().norm();

    /// command
//    obDouble_.segment(34 + 2 * nJoints_ * 4 + 4 * scanConfig_.sum(), 2) << targetRelBody[0], targetRelBody[1];
//    obDouble_[34 + 2 * nJoints_ * 4 + 4 * scanConfig_.sum() + 2] = std::min(3., dist);
  }

  inline void setRewardConfig(const Yaml::Node &cfg) {
    READ_YAML(double, commandTrackingRewardCoeff_, cfg["reward"]["command_tracking_reward_coeff"])
    READ_YAML(double, torqueRewardCoeff_, cfg["reward"]["torque_reward_coeff"])
    READ_YAML(double, smoothRewardCoeff_, cfg["reward"]["smooth_reward_coeff"])
    READ_YAML(double, orientationRewardCoeff_, cfg["reward"]["orientation_reward_coeff"])
    READ_YAML(double, jointVelocityRewardCoeff_, cfg["reward"]["joint_velocity_reward_coeff"])
    READ_YAML(double, slipRewardCoeff_, cfg["reward"]["slip_reward_coeff"])
    READ_YAML(double, airtimeRewardCoeff_, cfg["reward"]["airtime_reward_coeff"])
    READ_YAML(double, towardObjectRewardCoeff_, cfg["reward"]["towardObjectRewardCoeff_"])
    READ_YAML(double, stayObjectRewardCoeff_, cfg["reward"]["stayObjectRewardCoeff_"])
    READ_YAML(double, towardTargetRewardCoeff_, cfg["reward"]["towardTargetRewardCoeff_"])
    READ_YAML(double, stayTargetRewardCoeff_, cfg["reward"]["stayTargetRewardCoeff_"])

  }

  inline void accumulateRewards(double cf, const Eigen::Vector3d &cm) {
    torqueReward_ += cf * torqueRewardCoeff_ * (raibo_->getGeneralizedForce().e().tail(nJoints_).squaredNorm()) * simDt_;
//    std::cout << "Torque Reward : " << torqueReward_ << std::endl;
//    std::cout << "Generalized Force : " << raibo_->getGeneralizedForce().e().tail(nJoints_) << std::endl;
//    std::cout << "FeedForward Torque : " << raibo_->getFeedForwardGeneralizedForce().e().tail(nJoints_) << std::endl;
//    commandTrackingReward_ += cm[0] > 0 ? std::min(bodyLinVel_[0], cm[0]) : -std::max(bodyLinVel_[0], cm[0]);
//    commandTrackingReward_ += cm[1] > 0 ? std::min(bodyLinVel_[1], cm[1]) : -std::max(bodyLinVel_[1], cm[1]);
//    commandTrackingReward_ -= 2.0 * fabs(bodyLinVel_[2]);
//    commandTrackingReward_ += 0.5 * (cm[2] > 0 ? std::min(bodyAngVel_[2], cm[2]) : -std::max(bodyAngVel_[2], cm[2]));

//    commandTrackingReward_ += 6. - targetRel.norm();
//    commandTrackingReward_ += 0.3 * heading.dot(targetRel) / (targetRel.norm() * heading.norm());
//    commandTrackingReward_ *= commandTrackingRewardCoeff * simDt_;

//    orientationReward_ += cf * orientationRewardCoeff_ * simDt_ * std::asin(baseRot_[7]) * std::asin(baseRot_[7]);
    orientationReward_ += cf * orientationRewardCoeff_ * simDt_ * (gc_[7]*gc_[7] + gc_[10]*gc_[10] + gc_[13]*gc_[13] + gc_[16]*gc_[16]);

    jointVelocityReward_ += cf * jointVelocityRewardCoeff_ * simDt_ * jointVelocity_.squaredNorm();

//    std::cout << "JointVelocity Reward : " << jointVelocityReward_ << std::endl;

    /// move towards the object
    double toward_o = ((baseRot_.e().transpose() * Obj_Pos_.e() - ee_Pos_) / ((ee_Pos_w_.e() - Obj_Pos_.e()).norm() + 1e-8)).dot(ee_Vel_) - 1;
    towardObjectReward_ += cf * towardObjectRewardCoeff_ * simDt_ * exp(-std::pow(std::min(0.0, toward_o), 2));

    /// stay close to the object
    double stay_o = (Obj_Pos_.e() - ee_Pos_w_.e()).dot(Obj_Pos_.e() - ee_Pos_w_.e());
    stayObjectReward_ += cf * stayObjectRewardCoeff_ * simDt_ * exp(-stay_o);

    /// move the object towards the target
    double toward_t = ((baseRot_.e().transpose() * (command_Obj_Pos_ - Obj_Pos_.e())) / ((command_Obj_Pos_ - Obj_Pos_.e()).norm() + 1e-8)).dot(baseRot_.e().transpose() * Obj_Vel_.e()) - 1;
    towardTargetReward_ += cf * towardTargetRewardCoeff_ * simDt_ * exp(-std::pow(std::min(0.0, toward_t), 2));

    /// keep the object close to the target
    double stay_t = (command_Obj_Pos_ - Obj_Pos_.e()).dot(command_Obj_Pos_ - Obj_Pos_.e());
    stayTargetReward_ += cf * stayTargetRewardCoeff_ * simDt_ * exp(-stay_t);

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
  std::vector<size_t> footIndices_, footFrameIndicies_, armIndices_;
  Eigen::VectorXd nominalConfig_;
  static constexpr int nlegJoints_ = 12;
  static constexpr int narmJoints_ = 6;
  static constexpr int nJoints_ = nlegJoints_ + narmJoints_;
  static constexpr int nVargain_ = 0; /// Variable impedance control gain
  static constexpr int actionDim_ = nJoints_ + nVargain_ - 3; /// output dim : joint action 12 + task space action 6 + gain dim 4
  static constexpr size_t historyLength_ = 14;
  static constexpr size_t obDim_ = 55;
  static constexpr double simDt_ = .001;
  static constexpr int gcDim_ = 25;
  static constexpr int gvDim_ = 24;
  static constexpr int nPosHist_ = 3;
  static constexpr int nVelHist_ = 4;
  raisim::Box* Obj_;


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
  Eigen::Vector3d ee_Pos_, ee_Vel_, ee_Avel_;
  Eigen::Matrix3d eeRot_;


  // robot observation variables
  std::vector<raisim::VecDyn> heightScan_;
  Eigen::VectorXi scanConfig_;
  Eigen::VectorXd obDouble_, obMean_, obStd_;
  std::vector<std::vector<raisim::Vec<2>>> scanPoint_;
  Eigen::MatrixXd scanSin_;
  Eigen::MatrixXd scanCos_;
  Eigen::VectorXd Obj_Info_;
  raisim::Vec<3> Obj_Pos_, Obj_Vel_, Obj_AVel_;
  raisim::Mat<3,3> Obj_Rot_, Tar_Rot_;
  raisim::Vec<3> ee_Pos_w_, ee_Vel_w_, ee_Avel_w_;
  raisim::Mat<3,3> eeRot_w_;

  Eigen::VectorXd Gen_Force_;
  Eigen::MatrixXd Jb_a_, Jr_a_;
  Eigen::MatrixXd dJb_a_, dJr_a_;
  raisim::SparseJacobian dJb_a_s_, dJr_a_s_;
  raisim::Vec<3> point_;
  // control variables
  static constexpr double conDt_ = 0.005;
  bool standingMode_ = false;
  Eigen::VectorXd actionMean_, actionStd_, actionScaled_, previousAction_, prevprevAction_;
  Eigen::VectorXd pTarget_, vTarget_; // full robot gc dim
  Eigen::VectorXd jointTarget_, jointTargetDelta_;
  Eigen::VectorXd actionTarget_;
  Eigen::VectorXd jointPgain_, jointDgain_;
  Eigen::Vector3d command_Obj_Pos_;
  Eigen::Vector3d pos_des_;
  double posPgain_, posDgain_;
  double oriPgain_, oriDgain_;

  // reward variables
  double towardObjectRewardCoeff_ = 0., towardObjectReward_ = 0.;
  double stayObjectRewardCoeff_ = 0., stayObjectReward_ = 0.;
  double towardTargetRewardCoeff_ = 0., towardTargetReward_ = 0.;
  double stayTargetRewardCoeff_ = 0., stayTargetReward_ = 0.;
  double commandTrackingRewardCoeff_ = 0., commandTrackingReward_ = 0.;
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
