//
// Created by jemin on 2/25/20.
//

#ifndef _RAISIM_GYM_RAIBO_CONTROLLER_HPP
#define _RAISIM_GYM_RAIBO_CONTROLLER_HPP

#include "unsupported/Eigen/MatrixFunctions"
#include "raisim/RaisimServer.hpp"


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
  inline bool create(raisim::World *world, raisim::SingleBodyObject *obj) {
    raibo_ = reinterpret_cast<raisim::ArticulatedSystem *>(world->getObject("robot"));
    gc_.resize(raibo_->getGeneralizedCoordinateDim());
    gv_.resize(raibo_->getDOF());
    jointVelocity_.resize(nJoints_);
    nominalConfig_.setZero(nJoints_);
    nominalConfig_ << 0.0, 0.559836, -1.119672, -0.0, 0.559836, -1.119672, 0.0, 0.559836, -1.119672, -0.0, 0.559836, -1.119672;

    Obj_ = obj;
    /// foot scan config
//    scanConfig_.setZero(5);
//    scanConfig_ << 6, 8, 10, 12, 14;
//    scanPoint_.resize(4, std::vector<raisim::Vec<2>>(scanConfig_.sum()));
//    heightScan_.resize(4, raisim::VecDyn(scanConfig_.sum()));

    /// Observation
    actionTarget_.setZero(actionDim_);

    /// action
    actionMean_.setZero(actionDim_);
    actionStd_.setZero(actionDim_);
    actionScaled_.setZero(actionDim_);
    previousAction_.setZero(actionDim_);
    prev2Action_.setZero(actionDim_);
    prev3Action_.setZero(actionDim_);
    prev4Action_.setZero(actionDim_);

    actionMean_ << Eigen::VectorXd::Constant(actionDim_, -1); /// joint target
//    actionMean_.segment(nJoints_ - 6, 6) << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0; /// task space position & orientation residuals
//    actionMean_.tail(nVargain_) << 50, 0.5, 50, 0.5; /// positional jacobian p, d gain && orientation jacobian p, d gain
    actionStd_<< Eigen::VectorXd::Constant(actionDim_, 2); /// joint target
//    actionStd_.segment(nJoints_ - 3, 3) << Eigen::VectorXd::Constant(3, 0.1); /// orientation residual
//    actionStd_.tail(nVargain_) << 0.25, 0.25, 0.25, 0.25; /// positional jacobian p, d gain && orientation jacobian p, d gain

    obMean_.setZero(obDim_);
    obStd_.setZero(obDim_);
    obDouble_.setZero(obDim_);


    Obj_Info_.setZero(exteroceptiveDim_);
    objectInfoHistory_.setZero(historyLength_ * exteroceptiveDim_);
    stateInfoHistory_.setZero(historyLength_ * proprioceptiveDim_);


    /// observation
//    obMean_ << 0.47, /// average height
//        0.0, 0.0, 1.4, /// gravity axis 3
//        Eigen::VectorXd::Constant(6, 0.0), /// body lin/ang vel 6
//        nominalConfig_, /// joint pos
//        Eigen::VectorXd::Constant(nJoints_, 0.0), /// joint velocity
//        Eigen::VectorXd::Constant(3, 2.0), /// end-effector to object distance
//        Eigen::VectorXd::Constant(3, sqrt(2)), /// object to target distance
//        Eigen::VectorXd::Constant(3, 2.0), /// end-effector to target distance
//        Eigen::VectorXd::Constant(3, 0.0); /// object velocity
//
////        Eigen::VectorXd::Constant(nJoints_ * (4 - 1), 0.0), /// joint position error history
////        Eigen::VectorXd::Constant(nJoints_ * 4, 0.0), /// joint vel history
////        nominalConfig_, /// previous action
////        nominalConfig_; /// preprev action
////        Eigen::VectorXd::Constant(3, 0.0); /// command
//
//    obStd_ << 0.05, /// height
//        Eigen::VectorXd::Constant(3, 0.3), /// gravity axes
//        Eigen::VectorXd::Constant(3, 0.6), /// linear velocity
//        Eigen::VectorXd::Constant(3, 1.0), /// angular velocities
//        Eigen::VectorXd::Constant(nJoints_, 1.), /// joint angles
//        Eigen::VectorXd::Constant(nJoints_, 10.0), /// joint velocities
//        Eigen::VectorXd::Constant(3, 0.2), /// end-effector to object distance
//        Eigen::VectorXd::Constant(3, 0.2), /// object to target distance
//        Eigen::VectorXd::Constant(3, 0.2),
////        Eigen::VectorXd::Constant(3, 0.2), /// object to target orientation
//        Eigen::VectorXd::Constant(3, 0.2); /// object velocity
////        Eigen::VectorXd::Constant(nJoints_ * (4 - 1), 0.6), /// joint position error history
////        Eigen::VectorXd::Constant(nJoints_ * 4, 10.0), /// joint velocities
////        actionStd_ * 1.5, /// previous action
////        actionStd_ * 1.5; /// previous action
////        .5, 0.3, 0.6; /// command

    for (int i=0; i<historyNum_ ; i++) {
      obMean_.segment(proprioceptiveDim_*i, proprioceptiveDim_) << 0.0, 0.0, 1.4, /// gravity axis 3
          Eigen::VectorXd::Constant(6, 0.0); /// body lin/ang vel 6

      obMean_.segment(proprioceptiveDim_*historyNum_ + exteroceptiveDim_*i, exteroceptiveDim_) <<
          Eigen::VectorXd::Constant(2, 0),
          Eigen::VectorXd::Constant(1, 2), /// end-effector to object distance
          Eigen::VectorXd::Constant(2, 0),
          Eigen::VectorXd::Constant(1, sqrt(2)), /// object to target distance
          Eigen::VectorXd::Constant(2, 0),
          Eigen::VectorXd::Constant(1, 2), /// end-effector to target distance
          Eigen::VectorXd::Constant(3, 0.0), /// object to target velocity
          Eigen::VectorXd::Constant(1, 2), /// mass
          Eigen::VectorXd::Constant(3, 0), /// COM
          Eigen::VectorXd::Constant(9, 0), /// Inertia
          0.0, 0.0, 1.4, /// Orientation
          Eigen::VectorXd::Constant(4,0.5); /// one hot vector

      obMean_.segment(proprioceptiveDim_*historyNum_ + exteroceptiveDim_*historyNum_ + actionDim_ * i, actionDim_) <<
          Eigen::VectorXd::Constant(actionDim_, 0.0);
    }
//
//
//    obMean_ <<
//        0.0, 0.0, 1.4, /// gravity axis 3
//        Eigen::VectorXd::Constant(6, 0.0), /// body lin/ang vel 6
//
//        0.0, 0.0, 1.4, /// gravity axis 3
//        Eigen::VectorXd::Constant(6, 0.0), /// body lin/ang vel 6
//
//        0.0, 0.0, 1.4, /// gravity axis 3
//        Eigen::VectorXd::Constant(6, 0.0), /// body lin/ang vel 6
//
//        0.0, 0.0, 1.4, /// gravity axis 3
//        Eigen::VectorXd::Constant(6, 0.0), /// body lin/ang vel 6
//
//        Eigen::VectorXd::Constant(2, 0),
//        Eigen::VectorXd::Constant(1, 2), /// end-effector to object distance
//        Eigen::VectorXd::Constant(2, 0),
//        Eigen::VectorXd::Constant(1, sqrt(2)), /// object to target distance
//        Eigen::VectorXd::Constant(2, 0),
//        Eigen::VectorXd::Constant(1, 2), /// end-effector to target distance
//        Eigen::VectorXd::Constant(3, 0.0), /// object to target velocity
//        Eigen::VectorXd::Constant(1, 2),
//        Eigen::VectorXd::Constant(3, 0),
//        Eigen::VectorXd::Constant(9, 0),
//
//        Eigen::VectorXd::Constant(2, 0),
//        Eigen::VectorXd::Constant(1, 2), /// end-effector to object distance
//        Eigen::VectorXd::Constant(2, 0),
//        Eigen::VectorXd::Constant(1, sqrt(2)), /// object to target distance
//        Eigen::VectorXd::Constant(2, 0),
//        Eigen::VectorXd::Constant(1, 2), /// end-effector to target distance
//        Eigen::VectorXd::Constant(3, 0.0), /// object to target velocity
//        Eigen::VectorXd::Constant(1, 2),
//        Eigen::VectorXd::Constant(3, 0),
//        Eigen::VectorXd::Constant(9, 0),
//
//        Eigen::VectorXd::Constant(2, 0),
//        Eigen::VectorXd::Constant(1, 2), /// end-effector to object distance
//        Eigen::VectorXd::Constant(2, 0),
//        Eigen::VectorXd::Constant(1, sqrt(2)), /// object to target distance
//        Eigen::VectorXd::Constant(2, 0),
//        Eigen::VectorXd::Constant(1, 2), /// end-effector to target distance
//        Eigen::VectorXd::Constant(3, 0.0), /// object to target velocity
//        Eigen::VectorXd::Constant(1, 2),
//        Eigen::VectorXd::Constant(3, 0),
//        Eigen::VectorXd::Constant(9, 0),
//
//        Eigen::VectorXd::Constant(2, 0),
//        Eigen::VectorXd::Constant(1, 2), /// end-effector to object distance
//        Eigen::VectorXd::Constant(2, 0),
//        Eigen::VectorXd::Constant(1, sqrt(2)), /// object to target distance
//        Eigen::VectorXd::Constant(2, 0),
//        Eigen::VectorXd::Constant(1, 2), /// end-effector to target distance
//        Eigen::VectorXd::Constant(3, 0.0), /// object to target velocity
//        Eigen::VectorXd::Constant(1, 2),
//        Eigen::VectorXd::Constant(3, 0),
//        Eigen::VectorXd::Constant(9, 0),
//
//        Eigen::VectorXd::Constant(actionDim_, 0.0),
//        Eigen::VectorXd::Constant(actionDim_, 0.0),
//        Eigen::VectorXd::Constant(actionDim_, 0.0),
//        Eigen::VectorXd::Constant(actionDim_, 0.0); /// object velocity

    for (int i=0; i<historyNum_ ; i++) {
      obStd_.segment(proprioceptiveDim_*i, proprioceptiveDim_) <<
          Eigen::VectorXd::Constant(3, 0.3), /// gravity axes
          Eigen::VectorXd::Constant(3, 0.6), /// linear velocity
          Eigen::VectorXd::Constant(3, 1.0); /// angular velocities

      obStd_.segment(proprioceptiveDim_*historyNum_ + exteroceptiveDim_*i, exteroceptiveDim_) <<
          Eigen::VectorXd::Constant(2, 0.5),
          Eigen::VectorXd::Constant(1, 0.6), /// end-effector to object distance
          Eigen::VectorXd::Constant(2, 0.5),
          Eigen::VectorXd::Constant(1, 0.6), /// object to target distance
          Eigen::VectorXd::Constant(2, 0.5),
          Eigen::VectorXd::Constant(1, 0.6), /// end-effector to target distance
          Eigen::VectorXd::Constant(3, 0.5), /// object to target velocity
          Eigen::VectorXd::Constant(1, 0.2),
          Eigen::VectorXd::Constant(3, 0.1),
          Eigen::VectorXd::Constant(9, 0.1),
          Eigen::VectorXd::Constant(3,0.3), /// Orientation
          Eigen::VectorXd::Constant(4,0.2); /// one hot vector

      obStd_.segment(proprioceptiveDim_*historyNum_ + exteroceptiveDim_*historyNum_ + actionDim_ * i, actionDim_) <<
          Eigen::VectorXd::Constant(actionDim_, 0.5);
    }

//    obStd_ <<
//        Eigen::VectorXd::Constant(3, 0.3), /// gravity axes
//        Eigen::VectorXd::Constant(3, 0.6), /// linear velocity
//        Eigen::VectorXd::Constant(3, 1.0), /// angular velocities
//
//        Eigen::VectorXd::Constant(3, 0.3), /// gravity axes
//        Eigen::VectorXd::Constant(3, 0.6), /// linear velocity
//        Eigen::VectorXd::Constant(3, 1.0), /// angular velocities
//
//        Eigen::VectorXd::Constant(3, 0.3), /// gravity axes
//        Eigen::VectorXd::Constant(3, 0.6), /// linear velocity
//        Eigen::VectorXd::Constant(3, 1.0), /// angular velocities
//
//        Eigen::VectorXd::Constant(3, 0.3), /// gravity axes
//        Eigen::VectorXd::Constant(3, 0.6), /// linear velocity
//        Eigen::VectorXd::Constant(3, 1.0), /// angular velocities
//
//        Eigen::VectorXd::Constant(2, 0.5),
//        Eigen::VectorXd::Constant(1, 0.6), /// end-effector to object distance
//        Eigen::VectorXd::Constant(2, 0.5),
//        Eigen::VectorXd::Constant(1, 0.6), /// object to target distance
//        Eigen::VectorXd::Constant(2, 0.5),
//        Eigen::VectorXd::Constant(1, 0.6), /// end-effector to target distance
//        Eigen::VectorXd::Constant(3, 0.5), /// object to target velocity
//        Eigen::VectorXd::Constant(1, 0.2),
//        Eigen::VectorXd::Constant(3, 0.1),
//        Eigen::VectorXd::Constant(9, 0.05),
//
//        Eigen::VectorXd::Constant(2, 0.5),
//        Eigen::VectorXd::Constant(1, 0.6), /// end-effector to object distance
//        Eigen::VectorXd::Constant(2, 0.5),
//        Eigen::VectorXd::Constant(1, 0.6), /// object to target distance
//        Eigen::VectorXd::Constant(2, 0.5),
//        Eigen::VectorXd::Constant(1, 0.6), /// end-effector to target distance
//        Eigen::VectorXd::Constant(3, 0.5), /// object to target velocity
//        Eigen::VectorXd::Constant(1, 0.2),
//        Eigen::VectorXd::Constant(3, 0.1),
//        Eigen::VectorXd::Constant(9, 0.05),
//
//        Eigen::VectorXd::Constant(2, 0.5),
//        Eigen::VectorXd::Constant(1, 0.6), /// end-effector to object distance
//        Eigen::VectorXd::Constant(2, 0.5),
//        Eigen::VectorXd::Constant(1, 0.6), /// object to target distance
//        Eigen::VectorXd::Constant(2, 0.5),
//        Eigen::VectorXd::Constant(1, 0.6), /// end-effector to target distance
//        Eigen::VectorXd::Constant(3, 0.5), /// object to target velocity
//        Eigen::VectorXd::Constant(1, 0.2),
//        Eigen::VectorXd::Constant(3, 0.1),
//        Eigen::VectorXd::Constant(9, 0.05),
//
//        Eigen::VectorXd::Constant(2, 0.5),
//        Eigen::VectorXd::Constant(1, 0.6), /// end-effector to object distance
//        Eigen::VectorXd::Constant(2, 0.5),
//        Eigen::VectorXd::Constant(1, 0.6), /// object to target distance
//        Eigen::VectorXd::Constant(2, 0.5),
//        Eigen::VectorXd::Constant(1, 0.6), /// end-effector to target distance
//        Eigen::VectorXd::Constant(3, 0.5), /// object to target velocity
//        Eigen::VectorXd::Constant(1, 0.2),
//        Eigen::VectorXd::Constant(3, 0.1),
//        Eigen::VectorXd::Constant(9, 0.05),
//
//        Eigen::VectorXd::Constant(actionDim_, 0.5),
//        Eigen::VectorXd::Constant(actionDim_, 0.5),
//        Eigen::VectorXd::Constant(actionDim_, 0.5),
//        Eigen::VectorXd::Constant(actionDim_, 0.5); /// object velocity

    footIndices_.push_back(raibo_->getBodyIdx("LF_SHANK"));
    footIndices_.push_back(raibo_->getBodyIdx("RF_SHANK"));
    footIndices_.push_back(raibo_->getBodyIdx("LH_SHANK"));
    footIndices_.push_back(raibo_->getBodyIdx("RH_SHANK"));
    RSFATAL_IF(std::any_of(footIndices_.begin(), footIndices_.end(), [](int i){return i < 0;}), "footIndices_ not found")

    /// exported data
    stepDataTag_ = {"towardObject_rew",
                    "stayObject_rew",
                    "towardTarget_rew",
                    "stayTarget_rew",
                    "command_rew",
                    "torque_rew",
                    "stayObject_heading_rew"};
    stepData_.resize(stepDataTag_.size());


    classify_vector_.setZero(4);
    classify_vector_ << 1, 0, 0, 0;

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
//    Gen_Force_.setZero(raibo_->getDOF());
//    Jb_a_.setZero(3, raibo_->getDOF());
//    Jr_a_.setZero(3, raibo_->getDOF());
//    dJb_a_.setZero(3, raibo_->getDOF());
//    dJr_a_.setZero(3, raibo_->getDOF());
//
//    pos_des_ << 0.7, 0, 0.2;

    return true;
  };

  void updateHistory() {
    /// joint angles

  historyTempMemory_ = objectInfoHistory_;
    objectInfoHistory_.head((historyLength_ - 1) * exteroceptiveDim_) =
        historyTempMemory_.tail((historyLength_ - 1) * exteroceptiveDim_);
    objectInfoHistory_.tail(exteroceptiveDim_) = Obj_Info_;


    historyTempMemory_2 = stateInfoHistory_;
    stateInfoHistory_.head((historyLength_ - 1) * proprioceptiveDim_) =
        historyTempMemory_2.tail((historyLength_ - 1) * proprioceptiveDim_);
    Eigen::VectorXd stateInfo;
    stateInfo.setZero(proprioceptiveDim_);
    stateInfo.head(3) = baseRot_.e().row(2);
    stateInfo.segment(3,3) = bodyLinVel_;
    stateInfo.segment(6,3) = bodyAngVel_;
    stateInfoHistory_.tail(proprioceptiveDim_) = stateInfo;
    //
//    /// joint velocities
//    historyTempMemory_ = jointVelocityHistory_;
//    jointVelocityHistory_.head((historyLength_ - 1) * nJoints_) =
//        historyTempMemory_.tail((historyLength_ - 1) * nJoints_);
//    jointVelocityHistory_.tail(nJoints_) = gv_.tail(nJoints_);
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

//    /// foot info
//    for (size_t i = 0; i < 4; i++) {
//      raibo_->getFramePosition(footFrameIndicies_[i], footPos_[i]);
//      raibo_->getFrameVelocity(footFrameIndicies_[i], footVel_[i]);
//    }
    /// End effector info

    for (int i = 0; i < 3; i++) {
      ee_Pos_w_[i] = gc_[i];
      ee_Vel_w_[i] = gv_[i];
    }

//    switch (is_foot_contact_) {
//      case 1:
//        ee_Pos_w_ = footPos_[0];
//        ee_Vel_w_ = footVel_[0];
//        break;
//
//      case 0:
//        for (int i = 0; i < 3; i++) {
//          ee_Pos_w_[i] = gc_[i];
//          ee_Vel_w_[i] = gv_[i];
//
//        }
//          break;
//
//    }
    // Get EE position

//    ee_Pos_ = baseRot_.e().transpose() * (ee_Pos_w_.e() - gc_.head(3)); // World frame into robot frame
//
//    // Get EE orientation
//    raibo_->getBodyOrientation(armIndices_[0], eeRot_w_);
//    eeRot_ = baseRot_.e().transpose() * eeRot_w_.e(); // World frame into robot frame
//
    // Get EE velocity

//    ee_Vel_ = baseRot_.e().transpose() * ee_Vel_w_.e();
//
//    // Get EE angular velocity
//    raibo_->getAngularVelocity(armIndices_[0], ee_Avel_w_);
//    ee_Avel_ = baseRot_.e().transpose() * ee_Avel_w_.e();
//
    /// Object info

    Obj_->getPosition(Obj_Pos_);
//
    Obj_->getLinearVelocity(Obj_Vel_);

//    RSINFO(Obj_Vel_.e())
//    Obj_->getAngularVelocity(Obj_AVel_);
//    Obj_->getOrientation(Obj_->getIndexInWorld(), Obj_Rot_);
//
    //TODO

    Eigen::Vector3d ee_to_obj = (Obj_Pos_.e()-ee_Pos_w_.e());
    Eigen::Vector3d obj_to_target (command_Obj_Pos_ - Obj_Pos_.e());
    Eigen::Vector3d ee_to_target = (command_Obj_Pos_ - Obj_Pos_.e());
    ee_to_obj(2) = 0;
    obj_to_target(2) = 0;
    ee_to_target(2) = 0;
    ee_to_obj = baseRot_.e().transpose() * ee_to_obj;
    obj_to_target = baseRot_.e().transpose() * obj_to_target;
    ee_to_target = baseRot_.e().transpose() * ee_to_target;

    Eigen::Vector2d pos_temp_;
    double dist_temp_;

    dist_temp_ = ee_to_obj.head(2).norm();
    pos_temp_ = ee_to_obj.head(2) * (1./dist_temp_);

    Obj_Info_.segment(0, 2) << pos_temp_;
    Obj_Info_.segment(2, 1) << std::min(2., dist_temp_);

    dist_temp_ = obj_to_target.head(2).norm();
    pos_temp_ = obj_to_target.head(2) * (1./dist_temp_);

    Obj_Info_.segment(3, 2) << pos_temp_;
    Obj_Info_.segment(5, 1) << std::min(2., dist_temp_);

    dist_temp_ = ee_to_target.head(2).norm();
    pos_temp_ = ee_to_target.head(2) * (1./dist_temp_);

    Obj_Info_.segment(6, 2) << pos_temp_;
    Obj_Info_.segment(8, 1) << std::min(2., dist_temp_);

    Obj_Info_.segment(9, 3) << baseRot_.e().transpose() * Obj_Vel_.e();

    Obj_Info_.segment(12, 1) << Obj_->getMass();

    Obj_Info_.segment(13, 3) << Obj_->getCom().e();

    Obj_Info_.segment(16,3) = Obj_->getInertiaMatrix_B().row(0);

    Obj_Info_.segment(19,3) = Obj_->getInertiaMatrix_B().row(1);

    Obj_Info_.segment(22,3) = Obj_->getInertiaMatrix_B().row(2);

    Obj_Info_.segment(25,3) = Obj_->getOrientation().e().row(2);

    Obj_Info_.segment(28,4) = classify_vector_;
//        baseRot_.e().transpose() * Obj_AVel_.e();

    /// height map
    controlFrameX_ =
        {baseRot_[0], baseRot_[1], 0.}; /// body x axis projected on the world x-y plane, expressed in the world frame
    controlFrameX_ /= controlFrameX_.norm();
    raisim::cross(zAxis_, controlFrameX_, controlFrameY_);

    /// check if the feet are in contact with the ground

//    for (auto &fs: footContactState_) fs = false;
//    for (auto &contact: raibo_->getContacts()) {
//      auto it = std::find(footIndices_.begin(), footIndices_.end(), contact.getlocalBodyIndex());
//      size_t index = it - footIndices_.begin();
//      if (index < 4 && !contact.isSelfCollision())
//        footContactState_[index] = true;
//    }
  }

  void getObservation(Eigen::VectorXd &observation) {
    observation = (obDouble_ - obMean_).cwiseQuotient(obStd_);
//    std::cout << "Observation : " << observation << std::endl;
  }

  Eigen::VectorXf advance(raisim::World *world, const Eigen::Ref<EigenVec> &action) {
    Eigen::VectorXf position = action.cast<float>().cwiseQuotient(actionStd_.cast<float>());
    position += actionMean_.cast<float>();
    command_ = {position(0), position(1), 0};
    return command_;
  }

  bool advance(raisim::World *world, const Eigen::Ref<EigenVec> &action, double curriculumFactor) {
    /// action scaling
    prev4Action_ = prev3Action_;
    prev3Action_ = prev2Action_;
    prev2Action_ = previousAction_;
    previousAction_ = action.cast<double>();

//    actionTarget_ = action.cast<double>();
//
//    jointTarget_.head(nlegJoints_) << actionTarget_.head(nlegJoints_).cwiseProduct(actionStd_.head(nlegJoints_));
//    jointTarget_.head(nlegJoints_) += actionMean_.head(nlegJoints_);
//
//    pTarget_.segment(7,nlegJoints_) = jointTarget_.head(nlegJoints_);
//    raibo_->setPdTarget(pTarget_, vTarget_);

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



//    smoothReward_ = curriculumFactor * smoothRewardCoeff_ * (prevprevAction_ + jointTarget_ - 2 * previousAction_).squaredNorm();
    return true;
  }

  void reset(std::mt19937 &gen_,
             std::normal_distribution<double> &normDist_, Eigen::Vector3d command_obj_pos_) {
    raibo_->getState(gc_, gv_);
//    jointTarget_ = gc_.segment(7, nJoints_);
    previousAction_.setZero();
    prev2Action_.setZero();
    prev3Action_.setZero();
    prev4Action_.setZero();
    command_Obj_Pos_ = command_obj_pos_;

    // history
    for (int i = 0; i < exteroceptiveDim_ * historyLength_; i++)
      objectInfoHistory_[i] = normDist_(gen_) * .1;
//
    for (int i = 0; i < 9 * historyLength_; i++)
      stateInfoHistory_[i] = normDist_(gen_) * 0.1;
  }

  [[nodiscard]] float getRewardSum(bool visualize) {
    stepData_[0] = towardObjectReward_;
    stepData_[1] = stayObjectReward_;
    stepData_[2] = towardTargetReward_;
    stepData_[3] = stayTargetReward_;
    stepData_[4] = commandReward_;
    stepData_[5] = torqueReward_;
    stepData_[6] = stayObjectHeadingReward_;

    towardObjectReward_ = 0.;
    stayObjectReward_ = 0.;
    towardTargetReward_ = 0.;
    stayTargetReward_ = 0.;
    commandReward_ = 0.;
    torqueReward_ = 0.;
    stayObjectHeadingReward_ = 0.;

    return float(stepData_.sum());
  }

  [[nodiscard]] bool isTerminalState(float &terminalReward) {
    terminalReward = float(terminalRewardCoeff_);

    /// if the contact body is not feet
    for (auto &contact: raibo_->getContacts())
      if (std::find(footIndices_.begin(), footIndices_.end(), contact.getlocalBodyIndex()) == footIndices_.end() || contact.isSelfCollision())
        return true;


    terminalReward = 0.f;
//    Eigen::Vector3d obj_to_target = (command_Obj_Pos_ - Obj_Pos_.e());
//    if (obj_to_target.norm() < 0.03)
//      terminalReward = 0.1;



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
//    obDouble_[0] = gc_[2];
//    /// body orientation
    obDouble_.segment(0, 3) = baseRot_.e().row(2);

//    /// body velocities
    obDouble_.segment(3, 3) = bodyLinVel_;
    obDouble_.segment(6, 3) = bodyAngVel_;
    obDouble_.segment(9, 9) =
        stateInfoHistory_.segment((historyLength_ -1 - 6) * proprioceptiveDim_, proprioceptiveDim_);
    obDouble_.segment(18, 9) = stateInfoHistory_.segment((historyLength_ -1 - 4) * proprioceptiveDim_, proprioceptiveDim_);
    obDouble_.segment(27, 9) = stateInfoHistory_.segment((historyLength_ -1 - 2) * proprioceptiveDim_, proprioceptiveDim_);

//    /// except the first joints, the joint history stores target-position
//    obDouble_.segment(10, nJoints_) = gc_.tail(nJoints_);
//
//    /// base joint velocity
//    obDouble_.segment(22, nJoints_) = gv_.tail(nJoints_);

    /// Object information
    obDouble_.segment(36, exteroceptiveDim_) = Obj_Info_;


    /// proprioceptive information history


    /// object information history
    obDouble_.segment(36+exteroceptiveDim_, exteroceptiveDim_) =
        objectInfoHistory_.segment((historyLength_ -1 - 6) * exteroceptiveDim_, exteroceptiveDim_);
    obDouble_.segment(36+2*exteroceptiveDim_, exteroceptiveDim_) =
        objectInfoHistory_.segment((historyLength_ -1 - 4) * exteroceptiveDim_, exteroceptiveDim_);
    obDouble_.segment(36+3*exteroceptiveDim_, exteroceptiveDim_) =
        objectInfoHistory_.segment((historyLength_ -1 - 2) * exteroceptiveDim_, exteroceptiveDim_);


    /// action history
    obDouble_.segment(36+4*exteroceptiveDim_, actionDim_) = previousAction_;
    obDouble_.segment(36+4*exteroceptiveDim_ + actionDim_, actionDim_) = prev2Action_;
    obDouble_.segment(36+4*exteroceptiveDim_ + 2*actionDim_, actionDim_) = prev3Action_;
    obDouble_.segment(36+4*exteroceptiveDim_ + 3*actionDim_, actionDim_) = prev4Action_;



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

  inline void checkConfig(const Yaml::Node &cfg) {
    READ_YAML(int, proprioceptiveDim_, cfg["dimension"]["proprioceptiveDim_"])
    READ_YAML(int, exteroceptiveDim_, cfg["dimension"]["exteroceptiveDim_"])
    READ_YAML(int, historyNum_, cfg["dimension"]["historyNum_"])
    READ_YAML(int, actionNum_, cfg["dimension"]["actionhistoryNum_"])
  }

  inline void setRewardConfig(const Yaml::Node &cfg) {
    READ_YAML(double, towardObjectRewardCoeff_, cfg["reward"]["towardObjectRewardCoeff_"])
    READ_YAML(double, stayObjectRewardCoeff_, cfg["reward"]["stayObjectRewardCoeff_"])
    READ_YAML(double, towardTargetRewardCoeff_, cfg["reward"]["towardTargetRewardCoeff_"])
    READ_YAML(double, stayTargetRewardCoeff_, cfg["reward"]["stayTargetRewardCoeff_"])
    READ_YAML(double, commandRewardCoeff_, cfg["reward"]["commandRewardCoeff_"])
    READ_YAML(double, torqueRewardCoeff_, cfg["reward"]["torque_reward_coeff"])
    READ_YAML(double, stayObjectHeadingRewardCoeff_, cfg["reward"]["stayObjectHeadingRewardCoeff_"])
  }

  void updateObject(raisim::SingleBodyObject* obj) {
    Obj_ = obj;
  }

  void updateClassifyvector(Eigen::VectorXd &classify) {
    classify_vector_ = classify;
  }

  inline void accumulateRewards(double cf, const Eigen::Vector3d &cm) {
    /// move towards the object

    Eigen::Vector3d ee_to_obj = (Obj_Pos_.e()-ee_Pos_w_.e());
    Eigen::Vector3d obj_to_target (command_Obj_Pos_ - Obj_Pos_.e());
    Eigen::Vector3d ee_to_target = (command_Obj_Pos_ - Obj_Pos_.e());
    ee_to_obj(2) = 0;
    obj_to_target(2) = 0;
    ee_to_target(2) = 0;
    Obj_Vel_(2) = 0;
//    ee_to_obj = baseRot_.e().transpose() * ee_to_obj;
//    obj_to_target = baseRot_.e().transpose() * obj_to_target;
//    ee_to_target = baseRot_.e().transpose() * ee_to_target;

    double toward_o = (ee_to_obj * (1. / (ee_to_obj.norm() + 1e-8))).transpose()*(ee_Vel_w_.e() * (1. / (ee_Vel_w_.e().norm() + 1e-8))) - 1;
    towardObjectReward_ += cf * towardObjectRewardCoeff_ * simDt_ * exp(-std::pow(std::min(0.0, toward_o), 2));

    Eigen::Vector3d heading; heading << baseRot_[0], baseRot_[1], 0;

    /// stay close to the object
    double stay_o = ee_to_obj.norm(); /// max : inf, min : 0
    double stay_o_heading = Obj_Vel_.e().dot(heading) / (heading.norm() * Obj_Vel_.e().norm() + 1e-8) - 1; /// max : 1, min : 0
    stayObjectReward_ += cf * stayObjectRewardCoeff_ * simDt_ * exp(-stay_o);
    stayObjectHeadingReward_ += cf * stayObjectHeadingRewardCoeff_ * simDt_ * exp(stay_o_heading);

    /// move the object towards the target
    double toward_t = (obj_to_target * (1. / (obj_to_target.norm() + 1e-8))).transpose()*(Obj_Vel_.e() * (1./ (Obj_Vel_.e().norm() + 1e-8))) - 1;
//    std::cout << "Obj_Vel_ : " << Obj_Vel_.e() << std::endl;
    Eigen::Vector3d Obj_Vel_n = Obj_Vel_.e() * (1./ (Obj_Vel_.e().norm() + 1e-8));
//    std::cout << "Obj_Vel_ normalized : " << Obj_Vel_n << std::endl;
    towardTargetReward_ += cf * towardTargetRewardCoeff_ * simDt_ * exp(-std::pow(std::min(0.0, toward_t), 2));
//    std::cout << "towardTargetReward : " << towardTargetReward_ << std::endl;

    /// keep the object close to the target
    double stay_t = obj_to_target.norm();
    stayTargetReward_ += cf * stayTargetRewardCoeff_ * simDt_ * exp(-stay_t);

    double commandReward_tmp = std::max(5., static_cast<double>(command_.norm()));
    commandReward_ += cf * commandRewardCoeff_ * simDt_ * commandReward_tmp;

    torqueReward_ += cf * torqueRewardCoeff_ * simDt_ * raibo_->getGeneralizedForce().norm();

  }

//  void updateHeightScan(const raisim::HeightMap *map,
//                        std::mt19937 &gen_,
//                        std::normal_distribution<double> &normDist_) {
//    /// heightmap
//    for (int k = 0; k < scanConfig_.size(); k++) {
//      for (int j = 0; j < scanConfig_[k]; j++) {
//        const double distance = 0.07 * (k + 1);
//        for (int i = 0; i < 4; i++) {
//          scanPoint_[i][scanConfig_.head(k).sum() + j][0] =
//              footPos_[i][0] + controlFrameX_[0] * distance * scanCos_(k,j) + controlFrameY_[0] * distance * scanSin_(k,j);
//          scanPoint_[i][scanConfig_.head(k).sum() + j][1] =
//              footPos_[i][1] + controlFrameX_[1] * distance * scanCos_(k,j) + controlFrameY_[1] * distance * scanSin_(k,j);
//          heightScan_[i][scanConfig_.head(k).sum() + j] =
//              map->getHeight(scanPoint_[i][scanConfig_.head(k).sum() + j][0],
//                             scanPoint_[i][scanConfig_.head(k).sum() + j][1]) - footPos_[i][2] + normDist_(gen_) * 0.025;
//        }
//      }
//    }
//  }

  inline void setStandingMode(bool mode) { standingMode_ = mode; }

  [[nodiscard]] const Eigen::VectorXd &getJointPositionHistory() const { return jointPositionHistory_; }
  [[nodiscard]] const Eigen::VectorXd &getJointVelocityHistory() const { return jointVelocityHistory_; }

  [[nodiscard]] static constexpr int getObDim() { return obDim_; }
  [[nodiscard]] static constexpr int getActionDim() { return actionDim_; }
  [[nodiscard]] static constexpr double getSimDt() { return simDt_; }
  [[nodiscard]] static constexpr double getConDt() { return conDt_; }
  void getState(Eigen::Ref<EigenVec> gc, Eigen::Ref<EigenVec> gv) { gc = gc_.cast<float>(); gv = gv_.cast<float>(); }

  static void setSimDt(double dt) {
    RSFATAL_IF(fabs(dt - simDt_) > 1e-12, "sim dt is fixed to " << simDt_)
  };
  static void setConDt(double dt) {
    RSFATAL_IF(fabs(dt - conDt_) > 1e-12, "con dt is fixed to " << conDt_)};

  [[nodiscard]] inline const std::vector<std::string> &getStepDataTag() const { return stepDataTag_; }
  [[nodiscard]] inline const Eigen::VectorXd &getStepData() const { return stepData_; }

  // robot configuration variables
  raisim::ArticulatedSystem *raibo_;
  std::vector<size_t> footIndices_, footFrameIndicies_, armIndices_;
  Eigen::VectorXd nominalConfig_;
  static constexpr int actionDim_ = 2; /// output dim : joint action 12 + task space action 6 + gain dim 4
  static constexpr size_t historyLength_ = 14;

  int proprioceptiveDim_ = 9;
  int exteroceptiveDim_ = 32;
  int historyNum_ = 4;
  int actionNum_ = 4;

  static constexpr size_t obDim_ = 172;

//  static constexpr size_t obDim_ = (proprioceptiveDim_ + exteroceptiveDim_) * (historyNum_+1) +  actionDim_ * actionNum_;

  static constexpr double simDt_ = .001;
  static constexpr int gcDim_ = 19;
  static constexpr int gvDim_ = 18;
  static constexpr int nPosHist_ = 3;
  static constexpr int nVelHist_ = 4;
  raisim::SingleBodyObject* Obj_;
  static constexpr int nJoints_ = 12;
  static constexpr int is_foot_contact_ = 0;

  // robot state variables
  Eigen::VectorXd gc_, gv_;
  Eigen::Vector3d bodyLinVel_, bodyAngVel_; /// body velocities are expressed in the body frame
  Eigen::VectorXd jointVelocity_;
  std::array<raisim::Vec<3>, 4> footPos_, footVel_;
  raisim::Vec<3> zAxis_ = {0., 0., 1.}, controlFrameX_, controlFrameY_;
  Eigen::VectorXd jointPositionHistory_;
  Eigen::VectorXd jointVelocityHistory_;
  Eigen::VectorXd historyTempMemory_;
  Eigen::VectorXd objectInfoHistory_;
  Eigen::VectorXd stateInfoHistory_;
  Eigen::VectorXd historyTempMemory_2;
  std::array<bool, 4> footContactState_;
  raisim::Mat<3, 3> baseRot_;
  Eigen::Vector3f command_;

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

  // control variables
  static constexpr double conDt_ = 0.25;
  bool standingMode_ = false;
  Eigen::VectorXd actionMean_, actionStd_, actionScaled_, previousAction_, prev2Action_, prev3Action_, prev4Action_;
  Eigen::VectorXd actionTarget_;
  Eigen::Vector3d command_Obj_Pos_;
  Eigen::VectorXd classify_vector_;


  // reward variables
  double towardObjectRewardCoeff_ = 0., towardObjectReward_ = 0.;
  double stayObjectRewardCoeff_ = 0., stayObjectReward_ = 0.;
  double towardTargetRewardCoeff_ = 0., towardTargetReward_ = 0.;
  double stayTargetRewardCoeff_ = 0., stayTargetReward_ = 0.;
  double terminalRewardCoeff_ = 0.0;
  double commandRewardCoeff_ = 0., commandReward_ = 0.;
  double torqueRewardCoeff_ = 0., torqueReward_ = 0.;
  double stayObjectHeadingReward_ = 0., stayObjectHeadingRewardCoeff_ = 0.;

  // exported data
  Eigen::VectorXd stepData_;
  std::vector<std::string> stepDataTag_;
};

}

#endif //_RAISIM_GYM_RAIBO_CONTROLLER_HPP
