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
  RaiboController() {};

  inline bool create(raisim::World *world, raisim::SingleBodyObject *obj) {
    raibo_ = reinterpret_cast<raisim::ArticulatedSystem *>(world->getObject("robot"));
    gc_.setZero(raibo_->getGeneralizedCoordinateDim());
    gv_.setZero(raibo_->getDOF());
    gc_[3] = 1;
    jointVelocity_.resize(nJoints_);
    nominalConfig_.setZero(nJoints_);
    nominalConfig_ << 0.0, 0.559836, -1.119672, -0.0, 0.559836, -1.119672, 0.0, 0.559836, -1.119672, -0.0, 0.559836, -1.119672;

    Obj_ = obj;
    /// foot scan config

    /// Observation
    actionTarget_.setZero(actionDim_);

    /// action
    actionMean_.setZero(actionDim_);
    actionStd_.setZero(actionDim_);
    actionScaled_.setZero(actionDim_);

    actionMean_ << Eigen::VectorXd::Constant(actionDim_, 0.0); /// joint target
    actionStd_<< Eigen::VectorXd::Constant(actionDim_, 1.0); /// joint target
    start_rot_.setIdentity();
    obDouble_.setZero(obDim_);
    foot_contact_.setZero();
    state_Info_.setZero(proprioceptiveDim_);

    // Update History
    objectInfoHistory_.resize(historyNum_);
    stateInfoHistory_.resize(historyNum_);
    actionInfoHistory_.resize(actionNum_);
    anchorHistory_.resize(actionNum_);
    anchorHistory_e.setZero(actionNum_ * 8 * 3);

    for (int i =0; i<historyNum_; i++) {
      stateInfoHistory_[i].setZero(proprioceptiveDim_);

    }

    for (int i = 0; i<actionNum_; i++)
    {
      actionInfoHistory_[i].setZero(actionDim_);
      anchorHistory_[i].resize(8);
      for(int j = 0; j < 8; j ++) {
        anchorHistory_[i][j].setZero();
      }
    }

    for (auto &fs: footContactState_) fs = false;


    obBlockDim_ = proprioceptiveDim_ + exteroceptiveDim_ + actionDim_ + dynamicsInfoDim_;

    RSFATAL_IF(obBlockDim_*actionNum_ != obDim_, "Dimension is fucking bull shit")

    footIndices_.push_back(raibo_->getBodyIdx("LF_SHANK"));
    footIndices_.push_back(raibo_->getBodyIdx("RF_SHANK"));
    footIndices_.push_back(raibo_->getBodyIdx("LH_SHANK"));
    footIndices_.push_back(raibo_->getBodyIdx("RH_SHANK"));
    RSFATAL_IF(std::any_of(footIndices_.begin(), footIndices_.end(), [](int i){return i < 0;}), "footIndices_ not found")

    armIndices_.push_back(raibo_->getFrameIdxByLinkName("arm_link"));

    /// exported data
    stepDataTag_ = {"towardObject_rew",
                    "stayObject_rew",
                    "towardTarget_rew",
                    "stayTarget_rew",
                    "commandsmooth_rew",
                    "commandsmooth2_rew",
                    "torque_rew",
                    "stayObject_heading_rew",
                    "stayTarget_heading_rew",
                    "intrinsic_rew",
                    "extrinsic_rew"};
    stepData_.resize(stepDataTag_.size());

    classify_vector_.setZero(3);
    classify_vector_ << 1, 0, 0;
    pre_command_.setZero();
    prepre_command_.setZero();
    target_anchor_points.resize(8);
    object_anchor_points.resize(8);
    current_anchor_points.resize(8);
    next_anchor_points.resize(8);
    success_batch_.resize(success_batch_num_);

    return true;
  };

  void updateHistory() {
    /// joint angles

    std::rotate(stateInfoHistory_.begin(), stateInfoHistory_.begin()+1, stateInfoHistory_.end());
    stateInfoHistory_[historyNum_ - 1] = state_Info_;

    std::rotate(anchorHistory_.begin(), anchorHistory_.begin()+1, anchorHistory_.end());
    anchorHistory_[actionNum_ - 1] = current_anchor_points;
  }
  /// Simdt
  void updateStateVariables() {
    is_success_ = false;
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

    raibo_->getFramePosition(raibo_->getFrameIdxByLinkName("arm_link"), ee_Pos_w_);
    raibo_->getFrameVelocity(raibo_->getFrameIdxByLinkName("arm_link"), ee_Vel_w_);

    is_contact = false;

    raisim::Vec<3> LF_FOOT_Pos_w_, RF_FOOT_Pos_w_;
    raibo_->getFramePosition(raibo_->getFrameIdxByLinkName("LF_FOOT"), LF_FOOT_Pos_w_);
    raibo_->getFramePosition(raibo_->getFrameIdxByLinkName("RF_FOOT"), RF_FOOT_Pos_w_);

    for (auto &fs: footContactState_) fs = false;
    for (auto &contact: raibo_->getContacts()) {
      auto it = std::find(footIndices_.begin(), footIndices_.end(), contact.getlocalBodyIndex());
      size_t index = it - footIndices_.begin();
      if (index < 4 && !contact.isSelfCollision()) {
        footContactState_[index] = true;
      }
    }

    /// Update State Info
    state_Info_.segment(0,3) = baseRot_.e().row(2);
    state_Info_.segment(3,3) = bodyLinVel_;
    state_Info_.segment(6,3) = bodyAngVel_;
    state_Info_.segment(9,12) = gc_.tail(nJoints_);
    state_Info_.segment(21,12) = gv_.tail(nJoints_);
    state_Info_.segment(33, 4) << static_cast<double>(footContactState_[0]), static_cast<double>(footContactState_[1]), static_cast<double>(footContactState_[2]), static_cast<double>(footContactState_[3]);
    state_Info_.segment(37,1) <<  gc_[2];

    Eigen::Vector3d geometry{0.2,0.2,0.2};
    Eigen::Vector3d pos{gc_[0],gc_[1],0};
    raisim::Mat<3,3> yaw_rot;
    Eigen::Vector3d base_x_axis = baseRot_.e().row(0);
    base_x_axis(2) = 0;
    Eigen::Vector3d base_x_axis_norm = base_x_axis.normalized();
    raisim::angleAxisToRotMat({0,0,1}, std::atan2(base_x_axis(1), base_x_axis(0)), yaw_rot);
    get_anchor_points(current_anchor_points, pos, yaw_rot.e(), geometry);

    for (auto &contact : raibo_->getContacts()) {
      if (contact.getPairObjectIndex() == Obj_->getIndexInWorld()) {
        is_contact = true;
        contact_switch = true;
        break;
      }
    }
//
//    Obj_->getPosition(Obj_Pos_);
//    Obj_->getOrientation(0, Obj_Rot_);
//    Obj_->getLinearVelocity(Obj_Vel_);
//    Obj_->getAngularVelocity(Obj_AVel_);
//    //TODO
//
//    Eigen::Vector3d ee_to_obj = (Obj_Pos_.e()-ee_Pos_w_.e());
//    Eigen::Vector3d obj_to_target = (command_Obj_Pos_ - Obj_Pos_.e());
//
//    raisim::Mat<3,3> command_Obj_Rot_;
//    raisim::quatToRotMat(command_Obj_quat_, command_Obj_Rot_);
//
//    Eigen::Vector3d target_x_axis = command_Obj_Rot_.e().col(0);
//    Eigen::Vector3d base_x_axis = baseRot_.e().col(0);
//    Eigen::Vector3d obj_x_axis = Obj_Rot_.e().col(0);
//    base_x_axis(2) = 0;
//    obj_x_axis(2) = 0;
//    target_x_axis(2) = 0;
//    Eigen::Vector3d base_x_axis_norm = base_x_axis.normalized();
//    Eigen::Vector3d obj_x_axis_norm = obj_x_axis.normalized();
//    Eigen::Vector3d target_x_axis_norm = target_x_axis.normalized();
//    double stay_t_heading  = std::abs(std::acos(Eigen::Vector3d::UnitX().dot(((obj_x_axis_norm - base_x_axis_norm) + Eigen::Vector3d::UnitX()).normalized()))); /// 1 (align), -1 (180);

//    double robot_to_obj_heading_cos = base_x_axis.dot(obj_x_axis);
//    double robot_to_obj_heading_sin = (base_x_axis(0)*obj_x_axis(1) - base_x_axis(1)*obj_x_axis(0));
//
//    double robot_to_target_heading_cos = base_x_axis.dot(target_x_axis);
//    double robot_to_target_heading_sin = base_x_axis(0)*target_x_axis(1) - base_x_axis(1)*target_x_axis(0);
//
//    double obj_to_target_heading_cos = obj_x_axis.dot(target_x_axis);
//    double obj_to_target_heading_sin = obj_x_axis(0)*target_x_axis(1) - obj_x_axis(1) * target_x_axis(0);


//    get_anchor_points(object_anchor_points, Obj_Pos_.e(), Obj_Rot_.e(), obj_geometry_);

//    if(is_multiobject_)
//    {
//      if(obj_to_target.head(2).norm() < 0.03) {
//        is_success_ = true;
//      }
//    }
//
//    else
//    {
//      if(obj_to_target.head(2).norm() < 0.05 && obj_to_target_heading_cos > 0.98) {
//        is_success_ = true;
//      }
//    }
//    if(obj_to_target.head(2).norm() < 0.05 && obj_to_target_heading_cos > 0.98 ) {
//	    is_success_ = true;
//    }

//    std::rotate(success_batch_.begin(), success_batch_.begin()+1, success_batch_.end());
//    success_batch_[success_batch_num_ - 1] = is_success_;
//
//    Eigen::Vector3d ee_to_target = (command_Obj_Pos_ - ee_Pos_w_.e());
//    ee_to_obj(2) = 0;
//    obj_to_target(2) = 0;
//    ee_to_target(2) = 0;
//    ee_to_obj = baseRot_.e().transpose() * ee_to_obj;
//    obj_to_target = baseRot_.e().transpose() * obj_to_target;
//    ee_to_target = baseRot_.e().transpose() * ee_to_target;
//
//    Eigen::Vector2d pos_temp_;
//    double dist_temp_min_;
//
//    dist_temp_ = ee_to_obj.head(2).norm() + 1e-8;
//    pos_temp_ = ee_to_obj.head(2) * (1.0/dist_temp_);
//    Obj_Info_.segment(0, 2) << pos_temp_;
//    dist_temp_min_ = std::min(2.0, dist_temp_);
//    Obj_Info_.segment(2, 1) << dist_temp_min_;
//
//
//    dist_temp_ = obj_to_target.head(2).norm() + 1e-8;
//    pos_temp_ = obj_to_target.head(2) * (1.0/dist_temp_);
//
//    Obj_Info_.segment(3, 2) << pos_temp_;
//    dist_temp_min_ = std::min(2.0, dist_temp_);
//    Obj_Info_.segment(5, 1) << dist_temp_min_;
//
//
//    dist_temp_ = ee_to_target.head(2).norm() + 1e-8;
//    pos_temp_ = ee_to_target.head(2) * (1.0/dist_temp_);
//
//    Obj_Info_.segment(6, 2) << pos_temp_;
//    dist_temp_min_ = std::min(2.0, dist_temp_);
//    Obj_Info_.segment(8, 1) << dist_temp_min_;
//    Obj_Info_.segment(9, 3) << baseRot_.e().transpose() * Obj_Vel_.e();
//
////    baseRot_transform = rotMatTransform(baseRot_);
////    objRot_transform = rotMatTransform(Obj_Rot_);
////    base_to_obj_Rot_ = baseRot_transform.transpose() * objRot_transform;
//
//    Obj_Info_.segment(12,2) << robot_to_obj_heading_cos, robot_to_obj_heading_sin;
//    Obj_Info_.segment(14,2) << robot_to_target_heading_cos, robot_to_target_heading_sin;
//    Obj_Info_.segment(16,2) << obj_to_target_heading_cos, obj_to_target_heading_sin;
//    Obj_Info_.segment(18,3) = classify_vector_;
//    Obj_Info_.segment(21,3) = obj_geometry_;
//    Obj_Info_.segment(24, 1) << Obj_->getMass();
//    Obj_Info_.segment(25, 3) << Obj_->getCom().e();
//    Obj_Info_.segment(28,3) = Obj_->getInertiaMatrix_B().row(0);
//    Obj_Info_.segment(31,3) = Obj_->getInertiaMatrix_B().row(1);
//    Obj_Info_.segment(34,3) = Obj_->getInertiaMatrix_B().row(2);
//    Obj_Info_.segment(37,1) << friction_;
//    Obj_Info_.segment(38,1) << air_damping;
//    Obj_Info_.segment(39,1) << static_cast<double>(is_contact);
//
//
//    dynamics_Info_.segment(0,3) << Obj_Pos_.e();
//    dynamics_Info_.segment(3,3) << ee_Pos_w_.e();
//    dynamics_Info_.segment(6,3) << Obj_Vel_.e();
//    dynamics_Info_.segment(9,3) << gv_.segment(0,3);
//    dynamics_Info_.segment(12,3) << Obj_AVel_.e();
//    dynamics_Info_.segment(15,3) << gv_.segment(3,3);
//    dynamics_Info_.segment(18,3) = Obj_Rot_.e().row(0);
//    dynamics_Info_.segment(21,3) = baseRot_.e().row(0);
//    dynamics_Info_.segment(24,3) << obj_geometry_;

    /// height map
    controlFrameX_ =
        {baseRot_[0], baseRot_[1], 0.}; /// body x axis projected on the world x-y plane, expressed in the world frame
    controlFrameX_ /= (controlFrameX_.norm() + 1e-10);
    raisim::cross(zAxis_, controlFrameX_, controlFrameY_);


    /// Check if the distance between command pos and robot base are under offset.

//    Eigen::Vector3d current_base_pos = raibo_->getBasePosition().e();
//    current_base_pos[2] = 0;
//    desired_dist_ = (current_base_pos - desired_pos_).norm();
//    if(desired_dist_ < 0.1)
//      is_achieved = true;

  }



  Eigen::VectorXd get_desired_pos () {
    return desired_pos_;
  }

  bool getContact() {
    return contact_switch;
  }

  void get_anchor_points(std::vector<Eigen::Vector3d>& anchor_points, Eigen::Vector3d COM, Eigen::Matrix3d Rot, Eigen::Vector3d geometry) {
    /// i for x, j for y, k for z
    /// Extract 8 points in total.
    Eigen::Vector3d anchor_point;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        for (int k = 0; k < 2; k++)
        {
          anchor_point = {geometry(0) * (static_cast<double>((i)) - 0.5), geometry(1) * (static_cast<double>((j)) - 0.5), geometry(2) * (static_cast<double>((k)) - 0.5)};
          anchor_points[4*i + 2*j + k] = COM + Rot * anchor_point;
        }
      }
    }
  }
    void estimate_anchor_points(std::vector<Eigen::Vector3d>& current_anchor_points,std::vector<Eigen::Vector3d>& prev_anchor_points, const Eigen::Ref<EigenVec>& anchors,  Eigen::Matrix3d Rot) {
        /// i for x, j for y, k for z
        /// Extract 8 points in total.
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++)
                {
                    current_anchor_points[4*i + 2*j + k] = prev_anchor_points[4*i + 2*j + k] + Rot * anchors.segment(3*(4*i + 2*j + k),3).cast<double>();
                }
            }
        }
    }

  void getObservation(Eigen::VectorXd &observation) {
    observation = obDouble_;
  }

  Eigen::VectorXf advance(raisim::World *world, const Eigen::Ref<EigenVec> &action) {
    Eigen::VectorXf position;
    position = action.cast<float>().cwiseQuotient(actionStd_.cast<float>());
    Eigen::VectorXd current_pos_ = raibo_->getBasePosition().e();
    position += actionMean_.cast<float>();
    prepre_command_ = pre_command_;
    start_rot_ = baseRot_.e();
    pre_command_ = command_;
    if(is_position_goal)
      command_ = {position(0), position(1), 0};
    else
      command_ = {position(0), position(1), position(2)};
    desired_pos_ = command_.cast<double>();
//
//    desired_pos_ += current_pos_;
//    desired_pos_(2) = 0;

    is_achieved = false;

//    desired_dist_ = (raibo_->getBasePosition().e() - current_pos_).norm();
    return command_;
  }

  bool is_achieve() {
    return is_achieved;
  }

  bool update_actionHistory(raisim::World *world, const Eigen::Ref<EigenVec> &action, double curriculumFactor) {
    /// action scaling
//    std::rotate(actionInfoHistory_.begin(), actionInfoHistory_.begin()+1, actionInfoHistory_.end());
//    actionInfoHistory_[actionNum_ - 1] = action.cast<double>();
//
//    actionInfoHistory_.fill
    for (int i = 0; i < actionNum_; i++) {
      actionInfoHistory_[i] = command_.cast<double>();
    }

    return true;
  }

  Eigen::VectorXd get_com_pos() {
    return Obj_Pos_.e();
  }

  Eigen::VectorXd get_noisify_com_pos() {
    return Obj_Pos_.e() + Obj_->getCom().e();
  }

  void get_anchor_history(Eigen::Ref<EigenVec> &anchor_points, bool robotFrame) {
    if(robotFrame) {
      for(int i = 0; i < actionNum_; i ++) {
        for (int j = 0; j < 8; j++) {
          anchorHistory_e.segment(24 * i + 3 * j, 3) = start_rot_.transpose()*anchorHistory_[i][j];
        }
      }
    }

    else {
      for(int i = 0; i < actionNum_; i ++) {
        for (int j = 0; j < 8; j++) {
          anchorHistory_e.segment(24 * i + 3 * j, 3) = anchorHistory_[i][j];
        }
      }
    }

    anchor_points = anchorHistory_e.cast<float>();
  }

//  void reset_Rollout(Eigen::Vector3d command_obj_pos_, Eigen::Vector4d command_obj_quat_, Eigen::Vector3d obj_geometry, double friction) {
//    raibo_->getState(gc_, gv_);
//    is_success_ = false;
//    is_achieved = true;
//    command_Obj_Pos_ = command_obj_pos_;
//    command_Obj_quat_ = command_obj_quat_;
//    obj_geometry_ = obj_geometry;
//    friction_ = friction;
//
//    std::fill(success_batch_.begin(), success_batch_.end(), false);
//  }

  void reset(std::mt19937 &gen_,
             std::normal_distribution<double> &normDist_, Eigen::Vector3d command_obj_pos_, Eigen::Vector4d command_obj_quat_, Eigen::Vector3d obj_geometry, double friction, double damping) {
    raibo_->getState(gc_, gv_);
//    jointTarget_ = gc_.segment(7, nJoints_);
//    command_Obj_Pos_ = command_obj_pos_;
//    command_Obj_quat_ = command_obj_quat_;
//    obj_geometry_ = obj_geometry;

//    raisim::Mat<3,3> command_obj_rot;
//    raisim::quatToRotMat(command_Obj_quat_, command_obj_rot);
//    get_anchor_points(target_anchor_points, command_Obj_Pos_, command_obj_rot.e(), obj_geometry_);
//    Obj_->getPosition(Obj_Pos_);
//    distance = (command_obj_pos_.head(2) - Obj_Pos_.e().head(2)).norm();
//    is_success_ = false;
//    is_achieved = true;
//    contact_switch = false;
//    intrinsic_switch = true;
//    friction_ = friction;
//    air_damping = damping;
    pre_command_.setZero();
    prepre_command_.setZero();
    // history
    for (int i = 0; i < historyNum_; i++)
    {
      for (int j=0; j < proprioceptiveDim_; j++)
        stateInfoHistory_[i](j) = normDist_(gen_) * 0.1;
    }

    for (int i = 0; i < actionNum_; i++) {
      for (int j=0; j < actionDim_; j++)
        actionInfoHistory_[i](j) = normDist_(gen_) * 0.1;
    }


    std::fill(success_batch_.begin(), success_batch_.end(), false);

  }

  [[nodiscard]] float getRewardSum(bool visualize) {
    stepData_[0] = towardObjectReward_;
    stepData_[1] = stayObjectReward_;
    stepData_[2] = towardTargetReward_;
    stepData_[3] = stayTargetReward_;
    stepData_[4] = commandsmoothReward_;
    stepData_[5] = commandsmooth2Reward_;
    stepData_[6] = torqueReward_;
    stepData_[7] = stayObjectHeadingReward_;
    stepData_[8] = stayTargetHeadingReward_;
    stepData_[9] = intrinsicReward_;
    stepData_[10] = extrinsicReward_;

    towardObjectReward_ = 0.;
    stayObjectReward_ = 0.;
    towardTargetReward_ = 0.;
    stayTargetReward_ = 0.;
    commandsmoothReward_ = 0.;
    commandsmooth2Reward_ = 0.;
    torqueReward_ = 0.;
    stayObjectHeadingReward_ = 0.;
    stayTargetHeadingReward_ = 0.;
    intrinsicReward_ = 0.;
    extrinsicReward_ = 0.;
    stayTargetExtrinsicReward_ = 0.;
//    intrinsic_switch = true;

    return float(stepData_.tail(2).sum());
  }

  [[nodiscard]] bool isTerminalState(float &terminalReward) {
    terminalReward = float(terminalRewardCoeff_);

    return is_success_;
//    if(std::find(success_batch_.begin(), success_batch_.end(), false) == success_batch_.end())
//      return true;

    terminalReward = 0.f;

    return false;
  }

  void updateObservation(bool nosify,
                         const Eigen::Vector3d &command,
                         const raisim::HeightMap *map,
                         std::mt19937 &gen_,
                         std::normal_distribution<double> &normDist_) {


    // update History
    for (int i=0; i< historyNum_; i++) {
      obDouble_.segment((obBlockDim_)*i,
                        proprioceptiveDim_) = stateInfoHistory_[i];
      obDouble_.segment((obBlockDim_)*i + proprioceptiveDim_ + exteroceptiveDim_,
                        actionDim_) = actionInfoHistory_[i];
    }

    obDouble_.segment((obBlockDim_)*historyNum_, proprioceptiveDim_)
        = state_Info_;

    obDouble_.segment((obBlockDim_)*historyNum_ + proprioceptiveDim_+exteroceptiveDim_, actionDim_)
        = actionInfoHistory_.back();

    std::rotate(anchorHistory_.begin(), anchorHistory_.begin()+1, anchorHistory_.end());
    anchorHistory_[actionNum_ - 1] = current_anchor_points;
  }

  inline void checkConfig(const Yaml::Node &cfg) {
    READ_YAML(int, proprioceptiveDim_, cfg["dimension"]["proprioceptiveDim_"])
    READ_YAML(int, historyNum_, cfg["dimension"]["historyNum_"])
    READ_YAML(int, actionNum_, cfg["dimension"]["actionhistoryNum_"])
  }

  inline void setRewardConfig(const Yaml::Node &cfg) {
    READ_YAML(bool, is_multiobject_, cfg["MultiObject"])
    READ_YAML(double, towardObjectRewardCoeff_, cfg["reward"]["towardObjectRewardCoeff_"])
    READ_YAML(double, stayObjectRewardCoeff_, cfg["reward"]["stayObjectRewardCoeff_"])
    READ_YAML(double, towardTargetRewardCoeff_, cfg["reward"]["towardTargetRewardCoeff_"])
    READ_YAML(double, stayTargetRewardCoeff_, cfg["reward"]["stayTargetRewardCoeff_"])
    READ_YAML(double, commandsmoothRewardCoeff_, cfg["reward"]["commandsmoothRewardCoeff_"])
    READ_YAML(double, commandsmooth2RewardCoeff_, cfg["reward"]["commandsmooth2RewardCoeff_"])
    READ_YAML(double, torqueRewardCoeff_, cfg["reward"]["torque_reward_coeff"])
    READ_YAML(double, stayObjectHeadingRewardCoeff_, cfg["reward"]["stayObjectHeadingRewardCoeff_"])
    READ_YAML(double, stayTargetHeadingRewardCoeff_, cfg["reward"]["stayTargetHeadingRewardCoeff_"])
    READ_YAML(double, stayTargetHeadingRewardCoeff_alpha_, cfg["reward"]["stayTargetHeadingRewardCoeff_alpha_"])
    READ_YAML(double, stayTargetRewardCoeff_alpha_, cfg["reward"]["stayTargetRewardCoeff_alpha_"])
    READ_YAML(double, stayTargetExtrinsicRewardCoeff_, cfg["reward"]["stayTargetExtrinsicRewardCoeff_"])

    /// If multi object condition, dismiss the orientation reward
    if(is_multiobject_)
    {
      stayTargetHeadingRewardCoeff_ = 0.;
    }

  }

  void updateObject(raisim::SingleBodyObject* obj) {
    Obj_ = obj;
  }

  void updateClassifyvector(Eigen::VectorXd &classify) {
    classify_vector_ = classify;
  }

  inline void accumulateRewards(double cf, const Eigen::Vector3d &cm) {
    /// move towards the object
//
//    Eigen::Vector3d ee_to_obj = (Obj_Pos_.e()-ee_Pos_w_.e());
//    Eigen::Vector3d obj_to_target (command_Obj_Pos_ - Obj_Pos_.e());
//    Eigen::Vector3d ee_to_target = (command_Obj_Pos_ - ee_Pos_w_.e());
//
//    ee_to_obj(2) = 0;
//    obj_to_target(2) = 0;
//    ee_to_target(2) = 0;
//    Obj_Vel_(2) = 0;
//
////    ee_to_obj = baseRot_.e().transpose() * ee_to_obj;
////    obj_to_target = baseRot_.e().transpose() * obj_to_target;
////    ee_to_target = baseRot_.e().transpose() * ee_to_target;
//
//    double toward_o = (ee_to_obj * (1. / (ee_to_obj.norm() + 1e-8))).transpose()*(ee_Vel_w_.e() * (1. / (ee_Vel_w_.e().norm() + 1e-8))) - 1;
////    towardObjectReward_ += cf * towardObjectRewardCoeff_ * simDt_ * exp(-std::pow(std::min(0.0, toward_o), 2));
//
//    Eigen::Vector3d heading = baseRot_.e().col(0);
//
//
////    raisim::quatToEulerVec()
//
//    /// stay close to the object
//    double stay_o = ee_to_obj.norm(); /// max : inf, min : 0
////    double o_bound = obj_geometry_.head(2).norm() / 2 + 5e-2;
////    stayObjectReward_ += cf * stayObjectRewardCoeff_ * simDt_ * exp(-stay_o);
//
//    /// align robot head to the object
//    double stay_o_heading = Obj_Vel_.e().dot(heading) / (heading.norm() * Obj_Vel_.e().norm() + 1e-8) - 1; /// max : 0, min : -1
//    stayObjectHeadingReward_ += cf * stayObjectHeadingRewardCoeff_ * simDt_ * exp(stay_o_heading);
//
//    /// move the object towards the target
//    double toward_t = (obj_to_target * (1. / (obj_to_target.norm() + 1e-8))).transpose()*(Obj_Vel_.e() * (1./ (Obj_Vel_.e().norm() + 1e-8))) - 1;
////    towardTargetReward_ += cf * towardTargetRewardCoeff_ * simDt_ * exp(-std::pow(std::min(0.0, toward_t), 2));
//
//    /// keep the object close to the target (extrinsic)
//    double stay_t = 0;
//    for (int i = 0; i < object_anchor_points.size(); i++) {
//      stay_t += (target_anchor_points[i] - object_anchor_points[i]).norm();
//    }
//    /// Mean value
//    stay_t = stay_t / 8;
//
//    /// align object to the desired orientation (extrinsic)
//    raisim::Mat<3,3> command_Obj_Rot_;
//    raisim::quatToRotMat(command_Obj_quat_, command_Obj_Rot_);
//
//    Eigen::Vector3d target_x_axis = command_Obj_Rot_.e().col(0);
//    Eigen::Vector3d base_x_axis = baseRot_.e().col(0);
//    Eigen::Vector3d obj_x_axis = Obj_Rot_.e().col(0);
////    base_x_axis(2) = 0;
////    obj_x_axis(2) = 0;
////    target_x_axis(2) = 0;
////    Eigen::Vector3d base_x_axis_norm = base_x_axis.normalized();
////    Eigen::Vector3d obj_x_axis_norm = obj_x_axis.normalized();
////    Eigen::Vector3d target_x_axis_norm = target_x_axis.normalized();
//
//    double robot_to_obj_heading_cos = base_x_axis.dot(obj_x_axis);
//    double robot_to_obj_heading_sin = (base_x_axis(0)*obj_x_axis(1) - base_x_axis(1)*obj_x_axis(0));
//
//    double robot_to_target_heading_cos = base_x_axis.dot(target_x_axis);
//    double robot_to_target_heading_sin = base_x_axis(0)*target_x_axis(1) - base_x_axis(1)*target_x_axis(0);
//
//    double obj_to_target_heading_cos = obj_x_axis.dot(target_x_axis);
//    double obj_to_target_heading_sin = obj_x_axis(0)*target_x_axis(1) - obj_x_axis(1) * target_x_axis(0);
//
////    double stay_t_heading = 0;
////    stayTargetHeadingReward_ += cf * stayTargetHeadingRewardCoeff_ * simDt_ * exp(stay_t_heading)
////        * exp(-stayTargetHeadingRewardCoeff_alpha_ * obj_to_target.norm());
//
//    /// Smooth reward (intrinsic)
//    double command_smooth = (command_ - pre_command_).squaredNorm();
//    double command_smooth2 = (command_ - 2*pre_command_ + prepre_command_).squaredNorm();
////    commandsmoothReward_ += cf * commandsmoothRewardCoeff_ * simDt_ * exp(-command_smooth);
////    commandsmooth2Reward_ += cf * commandsmooth2RewardCoeff_ * simDt_ * exp(-command_smooth2);
////    torqueReward_ += cf * torqueRewardCoeff_ * simDt_ * raibo_->getGeneralizedForce().norm();
//
//    /// gathers intrinsic reward & extrinsic reward
//    /// if the distance between object and target below threshold, from that moment, we doesn't consider the intrinsic reward (saturate)
//    /// If reached rate == 0 => the object reached the target postion
//    double reached_rate = stay_t / distance;
//    if(stay_t < 0.2)
//    {
//      intrinsic_switch = false;
//    }
//
//    else
//    {
//      intrinsic_switch = true;
//    }
//
//    if (intrinsic_switch) {
//      towardObjectReward_ += towardObjectRewardCoeff_ * simDt_ * exp(-std::pow(std::min(0.0, toward_o), 2));
//      stayObjectReward_ += stayObjectRewardCoeff_ * simDt_ * exp(-stay_o);
//      towardTargetReward_ += towardTargetRewardCoeff_ * simDt_ * exp(-std::pow(std::min(0.0, toward_t), 2));
//      stayTargetReward_ += stayTargetRewardCoeff_ * simDt_ * (-log(stay_t + 0.05));
//      commandsmoothReward_ += cf * commandsmoothRewardCoeff_ * simDt_ * exp(-command_smooth);
//      commandsmooth2Reward_ += cf * commandsmooth2RewardCoeff_ * simDt_ * exp(-command_smooth2);
//      torqueReward_ += cf * torqueRewardCoeff_ * simDt_ * raibo_->getGeneralizedForce().norm();
//    }
//
//    else
//    {
//      stayTargetExtrinsicReward_ += stayTargetRewardCoeff_ * simDt_ * -log(stay_t + 0.05);
//      if (stay_t < 0.05)
//      {
//        stayTargetExtrinsicReward_ += stayTargetRewardCoeff_ * simDt_ * -log(stay_t + 0.05);
//      }
//
//    }
//
//    intrinsicReward_ = towardObjectReward_ + stayObjectReward_ + stayObjectHeadingReward_ + towardTargetReward_ + commandsmoothReward_ + commandsmooth2Reward_ + torqueReward_ + stayTargetHeadingReward_ + stayTargetReward_;
//    extrinsicReward_ = stayTargetExtrinsicReward_;
  }

  Eigen::Matrix3d rotMatTransform(raisim::Mat<3,3> rot) {
    Eigen::Vector3d x = rot.e().col(0);
    Eigen::Matrix3d rot_transform;
    x(2) = 0;
    Eigen::Vector3d x_norm = x.normalized();
    rot_transform.col(0) = x_norm;
    rot_transform.col(2) = Eigen::Vector3d::UnitZ();
    rot_transform.col(1) = rot_transform.col(2).cross(x);

    return rot_transform;
  }

  void get_privileged_information(Eigen::Ref<EigenVec> &privileged_information) {
    privileged_information = Obj_Info_.tail(privilegedDim_).cast<float>();
  }

  void set_History(std::vector<Eigen::VectorXd> &obj_info_history,
                   std::vector<Eigen::VectorXd> &state_info_history,
                   std::vector<Eigen::VectorXd> &action_info_history,
                   std::vector<Eigen::VectorXd> &dynamics_info_history) {
//    objectInfoHistory_ = obj_info_history;
//    stateInfoHistory_ = state_info_history;
//    actionInfoHistory_ = action_info_history;
//    dynamicsInfoHistory_ = dynamics_info_history;
  }

  void get_History(std::vector<Eigen::VectorXd> &obj_info_history,
                   std::vector<Eigen::VectorXd> &state_info_history,
                   std::vector<Eigen::VectorXd> &action_info_history,
                   std::vector<Eigen::VectorXd> &dynamics_info_history) {
//    obj_info_history = objectInfoHistory_;
//    state_info_history = stateInfoHistory_;
//    action_info_history = actionInfoHistory_;
//    dynamics_info_history = dynamicsInfoHistory_;
  }

  inline void setStandingMode(bool mode) { standingMode_ = mode; }

  [[nodiscard]] const Eigen::VectorXd &getJointPositionHistory() const { return jointPositionHistory_; }
  [[nodiscard]] const Eigen::VectorXd &getJointVelocityHistory() const { return jointVelocityHistory_; }

  [[nodiscard]] static constexpr int getObDim() { return obDim_; }
  [[nodiscard]] static constexpr int getActionDim() { return actionDim_; }
  [[nodiscard]] static constexpr double getSimDt() { return simDt_; }
  [[nodiscard]] static constexpr double getConDt() { return conDt_; }

  void is_success(bool &success) {
    success = is_success_;
  }

  void get_intrinsic_switch(bool &switch_){
    switch_ = not intrinsic_switch;
  }

  void getState(Eigen::Ref<EigenVec> gc, Eigen::Ref<EigenVec> gv) { gc = gc_.cast<float>(); gv = gv_.cast<float>(); }
  void setState(Eigen::Ref<EigenVec> gc, Eigen::Ref<EigenVec> gv) {
    raibo_->setState(gc.cast<double>(), gv.cast<double>());
    gc_ = gc.cast<double>();
    gv_ = gv.cast<double>();}

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
  /// output dim : joint action 12 + task space action 6 + gain dim 4

  int proprioceptiveDim_ = 38;
  int exteroceptiveDim_ = 0;
  int dynamicsInfoDim_ = 0;
  static constexpr int actionDim_ = 3;
  int historyNum_ = 20;
  int actionNum_ = 21;
  int obBlockDim_ = 0;
  int privilegedDim_ = 0;

  static constexpr size_t obDim_ = 861;

//  static constexpr size_t obDim_ = (proprioceptiveDim_ + exteroceptiveDim_) * (historyNum_+1) +  actionDim_ * actionNum_;

//  static constexpr double simDt_ = .001;
  static constexpr double simDt_ = .0025;
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
  std::vector<Eigen::VectorXd> objectInfoHistory_;
  std::vector<Eigen::VectorXd> stateInfoHistory_;
  std::vector<std::vector<Eigen::Vector3d>> anchorHistory_;
  Eigen::VectorXd rotHistory_e;
  Eigen::VectorXd anchorHistory_e;
  std::vector<Eigen::VectorXd> actionInfoHistory_;
  std::vector<Eigen::VectorXd> dynamicsInfoHistory_;
  Eigen::VectorXd historyTempMemory_2;
  std::array<bool, 4> footContactState_;
  raisim::Mat<3, 3> baseRot_;
  Eigen::Matrix3d baseRot_transform;
  Eigen::Matrix3d objRot_transform;
  Eigen::Matrix3d base_to_obj_Rot_;
  Eigen::Matrix3d start_rot_;
  Eigen::Vector3f command_, pre_command_, prepre_command_;
  Eigen::Vector3d desired_pos_;
  double desired_dist_;
  bool is_achieved = true;
  bool is_discrete_ = false;
  bool is_position_goal = false;
  int success_batch_num_ = 50;
  std::vector<bool> success_batch_;
  double dist_temp_;
  double friction_ = 1.1;
  double air_damping = 0.5;

  // robot observation variables
  std::vector<Eigen::Vector3d> target_anchor_points;
  std::vector<Eigen::Vector3d> object_anchor_points;
  std::vector<Eigen::Vector3d> current_anchor_points;
  std::vector<Eigen::Vector3d> next_anchor_points;
//  Eigen::VectorXd current_anchor_points_e;

  std::vector<raisim::VecDyn> heightScan_;
  Eigen::VectorXi scanConfig_;
  Eigen::VectorXd obDouble_;
  std::vector<std::vector<raisim::Vec<2>>> scanPoint_;
  Eigen::MatrixXd scanSin_;
  Eigen::MatrixXd scanCos_;
  Eigen::VectorXd Obj_Info_;
  Eigen::VectorXd state_Info_;
  Eigen::VectorXd dynamics_Info_;
  raisim::Vec<3> Obj_Pos_, Obj_Vel_, Obj_AVel_;
  raisim::Mat<3,3> Obj_Rot_, Tar_Rot_;
  raisim::Vec<3> ee_Pos_w_, ee_Vel_w_, ee_Avel_w_;
  raisim::Mat<3,3> eeRot_w_;
  double distance = 0.;
  bool is_contact = false;
  bool contact_switch = false;
  std::vector<Eigen::Vector2d> command_library_;

  // control variables
  static constexpr double conDt_ = 0.2;
  bool standingMode_ = false;
  Eigen::VectorXd actionMean_, actionStd_, actionScaled_;
  Eigen::VectorXd actionTarget_;
  Eigen::Vector3d command_Obj_Pos_;
  Eigen::Vector4d command_Obj_quat_;
  Eigen::Vector3d obj_geometry_;
  Eigen::VectorXd classify_vector_;
  Eigen::Vector4d foot_contact_;

  // For testing
  bool is_success_ = false;

  bool is_multiobject_ = true;

  // reward variables
  double towardObjectRewardCoeff_ = 0., towardObjectReward_ = 0.;
  double stayObjectRewardCoeff_ = 0., stayObjectReward_ = 0.;
  double towardTargetRewardCoeff_ = 0., towardTargetReward_ = 0.;
  double stayTargetRewardCoeff_ = 0., stayTargetReward_ = 0., stayTargetExtrinsicReward_ = 0., stayTargetExtrinsicRewardCoeff_ = 0., stayTargetRewardCoeff_alpha_ = 0.;
  double terminalRewardCoeff_ = 1000.0;
  double commandsmoothRewardCoeff_ = 0., commandsmoothReward_ = 0.;
  double commandsmooth2RewardCoeff_ = 0., commandsmooth2Reward_ = 0.;
  double torqueRewardCoeff_ = 0., torqueReward_ = 0.;
  double stayObjectHeadingReward_ = 0., stayObjectHeadingRewardCoeff_ = 0.;
  double stayTargetHeadingReward_ = 0.,  stayTargetHeadingRewardCoeff_ = 0., stayTargetHeadingRewardCoeff_alpha_ = 0. ;
  double intrinsicReward_ = 0.;
  bool intrinsic_switch = true;
  double extrinsicReward_ = 0.;
  // exported data
  Eigen::VectorXd stepData_;
  std::vector<std::string> stepDataTag_;
};

}

#endif //_RAISIM_GYM_RAIBO_CONTROLLER_HPP
