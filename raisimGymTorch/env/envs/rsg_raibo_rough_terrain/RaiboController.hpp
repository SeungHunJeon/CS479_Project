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

    obDouble_.setZero(obDim_);

    state_Info_.setZero(proprioceptiveDim_);
    Obj_Info_.setZero(exteroceptiveDim_);
    dynamics_Info_.setZero(dynamicsInfoDim_);

    // Update History
    objectInfoHistory_.resize(historyNum_);
    stateInfoHistory_.resize(historyNum_);
    actionInfoHistory_.resize(actionNum_);
    dynamicsInfoHistory_.resize(actionNum_);

    for (int i =0; i<historyNum_; i++) {
      objectInfoHistory_[i].setZero(exteroceptiveDim_);
      stateInfoHistory_[i].setZero(proprioceptiveDim_);
    }

    for (int i = 0; i<actionNum_; i++)
    {
      actionInfoHistory_[i].setZero(actionDim_);
      dynamicsInfoHistory_[i].setZero(dynamicsInfoDim_);
    }


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
                    "stayTarget_heading_rew"};
    stepData_.resize(stepDataTag_.size());


    classify_vector_.setZero(4);
    classify_vector_ << 1, 0, 0, 0;
    pre_command_.setZero();
    prepre_command_.setZero();

    success_batch_.resize(success_batch_num_);

    return true;
  };

  void updateHistory() {
    /// joint angles

    std::rotate(objectInfoHistory_.begin(), objectInfoHistory_.begin()+1, objectInfoHistory_.end());
    objectInfoHistory_[historyNum_ - 1] = Obj_Info_;


    std::rotate(stateInfoHistory_.begin(), stateInfoHistory_.begin()+1, stateInfoHistory_.end());
    stateInfoHistory_[historyNum_ - 1] = state_Info_;
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

    /// Update State Info
    state_Info_.segment(0,3) = baseRot_.e().row(2);
    state_Info_.segment(3,3) = bodyLinVel_;
    state_Info_.segment(6,3) = bodyAngVel_;
    state_Info_.segment(9,12) = gc_.segment(7, 12);

    for (auto &contact: raibo_->getContacts()) {
      if (contact.getlocalBodyIndex() == armIndices_.front()) {
        is_contact = true;
        break;
      }
    }

    Obj_->getPosition(Obj_Pos_);
    Obj_->getOrientation(0, Obj_Rot_);
    Obj_->getLinearVelocity(Obj_Vel_);
    Obj_->getAngularVelocity(Obj_AVel_);
    //TODO

    Eigen::Vector3d ee_to_obj = (Obj_Pos_.e()-ee_Pos_w_.e());
    Eigen::Vector3d obj_to_target = (command_Obj_Pos_ - Obj_Pos_.e());

    raisim::Mat<3,3> command_Obj_Rot_;
    raisim::quatToRotMat(command_Obj_quat_, command_Obj_Rot_);
    double stay_t_heading  = abs(Obj_Rot_.e().row(0).head(2).dot(command_Obj_Rot_.e().row(0).head(2))/(Obj_Rot_.e().row(0).head(2).norm()*command_Obj_Rot_.e().row(0).head(2).norm() + 1e-8));
    if(obj_to_target.head(2).norm() < 0.05 && stay_t_heading > 0.98)
      is_success_ = true;

    std::rotate(success_batch_.begin(), success_batch_.begin()+1, success_batch_.end());
    success_batch_[success_batch_num_ - 1] = is_success_;

    Eigen::Vector3d ee_to_target = (command_Obj_Pos_ - ee_Pos_w_.e());
    ee_to_obj(2) = 0;
    obj_to_target(2) = 0;
    ee_to_target(2) = 0;
    ee_to_obj = baseRot_.e().transpose() * ee_to_obj;
    obj_to_target = baseRot_.e().transpose() * obj_to_target;
    ee_to_target = baseRot_.e().transpose() * ee_to_target;

    Eigen::Vector2d pos_temp_;
    double dist_temp_min_;

    dist_temp_ = ee_to_obj.head(2).norm() + 1e-8;
    pos_temp_ = ee_to_obj.head(2) * (1.0/dist_temp_);
    Obj_Info_.segment(0, 2) << pos_temp_;
    dist_temp_min_ = std::min(2.0, dist_temp_);
    Obj_Info_.segment(2, 1) << dist_temp_min_;


    dist_temp_ = obj_to_target.head(2).norm() + 1e-8;
    pos_temp_ = obj_to_target.head(2) * (1.0/dist_temp_);

    Obj_Info_.segment(3, 2) << pos_temp_;
    dist_temp_min_ = std::min(2.0, dist_temp_);
    Obj_Info_.segment(5, 1) << dist_temp_min_;


    dist_temp_ = ee_to_target.head(2).norm() + 1e-8;
    pos_temp_ = ee_to_target.head(2) * (1.0/dist_temp_);

    Obj_Info_.segment(6, 2) << pos_temp_;
    dist_temp_min_ = std::min(2.0, dist_temp_);
    Obj_Info_.segment(8, 1) << dist_temp_min_;
    Obj_Info_.segment(9, 3) << baseRot_.e().transpose() * Obj_Vel_.e();
    Obj_Info_.segment(12, 3) << baseRot_.e().transpose() * Obj_AVel_.e();
    // TODO
    /// I think the orientation would be represented as relative orientation => rotation difference => cosine similarity has to be adapted
    /// This representation means that each body frame's (robot, object) x axis vector
    Obj_Info_.segment(15,3) = (baseRot_.e().row(0) - Obj_->getOrientation().e().row(0));
//    Obj_Info_.segment(21,4) = classify_vector_;
    Obj_Info_.segment(18,3) = obj_geometry_;
    Obj_Info_.segment(21, 1) << Obj_->getMass();
    Obj_Info_.segment(22, 3) << Obj_->getCom().e();
    Obj_Info_.segment(25,3) = Obj_->getInertiaMatrix_B().row(0);
    Obj_Info_.segment(28,3) = Obj_->getInertiaMatrix_B().row(1);
    Obj_Info_.segment(31,3) = Obj_->getInertiaMatrix_B().row(2);
    Obj_Info_.segment(34,1) << friction_;
    Obj_Info_.segment(35,1) << static_cast<double>(is_contact);


    dynamics_Info_.segment(0,3) << Obj_Pos_.e();
    dynamics_Info_.segment(3,3) << ee_Pos_w_.e();
    dynamics_Info_.segment(6,3) << Obj_Vel_.e();
    dynamics_Info_.segment(9,3) << gv_.segment(0,3);
    dynamics_Info_.segment(12,3) << Obj_AVel_.e();
    dynamics_Info_.segment(15,3) << gv_.segment(3,3);
    dynamics_Info_.segment(18,3) = Obj_Rot_.e().row(0);
    dynamics_Info_.segment(21,3) = baseRot_.e().row(0);
    dynamics_Info_.segment(24,3) << obj_geometry_;

    /// height map
    controlFrameX_ =
        {baseRot_[0], baseRot_[1], 0.}; /// body x axis projected on the world x-y plane, expressed in the world frame
    controlFrameX_ /= (controlFrameX_.norm() + 1e-10);
    raisim::cross(zAxis_, controlFrameX_, controlFrameY_);


    /// Check if the distance between command pos and robot base are under offset.

    Eigen::Vector3d current_base_pos = raibo_->getBasePosition().e();
    current_base_pos[2] = 0;
    desired_dist_ = (current_base_pos - desired_pos_).norm();
    if(desired_dist_ < 0.1)
      is_achieved = true;

  }

  Eigen::VectorXd get_desired_pos () {
    return desired_pos_;
  }

  void getObservation(Eigen::VectorXd &observation) {
    observation = obDouble_;
  }

  Eigen::VectorXf advance(raisim::World *world, const Eigen::Ref<EigenVec> &action) {
    Eigen::VectorXf position;
//    RSINFO(action)
    position = action.cast<float>().cwiseQuotient(actionStd_.cast<float>());

    Eigen::VectorXd current_pos_ = raibo_->getBasePosition().e();
    position += actionMean_.cast<float>();
    prepre_command_ = pre_command_;
    pre_command_ = command_;
    if(is_position_goal)
      command_ = {position(0), position(1), 0};
    else
      command_ = {position(0), position(1), position(2)};
    desired_pos_ = baseRot_.e() * command_.cast<double>();
    desired_pos_ += current_pos_;
    desired_pos_(2) = 0;

    is_achieved = false;

//    desired_dist_ = (raibo_->getBasePosition().e() - current_pos_).norm();
    return command_;
  }

  bool is_achieve() {
    return is_achieved;
  }

  bool update_actionHistory(raisim::World *world, const Eigen::Ref<EigenVec> &action, double curriculumFactor) {
    /// action scaling
    std::rotate(actionInfoHistory_.begin(), actionInfoHistory_.begin()+1, actionInfoHistory_.end());
    actionInfoHistory_[actionNum_ - 1] = action.cast<double>();

    std::rotate(dynamicsInfoHistory_.begin(), dynamicsInfoHistory_.begin()+1, dynamicsInfoHistory_.end());
    dynamicsInfoHistory_[actionNum_ - 1] = dynamics_Info_;

    return true;
  }

  Eigen::VectorXd get_com_pos() {
    return Obj_Pos_.e();
  }

  Eigen::VectorXd get_noisify_com_pos() {
    return Obj_Pos_.e() + Obj_->getCom().e();
  }

  void reset_Rollout(Eigen::Vector3d command_obj_pos_, Eigen::Vector4d command_obj_quat_, Eigen::Vector3d obj_geometry, double friction) {
    raibo_->getState(gc_, gv_);
    is_success_ = false;
    is_achieved = true;
    command_Obj_Pos_ = command_obj_pos_;
    command_Obj_quat_ = command_obj_quat_;
    obj_geometry_ = obj_geometry;
    friction_ = friction;

    std::fill(success_batch_.begin(), success_batch_.end(), false);
  }

  void reset(std::mt19937 &gen_,
             std::normal_distribution<double> &normDist_, Eigen::Vector3d command_obj_pos_, Eigen::Vector4d command_obj_quat_, Eigen::Vector3d obj_geometry, double friction) {
    raibo_->getState(gc_, gv_);
//    jointTarget_ = gc_.segment(7, nJoints_);
    command_Obj_Pos_ = command_obj_pos_;
    command_Obj_quat_ = command_obj_quat_;
    obj_geometry_ = obj_geometry;

    is_success_ = false;
    is_achieved = true;
    friction_ = friction;
    pre_command_.setZero();
    prepre_command_.setZero();
    // history
    for (int i = 0; i < historyNum_; i++)
    {
      for (int j=0; j < exteroceptiveDim_; j++)
        objectInfoHistory_[i](j) = normDist_(gen_) * 0.1;

      for (int j=0; j < proprioceptiveDim_; j++)
        stateInfoHistory_[i](j) = normDist_(gen_) * 0.1;
    }

    for (int i = 0; i < actionNum_; i++) {
      for (int j=0; j < actionDim_; j++)
        actionInfoHistory_[i](j) = normDist_(gen_) * 0.1;
    }


    for (int i = 0; i<actionNum_; i++)
    {
      dynamicsInfoHistory_[i].setZero();
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

    towardObjectReward_ = 0.;
    stayObjectReward_ = 0.;
    towardTargetReward_ = 0.;
    stayTargetReward_ = 0.;
    commandsmoothReward_ = 0.;
    commandsmooth2Reward_ = 0.;
    torqueReward_ = 0.;
    stayObjectHeadingReward_ = 0.;
    stayTargetHeadingReward_ = 0.;

    return float(stepData_.sum());
  }

  [[nodiscard]] bool isTerminalState(float &terminalReward) {
    terminalReward = float(terminalRewardCoeff_);

    if(std::find(success_batch_.begin(), success_batch_.end(), false) == success_batch_.end())
      return true;

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
      obDouble_.segment((obBlockDim_)*i + proprioceptiveDim_,
                        exteroceptiveDim_) = objectInfoHistory_[i];
      obDouble_.segment((obBlockDim_)*i + proprioceptiveDim_ + exteroceptiveDim_,
                        actionDim_) = actionInfoHistory_[i];
      obDouble_.segment((obBlockDim_)*i + proprioceptiveDim_ + exteroceptiveDim_ + actionDim_,
                        dynamicsInfoDim_) = dynamicsInfoHistory_[i];
    }

    obDouble_.segment((obBlockDim_)*historyNum_, proprioceptiveDim_)
    = state_Info_;

    obDouble_.segment((obBlockDim_)*historyNum_ + proprioceptiveDim_, exteroceptiveDim_)
    = Obj_Info_;

    obDouble_.segment((obBlockDim_)*historyNum_ + proprioceptiveDim_+exteroceptiveDim_, actionDim_)
    = actionInfoHistory_.back();

    obDouble_.segment((obBlockDim_)*historyNum_ + proprioceptiveDim_+exteroceptiveDim_+actionDim_, dynamicsInfoDim_)
    = dynamicsInfoHistory_.back();
  }

  inline void checkConfig(const Yaml::Node &cfg) {
    READ_YAML(int, proprioceptiveDim_, cfg["dimension"]["proprioceptiveDim_"])
    READ_YAML(int, exteroceptiveDim_, cfg["dimension"]["exteroceptiveDim_"])
    READ_YAML(int, historyNum_, cfg["dimension"]["historyNum_"])
    READ_YAML(int, actionNum_, cfg["dimension"]["actionhistoryNum_"])
    READ_YAML(int, dynamicsInfoDim_, cfg["dimension"]["dynamicsInfoDim_"])
  }

  inline void setRewardConfig(const Yaml::Node &cfg) {
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
    Eigen::Vector3d ee_to_target = (command_Obj_Pos_ - ee_Pos_w_.e());

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


    /// Reward for alignment
    raisim::Mat<3,3> command_Obj_Rot_;
    raisim::quatToRotMat(command_Obj_quat_, command_Obj_Rot_);
    double stay_t_heading  = abs(Obj_Rot_.e().row(0).head(2).dot(command_Obj_Rot_.e().row(0).head(2))/(Obj_Rot_.e().row(0).head(2).norm()*command_Obj_Rot_.e().row(0).head(2).norm() + 1e-8));
    if(stay_t_heading > 0.985){
      stayTargetHeadingReward_ += cf * stayTargetHeadingRewardCoeff_ * simDt_ * exp(1) * exp(- stayTargetHeadingRewardCoeff_alpha_ * obj_to_target.norm());
    }else {
      stayTargetHeadingReward_ += cf * stayTargetHeadingRewardCoeff_ * simDt_ * exp(stay_t_heading)
          * exp(-stayTargetHeadingRewardCoeff_alpha_ * obj_to_target.norm());
    }

    /// stay close to the object
    double stay_o = ee_to_obj.norm(); /// max : inf, min : 0
    double stay_o_heading = Obj_Vel_.e().dot(heading) / (heading.norm() * Obj_Vel_.e().norm() + 1e-8) - 1; /// max : 0, min : -1
    stayObjectReward_ += cf * stayObjectRewardCoeff_ * simDt_ * exp(-stay_o);
    stayObjectHeadingReward_ += cf * stayObjectHeadingRewardCoeff_ * simDt_ * exp(stay_o_heading);

    /// move the object towards the target
    double toward_t = (obj_to_target * (1. / (obj_to_target.norm() + 1e-8))).transpose()*(Obj_Vel_.e() * (1./ (Obj_Vel_.e().norm() + 1e-8))) - 1;
    towardTargetReward_ += cf * towardTargetRewardCoeff_ * simDt_ * exp(-std::pow(std::min(0.0, toward_t), 2));

    /// keep the object close to the target
    double stay_t = obj_to_target.norm();
    if(stay_t < 0.05) {
      stayTargetReward_ += cf * stayTargetRewardCoeff_ * simDt_ * exp(0);
    } else{
      stayTargetReward_ += cf * stayTargetRewardCoeff_ * simDt_ * exp(-stay_t);
    }
    double commandReward_tmp = std::max(5., static_cast<double>(command_.norm()));
    double command_smooth = (command_ - pre_command_).squaredNorm();
    double command_smooth2 = (command_ - 2*pre_command_ + prepre_command_).squaredNorm();

//    double remnant_dist =
    commandsmoothReward_ += cf * commandsmoothRewardCoeff_ * simDt_ * exp(-command_smooth);
    commandsmooth2Reward_ += cf * commandsmooth2RewardCoeff_ * simDt_ * exp(-command_smooth2);

    torqueReward_ += cf * torqueRewardCoeff_ * simDt_ * raibo_->getGeneralizedForce().norm();

  }

  void set_History(std::vector<Eigen::VectorXd> &obj_info_history,
                   std::vector<Eigen::VectorXd> &state_info_history,
                   std::vector<Eigen::VectorXd> &action_info_history,
                   std::vector<Eigen::VectorXd> &dynamics_info_history) {
    objectInfoHistory_ = obj_info_history;
    stateInfoHistory_ = state_info_history;
    actionInfoHistory_ = action_info_history;
    dynamicsInfoHistory_ = dynamics_info_history;
  }

  void get_History(std::vector<Eigen::VectorXd> &obj_info_history,
                   std::vector<Eigen::VectorXd> &state_info_history,
                   std::vector<Eigen::VectorXd> &action_info_history,
                   std::vector<Eigen::VectorXd> &dynamics_info_history) {
    obj_info_history = objectInfoHistory_;
    state_info_history = stateInfoHistory_;
    action_info_history = actionInfoHistory_;
    dynamics_info_history = dynamicsInfoHistory_;
  }

  inline void setStandingMode(bool mode) { standingMode_ = mode; }

  [[nodiscard]] const Eigen::VectorXd &getJointPositionHistory() const { return jointPositionHistory_; }
  [[nodiscard]] const Eigen::VectorXd &getJointVelocityHistory() const { return jointVelocityHistory_; }

  [[nodiscard]] static constexpr int getObDim() { return obDim_; }
  [[nodiscard]] static constexpr int getActionDim() { return actionDim_; }
  [[nodiscard]] static constexpr double getSimDt() { return simDt_; }
  [[nodiscard]] static constexpr double getConDt() { return conDt_; }

  bool is_success() {
    return is_success_;
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

  int proprioceptiveDim_ = 21;
  int exteroceptiveDim_ = 36;
  int dynamicsInfoDim_ = 27;
  static constexpr int actionDim_ = 3;
  int historyNum_ = 4;
  int actionNum_ = 5;
  int obBlockDim_ = 0;

  static constexpr size_t obDim_ = 435;

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
  std::vector<Eigen::VectorXd> actionInfoHistory_;
  std::vector<Eigen::VectorXd> dynamicsInfoHistory_;
  Eigen::VectorXd historyTempMemory_2;
  std::array<bool, 4> footContactState_;
  raisim::Mat<3, 3> baseRot_;
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

  // robot observation variables
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
  bool is_contact = false;
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

  // For testing
  bool is_success_ = false;

  // reward variables
  double towardObjectRewardCoeff_ = 0., towardObjectReward_ = 0.;
  double stayObjectRewardCoeff_ = 0., stayObjectReward_ = 0.;
  double towardTargetRewardCoeff_ = 0., towardTargetReward_ = 0.;
  double stayTargetRewardCoeff_ = 0., stayTargetReward_ = 0.;
  double terminalRewardCoeff_ = 0.0;
  double commandsmoothRewardCoeff_ = 0., commandsmoothReward_ = 0.;
  double commandsmooth2RewardCoeff_ = 0., commandsmooth2Reward_ = 0.;
  double torqueRewardCoeff_ = 0., torqueReward_ = 0.;
  double stayObjectHeadingReward_ = 0., stayObjectHeadingRewardCoeff_ = 0.;
  double stayTargetHeadingReward_ = 0.,  stayTargetHeadingRewardCoeff_ = 0., stayTargetHeadingRewardCoeff_alpha_ = 0. ;
  // exported data
  Eigen::VectorXd stepData_;
  std::vector<std::string> stepDataTag_;
};

}

#endif //_RAISIM_GYM_RAIBO_CONTROLLER_HPP
