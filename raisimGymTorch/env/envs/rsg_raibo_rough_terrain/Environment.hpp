// Copyright (c) 2020 Robotics and Artificial Intelligence Lab, KAIST
//
// Any unauthorized copying, alteration, distribution, transmission,
// performance, display or use of this material is prohibited.
//
// All rights reserved.

#pragma once

// raisim include
#include "raisim/World.hpp"
#include "raisim/RaisimServer.hpp"

// raisimGymTorch include
#include "../../Yaml.hpp"
#include "../../BasicEigenTypes.hpp"
#include "RaiboController.hpp"
#include "RandomHeightMapGenerator.hpp"
#include "../../../../default_controller_demo/module/controller/raibot_position_controller_sim_2/raibot_position_controller_sim_2.hpp"
#include "../../../../default_controller_demo/module/controller/raibot_default_controller/raibot_default_controller.hpp"
#include "RandomObjectGenerator.hpp"

namespace raisim {

class ENVIRONMENT {

 public:

  explicit ENVIRONMENT(const std::string &resourceDir, const Yaml::Node &cfg, bool visualizable, int id) :
      visualizable_(visualizable) {
    setSeed(id);
    READ_YAML(bool, is_multiobject_, cfg["MultiObject"])
    READ_YAML(double, curriculumFactor_, cfg["curriculum"]["initial_factor"])
    READ_YAML(double, curriculumDecayFactor_, cfg["curriculum"]["decay_factor"])
    READ_YAML(double, low_level_control_dt_, cfg["low_level_control_dt"])
    READ_YAML(bool, is_discrete_, cfg["discrete_action"])
    READ_YAML(bool, is_position_goal, cfg["position_goal"])
    READ_YAML(double, obj_mass, cfg["obj_mass"])
    READ_YAML(double, bound_ratio, cfg["bound_ratio"])
    if(is_discrete_) {
      READ_YAML(int, radial_, cfg["discrete"]["radial"])
      READ_YAML(int, tangential_, cfg["discrete"]["tangential"])
    }


    auto map = world_.addGround(0.0, "ground");
    map->setAppearance("hidden");
    world_.setDefaultMaterial(1.1, 0.0, 0.01);
    world_.setMaterialPairProp("ground", "object", friction, 0.1, 0.0);

    /// add objects
    raibo_ = world_.addArticulatedSystem(resourceDir + "/raibot/urdf/raibot_sensors.urdf");
    raibo_->setName("robot");
    raibo_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);


//    auto imu = raibo_->getSensor<raisim::InertialMeasurementUnit>("depth_camera_front_camera_parent:imu");
//    auto rgbCamera2 = raibo_->getSensor<raisim::RGBCamera>("depth_camera_front_camera_parent:color");
//    rgbCamera2->setMeasurementSource(raisim::Sensor::MeasurementSource::VISUALIZER);
    /// Object spawn
    Obj_ = objectGenerator_.generateObject(&world_, RandomObjectGenerator::ObjectShape(object_type), curriculumFactor_, gen_, uniDist_,
                                           normDist_, bound_ratio, obj_mass, object_radius, object_height, object_width_1, object_width_2);

    //웨안

    double height = objectGenerator_.get_height();
    Obj_->setPosition(2, 2, height/2);
    Obj_->setOrientation(1, 0, 0, 0);
    Obj_->setAppearance("hidden");

    /// create controller
    controller_.create(&world_, Obj_);

    /// set curriculum
    simulation_dt_ = RaiboController::getSimDt();
    high_level_control_dt_ = RaiboController::getConDt();


    /// create heightmap
//    groundType_ = (id+3) % 4;
//    heightMap_ = terrainGenerator_.generateTerrain(&world_, RandomHeightMapGenerator::GroundType(groundType_), curriculumFactor_, gen_, uniDist_);

    /// get robot data
    gcDim_ = int(raibo_->getGeneralizedCoordinateDim());
    gvDim_ = int(raibo_->getDOF());

    /// initialize containers
    gc_init_.setZero(gcDim_);
    gv_init_.setZero(gvDim_);
    nominalJointConfig_.setZero(nJoints_);
    gc_init_from_.setZero(gcDim_);
    gv_init_from_.setZero(gvDim_);


    current_anchor_points.resize(8);
    prev_anchor_points.resize(8);
    accum_anchor_points.resize(8);
    anchors_pred.resize(8);
    anchors_prev.resize(8);
    anchors_gt.resize(8);
    /// set pd gains
    jointPGain_.setZero(gvDim_);
    jointDGain_.setZero(gvDim_);
    jointPGain_.tail(nJoints_).setConstant(60.0);
    jointDGain_.tail(nJoints_).setConstant(0.5);
    raibo_->setPdGains(jointPGain_, jointDGain_);

    /// this is nominal configuration of anymal
    nominalJointConfig_<< 0, 0.56, -1.12, 0, 0.56, -1.12, 0, 0.56, -1.12, 0, 0.56, -1.12;
    gc_init_.head(7) << 0, 0, 0.4725, 1, 0.0, 0.0, 0.0;
    gc_init_.tail(nJoints_) << nominalJointConfig_;
    gc_init_from_ = gc_init_;
    raibo_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    // Reward coefficients
    controller_.setRewardConfig(cfg);
    command_Obj_Pos_ << 2, 2, command_object_height_/2;
    command_Obj_quat_ << 1, 0, 0, 0;

    rgbCameras[0] = raibo_->getSensor<raisim::RGBCamera>("d435i_R:color");
    rgbCameras[0]->setMeasurementSource(raisim::Sensor::MeasurementSource::VISUALIZER);
    depthCameras[0] = raibo_->getSensor<raisim::DepthCamera>("d435i_R:depth");
    depthCameras[0]->setMeasurementSource(raisim::Sensor::MeasurementSource::VISUALIZER);

    // visualize if it is the first environment
    if (visualizable_) {
      server_ = std::make_unique<raisim::RaisimServer>(&world_);
      server_->launchServer(8080);
      server_->focusOn(raibo_);
      server_->setMap("office1");
      std::cout << "Launch Server !!" << std::endl;
      command_Obj_ = server_->addVisualBox("command_Obj_", 0.1,0.1,0.1,0, 0, 0, 0.0);
      command_Obj_->setPosition(10, 10, 10);
      command_Obj_->setOrientation(command_Obj_quat_);

      for(int i =0; i<8; i++){
          anchors_pred[i] = server_->addVisualSphere("anchor"+std::to_string(i), 0.05,1-i/7,i/7,0,1.0);
          anchors_gt[i] = server_->addVisualSphere("anchor_gt"+std::to_string(i), 0.05,i/7,i/7,1-i/7,1.0);
          anchors_prev[i] = server_->addVisualSphere("anchor_prev"+std::to_string(i), 0.05,i/7,1-i/7,1-i/7,0.5);
      }

//      target_pos_ = server_->addVisualSphere("target_Pos_", 0.3, 1, 0, 0, 1.0);

    }
  }



  ~ENVIRONMENT() { if (server_) server_->killServer(); }

  void adapt_Low_position_controller (controller::raibotPositionController controller) {

    Low_controller_ = controller;
    Low_controller_.init(&world_);
  }

  void adapt_Low_velocity_controller (controller::raibotDefaultController controller) {
    Low_controller_2_ = controller;
    Low_controller_2_.init(&world_);
  }


  void Low_controller_create (bool is_position_goal) {
    if (is_position_goal) {
      Low_controller_.init(&world_);
      Low_controller_.create(&world_);

      RSINFO("create_test")
    }

    else {
      Low_controller_2_.init(&world_);
      Low_controller_2_.create(&world_);
      RSINFO("create_test")
    }
  }

  controller::raibotPositionController get_Low_position_controller () {
    return Low_controller_;
  }

  controller::raibotDefaultController get_Low_velocity_controller () {
    return Low_controller_2_;
  }

  void init () { }
  void close () { }
  void setSimulationTimeStep(double dt)
  { controller_.setSimDt(dt);
    if(is_position_goal)
      Low_controller_.setSimDt(dt);
    else
      Low_controller_2_.setSimDt(dt);
    world_.setTimeStep(dt);
  }
  void setControlTimeStep(double dt, double low_dt) {
    controller_.setConDt(dt);
    if(is_position_goal)
      Low_controller_.setConDt(low_dt);
    else
      Low_controller_2_.setConDt(low_dt);}
  void turnOffVisualization() { server_->hibernate(); }
  void turnOnVisualization() { server_->wakeup(); }
  void startRecordingVideo(const std::string& videoName ) { server_->startRecordingVideo(videoName); }
  void stopRecordingVideo() { server_->stopRecordingVideo(); }
  const std::vector<std::string>& getStepDataTag() { return controller_.getStepDataTag(); }
  const Eigen::VectorXd& getStepData() { return controller_.getStepData(); }

  void hard_reset () {
    friction = 0.2 + 0.1*curriculumFactor_ * (uniDist_(gen_) - 0.5);
    world_.setMaterialPairProp("ground", "object", friction, 0.1, 0.0);

    air_damping = 0.4 + 0.2*curriculumFactor_ * (uniDist_(gen_) - 0.5);
    Obj_->setAngularDamping({air_damping, air_damping, air_damping});
    Obj_->setLinearDamping(air_damping);

    /// Update Object damping coefficient
    /// Deprecated (box doesn't necessary)
//    if(is_multiobject_)
//    {

//    }

  }

  void getCameraPose(Eigen::Ref<EigenVec> cam_position, Eigen::Ref<EigenRowMajorMat> cam_rotMat) {
    raisim::Vec<3> pos;
    raisim::Mat<3,3> rot;
    raibo_->getFramePosition(raibo_->getFrameIdxByLinkName("d435i_R"), pos);
    raibo_->getFrameOrientation(raibo_->getFrameIdxByLinkName("d435i_R"), rot);
    cam_position = pos.e().cast<float>();
    cam_rotMat = rot.e().cast<float>();
  }

  void reset() {
    if(is_multiobject_)
      object_type = intuniDist_(gen_);
    else
      object_type = 2;
    updateObstacle();
    objectGenerator_.Inertial_Randomize(Obj_, bound_ratio, curriculumFactor_, gen_, uniDist_, normDist_);
//    if(curriculumFactor_ > 0.5)
//      hard_reset();
    /// set the state
    raibo_->setState(gc_init_, gv_init_); /// set it again to ensure that foot is in contact
    controller_.reset(gen_, normDist_, command_Obj_Pos_, command_Obj_quat_, objectGenerator_.get_geometry(), friction, air_damping);
    Eigen::VectorXd temp = objectGenerator_.get_classify_vector();
    controller_.updateClassifyvector(temp);
    prev_pos_={gc_init_[0],gc_init_[1],0};
    raisim::Vec<4> quat;
    raisim::Mat<3, 3> baseRot;
    quat[0] = gc_init_[3];
    quat[1] = gc_init_[4];
    quat[2] = gc_init_[5];
    quat[3] = gc_init_[6];
    raisim::quatToRotMat(quat, baseRot);
    Eigen::Vector3d base_x_axis = baseRot.e().row(0);
    base_x_axis(2) = 0;
    Eigen::Vector3d base_x_axis_norm = base_x_axis.normalized();
    raisim::angleAxisToRotMat({0,0,1}, std::atan2(-base_x_axis(1), base_x_axis(0)), prev_yaw_rot);

    if(is_position_goal) {
      Low_controller_.reset(&world_);
      controller_.updateStateVariables();
      Low_controller_.updateStateVariable();
    }

    else {
      Low_controller_2_.reset(&world_);
      controller_.updateStateVariables();
    }

    command(0) = uniDist_(gen_) * 1.0;
    command(1) = uniDist_(gen_) * 1.0;
    command(2) = uniDist_(gen_)-0.5;


  }



  double step(const Eigen::Ref<EigenVec>& action, bool visualize) {
    /// action scaling



    command(0) = std::clamp(command(0) + normDist_(gen_) * 0.2, -1.0, 1.0);
    command(1) = std::clamp(command(1) + normDist_(gen_) * 0.2, -1.0, 1.0);
    command(2) = std::clamp(command(2) + normDist_(gen_) * 0.2, -0.5, 0.5);
    command = controller_.advance(&world_, command);

    controller_.update_actionHistory(&world_, command, curriculumFactor_);

    if(is_position_goal)
      Low_controller_.setCommand(command);
    else
      Low_controller_2_.setCommand(command);

//    if (controller_.is_achieved)
//    {
//      command = controller_.advance(&world_, action);
//      Low_controller_.setCommand(command);
//    }


    float dummy;
    int howManySteps;
    int lowlevelSteps;



    /// Low level frequency 0.01
    for (lowlevelSteps = 0; lowlevelSteps < int(high_level_control_dt_ / low_level_control_dt_ + 1e-10); lowlevelSteps++) {
      controller_.updateStateVariables();
      controller_.updateHistory();
      if(is_position_goal) {
        Low_controller_.updateObservation(&world_);
        Low_controller_.advance(&world_);
      }
      else
        Low_controller_2_.advance(&world_);


      /// Simulation frequency
      for(howManySteps = 0; howManySteps< int(low_level_control_dt_ / simulation_dt_ + 1e-10); howManySteps++) {

        subStep();
//        if(visualize)
//          std::this_thread::sleep_for(std::chrono::microseconds(1000));
//
        if(isTerminalState(dummy)) {
          howManySteps++;
          break;
       }
      }
    }

    return controller_.getRewardSum(visualize);
  }


    double step_evaluate(const Eigen::Ref<EigenVec>& action, bool visualize, const Eigen::Ref<EigenVec>& anchors) {
        /// action scaling
        Eigen::Vector3d geometry{0.2,0.2,0.2};
        controller_.get_anchor_points(prev_anchor_points, prev_pos_, prev_yaw_rot.e(), geometry);
        controller_.estimate_anchor_points(current_anchor_points, prev_anchor_points, anchors,prev_yaw_rot.e());
        prev_prev_yaw_rot =prev_yaw_rot;

        for(int i=0; i<8;i++){
            anchors_prev[i]->setPosition(prev_anchor_points[i]);
        }
        for(int i=0; i<8;i++){
            anchors_pred[i]->setPosition(current_anchor_points[i]);
        }


        Eigen::VectorXd gc;
        gc.resize(raibo_->getGeneralizedCoordinateDim());
        gc = raibo_->getGeneralizedCoordinate().e();

        raisim::Vec<4> quat;
        raisim::Mat<3, 3> baseRot;
        quat[0] = gc[3];
        quat[1] = gc[4];
        quat[2] = gc[5];
        quat[3] = gc[6];
        raisim::quatToRotMat(quat, baseRot);
        Eigen::Vector3d base_x_axis = baseRot.e().row(0);
        base_x_axis(2) = 0;
        Eigen::Vector3d base_x_axis_norm = base_x_axis.normalized();
        raisim::Mat<3,3> yaw_rot;
        raisim::angleAxisToRotMat({0,0,1}, std::atan2(-base_x_axis(1), base_x_axis(0)), yaw_rot);

        controller_.get_anchor_points(prev_anchor_points, gc.head(3), yaw_rot.e(), geometry);
        for(int i=0; i<8;i++){
            anchors_gt[i]->setPosition(prev_anchor_points[i]);
        }

        prev_pos_=gc.head(3);
        prev_yaw_rot = yaw_rot;

        command(0) = std::clamp(command(0) + normDist_(gen_) * 0.2, -1.0, 1.0);
        command(1) = std::clamp(command(1) + normDist_(gen_) * 0.2, -1.0, 1.0);
        command(2) = std::clamp(command(2) + normDist_(gen_) * 0.2, -0.5, 0.5);
        command = controller_.advance(&world_, command);

        controller_.update_actionHistory(&world_, command, curriculumFactor_);

        if(is_position_goal)
            Low_controller_.setCommand(command);
        else
            Low_controller_2_.setCommand(command);

//    if (controller_.is_achieved)
//    {
//      command = controller_.advance(&world_, action);
//      Low_controller_.setCommand(command);
//    }

        float dummy;
        int howManySteps;
        int lowlevelSteps;

        /// Low level frequency 0.01
        for (lowlevelSteps = 0; lowlevelSteps < int(high_level_control_dt_ / low_level_control_dt_ + 1e-10); lowlevelSteps++) {
            controller_.updateStateVariables();
            controller_.updateHistory();
            if(is_position_goal) {
                Low_controller_.updateObservation(&world_);
                Low_controller_.advance(&world_);
            }
            else
                Low_controller_2_.advance(&world_);


            /// Simulation frequency
            for(howManySteps = 0; howManySteps< int(low_level_control_dt_ / simulation_dt_ + 1e-10); howManySteps++) {

                subStep();
                std::this_thread::sleep_for(std::chrono::microseconds(1000));
                if(isTerminalState(dummy)) {
                    howManySteps++;
                    break;
                }
            }
        }

        return controller_.getRewardSum(visualize);
    }

    double get_estimation_error(bool get, const Eigen::Ref<EigenVec>& anchors){


      Eigen::Vector3d delta;
      std::vector<Eigen::Vector3d> anchor_points;
      anchor_points.resize(8);
      delta.setZero();
      double error =0;
      if(get=false){

          for(int i=0; i<8; i++){
             error += (current_anchor_points[i]-prev_anchor_points[i]).squaredNorm();
          }
          accum_anchor_points = current_anchor_points;
          return error;
      }else{
          controller_.estimate_anchor_points(anchor_points, accum_anchor_points, anchors,prev_prev_yaw_rot.e());
          for(int i=0; i<8; i++){
              error += (anchor_points[i]-prev_anchor_points[i]).squaredNorm();
          }
          accum_anchor_points = anchor_points;
          return error;
      }


    }

  void updateObstacle(bool curriculum_Update = false) {

    world_.removeObject(Obj_);
    Obj_ = objectGenerator_.generateObject(&world_, RandomObjectGenerator::ObjectShape(object_type), curriculumFactor_, gen_, uniDist_,
                                           normDist_, bound_ratio, obj_mass, object_radius, object_height, object_width_1, object_width_2);
    Obj_->setAppearance("hidden");

    controller_.updateObject(Obj_);
    double height = objectGenerator_.get_height();

    double object_bound = objectGenerator_.get_dist();

    double x, y, x_command, y_command, offset, dist;
    double phi_;
    offset = 0.5;
//    dist = uniDist_(gen_) * curriculumFactor_;
    dist = uniDist_(gen_);
    phi_ = uniDist_(gen_);
    x = (object_bound + offset + dist * 3)*cos(phi_*2*M_PI);
    y = (object_bound + offset + dist * 3)*sin(phi_*2*M_PI);

    while (sqrt(std::pow(x,2) + std::pow(y,2)) < (object_bound + offset))
    {
//      dist = uniDist_(gen_) * curriculumFactor_;
      dist = uniDist_(gen_);
      phi_ = uniDist_(gen_);
      x = (object_bound + offset + dist * 3)*cos(phi_*2*M_PI);
      y = (object_bound + offset + dist * 3)*sin(phi_*2*M_PI);
    }

    x += gc_init_[0];
    y += gc_init_[1];

    phi_ = uniDist_(gen_) * M_PI * 2;;
    Obj_->setPosition(x, y, height/2+1e-2);
    Obj_->setOrientation(cos(phi_/2), 0, 0, sin(phi_/2));
    Obj_->setVelocity(0,0,0,0,0,0);

    /// Update Object damping coefficient
    /// Deprecated (box doesn't necessary)
//    Obj_->setAngularDamping({1.0,1.0,1.0});
//    Obj_->setLinearDamping(0.5);
    phi_ = uniDist_(gen_);
    dist = uniDist_(gen_) * 2;
//    dist = uniDist_(gen_) * curriculumFactor_ * 2;
    x_command = x + (0.3 + dist)*cos(phi_*2*M_PI);
    y_command = y + (0.3 + dist)*sin(phi_*2*M_PI);



    command_Obj_Pos_ << x_command, y_command, height/2;

    double alpha = uniDist_(gen_) * M_PI * 2;

    command_Obj_quat_ << cos(alpha / 2), 0, 0, sin(alpha/2);


    if(visualizable_) {
      server_->removeVisualObject("command_Obj_");
      command_Obj_ = server_->addVisualBox("command_Obj_", 0.1, 0.1, 0.1, 0, 0, 0, 0.0);
      command_Obj_->setPosition(10 , 10, 10);
      command_Obj_->setOrientation(command_Obj_quat_);

      ///TODO
      /// add raisim server bar chart
      ///

    }

  }


  std::vector<std::vector<float>> getDepthImage(){
    std::vector<std::vector<float>> image;
    for (raisim::DepthCamera* depthCamera : depthCameras){
      image.push_back(depthCamera->getDepthArray());
    }
    return image;
  }
  std::vector<std::vector<int>> getColorImage(){
    std::vector<std::vector<int>> image;
    for (raisim::RGBCamera* rgbCamera : rgbCameras){
      std::vector<char> charVec = rgbCamera->getImageBuffer();
      image.push_back(std::vector<int>{charVec.begin(), charVec.end()});
    }
    return image;
  }

  void subStep() {
    bool success;
    controller_.is_success(success);
//    if(success) {
//      Obj_->setAppearance("0, 0, 1, 0.7");
//    }
//
//    else{
//        Obj_->setAppearance("0, 1, 0, 0.3");
//    }

    if(is_position_goal)
      Low_controller_.updateHistory();

    if(visualizable_)
    {
      Eigen::Vector3d arrow_pos = raibo_->getBasePosition().e();
      arrow_pos(2) += 0.1;
      double arrow_angle = atan2(controller_.get_desired_pos()(1), controller_.get_desired_pos()(0));

      double robot_angle = atan2((raibo_->getBaseOrientation().e()(0) - raibo_->getBaseOrientation().e()(1))/2, (raibo_->getBaseOrientation().e()(1) + raibo_->getBaseOrientation().e()(0))/2);
      raisim::Mat<3,3> arrow_rotation_matrix;
      arrow_rotation_matrix = {cos(arrow_angle), sin(arrow_angle), 0,
                               -sin(arrow_angle), cos(arrow_angle), 0,
                               0, 0, 1};

      raisim::Mat<3,3> initial_matrix;
      initial_matrix = {cos(M_PI/2), 0, -sin(M_PI/2),
                        0, 1, 0,
                        sin(M_PI/2), 0, cos(M_PI/2)};

      raisim::Mat<3,3> robot_rotation_matrix = {
          cos(robot_angle), sin(robot_angle), 0,
          -sin(robot_angle), cos(robot_angle), 0,
          0, 0, 1
      };

//      double robot_angle =
      raisim::Mat<3,3> temp;
      temp = raibo_->getBaseOrientation() * arrow_rotation_matrix * initial_matrix;
      raisim::Vec<4> arrow_quat;
      raisim::rotMatToQuat(temp, arrow_quat);


    }

    world_.integrate1();
    if(server_) server_->lockVisualizationServerMutex();
    world_.integrate2();
    if(server_) server_->unlockVisualizationServerMutex();
    if(is_position_goal)
      Low_controller_.updateStateVariable();
    controller_.updateStateVariables();
    controller_.accumulateRewards(curriculumFactor_, command_);

  }

  void observe(Eigen::Ref<EigenVec> ob) {
    controller_.updateObservation(true, command_, heightMap_, gen_, normDist_);
    controller_.getObservation(obScaled_);
    ob = obScaled_.cast<float>();
  }

  bool get_contact() {
    return controller_.getContact();
  }

  bool isTerminalState(float& terminalReward) {
//    return controller_.isTerminalState(terminalReward);
    return false;
  }

  void setSeed(int seed) {
    gen_.seed(seed);
    terrainGenerator_.setSeed(seed);
    objectGenerator_.setSeed(seed);
  }

  void curriculumUpdate() {
//    object_type = (object_type+1) % 3; /// rotate ground type for a visualization purpose
    curriculumFactor_ = std::pow(curriculumFactor_, curriculumDecayFactor_);
    /// create heightmap
//    updateObstacle(true);
//    Eigen::VectorXd temp = objectGenerator_.get_classify_vector();
//    controller_.updateClassifyvector(temp);
  }

  void check_success(bool& success) {
    controller_.is_success(success);
  }

  void check_intrinsic_switch(bool& switch_) {
    controller_.get_intrinsic_switch(switch_);
  }

  void get_env_value(Eigen::Ref<EigenVec> value){
    controller_.get_privileged_information(value);
  }

  void moveControllerCursor(Eigen::Ref<EigenVec> pos) {
    controllerSphere_->setPosition(pos[0], pos[1], heightMap_->getHeight(pos[0], pos[1]));
  }

  void setCommand() {
    command_ = controllerSphere_->getPosition();
    commandSphere_->setPosition(command_);
  }

//  void get_camera_pose(Eigen::Ref<EigenVec> pose,){

//  }
  void get_anchor_history(Eigen::Ref<EigenVec> anchors, bool is_robotFrame) {
    controller_.get_anchor_history(anchors, is_robotFrame);
  }

  static constexpr int getObDim() { return RaiboController::getObDim(); }
  static constexpr int getActionDim() { return RaiboController::getActionDim(); }

  void getState(Eigen::Ref<EigenVec> gc, Eigen::Ref<EigenVec> gv) {
    controller_.getState(gc, gv);
  }

 protected:
  bool is_position_goal = true;
  double obj_mass = 10.0;
  double friction = 0.2;
  double air_damping = 0.5;
  static constexpr int nJoints_ = 12;
  raisim::World world_;
  double simulation_dt_;
  double high_level_control_dt_;
  double low_level_control_dt_;
  int gcDim_, gvDim_;
  std::array<size_t, 4> footFrameIndicies_;
  double bound_ratio = 0.5;
  double object_width_1 = 1.0;
  double object_width_2 = 1.0;
  raisim::ArticulatedSystem* raibo_;
  raisim::HeightMap* heightMap_;
  Eigen::VectorXd gc_init_, gv_init_, nominalJointConfig_;
  Eigen::VectorXd gc_init_from_, gv_init_from_;
  double curriculumFactor_, curriculumDecayFactor_;
  Eigen::VectorXd obScaled_;
  Eigen::Vector3d command_;
  bool visualizable_ = false;
  bool is_discrete_ = false;
  int object_type = 2;
  RandomHeightMapGenerator terrainGenerator_;
  RaiboController controller_;
  controller::raibotPositionController Low_controller_;
  controller::raibotDefaultController Low_controller_2_;
  Eigen::VectorXd jointDGain_, jointPGain_;
  RandomObjectGenerator objectGenerator_;
  raisim::RGBCamera* rgbCameras[1];
  raisim::DepthCamera* depthCameras[1];
  Eigen::Vector3f command;
  std::unique_ptr<raisim::RaisimServer> server_;
  raisim::Visuals *commandSphere_, *controllerSphere_;
  raisim::SingleBodyObject *Obj_, *Manipulate_;
  raisim::Visuals *command_Obj_, *cur_head_Obj_, *tar_head_Obj_, *command_ball_, *com_pos_, *com_noisify_;

  Eigen::Vector3d command_Obj_Pos_;
  Eigen::Vector4d command_Obj_quat_;
  Eigen::Vector3d Dist_eo_, Dist_og_;
  Eigen::Vector3d prev_pos_;
  raisim::Mat<3,3> prev_yaw_rot;
  raisim::Mat<3,3> prev_prev_yaw_rot;
  std::vector<Eigen::Vector3d> accum_anchor_points;
  std::vector<Eigen::Vector3d> prev_anchor_points;
  std::vector<Eigen::Vector3d> current_anchor_points;
  std::vector<raisim::Visuals *> anchors_pred;
  std::vector<raisim::Visuals *> anchors_prev;
  std::vector<raisim::Visuals *> anchors_gt;
  raisim::Vec<3> Pos_e_;
  int command_order = 0;
  double object_height = 0.6;
  double command_object_height_ = 0.55;
  double object_radius = 0.4;
  int radial_ = 0;
  int tangential_ = 0;
  bool is_multiobject_ = true;


  thread_local static std::mt19937 gen_;
  thread_local static std::normal_distribution<double> normDist_;
  thread_local static std::uniform_real_distribution<double> uniDist_;
  thread_local static std::uniform_int_distribution<int> intuniDist_;
};

thread_local std::mt19937 raisim::ENVIRONMENT::gen_;
thread_local std::normal_distribution<double> raisim::ENVIRONMENT::normDist_(0., 1.);
thread_local std::uniform_real_distribution<double> raisim::ENVIRONMENT::uniDist_(0., 1.);
thread_local std::uniform_int_distribution<int> raisim::ENVIRONMENT::intuniDist_(1, 2);
}
