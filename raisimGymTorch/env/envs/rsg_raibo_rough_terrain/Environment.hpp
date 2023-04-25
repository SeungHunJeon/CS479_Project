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


    world_.addGround(0.0, "ground");
    world_.setDefaultMaterial(1.1, 0.0, 0.01);
    world_.setMaterialPairProp("ground", "object", friction, 0.1, 0.0);

    /// add objects
    raibo_ = world_.addArticulatedSystem(resourceDir + "/raibot/urdf/raibot_simplified.urdf");
    raibo_->setName("robot");
    raibo_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
    /// Object spawn
    Obj_ = objectGenerator_.generateObject(&world_, RandomObjectGenerator::ObjectShape(object_type), curriculumFactor_, gen_, uniDist_,
                                           normDist_, bound_ratio, obj_mass, 0.5, 0.55, 1.0, 1.0);

    object_height = objectGenerator_.get_height();
    Obj_->setPosition(2, 2, object_height/2);
    Obj_->setOrientation(1, 0, 0, 0);
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

    // visualize if it is the first environment
    if (visualizable_) {
      server_ = std::make_unique<raisim::RaisimServer>(&world_);
      server_->launchServer(8080);
      server_->focusOn(raibo_);
      std::cout << "Launch Server !!" << std::endl;
      command_Obj_ = server_->addVisualBox("command_Obj_", 0.5, 0.5, command_object_height_, 1, 0, 0, 0.5);
      command_Obj_->setPosition(command_Obj_Pos_[0], command_Obj_Pos_[1], command_Obj_Pos_[2]);
      command_Obj_->setOrientation(command_Obj_quat_);
//      target_pos_ = server_->addVisualSphere("target_Pos_", 0.3, 1, 0, 0, 1.0);
      command_ball_ = server_->addVisualArrow("command_Arrow_", 0.2, 0.2, 1.0, 0.0, 0.0, 1.0);
      com_pos_ = server_->addVisualSphere("com_pos", 0.05, 0, 0.5, 0.5, 1.0);
      com_noisify_ = server_->addVisualSphere("com_nosify_pos", 0.05, 1.0, 0.5, 0.5, 1.0);
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
    friction = 0.6 + 0.2*curriculumFactor_ * 2 * (uniDist_(gen_) - 0.5);
    world_.setMaterialPairProp("ground", "object", friction, 0.1, 0.0);

    /// Update Object damping coefficient
    /// Deprecated (box doesn't necessary)
    air_damping = 0.3 + 0.5*curriculumFactor_ * uniDist_(gen_);
    Obj_->setAngularDamping({air_damping, air_damping, air_damping});
    Obj_->setLinearDamping(air_damping);
  }

  void reset() {
    if(is_multiobject_)
      object_type = intuniDist_(gen_);
    else
      object_type = 2;
    updateObstacle();
    objectGenerator_.Inertial_Randomize(Obj_, bound_ratio, curriculumFactor_, gen_, uniDist_, normDist_);
    if(curriculumFactor_ > 0.5)
      hard_reset();
    /// set the state
    raibo_->setState(gc_init_, gv_init_); /// set it again to ensure that foot is in contact
    controller_.reset(gen_, normDist_, command_Obj_Pos_, command_Obj_quat_, objectGenerator_.get_geometry(), friction, air_damping);
    Eigen::VectorXd temp = objectGenerator_.get_classify_vector();
    controller_.updateClassifyvector(temp);
    if(is_position_goal) {
      Low_controller_.reset(&world_);
      controller_.updateStateVariables();
      Low_controller_.updateStateVariable();
    }

    else {
      Low_controller_2_.reset(&world_);
      controller_.updateStateVariables();
    }


  }



  double step(const Eigen::Ref<EigenVec>& action, bool visualize) {
    /// action scaling
    Eigen::Vector3f command;

    command = controller_.advance(&world_, action);
    controller_.update_actionHistory(&world_, action, curriculumFactor_);

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



  void updateObstacle(bool curriculum_Update = false) {

    world_.removeObject(Obj_);
    Obj_ = objectGenerator_.generateObject(&world_, RandomObjectGenerator::ObjectShape(object_type), curriculumFactor_, gen_, uniDist_,
                                           normDist_, bound_ratio, obj_mass, 0.5, 0.55, 1.0, 1.0);
    Obj_->setAppearance("0, 1, 0, 0.3");
    controller_.updateObject(Obj_);
    object_height = objectGenerator_.get_height();

    object_radius = objectGenerator_.get_dist();

    double x, y, x_command, y_command, offset;
    double phi_;
    offset = 1.0;
    phi_ = uniDist_(gen_);
    x = (object_radius + offset*2)*cos(phi_*2*M_PI) + normDist_(gen_)*0.5*curriculumFactor_;
    y = (object_radius + offset*2)*sin(phi_*2*M_PI) + normDist_(gen_)*0.5*curriculumFactor_;

    while (sqrt(std::pow(x,2) + std::pow(y,2)) < (object_radius + offset))
    {
      x = (object_radius + offset*2)*cos(phi_*2*M_PI) + normDist_(gen_)*0.5*curriculumFactor_;
      y = (object_radius + offset*2)*sin(phi_*2*M_PI) + normDist_(gen_)*0.5*curriculumFactor_;
    }

    x += gc_init_[0];
    y += gc_init_[1];

    phi_ = uniDist_(gen_) * M_PI * 2;;
    Obj_->setPosition(x, y, object_height/2+1e-2);
    Obj_->setOrientation(cos(phi_/2), 0, 0, sin(phi_/2));
    Obj_->setVelocity(0,0,0,0,0,0);

    /// Update Object damping coefficient
    /// Deprecated (box doesn't necessary)
//    Obj_->setAngularDamping({1.0,1.0,1.0});
//    Obj_->setLinearDamping(0.5);

    phi_ = uniDist_(gen_);

    x_command = x + sqrt(2)*cos(phi_*2*M_PI) + normDist_(gen_)*1*curriculumFactor_;
    y_command = y + sqrt(2)*sin(phi_*2*M_PI) + normDist_(gen_)*1*curriculumFactor_;



    command_Obj_Pos_ << x_command, y_command, command_object_height_/2;

    double alpha = uniDist_(gen_) * M_PI * 2;

    command_Obj_quat_ << cos(alpha / 2), 0, 0, sin(alpha/2);


    if(visualizable_) {
      server_->removeVisualObject("command_Obj_");
      command_Obj_ = server_->addVisualBox("command_Obj_", objectGenerator_.get_geometry()[0], objectGenerator_.get_geometry()[1], objectGenerator_.get_geometry()[2], 1, 0, 0, 0.5);
      command_Obj_->setPosition(command_Obj_Pos_[0], command_Obj_Pos_[1], command_Obj_Pos_[2]);
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
    if(controller_.is_success()) {
      Obj_->setAppearance("0, 0, 1, 0.7");
    }

    else{
        Obj_->setAppearance("0, 1, 0, 0.3");
    }

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
      command_ball_->setPosition(arrow_pos);
      command_ball_->setOrientation(arrow_quat.e());

      com_pos_->setPosition(controller_.get_com_pos());
      com_noisify_->setPosition(controller_.get_noisify_com_pos());
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
    return controller_.isTerminalState(terminalReward);
//    return false;
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

  bool check_success() {
    return controller_.is_success();
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


  static constexpr int getObDim() { return RaiboController::getObDim(); }
  static constexpr int getActionDim() { return RaiboController::getActionDim(); }

  void getState(Eigen::Ref<EigenVec> gc, Eigen::Ref<EigenVec> gv) {
    controller_.getState(gc, gv);
  }

 protected:
  bool is_position_goal = true;
  double obj_mass = 6.0;
  double friction = 0.6;
  double air_damping = 0.5;
  static constexpr int nJoints_ = 12;
  raisim::World world_;
  double simulation_dt_;
  double high_level_control_dt_;
  double low_level_control_dt_;
  int gcDim_, gvDim_;
  std::array<size_t, 4> footFrameIndicies_;
  double bound_ratio = 0.5;
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

  std::unique_ptr<raisim::RaisimServer> server_;
  raisim::Visuals *commandSphere_, *controllerSphere_;
  raisim::SingleBodyObject *Obj_, *Manipulate_;
  raisim::Visuals *command_Obj_, *cur_head_Obj_, *tar_head_Obj_, *command_ball_, *com_pos_, *com_noisify_;
  Eigen::Vector3d command_Obj_Pos_;
  Eigen::Vector4d command_Obj_quat_;
  Eigen::Vector3d Dist_eo_, Dist_og_;
  raisim::Vec<3> Pos_e_;
  int command_order = 0;
  double object_height = 0.55;
  double command_object_height_ = 0.55;
  double object_radius;
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
thread_local std::uniform_int_distribution<int> raisim::ENVIRONMENT::intuniDist_(0, 2);
}