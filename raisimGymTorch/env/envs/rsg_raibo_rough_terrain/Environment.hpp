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

namespace raisim {

class ENVIRONMENT {

 public:

  explicit ENVIRONMENT(const std::string &resourceDir, const Yaml::Node &cfg, bool visualizable, int id) :
      visualizable_(visualizable) {
    setSeed(id);
    world_.addGround();
    world_.setDefaultMaterial(1.1, 0.0, 0.01);
    /// add objects
    raibo_ = world_.addArticulatedSystem(resourceDir + "/raibot/urdf/raibot_simplified.urdf");
    raibo_->setName("robot");
    raibo_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

    /// Object spawn
    Obj_ = world_.addCylinder(0.5, object_height, 4.0);
    Obj_->setName("Obj_");
    Obj_->setPosition(1, 1, object_height/2);
    Obj_->setOrientation(1, 0, 0, 0);

    /// create controller
    controller_.create(&world_, Obj_);
//    Low_controller_.create(&world_);

    /// set curriculum
    simulation_dt_ = RaiboController::getSimDt();
    high_level_control_dt_ = RaiboController::getConDt();

    READ_YAML(double, curriculumFactor_, cfg["curriculum"]["initial_factor"])
    READ_YAML(double, curriculumDecayFactor_, cfg["curriculum"]["decay_factor"])
    READ_YAML(double, low_level_control_dt_, cfg["low_level_control_dt"])
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
    command_Obj_Pos_ << 2, 2, object_height/2;

    command_set.push_back({1.5,0});
    command_set.push_back({-1.5, 0});
    command_set.push_back({0, 1.5});
    command_set.push_back({0, -1.5});


    // visualize if it is the first environment
    if (visualizable_) {
      server_ = std::make_unique<raisim::RaisimServer>(&world_);
      server_->launchServer(8080);
      server_->focusOn(raibo_);
      std::cout << "Launch Server !!" << std::endl;
      command_Obj_ = server_->addVisualCylinder("command_Obj_", 0.5, object_height, 1, 0, 0, 0.5);
      command_Obj_->setPosition(command_Obj_Pos_[0], command_Obj_Pos_[1], command_Obj_Pos_[2]);
      target_pos_ = server_->addVisualSphere("target_Pos_", 0.3, 1, 0, 0, 1.0);
    }
  }

  ~ENVIRONMENT() { if (server_) server_->killServer(); }


  void adapt_Low_controller (controller::raibotPositionController controller) {
    Low_controller_ = controller;
    Low_controller_.init(&world_);
    std::cout << "adapt test L " << std::endl;
    Low_controller_.test();
  }

  controller::raibotPositionController get_Low_controller () {
    return Low_controller_;
  }

  void Low_controller_create () {
    Low_controller_.create(&world_);
    Low_controller_.init(&world_);
    std::cout << "create test L " << std::endl;
    Low_controller_.test();
  }

  void init () { }
  void close () { }
  void setSimulationTimeStep(double dt)
  { controller_.setSimDt(dt);
    Low_controller_.setSimDt(dt);
    world_.setTimeStep(dt);
  }
  void setControlTimeStep(double dt, double low_dt) {
    controller_.setConDt(dt);
    Low_controller_.setConDt(low_dt);}
  void turnOffVisualization() { server_->hibernate(); }
  void turnOnVisualization() { server_->wakeup(); }
  void startRecordingVideo(const std::string& videoName ) { server_->startRecordingVideo(videoName); }
  void stopRecordingVideo() { server_->stopRecordingVideo(); }
  const std::vector<std::string>& getStepDataTag() { return controller_.getStepDataTag(); }
  const Eigen::VectorXd& getStepData() { return controller_.getStepData(); }

  void reset() {
    /// set the state
    raibo_->setState(gc_init_, gv_init_); /// set it again to ensure that foot is in contact
    updateObstacle();

    controller_.reset(gen_, normDist_, command_Obj_Pos_);
    controller_.updateStateVariables();
    Low_controller_.reset(&world_);
    Low_controller_.updateStateVariable();
  }



  double step(const Eigen::Ref<EigenVec>& action, bool visualize) {
    /// action scaling


    controller_.advance(&world_, action, curriculumFactor_);
    Eigen::Vector3f command;
    command = controller_.advance(&world_, action);

//    command = command_set[command_order%4];
//    std::cout << "command : " << command << std::endl;
//    command_order += 1;
    Low_controller_.setCommand(command);
    if(visualizable_)
    {
      target_pos_->setPosition(Low_controller_.getTargetPosition());
    }
    float dummy;
    int howManySteps;
    int lowlevelSteps;

    for (lowlevelSteps = 0; lowlevelSteps < int(high_level_control_dt_ / low_level_control_dt_ + 1e-10); lowlevelSteps++) {
//      sleep(0.01);
      controller_.updateHistory();
      Low_controller_.updateObservation(&world_);
      Low_controller_.advance(&world_);

      for(howManySteps = 0; howManySteps< int(low_level_control_dt_ / simulation_dt_ + 1e-10); howManySteps++) {

        subStep();
        if(visualize)
          sleep(simulation_dt_);

        if(isTerminalState(dummy)) {
          howManySteps++;
          break;
       }
      }
    }

    return controller_.getRewardSum(visualize);
  }



  void updateObstacle() {
    double x, y, x_command, y_command;
    double phi_;
    phi_ = uniDist_(gen_);

    while (true)
    {
      x = 2.0*cos(phi_*2*M_PI) + normDist_(gen_)*0.5*curriculumFactor_;
      y = 2.0*sin(phi_*2*M_PI) + normDist_(gen_)*0.5*curriculumFactor_;
      if(sqrt(std::pow(x,2) + std::pow(y,2)) > 1.8)
        break;
    }

    x += gc_init_[0];
    y += gc_init_[1];

    Obj_->setPosition(x, y, object_height/2);
    Obj_->setOrientation(1, 0, 0, 0);

    phi_ = uniDist_(gen_);

    x_command = x + sqrt(2)*cos(phi_*2*M_PI) + normDist_(gen_)*1*curriculumFactor_;
    y_command = y + sqrt(2)*sin(phi_*2*M_PI) + normDist_(gen_)*1*curriculumFactor_;

    command_Obj_Pos_ << x_command, y_command, object_height/2;

    if(visualizable_)
      command_Obj_->setPosition(command_Obj_Pos_[0], command_Obj_Pos_[1], command_Obj_Pos_[2]);

  }



  void subStep() {
    Low_controller_.updateHistory();


    world_.integrate1();
    world_.integrate2();

    Low_controller_.updateStateVariable();
    controller_.updateStateVariables();
    controller_.accumulateRewards(curriculumFactor_, command_);

  }

  void observe(Eigen::Ref<EigenVec> ob) {
    controller_.updateObservation(true, command_, heightMap_, gen_, normDist_);
    controller_.getObservation(obScaled_);
    ob = obScaled_.cast<float>();
  }

  bool isTerminalState(float& terminalReward) {
//    return controller_.isTerminalState(terminalReward);
    return false;
  }

  void setSeed(int seed) {
    gen_.seed(seed);
    terrainGenerator_.setSeed(seed);
  }

  void curriculumUpdate() {
//    groundType_ = (groundType_+1) % 4; /// rotate ground type for a visualization purpose
    curriculumFactor_ = std::pow(curriculumFactor_, curriculumDecayFactor_);
    /// create heightmap
//    world_.removeObject(heightMap_);
//    heightMap_ = terrainGenerator_.generateTerrain(&world_, RandomHeightMapGenerator::GroundType(groundType_), curriculumFactor_, gen_, uniDist_);
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
  static constexpr int nJoints_ = 12;
  raisim::World world_;
  double simulation_dt_;
  double high_level_control_dt_;
  double low_level_control_dt_;
  int gcDim_, gvDim_;
  std::array<size_t, 4> footFrameIndicies_;

  raisim::ArticulatedSystem* raibo_;
  raisim::HeightMap* heightMap_;
  Eigen::VectorXd gc_init_, gv_init_, nominalJointConfig_;
  Eigen::VectorXd gc_init_from_, gv_init_from_;
  double curriculumFactor_, curriculumDecayFactor_;
  Eigen::VectorXd obScaled_;
  Eigen::Vector3d command_;
  bool visualizable_ = false;
  int groundType_;
  RandomHeightMapGenerator terrainGenerator_;
  RaiboController controller_;
  controller::raibotPositionController Low_controller_;
  Eigen::VectorXd jointDGain_, jointPGain_;


  std::unique_ptr<raisim::RaisimServer> server_;
  raisim::Visuals *commandSphere_, *controllerSphere_;
  raisim::Cylinder* Obj_;
  raisim::Visuals *command_Obj_, *cur_head_Obj_, *tar_head_Obj_, *target_pos_;
  Eigen::Vector3d command_Obj_Pos_;
  Eigen::Vector3d Dist_eo_, Dist_og_;
  raisim::Vec<3> Pos_e_;
  std::vector<Eigen::Vector2f> command_set;
  int command_order = 0;
  double object_height = 0.55;


  thread_local static std::mt19937 gen_;
  thread_local static std::normal_distribution<double> normDist_;
  thread_local static std::uniform_real_distribution<double> uniDist_;
};

thread_local std::mt19937 raisim::ENVIRONMENT::gen_;
thread_local std::normal_distribution<double> raisim::ENVIRONMENT::normDist_(0., 1.);
thread_local std::uniform_real_distribution<double> raisim::ENVIRONMENT::uniDist_(0., 1.);
}