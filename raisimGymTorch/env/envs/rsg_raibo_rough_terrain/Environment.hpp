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
#include "RandomObjectGenerator.hpp"

namespace raisim {

class ENVIRONMENT {

 public:

  explicit ENVIRONMENT(const std::string &resourceDir, const Yaml::Node &cfg, bool visualizable, int id) :
      visualizable_(visualizable) {
    setSeed(id);

    READ_YAML(double, curriculumFactor_, cfg["curriculum"]["initial_factor"])
    READ_YAML(double, curriculumDecayFactor_, cfg["curriculum"]["decay_factor"])
    READ_YAML(double, low_level_control_dt_, cfg["low_level_control_dt"])
    READ_YAML(bool, is_discrete_, cfg["discrete_action"])

    if(is_discrete_) {
      READ_YAML(int, radial_, cfg["discrete"]["radial"])
      READ_YAML(int, tangential_, cfg["discrete"]["tangential"])
    }


    world_.addGround(0.0, "ground");
    world_.setDefaultMaterial(1.1, 0.0, 0.01);

    /// add objects
    raibo_ = world_.addArticulatedSystem(resourceDir + "/raibot/urdf/raibot_simplified.urdf");
    raibo_->setName("robot");
    raibo_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
//    raibo_->getCollisionBody("arm_link/0").setCollisionGroup(raisim::COLLISION(1));
//    raibo_->getCollisionBody("arm_link/0").setCollisionMask(raisim::COLLISION(1));

//    auto depthSensor1 = raibo_->getSensor<raisim::DepthCamera>("depth_camera_front_camera_parent:depth");
//    depthSensor1->setMeasurementSource(raisim::Sensor::MeasurementSource::VISUALIZER);
//    auto rgbCamera1 = raibo_->getSensor<raisim::RGBCamera>("depth_camera_front_camera_parent:color");
//    rgbCamera1->setMeasurementSource(raisim::Sensor::MeasurementSource::VISUALIZER);

//    depthCameras[0] = depthSensor1;
//    rgbCameras[0] = rgbCamera1;

//    raibo_->ignoreCollisionBetween(raibo_->getBodyIdx("base"), )

    /// Object spawn
    Obj_ = objectGenerator_.generateObject(&world_, RandomObjectGenerator::ObjectShape(object_type), curriculumFactor_, gen_, uniDist_,
                                           normDist_, bound_ratio, obj_mass, 0.5, 0.55, 1.0, 1.0);

    object_height = objectGenerator_.get_height();
    Obj_->setPosition(2, 2, object_height/2);
    Obj_->setOrientation(1, 0, 0, 0);

//    Manipulate_ = world_.addCylinder(0.1, 0.5, 1, "default", raisim::COLLISION(2), raisim::COLLISION(1)|raisim::COLLISION(2));


    /// create controller
    controller_.create(&world_, Obj_);
//    Low_controller_.create(&world_);

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

    command_set.push_back({1.5,0});
    command_set.push_back({-1.5, 0});
    command_set.push_back({0, 1.5});
    command_set.push_back({0, -1.5});


    if (is_discrete_)
      controller_.update_discrete_command_lib(radial_, tangential_);

    // visualize if it is the first environment
    if (visualizable_) {
      server_ = std::make_unique<raisim::RaisimServer>(&world_);
      server_->launchServer(8080);
      server_->focusOn(raibo_);
      std::cout << "Launch Server !!" << std::endl;
      command_Obj_ = server_->addVisualCylinder("command_Obj_", 0.5, command_object_height_, 1, 0, 0, 0.5);
      command_Obj_->setPosition(command_Obj_Pos_[0], command_Obj_Pos_[1], command_Obj_Pos_[2]);
      target_pos_ = server_->addVisualSphere("target_Pos_", 0.3, 1, 0, 0, 1.0);
      command_ball_ = server_->addVisualSphere("command_Ball", 0.1, 0, 1, 0, 1.0);
      com_pos_ = server_->addVisualSphere("com_pos", 0.05, 0, 0.5, 0.5, 1.0);
      com_noisify_ = server_->addVisualSphere("com_nosify_pos", 0.05, 1.0, 0.5, 0.5, 1.0);
    }
  }



  ~ENVIRONMENT() { if (server_) server_->killServer(); }

  void adapt_Low_controller (controller::raibotPositionController controller) {
    Low_controller_ = controller;
    Low_controller_.init(&world_);
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

  void hard_reset () {
    friction = 1.1 + 0.2*curriculumFactor_ * normDist_(gen_);
    if (friction < 0.6)
      friction = 1.1;
    world_.setMaterialPairProp("ground", "object", friction, 0.0, 0.01);

    Obj_->setAngularDamping({1.5*friction/1.1, 2.0*friction/1.1, 2.0*friction/1.1});
    Obj_->setLinearDamping(0.6*friction/1.1);
  }

  void reset() {
//    object_type = (object_type+1) % 3; /// rotate ground type for a visualization purpose

    object_type = 2;

    updateObstacle();
    objectGenerator_.Inertial_Randomize(Obj_, bound_ratio, curriculumFactor_, gen_, uniDist_, normDist_);
    if(curriculumFactor_ > 0.4)
      hard_reset();
    /// set the state
    raibo_->setState(gc_init_, gv_init_); /// set it again to ensure that foot is in contact
    controller_.reset(gen_, normDist_, command_Obj_Pos_, objectGenerator_.get_geometry(), friction);
    Low_controller_.reset(&world_);
    controller_.updateStateVariables();
    Low_controller_.updateStateVariable();

  }



  double step(const Eigen::Ref<EigenVec>& action, bool visualize) {
    /// action scaling
//    controller_.updateObservation(true, command_, heightMap_, gen_, normDist_);
    Eigen::Vector3f command;

    command = controller_.advance(&world_, action);
    Low_controller_.setCommand(command);


//    if (controller_.is_achieved)
//    {
//      command = controller_.advance(&world_, action);
//      Low_controller_.setCommand(command);
//    }

    if(visualizable_)
    {
      target_pos_->setPosition(Low_controller_.getTargetPosition());
      command_ball_->setPosition(controller_.get_desired_pos());
      com_pos_->setPosition(controller_.get_com_pos());
      com_noisify_->setPosition(controller_.get_noisify_com_pos());
    }
    float dummy;
    int howManySteps;
    int lowlevelSteps;

    /// Low level frequency
    for (lowlevelSteps = 0; lowlevelSteps < int(high_level_control_dt_ / low_level_control_dt_ + 1e-10); lowlevelSteps++) {

      /// per 0.02 sec, update history
      if(lowlevelSteps % (int(high_level_control_dt_/low_level_control_dt_ + 1e-10) / controller_.historyNum_))
      {
        controller_.updateHistory();
        controller_.update_actionHistory(&world_, action, curriculumFactor_);
      }
      Low_controller_.updateObservation(&world_);
      Low_controller_.advance(&world_);

      /// Simulation frequency
      for(howManySteps = 0; howManySteps< int(low_level_control_dt_ / simulation_dt_ + 1e-10); howManySteps++) {

        subStep();
        if(visualize)
//          std::this_thread::sleep_for(std::chrono::milliseconds(1));

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

    Obj_->setPosition(x, y, object_height/2);
    Obj_->setOrientation(1, 0, 0, 0);
    Obj_->setVelocity(0,0,0,0,0,0);
    Obj_->setAngularDamping({1.0,1.0,1.0});
    Obj_->setLinearDamping(0.5);

    phi_ = uniDist_(gen_);

    x_command = x + sqrt(2)*cos(phi_*2*M_PI) + normDist_(gen_)*1*curriculumFactor_;
    y_command = y + sqrt(2)*sin(phi_*2*M_PI) + normDist_(gen_)*1*curriculumFactor_;

    command_Obj_Pos_ << x_command, y_command, command_object_height_/2;

    if(visualizable_)
      command_Obj_->setPosition(command_Obj_Pos_[0], command_Obj_Pos_[1], command_Obj_Pos_[2]);

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

  void get_Controller_History(std::vector<Eigen::VectorXd> &obj_info_history,
                              std::vector<Eigen::VectorXd> &state_info_history,
                              std::vector<Eigen::VectorXd> &action_info_history,
                              std::vector<Eigen::VectorXd> &dynamics_info_history) {
    controller_.get_History(obj_info_history,
                            state_info_history,
                            action_info_history,
                            dynamics_info_history);
  }

  void get_Low_Controller_History(Eigen::VectorXd &joint_position_history,
                                  Eigen::VectorXd &joint_velocity_history,
                                  Eigen::VectorXd &prevAction,
                                  Eigen::VectorXd &prevprevAction) {
    Low_controller_.getJointPositionHistory(joint_position_history);
    Low_controller_.getJointVelocityHistory(joint_velocity_history);
    Low_controller_.getPrevAction(prevAction);
    Low_controller_.getPrevPrevAction(prevprevAction);
  }

  void get_obj_info_(Eigen::Vector3d &pos,
                     Eigen::Matrix3d &Rot,
                     Eigen::Vector3d &lin_vel,
                     Eigen::Vector3d &ang_vel,
                     Eigen::Matrix3d &inertia,
                     double &mass,
                     Eigen::Vector3d &com,
                     double &ratio,
                     double &height_ratio,
                     double &width_ratio_1,
                     double &width_ratio_2,
                     double &_friction
                     ) {
    pos = Obj_->getPosition();
    Rot = Obj_->getOrientation().e();
    lin_vel = Obj_->getLinearVelocity();
    ang_vel = Obj_->getAngularVelocity();
    inertia = Obj_->getInertiaMatrix_B();
    mass = Obj_->getMass();
    com = Obj_->getBodyToComPosition_rs().e();
    objectGenerator_.get_ratio(ratio, height_ratio, width_ratio_1, width_ratio_2);
    _friction = friction;
  }

  void export_info_(Eigen::Ref<EigenVec> gc,
                    Eigen::Ref<EigenVec> gv) {
    controller_.getState(gc, gv);
//    get_obj_info_();

  }

  void back_to_previous() {
    /// controller previous observation : obScaled_
    ///
  }

  void subStep() {
    if(controller_.is_success()) {
      Obj_->setAppearance("0, 0, 1, 0.7");
    }

    Low_controller_.updateHistory();


    world_.integrate1();
    if(server_) server_->lockVisualizationServerMutex();
    world_.integrate2();
    if(server_) server_->unlockVisualizationServerMutex();
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
    objectGenerator_.setSeed(seed);
  }

  void curriculumUpdate() {
//    object_type = (object_type+1) % 3; /// rotate ground type for a visualization purpose
    curriculumFactor_ = std::pow(curriculumFactor_, curriculumDecayFactor_);
    /// create heightmap
    updateObstacle(true);
    Eigen::VectorXd temp = objectGenerator_.get_classify_vector();
    controller_.updateClassifyvector(temp);
  }

  bool check_success() {
    return controller_.is_success();
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
  const double obj_mass = 2.0;
  double friction = 1.1;
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
  int object_type = 0;
  RandomHeightMapGenerator terrainGenerator_;
  RaiboController controller_;
  controller::raibotPositionController Low_controller_;
  Eigen::VectorXd jointDGain_, jointPGain_;
  RandomObjectGenerator objectGenerator_;
  raisim::RGBCamera* rgbCameras[1];
  raisim::DepthCamera* depthCameras[1];

  std::unique_ptr<raisim::RaisimServer> server_;
  raisim::Visuals *commandSphere_, *controllerSphere_;
  raisim::SingleBodyObject *Obj_, *Manipulate_;
  raisim::Visuals *command_Obj_, *cur_head_Obj_, *tar_head_Obj_, *target_pos_, *command_ball_, *com_pos_, *com_noisify_;
  Eigen::Vector3d command_Obj_Pos_;
  Eigen::Vector3d Dist_eo_, Dist_og_;
  raisim::Vec<3> Pos_e_;
  std::vector<Eigen::Vector2f> command_set;
  int command_order = 0;
  double object_height = 0.55;
  double command_object_height_ = 0.55;
  double object_radius;
  int radial_ = 0;
  int tangential_ = 0;


  thread_local static std::mt19937 gen_;
  thread_local static std::normal_distribution<double> normDist_;
  thread_local static std::uniform_real_distribution<double> uniDist_;
};

thread_local std::mt19937 raisim::ENVIRONMENT::gen_;
thread_local std::normal_distribution<double> raisim::ENVIRONMENT::normDist_(0., 1.);
thread_local std::uniform_real_distribution<double> raisim::ENVIRONMENT::uniDist_(0., 1.);
}