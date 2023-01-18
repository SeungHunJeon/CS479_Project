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

class ENVIRONMENT_ROLLOUT {

 public:

  explicit ENVIRONMENT_ROLLOUT(const std::string &resourceDir, const Yaml::Node &cfg, bool visualizable, int id) :
      visualizable_(visualizable) {
    setSeed(id);

    READ_YAML(double, curriculumFactor_, cfg["curriculum"]["initial_factor"])
    READ_YAML(double, curriculumDecayFactor_, cfg["curriculum"]["decay_factor"])
    READ_YAML(double, low_level_control_dt_, cfg["low_level_control_dt"])
    READ_YAML(bool, is_discrete_, cfg["discrete_action"])
    READ_YAML(int, n_samples, cfg["nSamples_"])

    if(is_discrete_) {
      READ_YAML(int, radial_, cfg["discrete"]["radial"])
      READ_YAML(int, tangential_, cfg["discrete"]["tangential"])
    }

    world_batch_.reserve(n_samples);
    controller_batch_.reserve(n_samples);
    Low_controller_batch_.reserve(n_samples);

    /// set curriculum
    simulation_dt_ = RaiboController::getSimDt();
    high_level_control_dt_ = RaiboController::getConDt();

    world_0.addGround(0.0, "ground");
    world_0.setDefaultMaterial(1.1, 0.0, 0.01);

    raibo_0 = world_0.addArticulatedSystem(resourceDir + "/raibot/urdf/raibot_simplified.urdf");
    raibo_0->setName("robot");
    raibo_0->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
    raibo_0->getCollisionBody("arm_link/0").setCollisionGroup(raisim::COLLISION(1));
    raibo_0->getCollisionBody("arm_link/0").setCollisionMask(raisim::COLLISION(1));

    Obj_ = objectGenerator_.generateObject(&world_0, RandomObjectGenerator::ObjectShape(object_type), curriculumFactor_, gen_, uniDist_,
                                           normDist_, bound_ratio, obj_mass_, 0.5, 0.55, 1.0, 1.0);

    object_height = objectGenerator_.get_height();
    Obj_->setPosition(2, 2, object_height/2);
    Obj_->setOrientation(1, 0, 0, 0);

    controller_0.create(&world_0, Obj_);

    /// get robot data
    gcDim_ = int(raibo_0->getGeneralizedCoordinateDim());
    gvDim_ = int(raibo_0->getDOF());

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
    raibo_0->setPdGains(jointPGain_, jointDGain_);

    /// this is nominal configuration of anymal
    nominalJointConfig_<< 0, 0.56, -1.12, 0, 0.56, -1.12, 0, 0.56, -1.12, 0, 0.56, -1.12;
    gc_init_.head(7) << 0, 0, 0.4725, 1, 0.0, 0.0, 0.0;
    gc_init_.tail(nJoints_) << nominalJointConfig_;
    gc_init_from_ = gc_init_;
    raibo_0->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    // Reward coefficients
    controller_0.setRewardConfig(cfg);
    command_Obj_Pos_ << 2, 2, command_object_height_/2;

    Low_controller_0.create(&world_0);
    Low_controller_0.init(&world_0);

    for (int i = 0; i<n_samples; i++) {
      raisim::World *world;
      RaiboController *controller;
      controller::raibotPositionController *low_controller;

      world->addGround(0.0, "ground");
      world->setDefaultMaterial(1.1, 0.0, 0.01);

      /// add objects
      auto raibo = world->addArticulatedSystem(resourceDir + "/raibot/urdf/raibot_simplified.urdf");
      raibo->setName("robot");
      raibo->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
      raibo->getCollisionBody("arm_link/0").setCollisionGroup(raisim::COLLISION(1));
      raibo->getCollisionBody("arm_link/0").setCollisionMask(raisim::COLLISION(1));

      /// Object spawn
      auto Obj = objectGenerator_.generateObject(world, RandomObjectGenerator::ObjectShape(object_type), curriculumFactor_, gen_, uniDist_,
                                             normDist_, bound_ratio, obj_mass_, 0.5, 0.55, 1.0, 1.0);

      object_height = objectGenerator_.get_height();
      Obj->setPosition(2, 2, object_height/2);
      Obj->setOrientation(1, 0, 0, 0);

      /// create controller
      controller->create(world, Obj);

      raibo->setPdGains(jointPGain_, jointDGain_);
      raibo->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

      // Reward coefficients
      controller->setRewardConfig(cfg);

      *low_controller = Low_controller_0;
      low_controller->init(world);


      world_batch_[i] = world;
      controller_batch_[i] = controller;
      Low_controller_batch_[i] = low_controller;

    }

    // visualize if it is the first environment
    if (visualizable_) {
      server_ = std::make_unique<raisim::RaisimServer>(&world_0);
      server_->launchServer(8080);
      server_->focusOn(raibo_0);
      std::cout << "Launch Server !!" << std::endl;
      command_Obj_ = server_->addVisualCylinder("command_Obj_", 0.5, command_object_height_, 1, 0, 0, 0.5);
      command_Obj_->setPosition(command_Obj_Pos_[0], command_Obj_Pos_[1], command_Obj_Pos_[2]);
      target_pos_ = server_->addVisualSphere("target_Pos_", 0.3, 1, 0, 0, 1.0);
      command_ball_ = server_->addVisualSphere("command_Ball", 0.1, 0, 1, 0, 1.0);
      com_pos_ = server_->addVisualSphere("com_pos", 0.05, 0, 0.5, 0.5, 1.0);
      com_noisify_ = server_->addVisualSphere("com_nosify_pos", 0.05, 1.0, 0.5, 0.5, 1.0);
    }



  }



  ~ENVIRONMENT_ROLLOUT() { if (server_) server_->killServer(); }

  void adapt_Low_controller (controller::raibotPositionController controller) {
//    Low_controller_ = controller;
//    Low_controller_.init(&world_);
//    Low_controller_.test();
  }

  controller::raibotPositionController get_Low_controller () {
    return Low_controller_0;
  }

  void Low_controller_create () {
//    Low_controller_0.create(&world_);
//    Low_controller_0.init(&world_);
//    std::cout << "create test L " << std::endl;
//    Low_controller_0.test();
  }

  void init () { }
  void close () { }
  void setSimulationTimeStep(double dt)
  { controller_0.setSimDt(dt);
    Low_controller_0.setSimDt(dt);
    world_0.setTimeStep(dt);

    for(int i = 0; i<n_samples; i++) {
      world_batch_[i]->setTimeStep(dt);
      controller_batch_[i]->setSimDt(dt);
      Low_controller_batch_[i]->setSimDt(dt);
    }

  }
  void setControlTimeStep(double dt, double low_dt) {
    controller_0.setConDt(dt);
    Low_controller_0.setConDt(low_dt);

    for(int i = 0; i<n_samples; i++) {
      controller_batch_[i]->setConDt(dt);
      Low_controller_batch_[i]->setConDt(low_dt);
    }

  }
  void turnOffVisualization() { server_->hibernate(); }
  void turnOnVisualization() { server_->wakeup(); }
  void startRecordingVideo(const std::string& videoName ) { server_->startRecordingVideo(videoName); }
  void stopRecordingVideo() { server_->stopRecordingVideo(); }
  const std::vector<std::string>& getStepDataTag() { return controller_0.getStepDataTag(); }
  const Eigen::VectorXd& getStepData() { return controller_0.getStepData(); }


  void hard_reset_Rollout() {
//#pragma omp parallel for schedule(auto)
    for(int i=0; i<n_samples; i++) {
      world_batch_[i]->setMaterialPairProp("ground", "object", friction, 0.0, 0.01);
      reinterpret_cast<SingleBodyObject *>(world_batch_[i]->getObject("object"))->setAngularDamping({1.5*friction/1.1, 2.0*friction/1.1, 2.0*friction/1.1});
      reinterpret_cast<SingleBodyObject *>(world_batch_[i]->getObject("object"))->setLinearDamping(0.6*friction/1.1);
    }
  }

  void hard_reset () {
    friction = 1.1 + 0.2*curriculumFactor_ * normDist_(gen_);
    if (friction < 0.6)
      friction = 1.1;
    world_0.setMaterialPairProp("ground", "object", friction, 0.0, 0.01);
    Obj_->setAngularDamping({1.5*friction/1.1, 2.0*friction/1.1, 2.0*friction/1.1});
    Obj_->setLinearDamping(0.6*friction/1.1);
  }

  void reset_Rollout() {
    updateObstacle_Rollout();
    if(curriculumFactor_ > 0.4)
      hard_reset_Rollout();
    /// set the state
//#pragma omp parallel for schedule(auto)
    for (int i =0; i< n_samples; i++) {
      auto obj = reinterpret_cast<SingleBodyObject *>(world_batch_[i]->getObject("object"));
      objectGenerator_.Inertial_Randomize(obj);
      auto raibo = reinterpret_cast<ArticulatedSystem *>(world_batch_[i]->getObject("robot"));
      raibo->setState(gc_init_, gv_init_); /// set it again to ensure that foot is in contact
      controller_batch_[i]->reset(gen_, normDist_, command_Obj_Pos_, objectGenerator_.get_geometry(), friction);
      Low_controller_batch_[i]->reset(world_batch_[i]);
      controller_batch_[i]->updateStateVariables();
      Low_controller_batch_[i]->updateStateVariable();
    }

  }

  void reset() {
    updateObstacle();
    objectGenerator_.Inertial_Randomize(Obj_, bound_ratio, curriculumFactor_, gen_, uniDist_, normDist_);
    if(curriculumFactor_ > 0.4)
      hard_reset();
    /// set the state
    raibo_0->setState(gc_init_, gv_init_); /// set it again to ensure that foot is in contact
    controller_0.reset(gen_, normDist_, command_Obj_Pos_, objectGenerator_.get_geometry(), friction);
    Low_controller_0.reset(&world_0);
    controller_0.updateStateVariables();
    Low_controller_0.updateStateVariable();

//    reset_Rollout();
  }


  void rollout_step(Eigen::Ref<EigenRowMajorMat> &action, bool b) {
//#pragma omp parallel for schedule(auto)
    for (int i = 0; i<n_samples; i++) {
      Eigen::Vector3f command;

      command = controller_batch_[i]->advance(world_batch_[i], action.row(i));
      Low_controller_batch_[i]->setCommand(command);

      float dummy;
      int howManySteps;
      int lowlevelSteps;

      /// Low level frequency
      for (lowlevelSteps = 0; lowlevelSteps < int(high_level_control_dt_ / low_level_control_dt_ + 1e-10); lowlevelSteps++) {

        /// per 0.02 sec, update history
        if(lowlevelSteps % (int(high_level_control_dt_/low_level_control_dt_ + 1e-10) / controller_batch_[i]->historyNum_))
        {
          controller_batch_[i]->updateHistory();
          controller_batch_[i]->update_actionHistory(world_batch_[i], action, curriculumFactor_);
        }
        Low_controller_batch_[i]->updateObservation(world_batch_[i]);
        Low_controller_batch_[i]->advance(world_batch_[i]);

        /// Simulation frequency
        for(howManySteps = 0; howManySteps< int(low_level_control_dt_ / simulation_dt_ + 1e-10); howManySteps++) {

          subStep_Rollout();
//          if(visualize)
//          std::this_thread::sleep_for(std::chrono::milliseconds(1));

            if(isTerminalState(dummy)) {
              howManySteps++;
              break;
            }
        }
      }
    }
  }



  double step(const Eigen::Ref<EigenVec>& action, bool visualize) {
    /// action scaling
//    controller_.updateObservation(true, command_, heightMap_, gen_, normDist_);
    Eigen::Vector3f command;

    command = controller_0.advance(&world_0, action);
    Low_controller_0.setCommand(command);


//    if (controller_.is_achieved)
//    {
//      command = controller_.advance(&world_, action);
//      Low_controller_.setCommand(command);
//    }

    if(visualizable_)
    {
      target_pos_->setPosition(Low_controller_0.getTargetPosition());
      command_ball_->setPosition(controller_0.get_desired_pos());
      com_pos_->setPosition(controller_0.get_com_pos());
      com_noisify_->setPosition(controller_0.get_noisify_com_pos());
    }
    float dummy;
    int howManySteps;
    int lowlevelSteps;

    /// Low level frequency
    for (lowlevelSteps = 0; lowlevelSteps < int(high_level_control_dt_ / low_level_control_dt_ + 1e-10); lowlevelSteps++) {

      /// per 0.02 sec, update history
      if(lowlevelSteps % (int(high_level_control_dt_/low_level_control_dt_ + 1e-10) / controller_0.historyNum_))
      {
        controller_0.updateHistory();
        controller_0.update_actionHistory(&world_0, action, curriculumFactor_);
      }
      Low_controller_0.updateObservation(&world_0);
      Low_controller_0.advance(&world_0);

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

    return controller_0.getRewardSum(visualize);
  }

  void updateObstacle_Rollout(bool curriculum_Update = false) {
//#pragma omp parallel for schedule(auto)
    for (int i=0; i<n_samples; i++) {
      world_batch_[i]->removeObject(world_batch_[i]->getObject("object"));
      auto Obj = objectGenerator_.generateObject(world_batch_[i], obj_mass_);
    }
  }

  void updateObstacle(bool curriculum_Update = false) {

    world_0.removeObject(Obj_);
    Obj_ = objectGenerator_.generateObject(&world_0, RandomObjectGenerator::ObjectShape(object_type), curriculumFactor_, gen_, uniDist_,
                                           normDist_, bound_ratio, obj_mass_, 0.5, 0.55, 1.0, 1.0);
    Obj_->setAppearance("0, 1, 0, 1.0");
    controller_0.updateObject(Obj_);
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

  void synchronize() {
    /// For Controller History
    std::vector<Eigen::VectorXd> obj_info_history;
    std::vector<Eigen::VectorXd> state_info_history;
    std::vector<Eigen::VectorXd> action_info_history;
    std::vector<Eigen::VectorXd> dynamics_info_history;

    /// For Low Controller History
    Eigen::VectorXd joint_position_history;
    Eigen::VectorXd joint_velocity_history;
    Eigen::VectorXd prevAction;
    Eigen::VectorXd prevprevAction;

    /// For Object Info
    Eigen::Vector3d pos;
    Eigen::Matrix3d Rot;
    Eigen::Vector3d lin_vel;
    Eigen::Vector3d ang_vel;
    Eigen::Matrix3d inertia;
    Eigen::Vector3d com;

    /// For Robot info
    Eigen::VectorXf gc;
    Eigen::VectorXf gv;

    get_Controller_History(obj_info_history,
                           state_info_history,
                           action_info_history,
                           dynamics_info_history);

    get_Low_Controller_History(joint_position_history,
                               joint_velocity_history,
                               prevAction,
                               prevprevAction);

    get_obj_info_(pos,
                  Rot,
                  lin_vel,
                  ang_vel
                  );

    get_robot_info_(gc,
                    gv);

//#pragma omp parallel for schedule(auto)
    for (int i=0; i<n_samples; i++) {
      controller_batch_[i]->set_History(obj_info_history,
                                        state_info_history,
                                        action_info_history,
                                        dynamics_info_history);

      Low_controller_batch_[i]->setJointPositionHistory(joint_position_history);
      Low_controller_batch_[i]->setJointVelocityHistory(joint_velocity_history);
      Low_controller_batch_[i]->setPrevAction(prevAction);
      Low_controller_batch_[i]->setPrevPrevAction(prevprevAction);

      controller_batch_[i]->setState(gc, gv);
      auto obj = reinterpret_cast<SingleBodyObject *>(world_batch_[i]->getObject("object"));
      obj->setPosition(pos);
      obj->setOrientation(Rot);
      obj->setLinearVelocity(lin_vel);
      obj->setAngularVelocity(ang_vel);
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


  void get_Controller_History(std::vector<Eigen::VectorXd> &obj_info_history,
                              std::vector<Eigen::VectorXd> &state_info_history,
                              std::vector<Eigen::VectorXd> &action_info_history,
                              std::vector<Eigen::VectorXd> &dynamics_info_history) {
    controller_0.get_History(obj_info_history,
                            state_info_history,
                            action_info_history,
                            dynamics_info_history);
  }

  void get_Low_Controller_History(Eigen::VectorXd &joint_position_history,
                                  Eigen::VectorXd &joint_velocity_history,
                                  Eigen::VectorXd &prevAction,
                                  Eigen::VectorXd &prevprevAction) {
    Low_controller_0.getJointPositionHistory(joint_position_history);
    Low_controller_0.getJointVelocityHistory(joint_velocity_history);
    Low_controller_0.getPrevAction(prevAction);
    Low_controller_0.getPrevPrevAction(prevprevAction);
  }

  void get_obj_info_(Eigen::Vector3d &pos,
                     Eigen::Matrix3d &Rot,
                     Eigen::Vector3d &lin_vel,
                     Eigen::Vector3d &ang_vel
                     ) {
    pos = Obj_->getPosition();
    Rot = Obj_->getOrientation().e();
    lin_vel = Obj_->getLinearVelocity();
    ang_vel = Obj_->getAngularVelocity();
  }

  void get_robot_info_(Eigen::Ref<EigenVec> gc,
                    Eigen::Ref<EigenVec> gv) {
    controller_0.getState(gc, gv);

  }

  void subStep_Rollout() {
//#pragma omp parallel for schedule(auto)
    for (int i=0; i<n_samples; i++) {
      Low_controller_batch_[i]->updateHistory();
      world_batch_[i]->integrate1();
      world_batch_[i]->integrate2();
      Low_controller_batch_[i]->updateStateVariable();
      controller_batch_[i]->updateStateVariables();
      controller_batch_[i]->accumulateRewards(curriculumFactor_, command_);
    }
  }

  void subStep() {
    Low_controller_0.updateHistory();


    world_0.integrate1();
    if(server_) server_->lockVisualizationServerMutex();
    world_0.integrate2();
    if(server_) server_->unlockVisualizationServerMutex();
    Low_controller_0.updateStateVariable();
    controller_0.updateStateVariables();
    controller_0.accumulateRewards(curriculumFactor_, command_);

  }

  void observe_Rollout(Eigen::Ref<EigenRowMajorMat> ob) {
//#pragma omp parallel for schedule(auto)
    for (int i=0; i<n_samples; i++) {
    controller_batch_[i]->updateObservation(true,
                                            command_,
                                            heightMap_,
                                            gen_,
                                            normDist_);
    controller_batch_[i]->getObservation(obScaled_);
    ob.row(i) = obScaled_.cast<float>();
    }
  }

  void observe(Eigen::Ref<EigenVec> ob) {
    controller_0.updateObservation(true, command_, heightMap_, gen_, normDist_);
    controller_0.getObservation(obScaled_);
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
    object_type = (object_type+1) % 3; /// rotate ground type for a visualization purpose
    curriculumFactor_ = std::pow(curriculumFactor_, curriculumDecayFactor_);
    /// create heightmap
    updateObstacle(true);
    Eigen::VectorXd temp = objectGenerator_.get_classify_vector();
    controller_0.updateClassifyvector(temp);
    for(int i = 0; i<n_samples; i++) {
      controller_batch_[i]->updateClassifyvector(temp);
    }
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
    controller_0.getState(gc, gv);
    RSINFO(gc)
    RSINFO(gv)
  }

  void getRolloutState(Eigen::Ref<EigenRowMajorMat> gc, Eigen::Ref<EigenRowMajorMat> gv) {
//#pragma omp parallel for schedule(auto)
    for (int i = 0; i<n_samples; i++) {
      controller_batch_[i]->getState(gc.row(i), gv.row(i));
    }
    RSINFO(gc)
    RSINFO(gv)
  }

 protected:
  double obj_mass_ = 3.0;
  double friction = 1.1;
  static constexpr int nJoints_ = 12;
  raisim::World world_0;
  std::vector<raisim::World *> world_batch_;
  double simulation_dt_;
  double high_level_control_dt_;
  double low_level_control_dt_;
  int gcDim_, gvDim_;
  int n_samples;
  std::array<size_t, 4> footFrameIndicies_;
  double bound_ratio = 0.5;
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
  RaiboController controller_0;
  ArticulatedSystem * raibo_0;
  std::vector<RaiboController *> controller_batch_;
  controller::raibotPositionController Low_controller_0;
  std::vector<controller::raibotPositionController *> Low_controller_batch_;
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

thread_local std::mt19937 raisim::ENVIRONMENT_ROLLOUT::gen_;
thread_local std::normal_distribution<double> raisim::ENVIRONMENT_ROLLOUT::normDist_(0., 1.);
thread_local std::uniform_real_distribution<double> raisim::ENVIRONMENT_ROLLOUT::uniDist_(0., 1.);
}