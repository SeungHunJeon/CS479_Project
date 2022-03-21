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


namespace raisim {

class ENVIRONMENT {

 public:

  explicit ENVIRONMENT(const std::string &resourceDir, const Yaml::Node &cfg, bool visualizable, int id) :
      visualizable_(visualizable) {
    setSeed(id);

    /// add objects
    raibo_ = world_.addArticulatedSystem(resourceDir + "/raibot/urdf/raibot_simplified.urdf");
    raibo_->setName("robot");
    raibo_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

    /// create controller
    controller_.create(&world_);

    /// indicies of the foot frame
    footFrameIndicies_[0] = raibo_->getFrameIdxByName("LF_S2F");
    footFrameIndicies_[1] = raibo_->getFrameIdxByName("RF_S2F");
    footFrameIndicies_[2] = raibo_->getFrameIdxByName("LH_S2F");
    footFrameIndicies_[3] = raibo_->getFrameIdxByName("RH_S2F");
    RSFATAL_IF(std::any_of(footFrameIndicies_.begin(), footFrameIndicies_.end(), [](int i){return i < 0;}), "footFrameIndicies_ not found")

    /// set curriculum
    simulation_dt_ = RaiboController::getSimDt();
    control_dt_ = RaiboController::getConDt();

    READ_YAML(double, curriculumFactor_, cfg["curriculum"]["initial_factor"])
    READ_YAML(double, curriculumDecayFactor_, cfg["curriculum"]["decay_factor"])

    /// create heightmap
    groundType_ = (id+3) % 4;
    heightMap_ = terrainGenerator_.generateTerrain(&world_, RandomHeightMapGenerator::GroundType(groundType_), curriculumFactor_, gen_, uniDist_);

    /// get robot data
    gcDim_ = int(raibo_->getGeneralizedCoordinateDim());
    gvDim_ = int(raibo_->getDOF());

    /// initialize containers
    gc_init_.setZero(gcDim_);
    gv_init_.setZero(gvDim_);
    nominalJointConfig_.setZero(nJoints_);
    gc_init_from_.setZero(gcDim_);
    gv_init_from_.setZero(gvDim_);

    /// this is nominal configuration of anymal
    nominalJointConfig_<< 0, 0.56, -1.12, 0, 0.56, -1.12, 0, 0.56, -1.12, 0, 0.56, -1.12;
    gc_init_.head(7) << 0, 0, 0.54, 1.0, 0.0, 0.0, 0.0;
    gc_init_.tail(12) = nominalJointConfig_;
    gc_init_from_ = gc_init_;
    raibo_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    // Reward coefficients
    controller_.setRewardConfig(cfg);

    // visualize if it is the first environment
    if (visualizable_) {
      server_ = std::make_unique<raisim::RaisimServer>(&world_);
      server_->launchServer();
      commandSphere_ = server_->addVisualSphere("commandSphere", 0.3, 1, 0, 0, 1);
      controllerSphere_ = server_->addVisualSphere("controllerSphere", 0.3, 0, 1, 0, 0.1);
    }
  }

  ~ENVIRONMENT() { if (server_) server_->killServer(); }
  void init () { }
  void close () { }
  void setSimulationTimeStep(double dt) { controller_.setSimDt(dt); };
  void setControlTimeStep(double dt) { controller_.setConDt(dt); };
  void turnOffVisualization() { server_->hibernate(); }
  void turnOnVisualization() { server_->wakeup(); }
  void startRecordingVideo(const std::string& videoName ) { server_->startRecordingVideo(videoName); }
  void stopRecordingVideo() { server_->stopRecordingVideo(); }
  const std::vector<std::string>& getStepDataTag() { return controller_.getStepDataTag(); }
  const Eigen::VectorXd& getStepData() { return controller_.getStepData(); }

  void reset() {
    // orientation
    raisim::Mat<3,3> rotMat, yawRot, pitchRollMat;
    raisim::Vec<4> quaternion;
    raisim::Vec<3> axis = {normDist_(gen_), normDist_(gen_), normDist_(gen_)};
    axis /= axis.norm();
    raisim::angleAxisToRotMat(axis, normDist_(gen_) * 0.2, pitchRollMat);
    raisim::angleAxisToRotMat({0,0,1}, uniDist_(gen_) * 2. * M_PI, yawRot);
    rotMat = pitchRollMat * yawRot;
    raisim::rotMatToQuat(rotMat, quaternion);
    gc_init_.segment(3, 4) = quaternion.e();

    // body position
    for(int i=0 ; i<2; i++)
      gc_init_[i] = 0.5 * normDist_(gen_);

    // joint angles
    for(int i=0 ; i<nJoints_; i++)
      gc_init_[i+7] = nominalJointConfig_[i] + 0.3 * normDist_(gen_);

    // command
//    const bool standingMode = normDist_(gen_) > 1.7;
    const bool standingMode = false;
    controller_.setStandingMode(standingMode);
    const double angle = 2. * (uniDist_(gen_) - 0.5) * M_PI;
    const double heading = 2. * (uniDist_(gen_) - 0.5) * M_PI;
    command_ << 5.0 * cos(angle), 5.0 * sin(angle), heading;

    /// randomize generalized velocities
    raisim::Vec<3> bodyVel_b, bodyVel_w;
    bodyVel_b[0] = 0.6 * normDist_(gen_) * curriculumFactor_;
    bodyVel_b[1] = 0.6 * normDist_(gen_) * curriculumFactor_;
    bodyVel_b[2] = 0.3 * normDist_(gen_) * curriculumFactor_;
    raisim::matvecmul(rotMat, bodyVel_b, bodyVel_w);

    // base angular velocities (just define this in the world frame since it is isometric)
    raisim::Vec<3> bodyAng_w;
    for(int i=0; i<3; i++) bodyAng_w[i] = 0.4 * normDist_(gen_) * curriculumFactor_;

    // joint velocities
    Eigen::VectorXd jointVel(12);
    for(int i=0; i<12; i++) jointVel[i] = 3. * normDist_(gen_) * curriculumFactor_;

    // combine
    gv_init_ << bodyVel_w.e(), bodyAng_w.e(), jointVel;

    // randomly initialize from previous trajectories
    if(uniDist_(gen_) < 0.25) {
      gc_init_ = gc_init_from_;
      gv_init_ = gv_init_from_;
      gc_init_.head(2).setZero();
    }
    raibo_->setGeneralizedCoordinate(gc_init_);

    // keep one foot on the terrain
    raisim::Vec<3> footPosition;
    double maxNecessaryShift = -1e20; /// some arbitrary high negative value
    for(auto& foot: footFrameIndicies_) {
      raibo_->getFramePosition(foot, footPosition);
      double terrainHeightMinusFootPosition = heightMap_->getHeight(footPosition[0], footPosition[1]) - footPosition[2];
      maxNecessaryShift = maxNecessaryShift > terrainHeightMinusFootPosition ? maxNecessaryShift : terrainHeightMinusFootPosition;
    }
    gc_init_[2] += maxNecessaryShift + 0.07;

    /// set the state
    raibo_->setState(gc_init_, gv_init_); /// set it again to ensure that foot is in contact
    controller_.reset(gen_, normDist_);
    controller_.updateStateVariables();
  }

  double step(const Eigen::Ref<EigenVec>& action, bool visualize) {
    /// action scaling
    controller_.advance(&world_, action, curriculumFactor_);

    float dummy;
    int howManySteps;

    for(howManySteps = 0; howManySteps< int(control_dt_ / simulation_dt_ + 1e-10); howManySteps++) {
      subStep();

      if(isTerminalState(dummy)) {
        howManySteps++;
        break;
      }
    }
    return controller_.getRewardSum(visualize);
  }

  void subStep() {
    controller_.updateHistory();
    world_.integrate1();
    world_.integrate2();
    controller_.updateStateVariables();
    controller_.accumulateRewards(curriculumFactor_, command_);

    if(uniDist_(gen_) < 0.005) {
      raibo_->getState(gc_init_from_, gv_init_from_);
      gc_init_from_[0] = 0.;
      gc_init_from_[1] = 0.;
    }
  }

  void observe(Eigen::Ref<EigenVec> ob) {
    controller_.updateObservation(true, command_, heightMap_, gen_, normDist_);
    controller_.getObservation(obScaled_);
    ob = obScaled_.cast<float>();
  }

  bool isTerminalState(float& terminalReward) {
    return controller_.isTerminalState(terminalReward);
  }

  void setSeed(int seed) {
    gen_.seed(seed);
    terrainGenerator_.setSeed(seed);
  }

  void curriculumUpdate() {
    groundType_ = (groundType_+1) % 4; /// rotate ground type for a visualization purpose
    curriculumFactor_ = std::pow(curriculumFactor_, curriculumDecayFactor_);
    /// create heightmap
    world_.removeObject(heightMap_);
    heightMap_ = terrainGenerator_.generateTerrain(&world_, RandomHeightMapGenerator::GroundType(groundType_), curriculumFactor_, gen_, uniDist_);
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
  double control_dt_;
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

  std::unique_ptr<raisim::RaisimServer> server_;
  raisim::Visuals *commandSphere_, *controllerSphere_;

  thread_local static std::mt19937 gen_;
  thread_local static std::normal_distribution<double> normDist_;
  thread_local static std::uniform_real_distribution<double> uniDist_;
};

thread_local std::mt19937 raisim::ENVIRONMENT::gen_;
thread_local std::normal_distribution<double> raisim::ENVIRONMENT::normDist_(0., 1.);
thread_local std::uniform_real_distribution<double> raisim::ENVIRONMENT::uniDist_(0., 1.);
}