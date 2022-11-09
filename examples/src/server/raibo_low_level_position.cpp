//
// Created by suyoung on 1/25/22.
//

#include "raisim/World.hpp"
#include "raisim/RaisimServer.hpp"
#include "raibot_position_controller_sim.hpp"

int main (int argc, char* argv[]) {
  auto binaryPath = raisim::Path::setFromArgv(argv[0]);

  /// create world and robot
  double control_dt = 0.005;
  double simulation_dt = 0.001;

  raisim::World world;
  world.setTimeStep(simulation_dt);
  world.addGround(0.0, "default");
  auto robot = world.addArticulatedSystem("/home/oem/workspace/default_controller_demo/rsc/raibot/urdf/raibot_simplified.urdf");
  robot->setName("robot");
  robot->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
  std::array<size_t, 4> footFrameIndicies_;
  footFrameIndicies_[0] = robot->getFrameIdxByName("LF_S2F");
  footFrameIndicies_[1] = robot->getFrameIdxByName("RF_S2F");
  footFrameIndicies_[2] = robot->getFrameIdxByName("LH_S2F");
  footFrameIndicies_[3] = robot->getFrameIdxByName("RH_S2F");

  raisim::RaisimServer server(&world);
  server.launchServer(8080);
  server.focusOn(robot);

  Eigen::VectorXd jointNominalConfig(robot->getGeneralizedCoordinateDim()), jointVelocityTarget(robot->getDOF());
  jointNominalConfig << 0, 0, 0.54, 1, 0.0, 0.0, 0.0, 0, 0.56, -1.12, 0, 0.56, -1.12, 0, 0.56, -1.12, 0, 0.56, -1.12;
  jointVelocityTarget.setZero();
  robot->setGeneralizedCoordinate(jointNominalConfig);
  Eigen::VectorXd jointPgain(robot->getDOF()), jointDgain(robot->getDOF());
  jointPgain.setZero();
  jointDgain.setZero();
  jointPgain.segment(6, 12).setConstant(60.0);
  jointDgain.segment(6, 12).setConstant(0.5);
  robot->setPdGains(jointPgain, jointDgain);
  robot->setGeneralizedForce(Eigen::VectorXd::Zero(robot->getDOF()));
  /// set ground contacting with body and thighs

  controller::raibotPositionController controller;
//

  // keep one foot on the terrain
  raisim::Vec<3> footPosition;
  double maxNecessaryShift = -1e20; /// some arbitrary high negative value
  for(auto& foot: footFrameIndicies_) {
    robot->getFramePosition(foot, footPosition);
//      double terrainHeightMinusFootPosition = heightMap_->getHeight(footPosition[0], footPosition[1]) - footPosition[2];
    double terrainHeightMinusFootPosition = 0 - footPosition[2];
    maxNecessaryShift = maxNecessaryShift > terrainHeightMinusFootPosition ? maxNecessaryShift : terrainHeightMinusFootPosition;
  }
  jointNominalConfig(2) += maxNecessaryShift + 0.07;

  robot->setState(jointNominalConfig, jointVelocityTarget);

  controller.create(&world);
  controller.setSimDt(simulation_dt);
  controller.setConDt(control_dt);
  controller.reset(&world);

  Eigen::Vector2f command = {5.0, 0.0};

  auto command_Obj_ = server.addVisualCylinder("command_Obj_", 0.5, 0.7, 1, 0, 0, 0.5);
  raisim::Vec<3> command_Obj_Pos_ = {2, 2, 0.35};
  command_Obj_->setPosition(command_Obj_Pos_[0], command_Obj_Pos_[1], command_Obj_Pos_[2]);

  controller.setCommand(command);
  int maxStep = 100000000;
  for (int i = 0; i < maxStep; ++i) {


    if (i%int(control_dt / simulation_dt) == 0) {
      controller.updateObservation(&world);
      controller.advance(&world);
    }
    controller.updateHistory();

    std::this_thread::sleep_for(std::chrono::microseconds(1000));
    server.integrateWorldThreadSafe();

  }
//  server.stopRecordingVideo();
  server.killServer();
}