//
// Created by suyoung on 1/25/22.
//

#include "raisim/World.hpp"
#include "raisim/RaisimServer.hpp"
#include "raibot_default_controller.hpp"

int main (int argc, char* argv[]) {
  auto binaryPath = raisim::Path::setFromArgv(argv[0]);

  /// create world and robot
  double control_dt = 0.01;
  double simulation_dt = 0.00025;

  raisim::World world;
  world.setTimeStep(simulation_dt);
  world.addGround(0.0, "default");
  auto robot = world.addArticulatedSystem("/home/user/Downloads/default_controller_demo/rsc/raibot/urdf/raibot_simplified.urdf");
  robot->setName("robot");
  robot->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);


  Eigen::VectorXd jointNominalConfig(robot->getGeneralizedCoordinateDim()), jointVelocityTarget(robot->getDOF());
  jointNominalConfig << 0, 0, 0.4725, 1, 0.0, 0.0, 0.0, 0, 0.56, -1.12, 0, 0.56, -1.12, 0, 0.56, -1.12, 0, 0.56, -1.12;
  jointVelocityTarget.setZero();
  robot->setGeneralizedCoordinate(jointNominalConfig);
  Eigen::VectorXd jointPgain(robot->getDOF()), jointDgain(robot->getDOF());
  jointPgain.setZero();
  jointDgain.setZero();
  jointPgain.segment(6, 12).setConstant(50.0);
  jointDgain.segment(6, 12).setConstant(0.5);
  robot->setPdGains(jointPgain, jointDgain);
  robot->setPdTarget(jointNominalConfig, jointVelocityTarget);
  /// set ground contacting with body and thighs

  controller::raibotDefaultController controller;
//
  controller.create(&world);
  controller.setSimDt(simulation_dt);
  controller.setConDt(control_dt);
  controller.reset(&world);

  Eigen::Vector3f command = {2.0, 0.0, 0.0};

  raisim::RaisimServer server(&world);
  server.launchServer(8080);
  server.focusOn(robot);

  int maxStep = 100000000;
  for (int i = 0; i < maxStep; ++i) {

    controller.setCommand(command);
    if (i%int(control_dt / simulation_dt) == 0) {
      controller.updateObservation(&world);
      controller.advance(&world);
    }




    std::this_thread::sleep_for(std::chrono::microseconds(250));
    server.integrateWorldThreadSafe();
  }
//  server.stopRecordingVideo();
  server.killServer();
}