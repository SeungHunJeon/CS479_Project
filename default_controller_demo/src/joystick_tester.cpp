//
// Created by suyoung on 1/25/22.
//

#include "raisim/World.hpp"
#include "raisim/RaisimServer.hpp"

#include "controller/raibot_default_controller/raibot_default_controller.hpp"

int main (int argc, char* argv[]) {
  auto binaryPath = raisim::Path::setFromArgv(argv[0]);

  /// create world and robot
  double control_dt = 0.01;
  double simulation_dt = 0.00025;

  raisim::World world;
  world.setTimeStep(simulation_dt);

  auto robot = world.addArticulatedSystem(binaryPath.getDirectory() + "../rsc/raibot/urdf/raibot_simplified.urdf");
  robot->setName("robot");
  robot->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

  /// set ground contacting with body and thighs
  std::vector<std::string> bodyColObjNames = {"base/0", "LF_THIGH/0", "RF_THIGH/0", "LH_THIGH/0", "RH_THIGH/0"};
  for (auto &name: bodyColObjNames) { robot->getCollisionBody(name).setCollisionGroup(raisim::COLLISION(10)); }
  world.setDefaultMaterial(1.1, 0.0, 0.01);
  auto ground = world.addGround(0.0, "default", raisim::COLLISION(10));
//  world.addGround(0.0, "default", raisim::COLLISION(2) | raisim::COLLISION(3) | raisim::COLLISION(4) | raisim::COLLISION(5));
  ground->setAppearance("dune");

  controller::raibotDefaultController controller;

  controller.create(&world);
  controller.setTimeConfig(control_dt, simulation_dt);
  controller.reset(&world);

  Eigen::Vector4f normalLimits = {1.0, 1.0, 1.0, 1.0};
  Eigen::Vector4f boostLimits = {3.0, 2.0, 1.5, 2.0};
  Eigen::Vector4f commandLimits = normalLimits;
  Eigen::Vector3f command = {0.0, 0.0, 0.0};

  raisim::RaisimServer server(&world);
  server.launchServer(8080);
  server.focusOn(robot);
//  server.addVisualBox("ground", 100.0, 0.001, 100.0, 0.7765, 0.5412, 0.0706, 0.5);

//  server.startRecordingVideo("robot.mp4");
  int maxStep = 100000000;
  for (int i = 0; i < maxStep; ++i) {


//    std::cout << commandLimits.transpose() << '\n' << command.transpose() << std::endl;
    controller.setCommand(command);
    controller.advance(&world);

    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
//  server.stopRecordingVideo();
  server.killServer();
}