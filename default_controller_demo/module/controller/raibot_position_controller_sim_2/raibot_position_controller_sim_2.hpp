//
// Created by suyoung on 1/25/22.
//

#ifndef _COMPLIANT_GYM_TESTER_SRC_RAIBOT_DEFAULT_CONTROLLER2_HPP_
#define _COMPLIANT_GYM_TESTER_SRC_RAIBOT_DEFAULT_CONTROLLER2_HPP_

#include "../common/RaiboPositionController_2.hpp"
#include "../common/neuralNet.hpp"
#include <filesystem>

namespace controller {

class raibotPositionController {

 public:
  bool create(raisim::World *world) {

    char tmp[256];
    getcwd(tmp, 256);

    std::string current_path = tmp;

//    std::cout << current_path << std::endl;

//    std::string network_path = current_path + "/../../../s/rsg_raibo_rough_terrain/default_controller_demo/module/controller/raibot_default_controller/network/network_100000";
    std::string network_path = current_path + "/../../../../default_controller_demo/module/controller/raibot_position_controller_sim_2/network/network_100000";
    actor_.readParamFromTxt(network_path + "/full_12800.txt");

    std::cout << network_path << std::endl;

//    std::cout << "Fuck" << std::endl;

    std::string in_line;
    std::ifstream obsMean_file(network_path + "/mean12800.csv");
    std::ifstream obsVariance_file(network_path + "/var12800.csv");

    obs_.setZero(raibotController_.getObDim());
    obsMean_.setZero(raibotController_.getObDim());
    obsVariance_.setZero(raibotController_.getObDim());
    actor_input_.setZero(raibotController_.getObDim());

    if (obsMean_file.is_open()) {
      for (int i = 0; i < obsMean_.size(); ++i) {
//        std::getline(obsMean_file, in_line, ' ');
        std::getline(obsMean_file, in_line);
        obsMean_(i) = std::stof(in_line);
      }
    }


    RSINFO(obsMean_)
//    std::cout << "obs Mean : " <<  obsMean_ << std::endl;

    if (obsVariance_file.is_open()) {
      for (int i = 0; i < obsVariance_.size(); ++i) {
//        std::getline(obsVariance_file, in_line, ' ');
        std::getline(obsVariance_file, in_line);
        obsVariance_(i) = std::stof(in_line);
      }
    }

//    std::cout << "obs Var : " <<  obsVariance_ << std::endl;

    obsMean_file.close();
    obsVariance_file.close();
    return true;
  }

  void init(raisim::World *world) {
    raibotController_.create(world);
    raibotController_.reset(world);
  }

  void setSimDt(double simulation_dt) {
    communication_dt_ = simulation_dt;
  }
  void setConDt(double control_dt) {
    control_dt_ = control_dt;
  }
  bool reset(raisim::World *world) {
    raibotController_.reset(world);
    clk_ = 0;
    return true;
  }

  Eigen::VectorXf obsScalingAndGetAction() {

    /// normalize the obs
    obs_ = raibotController_.getObservation().cast<float>();

//    std::cout << "obs : " << obs_ << std::endl;


    for (int i = 0; i < obs_.size(); ++i) {
      obs_(i) = (obs_(i) - obsMean_(i)) / std::sqrt(obsVariance_(i) + 1e-8);
//      if (obs_(i) > 10) { obs_(i) = 10.0; }
//      else if (obs_(i) < -10) { obs_(i) = -10.0; }
    }

//    std::cout << "obs Normed : " << obs_ << std::endl;



    /// concat obs and e_out and forward to the actor
    Eigen::Matrix<float, 133, 1> actor_input;
    actor_input << obs_;
    Eigen::VectorXf action = actor_.forward(actor_input);

//    std::cout << "action : " << action << std::endl;


    return action;
  }

  bool advance(raisim::World *world) {


    raibotController_.advance(obsScalingAndGetAction().head(12));

    return true;
  }

  void updateStateVariable() {
    raibotController_.updateStateVariables();
  }

  void updateObservation(raisim::World *world) {
    raibotController_.updateStateVariables();
    raibotController_.updateObservation(world);
  }

  void setCommand(const Eigen::Ref<raisim::EigenVec>& command) {
    raibotController_.setCommand(command);
  }

  Eigen::VectorXd getTargetPosition() {
    return raibotController_.getTargetPosition();
  }

  void updateHistory() {
    raibotController_.updateHistory();
  }



  void setJointPositionHistory(Eigen::VectorXd &joint_position_history) {
    raibotController_.setJointPositionHistory(joint_position_history);
  }

  void setJointVelocityHistory(Eigen::VectorXd &joint_velocity_history) {
    raibotController_.setJointVelocityHistory(joint_velocity_history);
  }

  void setPrevAction(Eigen::VectorXd &prevAction) {
    raibotController_.setPrevAction(prevAction);
  }

  void setPrevPrevAction(Eigen::VectorXd &prevprevAction) {
    raibotController_.setPrevPrevAction(prevprevAction);
  }

  void getJointPositionHistory(Eigen::VectorXd &joint_position_history) {
    joint_position_history = raibotController_.getJointPositionHistory();
  }

  void getJointVelocityHistory(Eigen::VectorXd &joint_velocity_history) {
    joint_velocity_history = raibotController_.getJointVelocityHistory();
  }

  void getPrevAction(Eigen::VectorXd &prevAction) {
    prevAction = raibotController_.getPrevAction();
  }

  void getPrevPrevAction(Eigen::VectorXd &prevprevAction) {
    prevprevAction = raibotController_.getPrevPrevAction();
  }

  void test() {
    Eigen::Matrix<float, 133, 1> actor_input;
    actor_input.setOnes();
    Eigen::VectorXf action = actor_.forward(actor_input);

//    std::cout << "test L " << action << std::endl;
  }

 private:
  raisim::RaiboPositionController raibotController_;
  Eigen::VectorXf obs_;
  Eigen::VectorXf actor_input_;

  Eigen::VectorXf obsMean_;
  Eigen::VectorXf obsVariance_;
  raisim::nn::Linear<float, 133, 12, raisim::nn::ActivationType::leaky_relu> actor_{{512, 400, 128}};

  int clk_ = 0;
  double control_dt_{0.005};
  double communication_dt_{0.001};
};

} // namespace raisim

#endif //_COMPLIANT_GYM_TESTER_SRC_RAIBOT_DEFAULT_CONTROLLER_HPP_
