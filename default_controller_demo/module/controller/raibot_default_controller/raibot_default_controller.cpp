//
// Created by suyoung on 1/25/22.
//

#include "raibot_default_controller.hpp"

namespace controller {

raibotDefaultController::raibotDefaultController()
: actor_(72, 2, {256, 128}),
  estimator_(36, 1, {128, 128}),
  control_dt_(0.01),
  communication_dt_(0.00025)
{
}

bool raibotDefaultController::create(raisim::World *world) {
  raibotController_.create(world);
  raibotController_.reset(world);

  std::string network_path = "../module/controller/raibot_default_controller/network/network_100000";
  actor_.readParamFromTxt(network_path + "/actor.txt");
  estimator_.readParamFromTxt(network_path + "/estimator.txt");

  std::string in_line;
  std::ifstream obsMean_file(network_path + "/obs_mean.csv");
  std::ifstream obsVariance_file(network_path + "/obs_var.csv");
  std::ifstream eoutMean_file(network_path + "/eout_mean.csv");
  std::ifstream eoutVariance_file(network_path + "/eout_var.csv");

  obs_.setZero(raibotController_.getObDim());
  obsMean_.setZero(raibotController_.getObDim());
  obsVariance_.setZero(raibotController_.getObDim());
  eoutMean_.setZero(raibotController_.getEstDim());
  eoutVariance_.setZero(raibotController_.getEstDim());
  actor_input_.setZero(raibotController_.getObDim() + raibotController_.getEstDim());

  if (obsMean_file.is_open()) {
    for (int i = 0; i < obsMean_.size(); ++i) {
      std::getline(obsMean_file, in_line, ' ');
      obsMean_(i) = std::stof(in_line);
    }
  }

  if (obsVariance_file.is_open()) {
    for (int i = 0; i < obsVariance_.size(); ++i) {
      std::getline(obsVariance_file, in_line, ' ');
      obsVariance_(i) = std::stof(in_line);
    }
  }

  if (eoutMean_file.is_open()) {
    for (int i = 0; i < eoutMean_.size(); ++i) {
      std::getline(eoutMean_file, in_line, ' ');
      eoutMean_(i) = std::stof(in_line);
    }
  }

  if (eoutVariance_file.is_open()) {
    for (int i = 0; i < eoutVariance_.size(); ++i) {
      std::getline(eoutVariance_file, in_line, ' ');
      eoutVariance_(i) = std::stof(in_line);
    }
  }

  obsMean_file.close();
  obsVariance_file.close();
  eoutMean_file.close();
  eoutVariance_file.close();
  return true;
}

void raibotDefaultController::setTimeConfig(double control_dt, double simulation_dt) {
  control_dt_ = control_dt;
  communication_dt_ = simulation_dt;
}

bool raibotDefaultController::reset(raisim::World *world) {
  raibotController_.reset(world);
  actor_.initHidden();
  estimator_.initHidden();
  clk_ = 0;
  return true;
}

bool raibotDefaultController::advance(raisim::World *world) {
//  if(clk_ % int(control_dt_ / communication_dt_ + 1e-10) == 0) {
  raibotController_.updateObservation(world);
  raibotController_.advance(world, obsScalingAndGetAction().head(12));
//  }

  clk_++;
  return true;
}

Eigen::VectorXf raibotDefaultController::obsScalingAndGetAction() {

  /// normalize the obs
  obs_ = raibotController_.getObservation().cast<float>();

  for (int i = 0; i < obs_.size(); ++i) {
    obs_(i) = (obs_(i) - obsMean_(i)) / std::sqrt(obsVariance_(i) + 1e-8);
    if (obs_(i) > 10) { obs_(i) = 10.0; }
    else if (obs_(i) < -10) { obs_(i) = -10.0; }
  }

  /// forward the obs to the estimator
  Eigen::Matrix<float, 30, 1> e_in;
  e_in = obs_.tail(obs_.size() - 3);
  Eigen::VectorXf e_out = estimator_.forward(e_in);

  /// normalize the output of estimator
  for (int i = 0; i < e_out.size(); ++i) {
    e_out(i) = (e_out(i) - eoutMean_(i)) / std::sqrt(eoutVariance_(i) + 1e-8);
    if (e_out(i) > 10) { e_out(i) = 10.0; }
    else if (e_out(i) < -10) { e_out(i) = -10.0; }
  }

  /// concat obs and e_out and forward to the actor
  Eigen::Matrix<float, 41, 1> actor_input;
  actor_input << obs_, e_out;
  Eigen::VectorXf action = actor_.forward(actor_input);

  return action;
}

void raibotDefaultController::setCommand(const Eigen::Ref<raisim::EigenVec>& command) {
  raibotController_.setCommand(command);
}

} // namespace controller
