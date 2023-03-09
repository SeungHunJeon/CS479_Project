//
// Created by suyoung on 1/25/22.
//

#ifndef _COMPLIANT_GYM_TESTER_SRC_RAIBOT_DEFAULT_CONTROLLER_HPP_
#define _COMPLIANT_GYM_TESTER_SRC_RAIBOT_DEFAULT_CONTROLLER_HPP_

#include "../common/BasicEigenTypes.hpp"
#include "../common/raibotController.hpp"
#include "../common/neuralNet.hpp"

namespace controller {

class raibotDefaultController {

 public:
  bool create(raisim::World *world) {

    char tmp[256];
    getcwd(tmp, 256);

    std::string current_path = tmp;

    std::string network_path = current_path + "/../../../../default_controller_demo/module/controller/raibot_default_controller/network/network_100000";

//    std::string network_path = "/home/oem/workspace/raisimGymForRaisin/default_controller_demo/module/controller/raibot_default_controller/network/network_100000";
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
        std::getline(obsMean_file, in_line);
        obsMean_(i) = std::stof(in_line);
      }
    }

    if (obsVariance_file.is_open()) {
      for (int i = 0; i < obsVariance_.size(); ++i) {
        std::getline(obsVariance_file, in_line, ' ');
        obsVariance_(i) = std::stof(in_line);
      }
    }

//    std::cout << obsVariance_ << std::endl;

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
  };

  void init (raisim::World *world) {
    raibotController_.create(world);
    raibotController_.reset(world);
  }

  bool reset(raisim::World *world) {
    raibotController_.reset(world);
    actor_.initHidden();
    estimator_.initHidden();
    clk_ = 0;
    return true;
  };

  void setSimDt(double simulation_dt) {
    communication_dt_ = simulation_dt;
  }
  void setConDt(double control_dt) {
    control_dt_ = control_dt;
  }

  Eigen::VectorXf obsScalingAndGetAction() {

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
  };

  void updateObservation(raisim::World *world) {
    raibotController_.updateObservation(world);
  }

  bool advance(raisim::World *world) {
    raibotController_.updateObservation(world);
    raibotController_.advance(world, obsScalingAndGetAction().head(12));

    clk_++;
    return true;
  };
  void setCommand(const Eigen::Ref<raisim::EigenVec>& command) {
    raibotController_.setCommand(command);
  }


  void update_model (raisim::nn::LSTM_MLP<float, 41, 12, raisim::nn::ActivationType::leaky_relu> actor,
                     raisim::nn::LSTM_MLP<float, 30, 8, raisim::nn::ActivationType::leaky_relu> estimator)
  {
    actor_ = actor;
    estimator_ = estimator;
  }

  raisim::nn::LSTM_MLP<float, 41, 12, raisim::nn::ActivationType::leaky_relu> export_actor ()
  {
    return actor_;
  }

  raisim::nn::LSTM_MLP<float, 30, 8, raisim::nn::ActivationType::leaky_relu> export_estimator ()
  {
    return estimator_;
  }

 private:
  raisim::raibotController raibotController_;
  Eigen::VectorXf obs_;
  Eigen::VectorXf actor_input_;

  Eigen::VectorXf obsMean_;
  Eigen::VectorXf obsVariance_;
  Eigen::VectorXf eoutMean_;
  Eigen::VectorXf eoutVariance_;
  raisim::nn::LSTM_MLP<float, 41, 12, raisim::nn::ActivationType::leaky_relu> actor_{72, 2, {256, 128}};
//  control_dt_(0.01),
//  communication_dt_(0.00025)
  raisim::nn::LSTM_MLP<float, 30, 8, raisim::nn::ActivationType::leaky_relu> estimator_{36, 1, {128, 128}};

  int clk_ = 0;
  double control_dt_ = 0.01;
  double communication_dt_ = 0.00025;
};

} // namespace raisim

#endif //_COMPLIANT_GYM_TESTER_SRC_RAIBOT_DEFAULT_CONTROLLER_HPP_
