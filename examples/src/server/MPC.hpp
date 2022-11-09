//
// Created by user on 22. 10. 31..
//

#ifndef RAISIM_EXAMPLES_SRC_SERVER_MPC_HPP_
#define RAISIM_EXAMPLES_SRC_SERVER_MPC_HPP_

#include "Eigen/Eigen"
#include "qpSWIFT.h"
#endif //RAISIM_EXAMPLES_SRC_SERVER_MPC_HPP_

class Sparse_Matrix {
 public:
  std::vector<int> Pjc;
  std::vector<int> Pir;
  std::vector<double> Ppr;
  std::vector<int> Ajc;
  std::vector<int> Air;
  std::vector<double> Apr;
  std::vector<int> Gjc;
  std::vector<int> Gir;
  std::vector<double> Gpr;

  void get_Sparse_P(Eigen::MatrixXd P, Eigen::MatrixXd R);

  void get_Sparse_A (Eigen::MatrixXd model_A, Eigen::MatrixXd model_B);

  void get_Sparse_G ();

};

class MPC {
 public:
  Eigen::MatrixXd P;
  Eigen::MatrixXd R;
  Eigen::MatrixXd A;
  Eigen::MatrixXd b;
  Eigen::MatrixXd G;
  Eigen::MatrixXd h;
  Eigen::MatrixXd P_con;
  Eigen::MatrixXd R_con;
  Eigen::MatrixXd Q_con;
  Eigen::MatrixXd model_A;
  Eigen::MatrixXd model_B;

  Eigen::VectorXd result;
  Eigen::VectorXd var;


  MPC(Eigen::MatrixXd P,
  Eigen::MatrixXd R,
  Eigen::MatrixXd model_A,
  Eigen::MatrixXd model_B
) {
    this->P = P;
    this->R = R;
    this->model_A = model_A;
    this->model_B = model_B;
  }


  void init();

  void update();

  Eigen::VectorXd Solve(Eigen::VectorXd Var);
};
