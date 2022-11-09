//
// Created by user on 22. 10. 20..
//

#ifndef RAISIM_EXAMPLES_SRC_SERVER_MATRIX_BOX_TEST_HPP_
#define RAISIM_EXAMPLES_SRC_SERVER_MATRIX_BOX_TEST_HPP_

#endif //RAISIM_EXAMPLES_SRC_SERVER_MATRIX_BOX_TEST_HPP_

#include "qpSWIFT.h"

class QP_Eigen {
 public:
  void init() {
    P << 5,1,0,
    1,2,1,
    0,1,4;

    c << 1, 2, 1;

    A << 1, -2, 1;

    b << 3;

    G << -4, -4, 0,
    0, 0, -1;

    h << -1, -1;
  }

  Eigen::Matrix<double, 3,3> P;
  Eigen::Matrix<double, 3,1> c;
  Eigen::Matrix<double, 1,3> A;
  Eigen::Matrix<double, 1,1> b;
  Eigen::Matrix<double, 2,3> G;
  Eigen::Matrix<double, 2,1> h;
};