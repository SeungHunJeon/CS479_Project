// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.

#include "raisim/RaisimServer.hpp"
#include "raisim/World.hpp"
#include "boost/math/interpolators/cubic_b_spline.hpp"
#include <matplot/matplot.h>
#include "Matrix_box_test.hpp"
#include "unsupported/Eigen/MatrixFunctions"
#include "Eigen/Sparse"

#if WIN32
#include <timeapi.h>
#endif

boost::math::cubic_b_spline<double> trajectory_generator (const std::vector<double> points_, const double t0, const double h) {
  boost::math::cubic_b_spline<double> spline(points_.data(), points_.size(), t0, h, 0, 0);
  return spline;

}

std::vector<double> transform_(const std::vector<double> &x,
                               boost::math::cubic_b_spline<double> fn) {
  std::vector<double> y(x.size());
  for (size_t i = 0; i < y.size(); ++i) {
    y[i] = fn(x[i]);
  }
  return y;
}



QP *myQP;
//Eigen::Matrix<double, 7,7> A;
//Eigen::Matrix<double, 7,2> B;
bool init = TRUE;

void dynamics_update_(const Eigen::VectorXd state, const Eigen::VectorXd control) {
  Eigen::VectorXd x = state; Eigen::VectorXd u = control;


}

int main(int argc, char* argv[]) {
  auto binaryPath = raisim::Path::setFromArgv(argv[0]);
  raisim::World::setActivationKey(binaryPath.getDirectory() + "\\rsc\\activation.raisim");

  /// Dynamics Iniialization

  QP_Eigen Matrix;
  Matrix.init();

  std::cout << Matrix.P << std::endl;

  Eigen::SparseMatrix<double> A = Matrix.P.sparseView();

  A.makeCompressed();

//  for (int i = 0; i<A.outerSize()+1 ; i++)
//    std::cout << A.outerIndexPtr()[i] << std::endl;
//
//  for (int i = 0; i<A.size()+1 ; i++)
//    std::cout << A.valuePtr()[i] << std::endl;

  for (int i = 0; i<A.size()-A.outerSize()+1 ; i++)
    std::cout << A.innerIndexPtr()[i] << std::endl;

  std::vector<int> outer;
  std::vector<int> outer2;

  outer.push_back(1);
  outer.push_back(2);

  outer2.push_back(3);
  outer2.push_back(12);

  outer2.insert(end(outer2), begin(outer), end(outer));

  for (int i=0; i<5; i++) {
    if (i<3) {
      std::cout << "bool" << std::endl;
      continue;
    }
    std::cout << "int" << std::endl;
  }

//  std::cout << outer2.back() << "back" << std::endl;
//
//  for (int i =0; i<outer.size() ; i++)
//    std::cout << outer[i] << std::endl;
//
//  for (int i =0; i<outer2.size() ; i++)
//    std::cout << outer2[i] << std::endl;
////  std::cout << Matrix.P.sparseView() << std::endl;

  myQP = QP_SETUP_dense(Matrix.P.rows(), Matrix.G.rows(), Matrix.A.rows(), Matrix.P.data(), Matrix.A.data(), Matrix.G.data(),
                        Matrix.c.data(), Matrix.h.data(), Matrix.b.data(), NULL, COLUMN_MAJOR_ORDERING);

  qp_int Exit_Code = QP_SOLVE(myQP);

  for (int i=0; i<Matrix.P.rows(); i++) {
    std::cout << myQP->x[i] << std::endl;
  }

//  if (init) {
//    A.setZero(); B.setZero();
//    A.block(0, 3, 3, 3) << Eigen::Matrix3d::Identity();
//    A.block(3, 6, 3, 1) << 0, -0.6, -0.6;
//    B.block(3, 0, 3, 1) << 0, 0, 0;
//    B.block(3, 1, 3, 1) << 0, 1, 0;
//  }

  QP_CLEANUP_dense(myQP);

  /// create raisim world
  raisim::World world;
  world.setTimeStep(0.001);

  /// create objects
  world.addGround();

  auto box = world.addBox(1, 1, 1, 1);
  box->setName("box");

  box->setPosition(0, 0, 0.5);
  box->setOrientation(1, 0, 0, 0);

  std::vector<double> f{0, -1, 1, -1, 1, -1, 2};
  double t0 = 0;
  double h = 1;

  auto spline = trajectory_generator(f, t0, h);


  std::vector<double> x = matplot::linspace(t0, (f.size()-1)*h);
  std::vector<double> y = transform_(x, spline);

  matplot::plot(x, y, "-o");
  matplot::show();

  /// launch raisim server
  raisim::RaisimServer server(&world);
  server.launchServer();

  raisim::Vec<3> Force_;

  Force_[0] = 10000;
  Force_[1] = 0;
  Force_[2] = 0;

  box->setExternalForce(0, Force_);
//  sleep(1);

  init = FALSE;

  for (int i=0; i<2000000; i++) {
    std::this_thread::sleep_for(std::chrono::microseconds(1000));

    server.integrateWorldThreadSafe();
  }


  server.killServer();
}


