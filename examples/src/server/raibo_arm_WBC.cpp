// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.

#include "raisim/RaisimServer.hpp"
//#include "../../../matplotlib-cpp/matplotlibcpp.h"
#if WIN32
#include <timeapi.h>
#endif

int main(int argc, char* argv[]) {
  auto binaryPath = raisim::Path::setFromArgv(argv[0]);
  raisim::World::setActivationKey(binaryPath.getDirectory() + "\\rsc\\activation.raisim");
#if WIN32
  timeBeginPeriod(1); // for sleep_for function. windows default clock speed is 1/64 second. This sets it to 1ms.
#endif

  /// create raisim world
  raisim::World world;
  world.setTimeStep(0.001);

  /// create objects
  auto ground = world.addGround(0, "gnd");
  ground->setAppearance("wheat");
//  auto raibo_arm = world.addArticulatedSystem(binaryPath.getDirectory() + "\\rsc\\raibo_arm\\urdf\\raibo_arm.urdf");
  auto raibo_arm = world.addArticulatedSystem(binaryPath.getDirectory() + "\\rsc\\raibo_arm_2\\urdf\\raibo_arm_2.urdf");

  /// anymalC joint PD controller
  Eigen::VectorXd jointNominalConfig(raibo_arm->getGeneralizedCoordinateDim()), jointVelocityTarget(raibo_arm->getDOF());
  jointNominalConfig << 0, 0, 0.54, 1.0, 0.0, 0.0, 0.0, 0, 0.56, -1.12, 0, 0.56, -1.12, 0, 0.56, -1.12, 0, 0.56, -1.12, -0.5, 1.0, 0, 2.3, 0, 0;
  jointVelocityTarget.setZero();

  Eigen::VectorXd jointPgain(raibo_arm->getDOF()), jointDgain(raibo_arm->getDOF());
  jointPgain.setZero();
  jointDgain.setZero();
  jointPgain.segment(6, 12).setConstant(100.0);
  jointDgain.segment(6, 12).setConstant(1.0);

  Eigen::VectorXd GenForce_;
  GenForce_.setZero(raibo_arm->getDOF());

  raibo_arm->setGeneralizedCoordinate(jointNominalConfig);
  raibo_arm->setGeneralizedForce(GenForce_);
  raibo_arm->setPdGains(jointPgain, jointDgain);
  raibo_arm->setPdTarget(jointNominalConfig, jointVelocityTarget);
  raibo_arm->setName("raibo_arm");

  Eigen::VectorXd Gen_arm_;
  Eigen::VectorXd Gen_leg_;
  Eigen::MatrixXd ee_Jaco_body_, ee_Jaco_rot_;
  Eigen::MatrixXd ee_Jaco_body_dot_, ee_Jaco_rot_dot_;
  Eigen::MatrixXd raibo_Mass_, raibo_Mass_simp_;
  Eigen::Vector3d ee_Vel_, ee_Avel_;
  Eigen::Vector3d ee_acc_, ee_Aacc_;
  raisim::Vec<3> ee_Vel_w_, ee_Avel_w_;
  raisim::Mat<3,3> baseRot_;
  Eigen::VectorXd gc_, gv_;

  raisim::SparseJacobian ee_Jaco_body_s_, ee_Jaco_rot_s_;
  raisim::SparseJacobian ee_Jaco_body_dot_s_, ee_Jaco_rot_dot_s_;
  raisim::SparseJacobian dJc_tmp_s_;
  std::vector<size_t> footIndices_;
  std::vector<size_t> armIndices_;
  std::vector<Eigen::Matrix<double, 3, 24>> Jc_batch_;
  std::vector<Eigen::Matrix<double, 3, 24>> dJc_batch_;
  Eigen::MatrixXd Jc_tmp_;
  Eigen::MatrixXd dJc_tmp_;
  Eigen::MatrixXd Jc_;
  Eigen::MatrixXd dJc_;
  Eigen::MatrixXd P_;
  Eigen::MatrixXd Q_;
  Eigen::MatrixXd R_;
  Eigen::VectorXd V_;
  Eigen::MatrixXd N_;
  Eigen::MatrixXd J_;
  Eigen::MatrixXd dJ_;
  Eigen::MatrixXd K_;
  Eigen::VectorXd Out_;


  /// Initialize
  Gen_leg_.setZero(12);
  Gen_arm_.setZero(6);
  ee_Jaco_body_.setZero(3, raibo_arm->getDOF());
  ee_Jaco_rot_.setZero(3, raibo_arm->getDOF());
  ee_Jaco_body_dot_.setZero(3, raibo_arm->getDOF());
  ee_Jaco_rot_dot_.setZero(3, raibo_arm->getDOF());
  ee_acc_.setZero();
  ee_Aacc_.setZero();
  gc_.resize(raibo_arm->getGeneralizedCoordinateDim());
  gv_.resize(raibo_arm->getDOF());
  footIndices_.push_back(raibo_arm->getBodyIdx("LF_SHANK"));
  footIndices_.push_back(raibo_arm->getBodyIdx("RF_SHANK"));
  footIndices_.push_back(raibo_arm->getBodyIdx("LH_SHANK"));
  footIndices_.push_back(raibo_arm->getBodyIdx("RH_SHANK"));
  armIndices_.push_back(raibo_arm->getBodyIdx("link6"));
  Jc_tmp_.setZero(3, raibo_arm->getDOF());
  dJc_tmp_.setZero(3, raibo_arm->getDOF());
  N_.setZero(raibo_arm->getDOF(), raibo_arm->getDOF());
  K_.setZero(raibo_arm->getDOF(), 6 + raibo_arm->getDOF());
  P_.setZero(raibo_arm->getDOF(), raibo_arm->getDOF());
  Out_.setZero(6 + raibo_arm->getDOF());

  ee_Vel_.setZero();

  Eigen::Vector3d pos_residual_;
  Eigen::Vector3d rot_residual_;
  Eigen::MatrixXd S_arm_, S_leg_;

  pos_residual_.setZero();
  rot_residual_.setZero();
  S_arm_.setZero(6, raibo_arm->getDOF());
  S_leg_.setZero(12, raibo_arm->getDOF());
  for (int i=0; i<6; i++)
  {
    S_arm_(i, raibo_arm->getDOF() - 6 + i) = 1;
  }

  for (int i=0; i<12; i++)
  {
    S_leg_(i, raibo_arm->getDOF() - 18 + i) = 1;
  }

  pos_residual_ << 0.0, 0.0, 0.0;
  rot_residual_ << 0.0, 0.0, 0.0;

  double posPgain_, posDgain_, oriPgain_, oriDgain_;


  posPgain_ = 10;
  posDgain_ = 1;
  oriPgain_ = 10;
  oriDgain_ = 1;


  /// launch raisim server
  raisim::RaisimServer server(&world);
  server.launchServer();
  server.focusOn(raibo_arm);

  auto sphere_ = server.addVisualSphere("body_pos_check", 0.05);

  for (int i=0; i<200000000; i++) {
    std::this_thread::sleep_for(std::chrono::microseconds(1000));
    server.integrateWorldThreadSafe();

//
    raibo_arm->getState(gc_, gv_);
//
//    /// initialize
    ee_Jaco_body_dot_.setZero();
    ee_Jaco_rot_dot_.setZero();
    dJc_tmp_.setZero();
//
    raisim::Vec<4> quat;
    quat[0] = gc_[3];
    quat[1] = gc_[4];
    quat[2] = gc_[5];
    quat[3] = gc_[6];
    raisim::quatToRotMat(quat, baseRot_);
    raisim::Vec<3> point_;

    for (auto &contact: raibo_arm->getContacts())
    {
      auto it =  std::find(footIndices_.begin(), footIndices_.end(), contact.getlocalBodyIndex());
      size_t index = it - footIndices_.begin();
      if (index < 4 && !contact.isSelfCollision())
      {

        raibo_arm->getBodyPosition(footIndices_[index], point_);
        raibo_arm->getDenseJacobian(footIndices_[index], point_, Jc_tmp_);
        raibo_arm->getTimeDerivativeOfSparseJacobian(footIndices_[index], raisim::ArticulatedSystem::WORLD_FRAME, point_, dJc_tmp_s_);

        int j = 0;
        for (auto i : dJc_tmp_s_.idx) {
          dJc_tmp_.col(i) << dJc_tmp_s_.v.e().col(j);
          j += 1;
        }
        Jc_batch_.push_back(Jc_tmp_);
        dJc_batch_.push_back(dJc_tmp_);
      }
    }

    Jc_.setZero(Jc_batch_.size()*3, raibo_arm->getDOF());
    dJc_.setZero(dJc_batch_.size()*3, raibo_arm->getDOF());
    Q_.setZero(2 * (Jc_batch_.size()*3 + 6), raibo_arm->getDOF());
    R_.setZero(2 * (Jc_batch_.size()*3 + 6), raibo_arm->getDOF());
    V_.setZero(2* (Jc_batch_.size()*3 + 6));
    J_.setZero((Jc_batch_.size()*3 + 6), raibo_arm->getDOF());
    dJ_.setZero((Jc_batch_.size()*3 + 6), raibo_arm->getDOF());


    for (int i=0; i<Jc_batch_.size(); i++) {
      Jc_(Eigen::seqN(3*i, 3), Eigen::all) << Jc_batch_[i];
    }

    std::cout << "Jc_ : " << std::endl << Jc_ << std::endl;


    for (int i=0; i<dJc_batch_.size(); i++) {
      dJc_(Eigen::seqN(3*i, 3), Eigen::all) << dJc_batch_[i];
    }
//
    raibo_arm->getBodyPosition(armIndices_[0], point_);
    raibo_arm->getDenseJacobian(armIndices_[0], point_, ee_Jaco_body_);
    raibo_arm->getDenseRotationalJacobian(armIndices_[0], ee_Jaco_rot_);
//
    raibo_arm->getTimeDerivativeOfSparseJacobian(armIndices_[0], raisim::ArticulatedSystem::WORLD_FRAME, point_, ee_Jaco_body_dot_s_);
    raibo_arm->getTimeDerivativeOfSparseRotationalJacobian(armIndices_[0], ee_Jaco_rot_dot_s_);

    sphere_->setPosition(point_.e());

//
    int j = 0;
    for (auto i : ee_Jaco_body_dot_s_.idx) {
      ee_Jaco_body_dot_.col(i) << ee_Jaco_body_dot_s_.v.e().col(j);
      j += 1;
    }

    j = 0;
    for (auto i : ee_Jaco_rot_dot_s_.idx) {
      ee_Jaco_rot_dot_.col(i) << ee_Jaco_rot_dot_s_.v.e().col(j);
      j += 1;
    }
//
    raibo_arm->getVelocity(armIndices_[0], ee_Vel_w_);
    ee_Vel_ = baseRot_.e().transpose() * ee_Vel_w_.e();
//
    raibo_arm->getAngularVelocity(armIndices_[0], ee_Avel_w_);
    ee_Avel_ = baseRot_.e().transpose() * ee_Avel_w_.e();
//
    raibo_arm->getMassMatrix();
//
    ee_acc_ << (posPgain_ * pos_residual_ - posDgain_ * ee_Vel_w_.e());
    ee_Aacc_ <<
             (oriPgain_ * rot_residual_ - oriDgain_ * ee_Avel_w_.e());

//

    J_(Eigen::seqN(0, Jc_batch_.size()*3), Eigen::all) << Jc_;
    J_(Eigen::seqN(Jc_batch_.size()*3, 3), Eigen::all) << ee_Jaco_body_;
    J_(Eigen::seqN(Jc_batch_.size()*3+3, 3), Eigen::all) << ee_Jaco_rot_;

    dJ_(Eigen::seqN(0, Jc_batch_.size()*3), Eigen::all) << dJc_;
    dJ_(Eigen::seqN(Jc_batch_.size()*3, 3), Eigen::all) << ee_Jaco_body_dot_;
    dJ_(Eigen::seqN(Jc_batch_.size()*3+3, 3), Eigen::all) << ee_Jaco_rot_dot_;

    Q_(Eigen::seqN(Jc_batch_.size()*3 + 6, Jc_batch_.size()*3 + 6), Eigen::all) << J_;

//    std::cout << "Q : " << std::endl << Q_ << std::endl;


    R_(Eigen::seqN(0, Jc_batch_.size()*3 + 6), Eigen::all) << J_;
    R_(Eigen::seqN(Jc_batch_.size()*3 + 6, Jc_batch_.size()*3 + 6), Eigen::all) << dJ_;

//    std::cout << "R : " << std::endl << R_ << std::endl;


    V_ << Eigen::VectorXd::Zero(Jc_batch_.size()*3), ee_Vel_w_.e(), ee_Avel_w_.e(), Eigen::VectorXd::Zero(Jc_batch_.size()*3), ee_acc_, ee_Aacc_;

    std::cout << "V : " << std::endl << V_ << std::endl;

    N_ = Eigen::MatrixXd::Identity(raibo_arm->getDOF(), raibo_arm->getDOF()) - Q_.completeOrthogonalDecomposition().pseudoInverse()*Q_;

//    std::cout << "N : " << std::endl << N_ << std::endl;

    P_ = Eigen::MatrixXd::Identity(raibo_arm->getDOF(), raibo_arm->getDOF()) - Jc_.transpose()*(Jc_ * raibo_arm->getInverseMassMatrix().e() * Jc_.transpose()).inverse()
        * Jc_ * raibo_arm->getInverseMassMatrix().e();

//    std::cout << "P : " << std::endl << P_ << std::endl;


    K_(Eigen::all, Eigen::seqN(0, 6)) << P_*S_arm_.transpose();
    K_(Eigen::all, Eigen::seqN(6, raibo_arm->getDOF())) << -raibo_arm->getMassMatrix().e()*N_;

    std::cout << "K : " << std::endl << K_ << std::endl;

    Out_  = K_.completeOrthogonalDecomposition().pseudoInverse() *
        (raibo_arm->getMassMatrix().e() * Q_.completeOrthogonalDecomposition().pseudoInverse() * V_ +
            P_*(raibo_arm->getNonlinearities(world.getGravity()).e() - S_leg_.transpose() * raibo_arm->getGeneralizedForce().e()(Eigen::seqN(6,12))) +
            (Jc_.transpose() * (Jc_ * raibo_arm->getInverseMassMatrix().e() * Jc_.transpose()).inverse() * dJc_ - raibo_arm->getMassMatrix().e() * Q_.completeOrthogonalDecomposition().pseudoInverse() * R_) * gv_);

    std::cout << "Out_ : " << std::endl << Out_ << std::endl;


    raibo_arm->setPdTarget(jointNominalConfig, jointVelocityTarget);
    raibo_arm->setGeneralizedForce(S_arm_.transpose() * Out_(Eigen::seqN(0, 6)));
    std::cout << "Gen_Force : " << std::endl << raibo_arm->getGeneralizedForce() << std::endl;
//    std::cout <<  << std::endl;

//    raibo_arm->setGeneralizedForce(S_arm_.transpose()*raibo_arm->getNonlinearities(world.getGravity()).e().tail(6) + ee_Jaco_body_.transpose()*ee_M_app_body_.inverse()*(-10*ee_Jaco_body_*gv_));

//    GenForce_ << ee_Jaco_body_.transpose() * ((ee_Jaco_body_ * raibo_arm->getInverseMassMatrix().e() * ee_Jaco_body_.transpose()).inverse() *(
//        (posPgain_ * pos_residual_ - posDgain_ * ee_Vel_)
//        - ee_Jaco_body_dot_ * gv_
//        + ee_Jaco_body_ * raibo_arm->getInverseMassMatrix().e() * raibo_arm->getNonlinearities(world.getGravity()).e()
//    )) +
//        ee_Jaco_rot_.transpose() * ((ee_Jaco_rot_ * raibo_arm->getInverseMassMatrix().e() * ee_Jaco_rot_.transpose()).inverse() * (
//            (oriPgain_ * rot_residual_ - oriDgain_ * ee_Avel_)
//            - ee_Jaco_rot_dot_ * gv_
//            + ee_Jaco_rot_ * raibo_arm->getInverseMassMatrix().e() * raibo_arm->getNonlinearities(world.getGravity()).e()
//            ));

//    GenForce_.head(18).setZero();
//    raibo_arm->setGeneralizedForce(GenForce_);


    std::vector<Eigen::Matrix<double, 3, 24>>().swap(Jc_batch_);
    std::vector<Eigen::Matrix<double, 3, 24>>().swap(dJc_batch_);


  }


  server.killServer();
}
