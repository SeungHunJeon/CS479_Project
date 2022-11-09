// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.

#include "raisim/RaisimServer.hpp"
#include "raisim/World.hpp"
#if WIN32
#include <timeapi.h>
#endif
#include "unsupported/Eigen/MatrixFunctions"
Eigen::Matrix3d hat(const Eigen::Vector3d R3) {
  Eigen::Matrix3d so3;
  so3 << 0, -R3(2), R3(1),
      R3(2), 0, -R3(0),
      -R3(1), R3(0), 0;

  return so3;
}

Eigen::Vector3d vee(const Eigen::Matrix3d so3) {
  Eigen::Vector3d R3;
  R3 << so3(2,1), -so3(2,0), so3(1,0);

  return R3;
}

Eigen::Matrix3d log(const Eigen::Matrix3d SO3) {
  Eigen::Matrix3d so3;
  so3 = SO3.log();

  return so3;
}

Eigen::Vector3d LOG(const Eigen::MatrixXd SO3) {
  Eigen::Vector3d R3;
  R3 = vee(SO3.log());

  return R3;
}

Eigen::Matrix3d exp(const Eigen::Matrix3d so3) {
  Eigen::Matrix3d SO3;
  SO3 = so3.exp();

  return SO3;
}

Eigen::Matrix3d EXP(const Eigen::Vector3d R3) {
  Eigen::Matrix3d SO3;
  SO3 = hat(R3).exp();

  return SO3;
}

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
  world.addGround();
  auto raibo_arm = world.addArticulatedSystem(binaryPath.getDirectory() + "\\rsc\\arm_2\\urdf\\arm_2.urdf");
  Eigen::VectorXd jointPgain(raibo_arm->getDOF()), jointDgain(raibo_arm->getDOF());

  int controlmode = 1;

  switch(controlmode) {
    case 0:
      raibo_arm->setControlMode(raisim::ControlMode::FORCE_AND_TORQUE);
      break;

    case 1:
      jointPgain.setZero();
      jointDgain.setZero();
      jointPgain << 0, 0, 0, 0, 0, 0;
      jointDgain << 0, 0, 0, 0, 0, 0;
      raibo_arm->setPdGains(jointPgain, jointDgain);
      break;
  }

  /// joint PD controller
  Eigen::VectorXd jointNominalConfig(raibo_arm->getGeneralizedCoordinateDim()), jointVelocityTarget(raibo_arm->getDOF());
  jointNominalConfig << -0.5, 1.0, 0, 2.3, 0, 0;
  jointVelocityTarget.setZero();



  raibo_arm->setGeneralizedCoordinate(jointNominalConfig);
  raibo_arm->setGeneralizedForce(Eigen::VectorXd::Zero(raibo_arm->getDOF()));




//  raibo_arm->setPdTarget(jointNominalConfig, jointVelocityTarget);
  raibo_arm->setName("raibo_arm");

  Eigen::MatrixXd ee_Jaco_body_, ee_Jaco_rot_;
  Eigen::MatrixXd ee_Jaco_body_dot_, ee_Jaco_rot_dot_;

  Eigen::MatrixXd J_;
  Eigen::MatrixXd dJ_;

  Eigen::MatrixXd raibo_Mass_, raibo_Mass_simp_;
  Eigen::Vector3d ee_Vel_, ee_Avel_;
  Eigen::Vector3d ee_acc_, ee_Aacc_;
  raisim::Vec<3> ee_Vel_w_, ee_Avel_w_;
  Eigen::VectorXd gc_, gv_;
  Eigen::VectorXd Gen_Force_;
  Eigen::VectorXd du_;

  raisim::SparseJacobian ee_Jaco_body_s_, ee_Jaco_rot_s_;
  raisim::SparseJacobian ee_Jaco_body_dot_s_, ee_Jaco_rot_dot_s_;
  raisim::Vec<3> point_;
  raisim::Mat<3,3> Ori_;

  ee_Jaco_body_.setZero(3, raibo_arm->getDOF());
  ee_Jaco_rot_.setZero(3, raibo_arm->getDOF());
  J_.setZero(6, raibo_arm->getDOF());
  dJ_.setZero(6, raibo_arm->getDOF());
  ee_Jaco_body_dot_.setZero(3, raibo_arm->getDOF());
  ee_Jaco_rot_dot_.setZero(3, raibo_arm->getDOF());
  Gen_Force_.setZero(raibo_arm->getDOF());
  du_.setZero(raibo_arm->getDOF());

  /// launch raisim server
  raisim::RaisimServer server(&world);
  server.launchServer();
  server.focusOn(raibo_arm);

  auto sphere_ = server.addVisualSphere("body_pos_check", 0.01, 1, 0, 0, 1.0);

  ee_acc_.setZero();
  ee_Aacc_.setZero();
  gc_.resize(raibo_arm->getGeneralizedCoordinateDim());
  gv_.resize(raibo_arm->getDOF());
  ee_Vel_.setZero();

  int Kp = 50;
  int Kd = 4;


  sleep(5);

  int dof_ = 0;

  for (int i=0; i<2000000; i++) {
    std::this_thread::sleep_for(std::chrono::microseconds(100));
    server.integrateWorldThreadSafe();

    raibo_arm->getState(gc_, gv_);
    ee_Jaco_body_dot_.setZero();
    ee_Jaco_rot_dot_.setZero();
    raisim::Vec<4> quat;

    raibo_arm->getBodyPosition(raibo_arm->getBodyIdx("link6"), point_);
    raibo_arm->getDenseJacobian(raibo_arm->getBodyIdx("link6"), point_, ee_Jaco_body_);
    raibo_arm->getDenseRotationalJacobian(raibo_arm->getBodyIdx("link6"), ee_Jaco_rot_);

    raibo_arm->getTimeDerivativeOfSparseJacobian(raibo_arm->getBodyIdx("link6"), raisim::ArticulatedSystem::WORLD_FRAME, point_, ee_Jaco_body_dot_s_);
    raibo_arm->getTimeDerivativeOfSparseRotationalJacobian(raibo_arm->getBodyIdx("link6"), ee_Jaco_rot_dot_s_);
    raibo_arm->getPosition(raibo_arm->getBodyIdx("link6"), point_);
    raibo_arm->getOrientation(raibo_arm->getBodyIdx("link6"), Ori_);


    sphere_->setPosition(point_.e());

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
    raibo_arm->getVelocity(raibo_arm->getBodyIdx("link6"), ee_Vel_w_);
////
    raibo_arm->getAngularVelocity(raibo_arm->getBodyIdx("link6"), ee_Avel_w_);
//
    raibo_arm->getMassMatrix();
//

    J_(Eigen::seqN(0, 3), Eigen::all) << ee_Jaco_body_;
    J_(Eigen::seqN(3, 3), Eigen::all) << ee_Jaco_rot_;

    dJ_(Eigen::seqN(0, 3), Eigen::all) << ee_Jaco_body_dot_;
    dJ_(Eigen::seqN(3, 3), Eigen::all) << ee_Jaco_rot_dot_;



//    Gen_Force_ << raibo_arm->getMassMatrix().e() * du_ + raibo_arm->getNonlinearities(world.getGravity()).e();


    Eigen::VectorXd Ext_F_;
    Eigen::Vector3d p_des;
    switch (dof_) {

      case 0:
        Ext_F_.setZero(3);
        Ext_F_ << 0, 0, 0;
//        Gen_Force_ << (J_(Eigen::seqN(0,3), Eigen::all)*raibo_arm->getInverseMassMatrix().e()).completeOrthogonalDecomposition().pseudoInverse() * (
//            (J_(Eigen::seqN(0,3), Eigen::all)*raibo_arm->getInverseMassMatrix().e()*raibo_arm->getNonlinearities(world.getGravity()).e()) +
//                (-dJ_(Eigen::seqN(0,3), Eigen::all)*gv_) +
//                (-J_(Eigen::seqN(0,3), Eigen::all)*raibo_arm->getInverseMassMatrix().e()*J_(Eigen::seqN(0,3), Eigen::all).transpose()*Ext_F_)
//        );

        p_des << 0.5,0,0.3;
        Gen_Force_ << ee_Jaco_body_.transpose() * (ee_Jaco_body_*raibo_arm->getInverseMassMatrix().e()*ee_Jaco_body_.transpose()).inverse() * (Kp*(p_des - point_.e()) - Kd*ee_Vel_w_.e()) +
        ee_Jaco_rot_.transpose() * (ee_Jaco_rot_ * raibo_arm->getInverseMassMatrix().e() * ee_Jaco_rot_.transpose()).inverse() * (- Kd*ee_Avel_w_.e()) +
        raibo_arm->getNonlinearities(world.getGravity()).e();

//        Gen_Force_ << raibo_arm->getMassMatrix().e() * (Kp*(jointNominalConfig-gc_) - Kd*gv_) + raibo_arm->getNonlinearities(world.getGravity()).e();

//        Gen_Force_ << raibo_arm->getNonlinearities(world.getGravity()).e() - J_(Eigen::seqN(0,3), Eigen::all).transpose() * Ext_F_;

        break;

      case 1:
        Ext_F_.setZero(6);
        Ext_F_ << 10, 10, 10, 0, 0, 0;
        Gen_Force_ << (J_*raibo_arm->getInverseMassMatrix().e()).inverse() * (
            (J_*raibo_arm->getInverseMassMatrix().e()*raibo_arm->getNonlinearities(world.getGravity()).e()) +
                (-dJ_*gv_) +
                (-J_*raibo_arm->getInverseMassMatrix().e()*J_.transpose()*Ext_F_)
        );
        break;

      default:
        Gen_Force_.setZero();
        break;
    }

//    std::cout << "Given GenForce : " << std::endl << Gen_Force_ << std::endl;

    std::cout << "position : " << point_ << std::endl;

    std::cout << "Orientaiton : " << Ori_ << std::endl;

    raibo_arm->setGeneralizedForce(Gen_Force_);

//    std::cout << "Gen Force : " << std::endl << raibo_arm->getGeneralizedForce() << std::endl;

  }


  server.killServer();
}
