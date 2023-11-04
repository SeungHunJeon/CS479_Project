//
// Created by oem on 22. 11. 18.
//

#ifndef _RAISIM_GYM_TORCH_ENV_ENVS_RSG_RAIBO_ROUGH_TERRAIN_RANDOMOBJECTGENERATOR_HPP_
#define _RAISIM_GYM_TORCH_ENV_ENVS_RSG_RAIBO_ROUGH_TERRAIN_RANDOMOBJECTGENERATOR_HPP_


#include "raisim/World.hpp"

namespace raisim {

class RandomObjectGenerator {
 public:

  enum class ObjectShape : int {
    BALL = 0,
    Cylinder = 1,
    BOX = 2,
    Capsule = 3
  };

  RandomObjectGenerator() {
    classify_vector.setZero(3);
    geometry.setZero(3);
  }

  void setSeed(int seed) {
    object_seed_ = seed;
  }

  double get_height () {
    return object_height;
  }

  double get_dist () {
    return geometry.segment(0,2).norm();
  }

  Eigen::VectorXd get_geometry () {
    return geometry;
  }

  Eigen::VectorXd get_classify_vector () {
    return classify_vector;
  }

  inline double soft_sample (double reference,
                             double bound_ratio,
                             double curriculumFactor,
                             std::mt19937& gen,
                             std::normal_distribution<double>& normDist
                             ) {

    double value;
    value = bound_ratio * reference * curriculumFactor * normDist(gen);
//    while (true)
//    {
//      value = bound_ratio * reference * curriculumFactor * normDist(gen); /// 0.5 * 1 * 0.2 * normDist
//      if (std::abs(value) < bound_ratio * reference * curriculumFactor)
//        break;
//    }
    return std::clamp(value, -bound_ratio*reference*curriculumFactor, bound_ratio*reference*curriculumFactor);
  }

  inline Eigen::Matrix3d sample_inertia (raisim::SingleBodyObject* object,
                                         double bound_ratio,
                                         double curriculumFactor,
                                         std::mt19937& gen,
                                         std::uniform_real_distribution<double>& uniDist,
                                         std::normal_distribution<double>& normDist)
  {
    Eigen::Matrix3d nominalI_ = object->getInertiaMatrix_B();
//    Eigen::EigenSolver<Eigen::Matrix3d> eigen_solver(nominalI_);
//    auto eigenvalue = eigen_solver.eigenvalues();
//    auto Dx = eigenvalue[0].real();
//    auto Dy = eigenvalue[1].real();
//    auto Dz = eigenvalue[2].real();
    auto Lx = 1.0 / 2 * nominalI_.trace() - nominalI_(0,0);
    auto Ly = 1.0 / 2 * nominalI_.trace() - nominalI_(1,1);
    auto Lz = 1.0 / 2 * nominalI_.trace() - nominalI_(2,2);


    Lx += soft_sample(Lx, bound_ratio, curriculumFactor, gen, normDist);
    Ly += soft_sample(Ly, bound_ratio, curriculumFactor, gen, normDist);
    Lz += soft_sample(Lz, bound_ratio, curriculumFactor, gen, normDist);

    Eigen::Vector3d L, D;

    L << Lx, Ly, Lz;
    D << L(1) + L(2), L(0) + L(2), L(0) + L(1);

    raisim::Vec<3> rpy;

    double r, p, y;
    r = soft_sample(10*M_PI/180, bound_ratio, curriculumFactor, gen, normDist); /// -5 ~ 5
    p = soft_sample(10*M_PI/180, bound_ratio, curriculumFactor, gen, normDist);
    y = soft_sample(10*M_PI/180, bound_ratio, curriculumFactor, gen, normDist);

    rpy.e() << r,p,y;
    raisim::Mat<3,3> rotMat;
    raisim::rpyToRotMat_extrinsic(rpy, rotMat);

    return rotMat.e() * D.asDiagonal() * rotMat.e().transpose();
  }

  inline Eigen::VectorXd sample_com (double bound_ratio,
                                       double curriculumFactor,
                                       std::mt19937& gen,
                                       std::uniform_real_distribution<double>& uniDist,
                                       std::normal_distribution<double>& normDist) {
    Eigen::Vector3d COM;

    switch (object_shape_) {
      case ObjectShape::BALL: /// 0
      {
        double radius, theta, phi;
        double obj_radius_ = geometry[0]/2;

        radius = soft_sample(obj_radius_, bound_ratio, curriculumFactor, gen, normDist);
        radius = std::abs(radius);
        theta = 2*M_PI*uniDist(gen);
        phi = M_PI*(uniDist(gen)-0.5);

//        COM << radius * sin(phi) * cos(theta),
//        radius * sin(phi) * sin(theta),
//        radius * cos(phi);
        COM << 0, 0, 0;

        return COM;
      }

      case ObjectShape::Cylinder: /// 1
      {
        double radius, phi, z;
        double obj_radius_ = geometry[0]/2;
        double obj_height_ = geometry[2]/2;

        radius = soft_sample(obj_radius_, bound_ratio, curriculumFactor, gen, normDist);
        radius = std::abs(radius);
        z = soft_sample(obj_height_, bound_ratio, curriculumFactor, gen, normDist);
        phi = 2*M_PI*uniDist(gen);

        COM << radius * cos(phi),
        radius * sin(phi),
        z;

        return COM;
      }

      case ObjectShape::BOX: /// 2
      {
        double x, y, z;
        double obj_x = geometry[0]/2;
        double obj_y = geometry[1]/2;
        double obj_z = geometry[2]/2;

        x = soft_sample(obj_x, bound_ratio, curriculumFactor, gen, normDist);
        y = soft_sample(obj_y, bound_ratio, curriculumFactor, gen, normDist);
        z = soft_sample(obj_z, bound_ratio, curriculumFactor, gen, normDist);

        COM << x,
        y,
        z;

        return COM;
      }

//      case ObjectShape::Capsule: /// 3
//      {
//        return
//      }
    }
  }

  void Inertial_Randomize(raisim::SingleBodyObject* object) {
    object->setMass(Mass);
    object->setInertia(Inertia);
    object->setCom(COM);
  }

  void Inertial_Randomize(raisim::SingleBodyObject* object,
                          double bound_ratio,
                          double curriculumFactor,
                          std::mt19937& gen,
                          std::uniform_real_distribution<double>& uniDist,
                          std::normal_distribution<double>& normDist) {
    /// Mass randomization (1-bound_ratio ~ 1+bound_ratio)
    double mass_ratio = 1;
    mass_ratio += soft_sample(mass_ratio, bound_ratio, curriculumFactor, gen, normDist);

    Mass = object->getMass()*mass_ratio;
    object->setMass(Mass);

    Inertia = sample_inertia(object, curriculumFactor, bound_ratio, gen, uniDist, normDist);
    object->setInertia(Inertia);

    /// COM randomization (1-bound_ratio ~ 1+bound_ratio)

    COM = sample_com(bound_ratio,
                     curriculumFactor,
                     gen,
                     uniDist,
                     normDist);

    object->setCom(COM);
  }

  void compute_ratio(double bound_ratio,
                     double curriculumFactor,
                     std::mt19937& gen,
                     std::normal_distribution<double>& normDist){
    ratio = 1;
    height_ratio = 1;
    width_ratio_1 = 1;
    width_ratio_2 = 1;
    ratio += soft_sample(ratio, bound_ratio, curriculumFactor, gen, normDist);
    height_ratio += soft_sample(1, bound_ratio, curriculumFactor, gen, normDist);
    width_ratio_1 += soft_sample(width_ratio_1, bound_ratio, curriculumFactor, gen, normDist);
    width_ratio_2 += soft_sample(width_ratio_2, bound_ratio, curriculumFactor, gen, normDist);
  }

  void get_ratio(double &ratio_,
                 double &height_ratio_,
                 double &width_ratio_1_,
                 double &width_ratio_2_) {
    ratio_ = ratio;
    height_ratio_ = height_ratio;
    width_ratio_1_ = width_ratio_1;
    width_ratio_2_ = width_ratio_2;
  }

  raisim::SingleBodyObject* generateObject(raisim::World* world, const double &mass, int &idx)
  {
    switch (object_shape_) {
      case ObjectShape::BALL:
      {
        auto Obj = world->addSphere(radius_, mass, "object");
        Obj->setName("Object");
        return Obj;
      }

      case ObjectShape::Cylinder:
      {
        auto Obj = world->addCylinder(radius_, height_, mass, "object");
        Obj->setName("Object");
        return Obj;
      }

      case ObjectShape::BOX:
      {
        auto Obj = world->addBox(width1_, width2_, height_, mass, "object");
        Obj->setName("Object");
        return Obj;
      }
    }
  }

  raisim::SingleBodyObject* generateObject(raisim::World* world,
                                 ObjectShape object_shape,
                                 double curriculumFactor,
                                 std::mt19937& gen,
                                 std::uniform_real_distribution<double>& uniDist,
                                 std::normal_distribution<double>& normDist,
                                 double bound_ratio,
                                 const double &mass,
                                 const double &radius,
                                 const double &height,
                                 const double &width1,
                                 const double &width2) {
    object_shape_ = object_shape;
    switch (object_shape) {
      case ObjectShape::BALL: /// 0
      {
        compute_ratio(bound_ratio, curriculumFactor, gen, normDist);
        radius_ = radius * ratio;
        object_height = 2*radius_;
        classify_vector << 1, 0, 0;
        geometry << 2*radius_, 2*radius_, 2*radius_;
//        return world->addSphere(radius_, mass, "object", raisim::COLLISION(1), raisim::COLLISION(1) | raisim::COLLISION(63));
        auto Obj = world->addSphere(radius_, mass, "object");

        Obj->setName("Object");
        return Obj;
      }


      case ObjectShape::Cylinder: /// 1
      {
        compute_ratio(bound_ratio, curriculumFactor, gen, normDist);
        radius_ = radius * ratio;
        height_ = height * height_ratio;
        object_height = height_;
        classify_vector << 0, 1, 0;
        geometry << 2*radius_, 2*radius_, height_;
        auto Obj =  world->addCylinder(radius_, height_, mass, "object", raisim::COLLISION(1), raisim::COLLISION(1) | raisim::COLLISION(63));
//        auto Obj = world->addCylinder(radius_, height_, mass, "object");
        Obj->setName("Object");

        return Obj;
      }


      case ObjectShape::BOX: /// 2
      {
        compute_ratio(bound_ratio, curriculumFactor, gen, normDist);
        width1_ = width1 * width_ratio_1;
        width2_ = width2 * width_ratio_2;
        height_ = height * height_ratio;

        object_height = height_;
        classify_vector << 0, 0, 1;
        geometry << width1_, width2_, height_;
        auto Obj = world->addBox(width1_, width2_, height_, mass, "object", raisim::COLLISION(1), raisim::COLLISION(1) | raisim::COLLISION(63));
//        auto Obj = world->addBox(width1_, width2_, height_, mass, "object");
        Obj->setName("Object");

        return Obj;
      }



//      case ObjectShape::Capsule: /// 3
//      {
//        double radius_ = radius * (1 + curriculumFactor * bound_ratio * normDist(gen));
//        double height_ = height * (1 + curriculumFactor * bound_ratio * normDist(gen));
//        object_height = 2 * radius_ + height_;
//        classify_vector << 0, 0, 0, 1;
//        geometry << 2*radius_, 2*radius_, height_+2*radius_;
//        return world->addCapsule(radius_, height_, mass, "object", raisim::COLLISION(1), raisim::COLLISION(1) | raisim::COLLISION(63));
//      }

    }
    return nullptr;
  }

 private:
  double Mass;
  Eigen::Vector3d COM;
  Eigen::Matrix3d Inertia;
  double ratio, height_ratio, width_ratio_1, width_ratio_2;
  double height_, width2_, width1_, radius_;
  int object_seed_;
  double object_height;
  ObjectShape object_shape_;
  Eigen::VectorXd classify_vector;
  Eigen::VectorXd geometry; // width, width, height
};

}

#endif //_RAISIM_GYM_TORCH_ENV_ENVS_RSG_RAIBO_ROUGH_TERRAIN_RANDOMOBJECTGENERATOR_HPP_
