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
    classify_vector.setZero(4);
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

  void Inertial_Randomize(raisim::SingleBodyObject* object, double bound_ratio, double curriculumFactor, std::mt19937& gen, std::uniform_real_distribution<double>& uniDist, std::normal_distribution<double>& normDist) {
    /// Mass randomization (1-bound_ratio ~ 1+bound_ratio)
    double ratio = (1 + curriculumFactor * bound_ratio * normDist(gen));
    while (ratio < 0.5)
      ratio = (1 + curriculumFactor * bound_ratio * normDist(gen));

    double Mass = object->getMass()*ratio;
    object->setMass(Mass);

    /// Inertia randomization (1-bound_ratio ~ 1+bound_ratio)
    Eigen::Matrix3d inertia_residual;
    inertia_residual.setOnes();
    for (int i=0; i<9; i++) {
      if(i % 3 == 0)
        inertia_residual(i) += 0.1 + curriculumFactor*bound_ratio * uniDist(gen);
      else
        inertia_residual(i) += 0.1*curriculumFactor*bound_ratio * normDist(gen);
    }

    Eigen::Matrix3d Inertia = object->getInertiaMatrix_B().cwiseProduct(inertia_residual);
    object->setInertia(Inertia);

    /// COM randomization (1-bound_ratio ~ 1+bound_ratio)
    Eigen::Vector3d COM_residual;
    COM_residual.setZero();
    for (int i=0; i<3; i++) {
      COM_residual(i) += 0.2*curriculumFactor*bound_ratio * normDist(gen);
    }
    Eigen::Vector3d COM = COM_residual;
    object->setCom(COM);
//    RSINFO(COM)
//    RSINFO(object->getCom().e())
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
    switch (object_shape) {
      case ObjectShape::BALL: /// 0
      {
        double ratio = (1 + curriculumFactor * bound_ratio * normDist(gen));
        while (ratio < 0.5)
          ratio = (1 + curriculumFactor * bound_ratio * normDist(gen));
        double radius_ = radius * ratio;
        object_height = 2*radius_;
        classify_vector << 1, 0, 0, 0;
        geometry << 2*radius_, 2*radius_, 2*radius_;
        return world->addSphere(radius_, mass, "default", raisim::COLLISION(1), raisim::COLLISION(1) | raisim::COLLISION(63));
      }


      case ObjectShape::Cylinder: /// 1
      {
        double ratio = (1 + curriculumFactor * bound_ratio * normDist(gen));
        while (ratio < 0.5)
          ratio = (1 + curriculumFactor * bound_ratio * normDist(gen));

        double height_ratio = (1 + curriculumFactor * bound_ratio * normDist(gen));
        while (height_ratio < 0.5)
          height_ratio = (1 + curriculumFactor * bound_ratio * normDist(gen));

        double radius_ = radius * ratio;
        double height_ = height * height_ratio;
        object_height = height;
        classify_vector << 0, 1, 0, 0;
        geometry << 2*radius_, 2*radius_, height_;
        return world->addCylinder(radius_, height_, mass, "default", raisim::COLLISION(1), raisim::COLLISION(1) | raisim::COLLISION(63));
      }


      case ObjectShape::BOX: /// 2
      {
        double width_ratio_1 = (1 + curriculumFactor * bound_ratio * normDist(gen));
        while (width_ratio_1 < 0.5)
          width_ratio_1 = (1 + curriculumFactor * bound_ratio * normDist(gen));

        double width_ratio_2 = (1 + curriculumFactor * bound_ratio * normDist(gen));
        while (width_ratio_2 < 0.5)
          width_ratio_2 = (1 + curriculumFactor * bound_ratio * normDist(gen));

        double height_ratio = (1 + curriculumFactor * bound_ratio * normDist(gen));
        while (height_ratio < 0.5)
          height_ratio = (1 + curriculumFactor * bound_ratio * normDist(gen));

        double width1_ = width1 * width_ratio_1;
        double width2_ = width2 * width_ratio_2;
        double height_ = height * height_ratio;
        object_height = height_;
        classify_vector << 0, 0, 1, 0;
        geometry << width1_, width2_, height_;
        return world->addBox(width1_, width2_, height_, mass, "default", raisim::COLLISION(1), raisim::COLLISION(1) | raisim::COLLISION(63));
      }



      case ObjectShape::Capsule: /// 3
      {
        double radius_ = radius * (1 + curriculumFactor * bound_ratio * normDist(gen));
        double height_ = height * (1 + curriculumFactor * bound_ratio * normDist(gen));
        object_height = 2 * radius_ + height_;
        classify_vector << 0, 0, 0, 1;
        geometry << 2*radius_, 2*radius_, height_+2*radius_;
        return world->addCapsule(radius_, height_, mass, "default", raisim::COLLISION(1), raisim::COLLISION(1) | raisim::COLLISION(63));
      }

    }
    return nullptr;
  }

 private:
  int object_seed_;
  double object_height;
  Eigen::VectorXd classify_vector;
  Eigen::VectorXd geometry; // width, width, height
};

}

#endif //_RAISIM_GYM_TORCH_ENV_ENVS_RSG_RAIBO_ROUGH_TERRAIN_RANDOMOBJECTGENERATOR_HPP_
