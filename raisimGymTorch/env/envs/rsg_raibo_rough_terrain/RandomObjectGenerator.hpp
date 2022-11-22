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

  RandomObjectGenerator() = default;

  void setSeed(int seed) {
    object_seed_ = seed;
  }

  double get_height () {
    return object_height;
  }

  void Inertial_Randomize(raisim::SingleBodyObject* object, double bound_ratio, double curriculumFactor, std::mt19937& gen, std::uniform_real_distribution<double>& uniDist, std::normal_distribution<double>& normDist) {
    /// Mass randomization (1-bound_ratio ~ 1+bound_ratio)
    double Mass = object->getMass()*(1 + curriculumFactor*bound_ratio * normDist(gen));
    object->setMass(Mass);

    /// Inertia randomization (1-bound_ratio ~ 1+bound_ratio)
    Eigen::Matrix3d inertia_residual;
    inertia_residual.setOnes();
    for (int i=0; i<9; i++)
      inertia_residual(i) += curriculumFactor*bound_ratio * normDist(gen);
    Eigen::Matrix3d Inertia = object->getInertiaMatrix_B().cwiseProduct(inertia_residual);
    object->setInertia(Inertia);

    /// COM randomization (1-bound_ratio ~ 1+bound_ratio)
    Eigen::Vector3d COM_residual;
    COM_residual.setOnes();
    for (int i=0; i<3; i++) {
      COM_residual(i) += curriculumFactor*bound_ratio * normDist(gen);
    }
    Eigen::Vector3d COM = object->getCom().e().cwiseProduct(COM_residual);
    object->setCom(COM);
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
      case ObjectShape::BALL:
      {
        double radius_ = radius * (1 + curriculumFactor * bound_ratio * normDist(gen));
        object_height = 2*radius_;
        return world->addSphere(radius_, mass, "default", raisim::COLLISION(1), raisim::COLLISION(1) | raisim::COLLISION(63));
      }


      case ObjectShape::Cylinder:
      {
        double radius_ = radius * (1 + curriculumFactor * bound_ratio * normDist(gen));
        double height_ = height * (1 + curriculumFactor * bound_ratio * normDist(gen));
        object_height = height;
        return world->addCylinder(radius_, height_, mass, "default", raisim::COLLISION(1), raisim::COLLISION(1) | raisim::COLLISION(63));
      }


      case ObjectShape::BOX:
      {
        double width1_ = 2 * width1 * (1 + curriculumFactor * bound_ratio * normDist(gen));
        double width2_ = 2 * width2 * (1 + curriculumFactor * bound_ratio * normDist(gen));
        double height_ = height * (1 + curriculumFactor * bound_ratio * normDist(gen));
        object_height = height_;
        return world->addBox(width1_, width2_, height_, mass, "default", raisim::COLLISION(1), raisim::COLLISION(1) | raisim::COLLISION(63));
      }



      case ObjectShape::Capsule:
      {
        double radius_ = radius * (1 + curriculumFactor * bound_ratio * normDist(gen));
        double height_ = height * (1 + curriculumFactor * bound_ratio * normDist(gen));
        object_height = height_;
        return world->addCapsule(radius_, height_, mass, "default", raisim::COLLISION(1), raisim::COLLISION(1) | raisim::COLLISION(63));
      }

    }
    return nullptr;
  }

 private:
  int object_seed_;
  double object_height;
};

}

#endif //_RAISIM_GYM_TORCH_ENV_ENVS_RSG_RAIBO_ROUGH_TERRAIN_RANDOMOBJECTGENERATOR_HPP_
