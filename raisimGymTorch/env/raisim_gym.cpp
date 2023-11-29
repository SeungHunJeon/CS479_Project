//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "Environment.hpp"
#include "VectorizedEnvironment.hpp"

namespace py = pybind11;
using namespace raisim;

int THREAD_COUNT = 1;

PYBIND11_MODULE(RAISIMGYM_TORCH_ENV_NAME, m) {
  py::class_<VectorizedEnvironment<ENVIRONMENT>>(m, "RaisimGymRaiboRoughTerrain")
    .def(py::init<std::string, std::string>())
    .def("init", &VectorizedEnvironment<ENVIRONMENT>::init)
    .def("reset", &VectorizedEnvironment<ENVIRONMENT>::reset)
    .def("observe", &VectorizedEnvironment<ENVIRONMENT>::observe)
    .def("step", &VectorizedEnvironment<ENVIRONMENT>::step)
    .def("step_visualize", &VectorizedEnvironment<ENVIRONMENT>::step_visualize)
    .def("step_evaluate", &VectorizedEnvironment<ENVIRONMENT>::step_evaluate)
    .def("get_error", &VectorizedEnvironment<ENVIRONMENT>::get_error)
    .def("setSeed", &VectorizedEnvironment<ENVIRONMENT>::setSeed)
    .def("close", &VectorizedEnvironment<ENVIRONMENT>::close)
    .def("isTerminalState", &VectorizedEnvironment<ENVIRONMENT>::isTerminalState)
    .def("setSimulationTimeStep", &VectorizedEnvironment<ENVIRONMENT>::setSimulationTimeStep)
    .def("setControlTimeStep", &VectorizedEnvironment<ENVIRONMENT>::setControlTimeStep)
    .def("getObDim", &VectorizedEnvironment<ENVIRONMENT>::getObDim)
    .def("getActionDim", &VectorizedEnvironment<ENVIRONMENT>::getActionDim)
    .def("getNumOfEnvs", &VectorizedEnvironment<ENVIRONMENT>::getNumOfEnvs)
    .def("turnOnVisualization", &VectorizedEnvironment<ENVIRONMENT>::turnOnVisualization)
    .def("turnOffVisualization", &VectorizedEnvironment<ENVIRONMENT>::turnOffVisualization)
    .def("stopRecordingVideo", &VectorizedEnvironment<ENVIRONMENT>::stopRecordingVideo)
    .def("startRecordingVideo", &VectorizedEnvironment<ENVIRONMENT>::startRecordingVideo)
    .def("curriculumUpdate", &VectorizedEnvironment<ENVIRONMENT>::curriculumUpdate)
    .def("getStepDataTag", &VectorizedEnvironment<ENVIRONMENT>::getStepDataTag)
    .def("getStepData", &VectorizedEnvironment<ENVIRONMENT>::getStepData)
    .def("setCommand", &VectorizedEnvironment<ENVIRONMENT>::setCommand)
    .def("moveControllerCursor", &VectorizedEnvironment<ENVIRONMENT>::moveControllerCursor)
    .def("getState", &VectorizedEnvironment<ENVIRONMENT>::getState)
    .def("getObStatistics", &VectorizedEnvironment<ENVIRONMENT>::getObStatistics)
    .def("setObStatistics", &VectorizedEnvironment<ENVIRONMENT>::setObStatistics)
    .def("getDepthImage", &VectorizedEnvironment<ENVIRONMENT>::getDepthImage)
    .def("getColorImage", &VectorizedEnvironment<ENVIRONMENT>::getColorImage)
    .def("getSuccess", &VectorizedEnvironment<ENVIRONMENT>::getSuccess)
    .def("getIntrinsicSwitch", &VectorizedEnvironment<ENVIRONMENT>::getIntrinsicSwitch)
    .def("getPrivilegedInformation", &VectorizedEnvironment<ENVIRONMENT>::getPrivilegedInformation)
    .def("step_visualize_success", &VectorizedEnvironment<ENVIRONMENT>::step_visualize_success)
    .def("observationDeNormalize", &VectorizedEnvironment<ENVIRONMENT>::ObservationDeNormalize)
    .def("getContact", &VectorizedEnvironment<ENVIRONMENT>::getContact)
    .def("getCameraPose", &VectorizedEnvironment<ENVIRONMENT>::getCameraPose)
    .def("getAnchorHistory", &VectorizedEnvironment<ENVIRONMENT>::getAnchorHistory);
//    .def("synchronize", &VectorizedEnvironment<ENVIRONMENT>::synchronize);

//  py::class_<VectorizedEnvironment<ENVIRONMENT_ROLLOUT>>(m, "RaisimGymRaiboRoughTerrain_ROLLOUT")
//      .def(py::init<std::string, std::string>())
//      .def("init", &VectorizedEnvironment<ENVIRONMENT_ROLLOUT>::init)
//      .def("reset", &VectorizedEnvironment<ENVIRONMENT_ROLLOUT>::reset)
//      .def("observe", &VectorizedEnvironment<ENVIRONMENT_ROLLOUT>::observe)
//      .def("step", &VectorizedEnvironment<ENVIRONMENT_ROLLOUT>::step)
//      .def("step_visualize", &VectorizedEnvironment<ENVIRONMENT_ROLLOUT>::step_visualize)
//      .def("setSeed", &VectorizedEnvironment<ENVIRONMENT_ROLLOUT>::setSeed)
//      .def("close", &VectorizedEnvironment<ENVIRONMENT_ROLLOUT>::close)
//      .def("isTerminalState", &VectorizedEnvironment<ENVIRONMENT_ROLLOUT>::isTerminalState)
//      .def("setSimulationTimeStep", &VectorizedEnvironment<ENVIRONMENT_ROLLOUT>::setSimulationTimeStep)
//      .def("setControlTimeStep", &VectorizedEnvironment<ENVIRONMENT_ROLLOUT>::setControlTimeStep)
//      .def("getObDim", &VectorizedEnvironment<ENVIRONMENT_ROLLOUT>::getObDim)
//      .def("getActionDim", &VectorizedEnvironment<ENVIRONMENT_ROLLOUT>::getActionDim)
//      .def("getNumOfEnvs", &VectorizedEnvironment<ENVIRONMENT_ROLLOUT>::getNumOfEnvs)
//      .def("turnOnVisualization", &VectorizedEnvironment<ENVIRONMENT_ROLLOUT>::turnOnVisualization)
//      .def("turnOffVisualization", &VectorizedEnvironment<ENVIRONMENT_ROLLOUT>::turnOffVisualization)
//      .def("stopRecordingVideo", &VectorizedEnvironment<ENVIRONMENT_ROLLOUT>::stopRecordingVideo)
//      .def("startRecordingVideo", &VectorizedEnvironment<ENVIRONMENT_ROLLOUT>::startRecordingVideo)
//      .def("curriculumUpdate", &VectorizedEnvironment<ENVIRONMENT_ROLLOUT>::curriculumUpdate)
//      .def("getStepDataTag", &VectorizedEnvironment<ENVIRONMENT_ROLLOUT>::getStepDataTag)
//      .def("getStepData", &VectorizedEnvironment<ENVIRONMENT_ROLLOUT>::getStepData)
//      .def("setCommand", &VectorizedEnvironment<ENVIRONMENT_ROLLOUT>::setCommand)
//      .def("moveControllerCursor", &VectorizedEnvironment<ENVIRONMENT_ROLLOUT>::moveControllerCursor)
//      .def("getState", &VectorizedEnvironment<ENVIRONMENT_ROLLOUT>::getState)
//      .def("getObStatistics", &VectorizedEnvironment<ENVIRONMENT_ROLLOUT>::getObStatistics)
//      .def("setObStatistics", &VectorizedEnvironment<ENVIRONMENT_ROLLOUT>::setObStatistics)
//      .def("getDepthImage", &VectorizedEnvironment<ENVIRONMENT_ROLLOUT>::getDepthImage)
//      .def("getColorImage", &VectorizedEnvironment<ENVIRONMENT_ROLLOUT>::getColorImage)
//      .def("synchronize", &VectorizedEnvironment<ENVIRONMENT_ROLLOUT>::synchronize)
//      .def("step_Rollout", &VectorizedEnvironment<ENVIRONMENT_ROLLOUT>::step_Rollout)
//      .def("getState_Rollout", &VectorizedEnvironment<ENVIRONMENT_ROLLOUT>::getState_Rollout)
//      .def("get_target_pos", &VectorizedEnvironment<ENVIRONMENT_ROLLOUT>::get_target_pos)
//      .def("get_obj_pos", &VectorizedEnvironment<ENVIRONMENT_ROLLOUT>::get_obj_pos)
//      .def("observe_Rollout", &VectorizedEnvironment<ENVIRONMENT_ROLLOUT>::observe_Rollout)
//      .def("predict_obj_update", &VectorizedEnvironment<ENVIRONMENT_ROLLOUT>::predict_obj_update)
//      .def("observationDeNormalize", &VectorizedEnvironment<ENVIRONMENT_ROLLOUT>::ObservationDeNormalize)
//      ;

  py::class_<NormalSampler>(m, "NormalSampler")
      .def(py::init<int>(), py::arg("dim"))
      .def("seed", &NormalSampler::seed)
      .def("sample", &NormalSampler::sample);

  py::class_<DiscreteSampler>(m, "DiscreteSampler")
      .def(py::init<int>(), py::arg("dim"))
      .def("seed", &DiscreteSampler::seed)
      .def("sample", &DiscreteSampler::sample);
}
