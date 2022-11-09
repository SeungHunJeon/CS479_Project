cmake_minimum_required(VERSION 3.5)
project(raisim_examples LANGUAGES CXX)

#==
# Dependencies
#==
set (workspace "${CMAKE_CURRENT_SOURCE_DIR}/../..")
#set(low "${workspace}/default_controller_demo")
set(plot_dir "${workspace}/matplotplusplus")
set(qpsolver_dir "${workspace}/qpSWIFT")

find_package(raisim CONFIG REQUIRED)
find_package(Eigen3 3.4 REQUIRED HINTS ${Eigen3_HINT})
find_package(qpSWIFT)
#add_subdirectory(${low} ${CMAKE_BINARY_DIR}/low_level)
#add_subdirectory(${plot_dir} ${CMAKE_BINARY_DIR}/matplot)
add_subdirectory(${plot_dir} ${CMAKE_BINARY_DIR}/matplotplusplus)
message(SISISIBAL)
message(${CMAKE_BINARY_DIR})
add_subdirectory(${qpsolver_dir} ${CMAKE_BINARY_DIR}/qpSWIFT)

if(WIN32)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/../raisim/win32/mt_release/bin")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/../raisim/win32/mt_debug/bin")
    file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/../rsc" DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE})
    file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/../rsc" DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG})
else()
    file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/../rsc" DESTINATION ${CMAKE_BINARY_DIR}/examples)
endif()

include_directories ( ${EIGEN3_INCLUDE_DIRS} )

#==
# Build Targets
#==

function(create_executable app_name file_name)
    add_executable(${app_name} ${file_name})
    set_target_properties(${app_name} PROPERTIES MACOSX_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/../raisim/mac/lib")
    if(WIN32)
        target_link_libraries(${app_name} PUBLIC raisim::raisim Ws2_32 Winmm)
        target_compile_options(${app_name} PRIVATE "/MP")
    else()
        target_link_libraries(${app_name} PUBLIC raisim::raisim pthread)
    endif()

    if(APPLE)
        execute_process(COMMAND sysctl -q hw.optional.arm64
                OUTPUT_VARIABLE _sysctl_stdout
                ERROR_VARIABLE _sysctl_stderr
                RESULT_VARIABLE _sysctl_result
                )
        if(_sysctl_result EQUAL 0 AND _sysctl_stdout MATCHES "hw.optional.arm64: 1")
            target_compile_options(${app_name} PRIVATE -mcpu=apple-m1)
        endif()
    endif()

    target_include_directories(${app_name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
#    target_include_directories(${app_name} PUBLIC ${low}/module/controller/raibot_position_controller_sim)
    target_link_libraries(${app_name} PUBLIC matplot)
    target_link_libraries(${app_name} PRIVATE qpSWIFT::qpSWIFT-shared)
endfunction()

# benchmark examples
create_executable(speed_test src/benchmark/speed_test.cpp)

# server examples
create_executable(primitives src/server/primitives.cpp)
create_executable(meshes src/server/meshes.cpp)
create_executable(laikago src/server/laikago.cpp)
create_executable(balls src/server/balls.cpp)
create_executable(heightmap src/server/heightmap.cpp)
create_executable(aliengo src/server/aliengo.cpp)
create_executable(materials src/server/material.cpp)
create_executable(materialStaticFriction src/server/materialStaticFriction.cpp)
create_executable(heightMapUsingPng src/server/heightMapUsingPng.cpp)
create_executable(wheeledRobot src/server/wheeledRobot.cpp)
create_executable(newtonsCradle src/server/newtonsCradle.cpp)
create_executable(visualObjects src/server/visualObjects.cpp)
create_executable(rayDemo src/server/rayDemo.cpp)
create_executable(rayDemo2 src/server/rayDemo2.cpp)
create_executable(anymal_stress_test src/server/anymals_stress_test.cpp)
create_executable(compound src/server/compound.cpp)
create_executable(cartPole src/server/cartPole.cpp)
create_executable(springTest src/server/springTest.cpp)
create_executable(trackedRobotAndTemplatedURDF src/server/trackedRobotAndTemplatedURDF.cpp)
create_executable(kinematicObject src/server/kinematicObject.cpp)
create_executable(sensors src/server/sensors.cpp)
create_executable(synchronousServerUpdate src/server/synchronousServerUpdate.cpp)
create_executable(raibo_arm src/server/raibo_arm.cpp)
create_executable(arm_2 src/server/arm_2.cpp)
create_executable(raibo2_arm src/server/raibo2_arm.cpp)
create_executable(raibo_arm_WBC src/server/raibo_arm_WBC.cpp)
create_executable(raibo_arm_WBC_2 src/server/raibo_arm_WBC_2.cpp)
create_executable(raibo_arm_Impedance src/server/raibo_arm_Impedance.cpp)
#create_executable(raibo_low_controller src/server/raibo_low_level.cpp)
#create_executable(raibo_low_controller_position src/server/raibo_low_level_position.cpp)
create_executable(box_test src/server/box_test.cpp)
# xml reader
create_executable(xmlRader src/xml/xmlReader.cpp)

# mjcf files
create_executable(mjcf_humanoid src/mjcf/humanoid.cpp)
create_executable(mjcf_ant src/mjcf/ant.cpp)
create_executable(mjcf_cassie src/mjcf/cassie.cpp)

# maps
create_executable(mountain1 src/maps/mountain1.cpp)
create_executable(office1 src/maps/office1.cpp)
create_executable(lake1 src/maps/lake1.cpp)
create_executable(atlas src/maps/atlas.cpp)
create_executable(anymals src/maps/anymals.cpp)
create_executable(hill1 src/maps/hill1.cpp)
create_executable(kinova src/maps/kinova.cpp)

