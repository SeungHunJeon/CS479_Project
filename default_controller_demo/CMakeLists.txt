cmake_minimum_required(VERSION 3.10)
project(_compliant_gym_tester LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

#list(APPEND CMAKE_PREFIX_PATH $LOCAL_INSTALL)
#list(APPEND CMAKE_PREFIX_PATH "~/workspace/raisin_ws/src/raisin/raisin_torch/pytorch/libtorch")
#list(APPEND CMAKE_PREFIX_PATH "~/workspace/Libraries/libtorch_cpu/libtorch")


find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)
#find_package(Torch REQUIRED)

message("ASDQWFSAD Build done .. ")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../raisim_ws/raisimLib/raisim/linux")

find_package(raisim REQUIRED PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../../../raisim_ws/raisimLib/raisim/linux)
#find_package(raisim REQUIRED PATHS /home/oem/workspace/raisimLib/raisim/linux)

include_directories (${EIGEN3_INCLUDE_DIRS})
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/rsc" DESTINATION ${CMAKE_BINARY_DIR})

macro(subdir_names result srcdir)
    file(GLOB children ${srcdir}/*)
    set(dirlist "")
    foreach(child ${children})
        get_filename_component(name ${child} NAME)
        if(IS_DIRECTORY ${child})
            list(APPEND dirlist ${name})
        endif()
    endforeach()
    set(${result} ${dirlist})
endmacro()

subdir_names(CONTROLLERS module/controller)

list(REMOVE_ITEM CONTROLLERS common)

add_subdirectory(module/controller)