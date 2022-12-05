cmake_minimum_required(VERSION 3.10)
project(controller LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)


find_package(Eigen3 REQUIRED)
#find_package(Torch REQUIRED)

#list(APPEND CMAKE_PREFIX_PATH "~/workspace/raisin_ws/src/raisin/raisin_torch/pytorch/libtorch")
#list(APPEND CMAKE_PREFIX_PATH "~/workspace/Libraries/libtorch_cpu/libtorch")

list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../raisimLib/raisim/linux")
find_package(raisim REQUIRED PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../../raisimLib/raisim/linux)
#find_package(raisim REQUIRED PATHS /home/oem/workspace/raisimLib/raisim/linux)

include_directories (${EIGEN3_INCLUDE_DIRS})

subdir_names(CONTROLLERS .)

list(REMOVE_ITEM CONTROLLERS common)
message(${CONTROLLERS})
message("RRRRR Build done .. ")

function(create_controller name)
    add_library(${name} STATIC ${name}/${name}.hpp)
    set_target_properties(${name} PROPERTIES LINKER_LANGUAGE CXX)
    target_link_libraries(${name} raisim::raisim pthread ${TORCH_LIBRARIES})
    target_include_directories(${name} PUBLIC ${EIGEN3_INCLUDE_DIRS} common ../../include)
    if (CMAKE_BUILD_TYPE STREQUAL "RELEASE" OR CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(${name} PRIVATE -mtune=native -fPIC -O3 -g -march=native)
    endif()
endfunction()

foreach(con ${CONTROLLERS})
    create_controller(${con})
    message(-- complete: ${con})
endforeach()
