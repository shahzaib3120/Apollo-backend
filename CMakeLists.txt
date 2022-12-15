cmake_minimum_required(VERSION 3.23)
project(Apollo)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(HIGH_FIVE_INCLUDE_DIR F:/Learning/vcpkg/buildtrees/highfive/src/v2.3-6b271ad31e.clean/include)
# add flag to enable C++17
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
#// load the Eigen library
find_package(Eigen3 REQUIRED)
# include highFive directory
#include_directories(${HIGH_FIVE_INCLUDE_DIR})
add_executable(Apollo main.cpp Layers/Layer.cpp Layers/Layer.h Layers/Dense.cpp Layers/Dense.h Layers/Sigmoid.cpp Layers/Sigmoid.h Layers/Softmax.cpp Layers/Softmax.h Utils/Dataloader.cpp Utils/Dataloader.h Loss/Loss.cpp Model/Model.cpp Model/Model.h Loss/Loss.h Utils/linalg.h Utils/linalg.cpp)
target_link_libraries(Apollo Eigen3::Eigen)
add_executable(Apollo-test Tests/Tests.cpp Layers/Layer.cpp Layers/Layer.h Layers/Dense.cpp Layers/Dense.h Layers/Sigmoid.cpp Layers/Sigmoid.h Layers/Softmax.cpp Layers/Softmax.h Utils/Dataloader.cpp Utils/Dataloader.h Tests/main.cpp Tests/Tests.h)
#add_executable(Apollo-test Tests/main.cpp)
target_link_libraries(Apollo-test Eigen3::Eigen)
