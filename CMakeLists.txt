cmake_minimum_required(VERSION 3.23)
project(Apollo)

set(CMAKE_CXX_STANDARD 14)
#// load the Eigen library
find_package(Eigen3 REQUIRED)
add_executable(Apollo main.cpp Layers/Layer.cpp Layers/Layer.h Layers/Dense.cpp Layers/Dense.h Layers/Sigmoid.cpp Layers/Sigmoid.h Layers/Softmax.cpp Layers/Softmax.h Utils/Dataloader.cpp Utils/Dataloader.h)
find_package(Eigen3 REQUIRED)
add_executable(Apollo-test Tests/SigmoidTest.cpp)
target_link_libraries(Apollo Eigen3::Eigen)

