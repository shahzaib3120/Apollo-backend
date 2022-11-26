cmake_minimum_required(VERSION 3.23)
project(Apollo)

set(CMAKE_CXX_STANDARD 14)
#// load the Eigen library
find_package(Eigen3 REQUIRED)
add_executable(Apollo main.cpp Layers/Layer.cpp Layers/Layer.h Layers/Dense.cpp Layers/Dense.h Layers/Sigmoid.cpp Layers/Sigmoid.h Layers/Softmax.cpp Layers/Softmax.h Utils/Dataloader.cpp Utils/Dataloader.h includes.h)
target_link_libraries(Apollo Eigen3::Eigen)
find_package(Eigen3 REQUIRED)
add_executable(Apollo-test Tests/Tests.cpp Layers/Layer.cpp Layers/Layer.h Layers/Dense.cpp Layers/Dense.h Layers/Sigmoid.cpp Layers/Sigmoid.h Layers/Softmax.cpp Layers/Softmax.h Utils/Dataloader.cpp Utils/Dataloader.h includes.h Tests/main.cpp Tests/Tests.h)
#add_executable(Apollo-test Tests/main.cpp)
target_link_libraries(Apollo-test Eigen3::Eigen)
