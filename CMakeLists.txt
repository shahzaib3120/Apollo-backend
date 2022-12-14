cmake_minimum_required(VERSION 3.23)
project(Apollo)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# add flag to enable C++17
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
#// load the Eigen library
find_package(Eigen3 REQUIRED)

add_executable(Apollo main.cpp Layers/Layer.cpp Layers/Layer.h Layers/Dense.cpp Layers/Dense.h Layers/Sigmoid.cpp Layers/Sigmoid.h Layers/Softmax.cpp Layers/Softmax.h Utils/Dataloader.cpp Utils/Dataloader.h Loss/Loss.cpp Model/Model.cpp Model/Model.h Loss/Loss.h Utils/linalg.h Utils/linalg.cpp Preprocessing/Preprocessing.h Preprocessing/Preprocessing.cpp Apollo.h )
target_link_libraries(Apollo Eigen3::Eigen)

add_executable(Apollo-test Tests/Tests.cpp Layers/Layer.cpp Layers/Layer.h Layers/Dense.cpp Layers/Dense.h Layers/Sigmoid.cpp Layers/Sigmoid.h Layers/Softmax.cpp Layers/Softmax.h Utils/Dataloader.cpp Utils/Dataloader.h Tests/main.cpp Tests/Tests.h)
target_link_libraries(Apollo-test Eigen3::Eigen)
