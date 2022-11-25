cmake_minimum_required(VERSION 3.23)
project(Apollo)

set(CMAKE_CXX_STANDARD 14)
#// load the Eigen library
find_package(Eigen3 REQUIRED)
add_executable(Apollo main.cpp)
target_link_libraries(Apollo Eigen3::Eigen)

