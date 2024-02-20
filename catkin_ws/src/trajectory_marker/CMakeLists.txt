cmake_minimum_required(VERSION 3.0.2)
project(trajectory_marker)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

# include(CheckCXXCompilerFlag)
# CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
# CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
# if(COMPILER_SUPPORTS_CXX11)
#     set(CMAKE_CXX_FLAGS "-std=c++11")
# elseif(COMPILER_SUPPORTS_CXX0X)
#     set(CMAKE_CXX_FLAGS "-std=c++0x")
# else()
#     message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
# endif()

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  gazebo
  gazebo_ros
  roscpp
  std_msgs
  visualization_msgs
)

find_package(gazebo REQUIRED)


link_directories(${GAZEBO_LIBRARY_DIRS})

include_directories(${catkin_INCLUDE_DIRS} ${GAZEBO_INCLUDE_DIRS})

list(APPEND CMAKE_CXX_FLAGS "${GAZEBO_CXX_FLAGS}")

catkin_package(
  INCLUDE_DIRS include
  DEPENDS
    roscpp
    gazebo_ros
)

add_library(${PROJECT_NAME} SHARED src/trajectory_marker_plugin.cpp)
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})
add_dependencies(${PROJECT_NAME}  ${catkin_EXPORTED_TARGETS})