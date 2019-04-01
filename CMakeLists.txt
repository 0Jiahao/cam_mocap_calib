# What version of CMake is needed ?
cmake_minimum_required(VERSION 2.8.3 )

# Name of this package .
project( cam_mocap_calib )

# Find the catkin build system , and any other packages on
# which we depend .
find_package(catkin REQUIRED COMPONENTS roscpp rospy tf cv_bridge image_transport sensor_msgs geometry_msgs std_msgs)
find_package(OpenCV REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Declare our catkin package .
catkin_package( )

# Specify locations of header files .
include_directories( include ${catkin_INCLUDE_DIRS} )
include_directories(${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

# Declare the executable , along with its sourcefiles . If
# there are multiple executables ,use multiple copies of
# this line .
file(GLOB_RECURSE UV_HDRS_FILES "src/*.h" "src/*.hpp" "include/*.h" "include/*.hpp")
file(GLOB_RECURSE UV_SRCS_FILES "src/*.c" "src/*.cpp" "include/*.c" "include/*.cpp")

add_executable( cam_mocap_calib cam_mocap_calib.cpp ${UV_SRCS_FILES} ${UV_HDRS_FILES})

# Specify libraries against which to link. Again , this
# line should be copied for each distinct executable in
# the package .
target_link_libraries( cam_mocap_calib ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} Eigen3::Eigen)

 
