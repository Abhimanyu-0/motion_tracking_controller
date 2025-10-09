cmake_minimum_required(VERSION 3.8)
project(motion_tracking_controller)

add_compile_options(-Wall -Wextra -Wpedantic)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()


find_package(Eigen3 REQUIRED)

# Find cnpy for .npz file loading
find_library(CNPY_LIBRARY cnpy HINTS $ENV{HOME}/.local/lib)
find_path(CNPY_INCLUDE_DIR cnpy.h HINTS $ENV{HOME}/.local/include)

if(NOT CNPY_LIBRARY OR NOT CNPY_INCLUDE_DIR)
    message(FATAL_ERROR "cnpy library not found. Install it from https://github.com/rogersce/cnpy")
endif()

# Find LibTorch for .pt model loading
set(CMAKE_PREFIX_PATH "$ENV{HOME}/.local/libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

ament_auto_add_library(
    ${PROJECT_NAME}
    SHARED
    DIRECTORY src
)

# prevent pluginlib from using boost
target_compile_definitions(${PROJECT_NAME} PUBLIC "PLUGINLIB__DISABLE_BOOST_FUNCTIONS")
pluginlib_export_plugin_description_file(controller_interface motion_tracking_controller.xml)

target_include_directories(
    ${PROJECT_NAME} PUBLIC
    ${EIGEN3_INCLUDE_DIRS}
    ${CNPY_INCLUDE_DIR}
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}
    Eigen3::Eigen
    ${CNPY_LIBRARY}
    z  # zlib required by cnpy
    ${TORCH_LIBRARIES}
)

ament_auto_package(
    INSTALL_TO_SHARE config launch
)
