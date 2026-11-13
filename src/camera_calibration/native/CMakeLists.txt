cmake_minimum_required(VERSION 3.16)
project(wato_koide_calibrate LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(Ceres 2.2.0 EXACT REQUIRED)
find_path(OPENCV_INCLUDE_DIR opencv2/core.hpp PATH_SUFFIXES opencv4 REQUIRED)
find_library(OPENCV_CORE opencv_core REQUIRED)
if(NOT KOIDE_SOURCE_DIR)
  message(FATAL_ERROR "KOIDE_SOURCE_DIR must point to pinned Koide source")
endif()
add_executable(wato_koide_calibrate calibrate.cpp)
target_include_directories(wato_koide_calibrate PRIVATE
  ${OPENCV_INCLUDE_DIR} ${KOIDE_SOURCE_DIR}/include
  ${KOIDE_SOURCE_DIR}/thirdparty/Sophus
  ${KOIDE_SOURCE_DIR}/thirdparty/json/single_include)
target_link_libraries(wato_koide_calibrate PRIVATE Ceres::ceres ${OPENCV_CORE})
install(TARGETS wato_koide_calibrate RUNTIME DESTINATION bin)
