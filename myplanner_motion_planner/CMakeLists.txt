cmake_minimum_required(VERSION 3.22)
project(myplanner_motion_planner)

# Common cmake code applied to all moveit packages
find_package(moveit_common REQUIRED)
moveit_package()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(moveit_core REQUIRED)
find_package(rclcpp REQUIRED)
find_package(trajectory_msgs REQUIRED)

pluginlib_export_plugin_description_file(moveit_core myplanner_motion_planner_plugin_description.xml)

# ========================
# 查找 ONNX Runtime
# ========================
find_path(ONNXRUNTIME_INCLUDE_DIR
  NAMES onnxruntime_cxx_api.h
  PATHS /opt/onnxruntime-linux-x64-gpu-1.23.1/include
        /usr/include/onnxruntime
  NO_DEFAULT_PATH
)

find_library(ONNXRUNTIME_LIBRARY
  NAMES onnxruntime
  PATHS /opt/onnxruntime-linux-x64-gpu-1.23.1/lib
        /usr/lib/x86_64-linux-gnu
  NO_DEFAULT_PATH
)

# 检查是否找到
if(NOT ONNXRUNTIME_INCLUDE_DIR)
  message(FATAL_ERROR "❌ 未找到 onnxruntime_cxx_api.h！请检查路径是否正确：/opt/onnxruntime-linux-x64-gpu-1.23.1/include")
endif()

if(NOT ONNXRUNTIME_LIBRARY)
  message(FATAL_ERROR "❌ 未找到 libonnxruntime.so！请检查路径是否正确：/opt/onnxruntime-linux-x64-gpu-1.23.1/lib")
endif()

# 输出找到的路径（用于调试）
message(STATUS "✅ 找到 ONNX Runtime 头文件: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "✅ 找到 ONNX Runtime 库: ${ONNXRUNTIME_LIBRARY}")

set(THIS_PACKAGE_INCLUDE_DEPENDS
  moveit_core
  rclcpp
  trajectory_msgs
  visualization_msgs
)

include_directories(
  include
)

add_library(${PROJECT_NAME} SHARED
  src/my_parameters.cpp
  src/my_trajectory.cpp
  src/iRRTCUpId_planner.cpp
  src/my_planner.cpp
)


target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC
  ${ONNXRUNTIME_INCLUDE_DIR}
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(${PROJECT_NAME} ${THIS_PACKAGE_INCLUDE_DEPENDS})
target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${Boost_LIBRARIES}
  ${PCL_LIBRARIES}
  ${ONNXRUNTIME_LIBRARY}  # 👈 链接 ONNX Runtime
)

set_target_properties(${PROJECT_NAME} PROPERTIES VERSION "${${PROJECT_NAME}_VERSION}")


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)

  # These don't pass yet, disable them for now
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  set(ament_cmake_flake8_FOUND TRUE)
  set(ament_cmake_uncrustify_FOUND TRUE)
  set(ament_cmake_lint_cmake_FOUND TRUE)

  # Run all lint tests in package.xml except those listed above
  ament_lint_auto_find_test_dependencies()
endif()

install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}Targets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)
install(DIRECTORY include/ DESTINATION include)

ament_export_targets(${PROJECT_NAME}Targets HAS_LIBRARY_TARGET)
ament_export_dependencies(${THIS_PACKAGE_EXPORT_DEPENDS})

ament_package()
