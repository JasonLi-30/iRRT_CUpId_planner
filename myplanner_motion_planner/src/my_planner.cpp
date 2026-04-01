/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2012, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/

/* Author: Shengjie Li */

// my_planner.cpp

#include <my_motion_planner/iRRTCUpId_planner.h>
#include <my_motion_planner/my_planner.h>
#include <my_motion_planner/my_trajectory.h>
#include <moveit/robot_state/conversions.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include "onnxruntime_cxx_api.h"  // 包含 ONNX Runtime C++ API
#include <chrono>
#include <fstream>
#include <nlohmann/json.hpp>
#include <future>        // ← 新增头文件

namespace my_planner
{
static const rclcpp::Logger LOGGER = rclcpp::get_logger("my_planner");

// 静态地图生成函数（线程安全）
std::vector<Eigen::MatrixXd> my_planner::iRRT_CUpId_Planner::createOccupancyMapVolumeStatic(
    const planning_scene::PlanningSceneConstPtr& scene,
    double resolution)
{
    const collision_detection::World& world = *(scene->getWorld());

    const double min_x = -0.6, max_x = 0.6;
    const double min_y = -0.6, max_y = 0.6;
    const double min_z = -0.45, max_z = 1.1;

    int nx = static_cast<int>(std::ceil((max_x - min_x) / resolution));
    int ny = static_cast<int>(std::ceil((max_y - min_y) / resolution));
    int nz = static_cast<int>(std::floor((max_z - min_z) / resolution));

    std::vector<Eigen::MatrixXd> obs_map_volume(nz, Eigen::MatrixXd::Zero(nx, ny));

    for (const auto& pair : world) {
        const auto& obj = pair.second;
        for (size_t i = 0; i < obj->shapes_.size(); ++i) {
            const auto& shape = obj->shapes_[i];
            const Eigen::Isometry3d& pose = obj->global_shape_poses_[i];

            if (shape->type != shapes::BOX) continue;

            const auto* box = static_cast<const shapes::Box*>(shape.get());
            Eigen::Vector3d size(box->size[0], box->size[1], box->size[2]);

            double cx = pose.translation().x();
            double cy = pose.translation().y();
            double cz = pose.translation().z();

            double half_l = size.x() / 2;
            double half_w = size.y() / 2;
            double half_h = size.z() / 2;

            double bx_min = cx - half_l;
            double bx_max = cx + half_l;
            double by_min = cy - half_w;
            double by_max = cy + half_w;
            double bz_min = cz - half_h;
            double bz_max = cz + half_h;

            int z_start_idx = std::max(0, static_cast<int>(std::floor((bz_min - min_z) / resolution)));
            int z_end_idx   = std::min(nz - 1, static_cast<int>(std::floor((bz_max - min_z) / resolution)));

            for (int z_idx = z_start_idx; z_idx <= z_end_idx; ++z_idx) {
                double layer_z_min = min_z + z_idx * resolution;
                double layer_z_max = layer_z_min + resolution;

                if (bz_max >= layer_z_min && bz_min <= layer_z_max) {
                    Eigen::MatrixXd& map = obs_map_volume[z_idx];
                    int x_start = std::max(0, static_cast<int>(std::floor((bx_min - min_x) / resolution)));
                    int x_end   = std::min(nx - 1, static_cast<int>(std::floor((bx_max - min_x) / resolution)));
                    int y_start = std::max(0, static_cast<int>(std::floor((by_min - min_y) / resolution)));
                    int y_end   = std::min(ny - 1, static_cast<int>(std::floor((by_max - min_y) / resolution)));

                    for (int i = x_start; i <= x_end; ++i)
                        for (int j = y_start; j <= y_end; ++j)
                            map(i, j) = 1.0;
                }
            }
        }
    }

    return obs_map_volume;
}

bool my_planner::iRRT_CUpId_Planner::solve(const planning_scene::PlanningSceneConstPtr& planning_scene,
                         const planning_interface::MotionPlanRequest& req, const iRRT_CUpId_Parameters& params,
                         planning_interface::MotionPlanDetailedResponse& res) const
{
    int count = 0;
    double avg_time = 0;
    double max_time = 0;
    double min_time = 1000000;
    int loopNum = 100;
    int successNum = 0;
    double successRate = 0.0;
    double avg_length = 0;
    double max_length = 0;
    double min_length = 1000000;

    const std::string MODEL_PATH = "/home/jasonli/ws_moveit2/src/moveit2/moveit_planners/iRRT_CUpId_planner/myplanner_motion_planner/src/best_ppo_sampler_model_quantized.onnx";
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntime_Inference");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetLogSeverityLevel(1);
    Ort::Session session_presample(env, MODEL_PATH.c_str(), session_options);

    std::string config_path = "/home/jasonli/ws_moveit2/irrt_config.json";
    bool do_presample_ = false;
    bool do_export_dataset_ = false;
    bool upsampling_ = true;
    bool DPS_ = true;
    bool incremental_ = true;

    try 
    {
        std::ifstream config_file(config_path);
        if (config_file.is_open()) {
            nlohmann::json j;
            config_file >> j;
            do_presample_ = j.value("do_presample", false);
            do_export_dataset_ = j.value("do_export_dataset", false);
            upsampling_ = j.value("upsampling", true);
            DPS_ = j.value("DPS", true);
            incremental_ = j.value("incremental", true);

            std::cout << "=============================" << std::endl;
            std::cout << "do_presample: " << do_presample_ << std::endl;
            std::cout << "do_export_dataset: " << do_export_dataset_ << std::endl;
            std::cout << "upsampling: " << upsampling_ << std::endl;
            std::cout << "DPS: " << DPS_ << std::endl;
            std::cout << "incremental: " << incremental_ << std::endl;
            std::cout << "=============================" << std::endl;
        }
    } 
    catch (const std::exception& e) {
        std::cout << "Error parsing JSON config: " << e.what() << ". Using defaults." << std::endl;
    }

    // ===== 初始化 obs_map_current（在 while 外）=====
    std::vector<Eigen::MatrixXd> obs_map_current;
    const double min_x = -0.6, max_x = 0.6;
    const double min_y = -0.6, max_y = 0.6;
    const double min_z = -0.45, max_z = 1.1;
    int nx = static_cast<int>(std::ceil((max_x - min_x) / 0.05));
    int ny = static_cast<int>(std::ceil((max_y - min_y) / 0.05));
    int nz = static_cast<int>(std::floor((max_z - min_z) / 0.05));

    if (do_presample_) {
        try {
            planning_scene::PlanningScenePtr initial_scene = planning_scene::PlanningScene::clone(planning_scene);
            obs_map_current = createOccupancyMapVolumeStatic(initial_scene, 0.05);
        } catch (...) {
            RCLCPP_ERROR(LOGGER, "Failed to initialize occupancy map on first try.");
            return false;
        }
    } else {
        obs_map_current = std::vector<Eigen::MatrixXd>(nz, Eigen::MatrixXd::Zero(nx, ny));
    }

    // ===== 异步 future 变量（局部）=====
    std::future<std::vector<Eigen::MatrixXd>> next_map_future;

    // 启动第一次预取（用于第 1 次迭代）
    if (do_presample_) {
        planning_scene::PlanningScenePtr scene_clone = planning_scene::PlanningScene::clone(planning_scene);
        next_map_future = std::async(std::launch::async, [scene_clone]() {
            return createOccupancyMapVolumeStatic(scene_clone, 0.05);
        });
    }

    while (count < loopNum)
    {
        // ===== 尝试获取新地图，但不等待 =====
        std::vector<Eigen::MatrixXd> current_obs_map = obs_map_current; // 默认用旧的

        if (do_presample_ && next_map_future.valid()) {
            if (next_map_future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
                try {
                    current_obs_map = next_map_future.get(); // 成功拿到新地图
                    obs_map_current = current_obs_map;       // 更新缓存
                } catch (...) {
                    RCLCPP_WARN(LOGGER, "Async map update failed, using old map.");
                    // keep current_obs_map = obs_map_current
                }
            }
            // 如果未 ready，直接跳过，用旧地图
        }

        // 克隆当前场景用于本次规划
        planning_scene::PlanningScenePtr planning_scene_current = planning_scene::PlanningScene::clone(planning_scene);

        // 启动下一次地图预取（如果不是最后一轮）
        if (count < loopNum - 1 && do_presample_) {
            planning_scene::PlanningScenePtr next_scene = planning_scene::PlanningScene::clone(planning_scene);
            next_map_future = std::async(std::launch::async, [next_scene]() {
                return createOccupancyMapVolumeStatic(next_scene, 0.05);
            });
        }

        // ==================== 规划流程开始 ====================
        auto start_time = std::chrono::system_clock::now();

        if (!planning_scene_current)
        {
            RCLCPP_ERROR(LOGGER, "No planning scene initialized.");
            res.error_code_.val = moveit_msgs::msg::MoveItErrorCodes::FAILURE;
            return false;
        }

        moveit::core::RobotState start_state = planning_scene_current->getCurrentState();
        moveit::core::robotStateMsgToRobotState(planning_scene_current->getTransforms(), req.start_state, start_state);

        if (!start_state.satisfiesBounds())
        {
            RCLCPP_ERROR(LOGGER, "Start state violates joint limits");
            res.error_code_.val = moveit_msgs::msg::MoveItErrorCodes::INVALID_ROBOT_STATE;
            return false;
        }

        my_planner::iRRTCUpId_Trajectory trajectory(planning_scene_current->getRobotModel(), 3.0, .03, req.group_name);
        robotStateToArray(start_state, req.group_name, trajectory.getTrajectoryPoint(0));

        if (req.goal_constraints.size() != 1)
        {
            RCLCPP_ERROR(LOGGER, "Expecting exactly one goal constraint, got: %zd", req.goal_constraints.size());
            res.error_code_.val = moveit_msgs::msg::MoveItErrorCodes::INVALID_GOAL_CONSTRAINTS;
            return false;
        }

        if (req.goal_constraints[0].joint_constraints.empty() || !req.goal_constraints[0].position_constraints.empty() ||
            !req.goal_constraints[0].orientation_constraints.empty())
        {
            RCLCPP_ERROR(LOGGER, "Only joint-space goals are supported");
            res.error_code_.val = moveit_msgs::msg::MoveItErrorCodes::INVALID_GOAL_CONSTRAINTS;
            return false;
        }

        const size_t goal_index = trajectory.getNumPoints() - 1;
        moveit::core::RobotState goal_state(start_state);
        for (const moveit_msgs::msg::JointConstraint& joint_constraint : req.goal_constraints[0].joint_constraints)
            goal_state.setVariablePosition(joint_constraint.joint_name, joint_constraint.position);
        if (!goal_state.satisfiesBounds())
        {
            RCLCPP_ERROR(LOGGER, "Goal state violates joint limits");
            res.error_code_.val = moveit_msgs::msg::MoveItErrorCodes::INVALID_ROBOT_STATE;
            return false;
        }
        robotStateToArray(goal_state, req.group_name, trajectory.getTrajectoryPoint(goal_index));

        const moveit::core::JointModelGroup* model_group =
            planning_scene_current->getRobotModel()->getJointModelGroup(req.group_name);
        for (size_t i = 0; i < model_group->getActiveJointModels().size(); ++i)
        {
            const moveit::core::JointModel* model = model_group->getActiveJointModels()[i];
            const moveit::core::RevoluteJointModel* revolute_joint =
                dynamic_cast<const moveit::core::RevoluteJointModel*>(model);

            if (revolute_joint != nullptr && revolute_joint->isContinuous())
            {
                double start = (trajectory)(0, i);
                double end = (trajectory)(goal_index, i);
                (trajectory)(goal_index, i) = start + shortestAngularDistance(start, end);
            }
        }

        trajectory.fillInLinearInterpolation();

        my_planner::iRRT_CUpId_Parameters params_nonconst = params;
        std::unique_ptr<my_planner::iRRT_CUpId> planner = std::make_unique<my_planner::iRRT_CUpId>(
            &trajectory, planning_scene_current, req.group_name, start_state, params_nonconst, &session_presample);

        planner->do_presample_ = do_presample_;
        planner->do_export_dataset_ = do_export_dataset_;
        planner->upsampling_ = upsampling_;
        planner->DPS_ = DPS_;
        planner->incremental_ = incremental_;
        planner->obs_map_current_ = std::move(current_obs_map); // ← 关键：使用（可能更新的）地图

        if (!planner->isInitialized())
        {
            RCLCPP_ERROR(LOGGER, "Could not initialize planner");
            res.error_code_.val = moveit_msgs::msg::MoveItErrorCodes::PLANNING_FAILED;
            return false;
        }

        double length = 0;
        bool planning_result = planner->planning(length);

        // 轨迹结果填充（略作简化）
        auto result = std::make_shared<robot_trajectory::RobotTrajectory>(planning_scene_current->getRobotModel(), req.group_name);
        for (int i = 0; i < trajectory.getTrajectory().rows(); ++i)
        {
            const Eigen::MatrixXd::RowXpr source = trajectory.getTrajectoryPoint(i);
            auto state = std::make_shared<moveit::core::RobotState>(start_state);
            size_t joint_index = 0;
            for (const moveit::core::JointModel* jm : result->getGroup()->getActiveJointModels())
            {
                assert(jm->getVariableCount() == 1);
                state->setVariablePosition(jm->getFirstVariableIndex(), source[joint_index++]);
            }
            result->addSuffixWayPoint(state, 0.0);
        }

        res.trajectory_.resize(1);
        res.trajectory_[0] = result;
        res.processing_time_.resize(1);
        res.processing_time_[0] = std::chrono::duration<double>(std::chrono::system_clock::now() - start_time).count();
        res.error_code_.val = moveit_msgs::msg::MoveItErrorCodes::SUCCESS;

        // 统计
        if (planning_result && planner->isCollisionFree())
        {
            if(res.processing_time_[0] > max_time) max_time = res.processing_time_[0];
            if(res.processing_time_[0] < min_time) min_time = res.processing_time_[0];
            avg_time += res.processing_time_[0];

            if(length > max_length) max_length = length;
            if(length < min_length) min_length = length;
            avg_length += length;
            successNum++;
        }

        count++;
    }

    if (successNum > 0) {
        avg_time /= successNum;
        avg_length /= successNum;
    }
    successRate = static_cast<double>(successNum) / loopNum;

    std::cout << "Processing_avg_time: " << avg_time << std::endl;
    std::cout << "Processing_max_time: " << max_time << std::endl;
    std::cout << "Processing_min_time: " << min_time << std::endl;
    std::cout << "Processing_avg_length: " << avg_length << std::endl;
    std::cout << "Processing_max_length: " << max_length << std::endl;
    std::cout << "Processing_min_length: " << min_length << std::endl;
    std::cout << "SuccessRate: " << successRate << std::endl;

    return true;
}

}  // namespace my_planner