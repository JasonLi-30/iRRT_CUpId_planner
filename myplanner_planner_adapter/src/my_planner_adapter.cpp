/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2018, Raghavender Sahdev.
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
 *   * Neither the name of Raghavender Sahdev nor the names of its
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

#include <my_motion_planner/my_parameters.h>
#include <my_motion_planner/my_planner.h>
#include <moveit/collision_distance_field/collision_detector_allocator_hybrid.h>
#include <moveit/planning_interface/planning_interface.h>
#include <moveit/planning_request_adapter/planning_request_adapter.h>
#include <moveit/robot_state/conversions.h>
#include <moveit/trajectory_processing/trajectory_tools.h>

#include <eigen3/Eigen/Core>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <rclcpp/logger.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp/parameter_value.hpp>
#include <vector>

namespace my_planner
{
static rclcpp::Logger LOGGER = rclcpp::get_logger("my_planner");

class PlannerAdapter : public planning_request_adapter::PlanningRequestAdapter
{
public:
  PlannerAdapter() : planning_request_adapter::PlanningRequestAdapter()
  {
  }

  void initialize(const rclcpp::Node::SharedPtr& node, const std::string& /* unused */) override
  {
    if (!node->get_parameter("iRRT_CUpId.planning_time_limit", params_.planning_time_limit_))
    {
      params_.planning_time_limit_ = 10.0;
      RCLCPP_DEBUG(LOGGER, "Param planning_time_limit was not set. Using default value: %f",
                   params_.planning_time_limit_);
    }
    if (!node->get_parameter("iRRT_CUpId.max_iterations", params_.max_iterations_))
    {
      params_.max_iterations_ = 300;
      RCLCPP_DEBUG(LOGGER, "Param max_iterations was not set. Using default value: %d", params_.max_iterations_);
    }
  }

  std::string getDescription() const override
  {
    return "My Planner";
  }

  bool adaptAndPlan(const PlannerFn& planner, const planning_scene::PlanningSceneConstPtr& ps,
                    const planning_interface::MotionPlanRequest& req, planning_interface::MotionPlanResponse& res,
                    std::vector<std::size_t>& /*added_path_index*/) const override
  {
    RCLCPP_DEBUG(LOGGER, "My Planner: adaptAndPlan ...");

    if (!planner(ps, req, res))
      return false;

    // create a hybrid collision detector to set the collision checker as hybrid
    collision_detection::CollisionDetectorAllocatorPtr hybrid_cd(
        collision_detection::CollisionDetectorAllocatorHybrid::create());

    // create a writable planning scene
    planning_scene::PlanningScenePtr planning_scene = ps->diff();
    RCLCPP_DEBUG(LOGGER, "Configuring Planning Scene for My Planner ...");
    planning_scene->allocateCollisionDetector(hybrid_cd);

    my_planner::iRRT_CUpId_Planner iRRT_CUpId_planner;
    planning_interface::MotionPlanDetailedResponse res_detailed;
    res_detailed.trajectory_.push_back(res.trajectory_);

    bool planning_success = iRRT_CUpId_planner.solve(planning_scene, req, params_, res_detailed);

    if (planning_success)
    {
      res.trajectory_ = res_detailed.trajectory_[0];
      res.planning_time_ += res_detailed.processing_time_[0];
    }
    res.error_code_ = res_detailed.error_code_;

    return planning_success;
  }

private:
  my_planner::iRRT_CUpId_Parameters params_;
};
}  // namespace my_planner

PLUGINLIB_EXPORT_CLASS(my_planner::PlannerAdapter, planning_request_adapter::PlanningRequestAdapter)
