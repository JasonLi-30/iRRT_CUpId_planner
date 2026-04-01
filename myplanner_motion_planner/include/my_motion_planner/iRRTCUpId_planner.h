/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2009, Willow Garage, Inc.
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

// iRRT_CUpId_Planner.h

#pragma once

#include <my_motion_planner/my_parameters.h>
#include <my_motion_planner/my_trajectory.h>
#include <moveit/collision_distance_field/collision_env_hybrid.h>
#include <moveit/collision_detection_fcl/collision_env_fcl.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_model/robot_model.h>

#include <Eigen/Core>
#include <Eigen/StdVector>
#include <vector>
#include <H5Cpp.h>
#include "onnxruntime_cxx_api.h"  // 包含 ONNX Runtime C++ API

namespace my_planner
{
struct Dataset {
    std::vector<Eigen::VectorXd> parent_q_list;
    std::vector<Eigen::VectorXd> goal_q_list;
    std::vector<Eigen::VectorXd> sampled_q_list;
    std::vector<std::vector<Eigen::MatrixXd>> obs_map_list;
    std::vector<double>          reward_list;

    int n_dof;        // 关节自由度 (dof)
    int map_rows;     // obs_map 行数 (height)
    int map_cols;     // obs_map 列数 (width)
    int num_samples;  // 样本数量

    Dataset(int dof = 6, int h = 32, int w = 32) 
        : n_dof(dof), map_rows(h), map_cols(w), num_samples(0) {}
};


class Node_Info
{
public:
  Node_Info() = default;    //必须有 = default，否则初始化会失败
  ~Node_Info() = default;

  std::vector<double> configuration; 
  bool in_Coli = false;
};

class RRT_Node
{
public:
  RRT_Node() = default;
  ~RRT_Node() = default;

  Node_Info node;
  int iParent = 0;
  double G_n = 0;
  double F_n = 0;
  double H_n = 0;
  double length = 0;
  double Co_colli_prob = 1.0;       //尝试增加碰撞概率系数，以实现效率和安全性的兼顾
  double Num_colli_prob = 1;
  bool isCCDfree = false;
};

class iRRT_CUpId
{
public:
  iRRT_CUpId() = default; 

  iRRT_CUpId(iRRTCUpId_Trajectory* trajectory, planning_scene::PlanningScenePtr& planning_scene,
                 const std::string& planning_group, moveit::core::RobotState& start_state, 
                 iRRT_CUpId_Parameters params, Ort::Session* session_presample);

  virtual ~iRRT_CUpId();

  bool planning(double& length);

  inline void destroy()
  {
    // Nothing for now.
  }

  bool isInitialized() const
  {
    return initialized_;
  }

  bool isCollisionFree() const
  {
    return is_collision_free_;
  }
  
public:

  void setRobotState(std::vector<double>& joint_states);

  int num_joints_;
  int num_vars_free_;
  int num_vars_all_;
  int num_collision_points_;
  int free_vars_start_;
  int free_vars_end_;
  int max_iterations_;
  bool upsampling_;
  bool DPS_;
  bool incremental_;
  Ort::Session* session_presample_;

  iRRTCUpId_Trajectory* full_trajectory_;
  iRRTCUpId_Trajectory group_trajectory_;
  std::string planning_group_;
  
  planning_scene::PlanningScenePtr planning_scene_;
  moveit::core::RobotState state_;
  const moveit::core::JointModelGroup* joint_model_group_;
  collision_detection::CollisionEnvFCL* hy_env_;

  collision_detection::GroupStateRepresentationPtr gsr_;
  bool initialized_;

  std::vector<std::vector<double> > collision_point_potential_;
  std::vector<std::vector<double> > collision_point_vel_mag_;
  std::vector<EigenSTL::vector_Vector3d> collision_point_potential_gradient_;
  std::vector<EigenSTL::vector_Vector3d> joint_axes_;
  std::vector<EigenSTL::vector_Vector3d> joint_positions_;
  Eigen::MatrixXd group_trajectory_backup_;
  Eigen::MatrixXd best_group_trajectory_;

  std::vector<int> state_is_in_collision_; /**< Array containing a boolean about collision info for each point in the
                                                                            trajectory */
  std::vector<std::vector<int> > point_is_in_collision_;
  bool is_collision_free_;
  bool Tree1_add_, Tree2_add_;
  bool do_export_dataset_, forward_presample_;
  bool do_presample_;
  bool is_variable_step_;
  std::vector<Eigen::VectorXd> parent_q_list_, goal_q_list_, sampled_q_list_;
  std::vector<std::vector<Eigen::MatrixXd>> obs_map_list_;
  std::vector<Eigen::MatrixXd> obs_map_current_;
  std::vector<double> reward_list_;
  std::vector<float> parent_q_flat_, goal_q_flat_, sampled_q_flat_, reward_list_flat_, obs_map_flat_;
  double worst_collision_cost_state_;
  int batch_size_;
  

  Eigen::MatrixXd smoothness_increments_;
  Eigen::MatrixXd collision_increments_;
  Eigen::MatrixXd final_increments_;

  std::vector<double> joint_factor_;
  std::vector<std::string> joint_names_;
  std::map<std::string, std::map<std::string, bool> > joint_parent_map_;

  Dataset dataset_;

  void initialize();
  void printObstacleInfo(const planning_scene::PlanningScenePtr& scene);
  bool getPointColi(std::vector<double>& joint_states);
  bool getCCD(RRT_Node* Node1, RRT_Node* Node2, double Stepsize, bool mode = false);
  Eigen::MatrixXd PathSplicing(Eigen::MatrixXd Path1, std::vector<RRT_Node> Path2, int iStart, int iEnd);
  bool RRTConnectToTrajectory(std::vector<RRT_Node>& output, double& length);
  int RRTConnectOnce(std::vector<RRT_Node>& output, RRT_Node* LocalStart, RRT_Node* LocalEnd, double maxDis, bool flag_CCD, bool init = false);
  RRT_Node RRTConnectGenNode(std::vector<RRT_Node>& parent, double* RND_Mean_, double* STD_dev, RRT_Node* goal, double maxDis = 0.6, bool flag_CCD = false);
  RRT_Node RLPresampleNode(RRT_Node& parent, RRT_Node& goal);
  int RRTConnectSteerNode(RRT_Node* goal, std::vector<RRT_Node>& Tree, double maxDis = 0.5, bool flag_CCD = false);
  bool FindParentNode(RRT_Node* node, std::vector<RRT_Node>& Tree, bool flag_CCD);
  double FindNearestNodes(std::vector<RRT_Node>& Tree1, std::vector<RRT_Node>& Tree2, int& iTree1Node, int& iTree2Node, bool flag_CCD, double threshold);
  double SED(RRT_Node* Node1, RRT_Node* Node2);
  void Simplify(std::vector<RRT_Node>& input, std::vector<RRT_Node>& output, double maxDis, int flag_CCD, double length_total);
  void Simplify_longest(std::vector<RRT_Node>& input, std::vector<RRT_Node>& output, double maxDis);
  void updateFullTrajectory();
  void append_data(bool is_collision, std::vector<double> parent_q, std::vector<double> goal_q, std::vector<double> start_q, std::vector<double> sampled_q);
  void saveToBinary(const Dataset& data, const std::string& filename);
  std::vector<Eigen::MatrixXd> createOccupancyMapVolume(const planning_scene::PlanningScenePtr& scene, double resolution = 0.05);
};
}  // namespace my_planner
