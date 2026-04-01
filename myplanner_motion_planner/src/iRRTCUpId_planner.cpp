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

//iRRT_CUpId_Planner.cpp

#include <my_motion_planner/iRRTCUpId_planner.h>
#include <my_motion_planner/my_utils.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_state/conversions.h>

#include <rclcpp/logger.hpp>
#include <rclcpp/logging.hpp>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>
#include <random>
#include <H5Cpp.h>
#include <visualization_msgs/msg/marker_array.hpp>


namespace my_planner
{
static const rclcpp::Logger LOGGER = rclcpp::get_logger("iRRT_CUpId planner");
iRRT_CUpId::iRRT_CUpId(iRRTCUpId_Trajectory* trajectory, planning_scene::PlanningScenePtr& planning_scene,
                  const std::string& planning_group, moveit::core::RobotState& start_state, iRRT_CUpId_Parameters params, Ort::Session* session_presample)
  : group_trajectory_(*trajectory, planning_group, DIFF_RULE_LENGTH)
  , state_(start_state)
  , session_presample_(session_presample)
{
  full_trajectory_ = trajectory;
  planning_group_ = planning_group;
  planning_scene_ = planning_scene;
  initialized_ = false;
  max_iterations_ = params.max_iterations_;
  // session_presample_ = session_presample;
  initialize();
}

void iRRT_CUpId::initialize()
{
  // init some variables:
  num_vars_free_ = group_trajectory_.getNumFreePoints();
  num_vars_all_ = group_trajectory_.getNumPoints();
  num_joints_ = group_trajectory_.getNumJoints();

  free_vars_start_ = group_trajectory_.getStartIndex();
  free_vars_end_ = group_trajectory_.getEndIndex();

  collision_detection::CollisionRequest req;
  collision_detection::CollisionResult res;
  req.group_name = planning_group_;
  num_collision_points_ = 1;

  double max_cost_scale = 0.0;

  joint_model_group_ = planning_scene_->getRobotModel()->getJointModelGroup(planning_group_);

  const std::vector<const moveit::core::JointModel*> joint_models = joint_model_group_->getActiveJointModels();

  joint_axes_.resize(num_vars_all_, EigenSTL::vector_Vector3d(num_joints_));
  joint_positions_.resize(num_vars_all_, EigenSTL::vector_Vector3d(num_joints_));

  is_collision_free_ = false;
  Tree1_add_ = false;
  Tree2_add_ = false;
  do_presample_ = false;
  do_export_dataset_ = false;
  forward_presample_ = false;
  is_variable_step_ = false;
  int start = free_vars_start_;
  int end = free_vars_end_;

  initialized_ = true;
  upsampling_ = true;     //是否启用升采样策略
  DPS_ = true;            //是否启用动态路径简化
  incremental_ = true;    //是否启用增量规划


  joint_factor_.resize(num_joints_);
  for(int i = 0;i < num_joints_;i++)
    joint_factor_[i] = 1.0;

}

iRRT_CUpId::~iRRT_CUpId()
{
  destroy();
}


bool iRRT_CUpId::planning(double& length)   //路径规划算法主干内容
{
  const auto start_time = std::chrono::system_clock::now();
  bool should_break_out = false;
  double c_cost_last = 0;
  int last_n = 3;
  std::vector<int> last_n_iteration(last_n, 0);
  std::vector<RRT_Node> TEMP_Path;
  // printObstacleInfo(planning_scene_);

 
  // std::cout << "===RRTConnectToTrajectory start===" << std::endl;
  bool result = RRTConnectToTrajectory(TEMP_Path, length);

  group_trajectory_.getTrajectory() = PathSplicing(group_trajectory_.getTrajectory(), TEMP_Path, free_vars_start_, free_vars_end_);

  updateFullTrajectory();

  // 在此处加入导出数据集的内容，并用bool变量do_export_dataset_控制是否导出数据集
  if(do_export_dataset_)
  {
    if(forward_presample_)
    {
      for(int i = 1;i < TEMP_Path.size();i++)
      {
        append_data(false, TEMP_Path[i-1].node.configuration, TEMP_Path.back().node.configuration, TEMP_Path[0].node.configuration, TEMP_Path[i].node.configuration);
      }
    }
    else
    {
      for(int i = TEMP_Path.size()-2;i > 0;i--)
      {
        append_data(false, TEMP_Path[i+1].node.configuration, TEMP_Path[0].node.configuration, TEMP_Path.back().node.configuration, TEMP_Path[i].node.configuration);
      }
    }
    saveToBinary(dataset_, "irrt_training_data.bin");
    std::cout << "===irrt_training_data.bin has been exported===" << std::endl;
  }

  is_collision_free_ = result;

  return is_collision_free_;
  
}

int iRRT_CUpId::RRTConnectSteerNode(RRT_Node* goal, std::vector<RRT_Node>& Tree, double maxDis, bool flag_CCD)
{
  RRT_Node start, temp;
  std::vector<double> tempV(num_joints_, 0);
  temp.node.configuration = tempV;
  double min_dist = INFINITY;  
  double dist;
  int j_start = 0;
  int output = 0;
  double norm = 0;
  double RND_Temp = 0;
  const std::vector<const moveit::core::JointModel*> joint_models = joint_model_group_->getActiveJointModels();
  bool colli_flag = false;
  std::random_device rd;
  std::mt19937 gen2(rd());                         // 定义随机数生成器对象gen，使用time(nullptr)作为随机数生成器的种子
  std::normal_distribution<double> dis2(0.0, 1.0); // 定义随机数分布器对象dis，期望为0.0，标准差为1.0的正态分布

  {
    for(int j = 0;j < Tree.size();j++)
    {
      dist = SED(goal, &Tree[j]);  
      if(dist < min_dist)
      {
        min_dist = dist;
        j_start = j;
        output = j;
        start = Tree[j];
      }
    }


    for(int j = 0;j < num_joints_;j++)
    {
      norm += pow(goal->node.configuration[j] - start.node.configuration[j],2);
    }
    norm = sqrt(norm);

    RND_Temp = 1;   // (rand() % 1000 / 1000.0); // dis2(gen2);
    for(int j = 0;j < num_joints_;j++)
    {
      const moveit::core::JointModel* joint_model1 = joint_models[j];
      const moveit::core::JointModel::Bounds& bounds = joint_model1->getVariableBounds();

      temp.node.configuration[j] = start.node.configuration[j] + (goal->node.configuration[j]-start.node.configuration[j]) / norm * maxDis * RND_Temp;  //(rand() % 100 / 100.0);
      if(bounds[0].min_position_ > temp.node.configuration[j])
        temp.node.configuration[j] = bounds[0].min_position_;

      if(bounds[0].max_position_ < temp.node.configuration[j])
        temp.node.configuration[j] = bounds[0].max_position_;
    }

    if(flag_CCD)
      colli_flag = getCCD(&Tree[j_start], &temp, 0.05, false);
    else
      colli_flag = getPointColi(temp.node.configuration);

    if(!colli_flag)
    {
      temp.iParent = j_start;
      Tree.push_back(temp);
      output = Tree.size() - 1;
    }
  }
  return output;

}

double iRRT_CUpId::SED(RRT_Node* Node1, RRT_Node* Node2)
{
  double output = 0;
  
  for(int i = 0;i < num_joints_;i++)
    output += joint_factor_[i] * pow(Node1->node.configuration[i] - Node2->node.configuration[i], 2);

  return output;
}


double iRRT_CUpId::FindNearestNodes(std::vector<RRT_Node>& Tree1, std::vector<RRT_Node>& Tree2, int& iTree1Node, int& iTree2Node, bool flag_CCD, double threshold)
{
  double MinDist = INFINITY;
  double dist;
  bool flag = false;
  if(Tree1.size() < 1 || Tree2.size() < 1)
    return -1;

  if(Tree1_add_ == true)
  {
    for(int i = Tree1.size() - 1;i < Tree1.size();i++)
    {
      for(int j = 0;j < Tree2.size();j++)
      {
        dist = SED(&Tree1[i],&Tree2[j]);
        if(MinDist > dist)
        {
          if(!flag_CCD || (flag_CCD && !getCCD(&Tree1[i],&Tree2[j], 0.05, false)))
          {
            MinDist = dist;
            iTree1Node = i;
            iTree2Node = j;
            if(flag_CCD)
              return MinDist;
          }
          else
          {
            flag = true;
          }
        }
      }
    }
  }
  
  if(Tree2_add_ == true)
  {
    for(int i = 0;i < Tree1.size();i++)
    {
      for(int j = Tree2.size() - 1;j < Tree2.size();j++)
      {
        dist = SED(&Tree1[i],&Tree2[j]);
        if(MinDist > dist)
        {
          if(!flag_CCD || (flag_CCD && !getCCD(&Tree1[i],&Tree2[j], 0.05, false)))
          {
            MinDist = dist;
            iTree1Node = i;
            iTree2Node = j;
            if(flag_CCD)
              return MinDist;
          }
          else
          {
            flag = true;
          }
        }
      }
    }
  }
  
  if(MinDist == INFINITY)    //if(flag && MinDist > threshold)
  {
    // std::cout << "Tree1.size(): " << Tree1.size() << std::endl; 
    // std::cout << "Tree2.size(): " << Tree2.size() << std::endl; 
    // std::cout << "MinDist: " << MinDist << std::endl; 
    return -1;
  }
  return MinDist;
}

bool iRRT_CUpId::FindParentNode(RRT_Node* node, std::vector<RRT_Node>& Tree, bool flag_CCD)
{
  double min_dist = INFINITY;  
  double dist;
  bool output = false;
  for(int i = 0;i < Tree.size();i++)
  {
    dist = SED(node, &Tree[i]);  
    if(dist < min_dist && (!flag_CCD || (flag_CCD && !getCCD(node, &Tree[i], 0.05, false)))) 
    {
      min_dist = dist;
      node->iParent = i;
      // node->length = Tree[i].length + sqrt(dist);
      output = true;
    }
  }
  return output;
}


RRT_Node iRRT_CUpId::RLPresampleNode(RRT_Node& parent, RRT_Node& goal)
{
  RRT_Node output;
  output.node.in_Coli = false;
  std::vector<double> temp(num_joints_, 0);
  output.node.configuration = temp;
  output.iParent = 0;
  std::vector<double> parent_config = parent.node.configuration;
  std::vector<float> parent_q(parent_config.begin(), parent_config.end());
  std::vector<double> goal_config = goal.node.configuration;
  std::vector<float> goal_q(goal_config.begin(), goal_config.end());
  auto output_names = session_presample_->GetOutputNames();
  

  // 5. 创建输入张量
  std::vector<int64_t> parent_q_shape = {1, num_joints_};
  std::vector<int64_t> goal_q_shape   = {1, num_joints_};
  std::vector<int64_t> obs_map_shape  = {1, 31, 24, 24};

  // 假设每个 MatrixXf 是 (H, W)，总共有 N 个
  std::vector<float> flat_data;
  int H = 24, W = 24;
  int N = 31;

  if (N > 0) {
      H = obs_map_current_[0].rows();
      W = obs_map_current_[0].cols();

      // 预分配空间
      flat_data.reserve(N * H * W);

      for (const auto& mat : obs_map_current_) {
          // 将每个 MatrixXf 的数据添加到 flat_data
          for (int i = 0; i < mat.rows(); ++i) {
              for (int j = 0; j < mat.cols(); ++j) {
                  flat_data.push_back(mat(i, j));
              }
          }
      }
  }
  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  Ort::Value parent_q_tensor = Ort::Value::CreateTensor<float>(
      memory_info, parent_q.data(), parent_q.size(), parent_q_shape.data(), parent_q_shape.size());

  Ort::Value goal_q_tensor = Ort::Value::CreateTensor<float>(
      memory_info, goal_q.data(), goal_q.size(), goal_q_shape.data(), goal_q_shape.size());

  // Ort::Value obs_map_tensor = Ort::Value::CreateTensor<float>(
  //     memory_info, obs_map_current_.data(), obs_map_current_.size(), obs_map_shape.data(), obs_map_shape.size());
  Ort::Value obs_map_tensor = Ort::Value::CreateTensor(
      memory_info,
      flat_data.data(),           // ✅ float*
      flat_data.size(),
      obs_map_shape.data(),
      obs_map_shape.size()
  );
  const char* input_names_c[] = { "parent_q", "goal_q", "obstacle_map" };
  Ort::Value* inputs[] = { &parent_q_tensor, &goal_q_tensor, &obs_map_tensor };
  // 替换你原来的 inputs 定义
  Ort::Value input_tensors[] = {
      std::move(parent_q_tensor),
      std::move(goal_q_tensor),
      std::move(obs_map_tensor)
  };
  std::vector<const char*> output_names_c;
  for (size_t i = 0; i < output_names.size(); ++i) {
      output_names_c.push_back(output_names[i].c_str());
  }
  auto output_tensors = session_presample_->Run(
      Ort::RunOptions{nullptr},
      input_names_c, input_tensors, 3,
      output_names_c.data(), output_names_c.size()  // ✅ 修正这里
  );
  auto& output_tensor = output_tensors[0];  // sampled_joint
  auto* output_data = output_tensor.template GetTensorMutableData<float>();
  for (size_t i = 0; i < num_joints_; ++i) {
    output.node.configuration[i] = output_data[i];
  }
  return output;
}


RRT_Node iRRT_CUpId::RRTConnectGenNode(std::vector<RRT_Node>& parent, double* RND_Mean_, double* STD_dev, RRT_Node* goal, double maxDis, bool flag_CCD)
{
  // 先基于强化学习根据障碍物的实时位置和速度方向进行非随机的预采样，如果采样失败了，再采用随机采样
  int count = 0, MaxIter = 700;
  RRT_Node output;
  output.node.in_Coli = false;
  std::vector<double> temp(num_joints_, 0);
  output.node.configuration = temp;
  output.iParent = 0;
  double norm = 0;
  double normMin = INFINITY;
  int point_i_Min;
  const std::vector<const moveit::core::JointModel*> joint_models = joint_model_group_->getActiveJointModels();
  std::random_device rd;
  std::mt19937 gen2(rd());                                      // 定义随机数生成器对象gen，使用time(nullptr)作为随机数生成器的种子
  std::normal_distribution<double> dis2(*RND_Mean_, *STD_dev);  // 定义随机数分布器对象dis，期望为0.0，标准差为1.0的正态分布
  double randTemp = 0;
  bool colli_flag = false;
  double maxDis_min = maxDis;
  double ratio = 0.95;
  bool RL_temp = do_presample_;

  while(count < MaxIter)
  {
    int point_i = floor((rand() % 200 / 200.0)*parent.size());
    if(point_i >= parent.size())
      point_i = parent.size() - 1;
    point_i_Min = point_i;
    if (is_variable_step_ == true)
    {
      if(count == 0)
        maxDis = sqrt(SED(&parent[point_i_Min], goal)) * 0.3;
      else
        maxDis = maxDis * ratio;

      if(maxDis < maxDis_min)
        maxDis = maxDis_min;
    }

    // 强化学习预采样,RLPresampleNode生成的点要在关节范围内,强化学习智能体输入部分主要考虑parent.back()、goal、障碍物的位置、尺寸及速度
    if (do_presample_ == true && RL_temp == true)
    {
      RRT_Node rl_sample = RLPresampleNode(parent[point_i_Min], *goal);
      output.node.configuration = rl_sample.node.configuration;
      RL_temp = false;
    }
    else
    {
      for (size_t joint_i = 0; joint_i < joint_models.size(); ++joint_i)
      {
        randTemp = dis2(gen2);
        output.node.configuration[joint_i] = randTemp;      //(rand() % 200 / 200.0)
      }
    }
    
    norm = 0;
    for(int joint_i = 0;joint_i < num_joints_;joint_i++)
    {
      norm += pow(output.node.configuration[joint_i] - parent[point_i].node.configuration[joint_i],2);
    }

    temp = output.node.configuration;
    
    normMin = sqrt(norm);

    if(upsampling_ == true)
      randTemp = (rand() % 100 / 100.0);
    else
      randTemp = 1.0;

    for(int j = 0;j < num_joints_;j++)
    {
      const moveit::core::JointModel* joint_model1 = joint_models[j];
      const moveit::core::JointModel::Bounds& bounds = joint_model1->getVariableBounds();

      output.node.configuration[j] = parent[point_i_Min].node.configuration[j] + (temp[j]-parent[point_i_Min].node.configuration[j]) / normMin * maxDis * randTemp;  //dis2(gen2);     //(rand() % 100 / 100.0);
      
      //确保不会超限
      if(bounds[0].min_position_ > output.node.configuration[j])
        output.node.configuration[j] = bounds[0].min_position_;

      if(bounds[0].max_position_ < output.node.configuration[j])
        output.node.configuration[j] = bounds[0].max_position_;
    }

    colli_flag = getPointColi(output.node.configuration);

    // 在此处加入扩充数据集的内容，并用bool变量do_export_dataset_控制是否导出数据集
    if(do_export_dataset_)
    {
      append_data(colli_flag, parent[point_i_Min].node.configuration, goal->node.configuration, parent[0].node.configuration, output.node.configuration);
    }

    if(!colli_flag)
    {
      return output;
    }
    count++;
  }

  output.node.in_Coli = true;
  return output;
}

Eigen::MatrixXd iRRT_CUpId::PathSplicing(Eigen::MatrixXd Path1, std::vector<RRT_Node> Path2, int iStart, int iEnd)
{
  int num_points = iStart + Path2.size() + (Path1.rows() - iEnd);
  Eigen::MatrixXd output(num_points, num_joints_);

  for(int i = 0;i < iStart;i++)
  {
    output.row(i) = Path1.row(i);
  }

  for(int i = 0;i < Path2.size();i++)
  {
    for(int j = 0;j < num_joints_;j++)
      output.row(i + iStart)[j] = Path2[i].node.configuration[j];
  }

  for(int i = 0;i < Path1.rows() - iEnd;i++)
  {
    output.row(i + iStart + Path2.size()) = Path1.row(i + iEnd);
  }

  return output;
}


bool iRRT_CUpId::getPointColi(std::vector<double>& joint_states)
{
  setRobotState(joint_states);

  collision_detection::CollisionRequest req;
  collision_detection::CollisionResult res;
  req.group_name = planning_group_;
  planning_scene_->checkCollision(req, res, state_, planning_scene_->getAllowedCollisionMatrix());
  return res.collision;
}

void iRRT_CUpId::printObstacleInfo(const planning_scene::PlanningScenePtr& scene)
{
  // 获取 world 的 const 引用，使用 begin()/end() 遍历
  const collision_detection::World& world = *scene->getWorld();
  scene->getWorld();
  RCLCPP_INFO(rclcpp::get_logger("obstacle_info"), "Found %zu collision objects in the planning scene:", world.size());

  // 使用迭代器遍历所有对象
  for (const auto& pair : world)
  {
    const std::string& object_id = pair.first;
    const collision_detection::World::ObjectConstPtr& obj = pair.second;

    RCLCPP_INFO(rclcpp::get_logger("obstacle_info"), "Obstacle ID: %s", object_id.c_str());

    // 遍历该物体的每一个形状
    for (size_t i = 0; i < obj->shapes_.size(); ++i)
    {
      std::cout << object_id.c_str() << "的信息：" << std::endl;
      // const shapes::ShapeConstPtr& shape = obj->shapes_[i];
      const Eigen::Isometry3d& pose = obj->global_shape_poses_[i];

      double x = pose.translation().x();
      double y = pose.translation().y();
      double z = pose.translation().z();
      
      std::cout << "x: " << x  << ", y: " << y  << ", z: " << z << std::endl;

      const shapes::ShapeConstPtr& shape = obj->shapes_[i];

      // 强制转换为 Box 类型
      const shapes::Box* box = static_cast<const shapes::Box*>(shape.get());

      double length = box->size[0];  // x
      double width  = box->size[1];  // y
      double height = box->size[2];  // z

      if(length * width * height > 0)
      {
        std::cout << "length: " << length << ", width: " << width << ", height: " << height << std::endl;
      }
    }
  }
}

bool iRRT_CUpId::getCCD(RRT_Node* Node1, RRT_Node* Node2, double Stepsize, bool mode)
{
  double TEMP = SED(Node1, Node2);
  int num = ceil(TEMP / Stepsize);   //* Node1->Co_colli_prob
  std::vector<double> joint_states(num_joints_, 0);
  // std::cout << "num: " << num  << ", TEMP: " << TEMP  << ", Stepsize: " << Stepsize << std::endl;
  for(int ii = 0;ii < num;ii++)
  {
    for(int j = 0;j < num_joints_;j++)
    {
      joint_states[j] = Node1->node.configuration[j] + (Node2->node.configuration[j] - Node1->node.configuration[j]) / (1.0 * num) * ii;
    }
    if(getPointColi(joint_states))
    {
      if(mode)
      {
        Node1->Co_colli_prob = (Node1->Co_colli_prob * Node1->Num_colli_prob + 1.0) / (Node1->Num_colli_prob + 1);
        Node1->Num_colli_prob += 1;
        Node2->Co_colli_prob = (Node2->Co_colli_prob * Node2->Num_colli_prob + 1.0) / (Node2->Num_colli_prob + 1);
        Node2->Num_colli_prob += 1;
      }
      return true;      //有碰撞
    }
  }
  if(mode)
  {
    Node1->Co_colli_prob = (Node1->Co_colli_prob * Node1->Num_colli_prob + 0.0) / (Node1->Num_colli_prob + 1);
    Node1->Num_colli_prob += 1;
    Node2->Co_colli_prob = (Node2->Co_colli_prob * Node2->Num_colli_prob + 0.0) / (Node2->Num_colli_prob + 1);
    Node2->Num_colli_prob += 1;
  }
  return false;         //无碰撞
}

int iRRT_CUpId::RRTConnectOnce(std::vector<RRT_Node>& output, RRT_Node* LocalStart, RRT_Node* LocalEnd, double maxDis, bool flag_CCD, bool init)
{
  //报错“free(): double free detected in tcache 2“的一个可能原因是没有return结果。
  //std::vector<double> temp(num_joints_, 0);
  std::vector<RRT_Node> Tree1, Tree2, Path1, Path2;

  int iTree1_out = 0, iTree2_out = 0;
  double threshold = 0.10;

  LocalStart->iParent = -1;
  LocalEnd->iParent = -1;

  Tree1.push_back(*LocalStart);
  Tree2.push_back(*LocalEnd);

  int count = 0;
  double distNearest = 0;
  RRT_Node Samp1;
  double RND_Mean;
  double STD_dev;
  int num_fail = 0;
  
  RND_Mean = 0.0;
  STD_dev = 1.0;
  output.clear();

  // std::cout << "RRTConnectOnce" << std::endl;

  // std::cout << "max_iterations_: " << max_iterations_ << std::endl;

  while(count < max_iterations_)
  {
    // std::cout << "count: " << count << std::endl;
    // std::cout << "===1===" << std::endl;
    if(count % 2 == 0)         
    {
      if(incremental_ && (flag_CCD || init) && count == 0)
      {
        if(!getCCD(LocalStart, LocalEnd, 0.05))
        {
          output.push_back(*LocalStart);
          output.push_back(*LocalEnd);
          // std::cout << "CCD: collision-free" << std::endl;
          // std::cout << "output.size(): " << output.size() << std::endl;
          return 2;
        }
      }
      
      auto inference_start = std::chrono::high_resolution_clock::now();
      Samp1 = RRTConnectGenNode(Tree1, &RND_Mean, &STD_dev, LocalEnd, maxDis, flag_CCD);
      auto inference_end = std::chrono::high_resolution_clock::now();
      auto duration_ms = std::chrono::duration<double, std::milli>(inference_end - inference_start).count();
      // std::cout << "⏱️ 采样耗时: " << duration_ms << " ms" << std::endl;
      if(Samp1.node.in_Coli == true) 
      {
        count++;
        continue;
      }
      
      if(!FindParentNode(&Samp1, Tree1, flag_CCD))
      {
        count++;
        continue;
      }
      Tree1.push_back(Samp1);
      Tree1_add_ = true;
      int iNearest = RRTConnectSteerNode(&Tree1.back(), Tree2, maxDis, flag_CCD);
    }
    else
    {
      Samp1 = RRTConnectGenNode(Tree2, &RND_Mean, &STD_dev, LocalStart, maxDis, flag_CCD);
      if(Samp1.node.in_Coli == true) 
      {
        count++;
        continue;
      }

      if(!FindParentNode(&Samp1, Tree2, flag_CCD))
      {
        count++;
        continue;
      }
      Tree2.push_back(Samp1);
      Tree2_add_ = true;
      int iNearest = RRTConnectSteerNode(&Tree2.back(), Tree1, maxDis, flag_CCD);
    }
    // std::cout << "===2===" << std::endl;
    //尝试将找到的两个近点用于引导随机点生成
    distNearest = FindNearestNodes(Tree1,Tree2,iTree1_out,iTree2_out,flag_CCD,threshold);
    // std::cout << "Distance_of_NearestNodes: " << distNearest << std::endl;
    Tree1_add_ = false;
    Tree2_add_ = false;
    if(distNearest < 0)
    {
      num_fail++;
      // std::cout << "num_fail: " << num_fail << std::endl;
      if(num_fail > 10 && upsampling_ == true)
      {
        // std::cout << "failed" << std::endl;
        return 0;
      }
      count++;
      continue;       //不continue的话会在后面的条件结构中跳出
    }
    else
    {
      num_fail = 0;
      if(flag_CCD)
        break;
    }
    // std::cout << "===3===" << std::endl;
    if(distNearest < maxDis)
    {
      break;
    }
    count++;
  }
  if(count >= max_iterations_)
  {
    std::cout << "count >= max_iterations_" << std::endl;
    return 0;
  }

  Path1.push_back(Tree1[iTree1_out]);
  Path2.push_back(Tree2[iTree2_out]);

  RRT_Node temp1 = Path1.back();
  RRT_Node temp2 = Path2.back();

  while(temp1.iParent != -1 && Tree1[temp1.iParent].iParent != -1)    //倒序
  {
    Path1.push_back(temp1);
    temp1 = Tree1[temp1.iParent];
  }
  Path1.push_back(Tree1[0]);

  while(temp2.iParent != -1 && Tree2[temp2.iParent].iParent != -1)    //正序
  {
    Path2.push_back(temp2);
    temp2 = Tree2[temp2.iParent];
  }
  Path2.push_back(Tree2[0]);

  // std::cout << "Path1.size(): " << Path1.size() << std::endl;
  // std::cout << "Path2.size(): " << Path2.size() << std::endl;

  for(int i = Path1.size() - 1;i > -1;i--)
  {
    output.push_back(Path1[i]);
    // std::cout << "Path1: " << std::endl;
    // for(int j = 0;j < num_joints_;j++)
    // {
    //   std::cout << Path1[i].node.configuration[j] << ", ";
    // }
    // std::cout << std::endl;
  }
  for(int i = 0;i < Path2.size();i++)
  {
    output.push_back(Path2[i]);
    // std::cout << "Path2: " << std::endl;
    // for(int j = 0;j < num_joints_;j++)
    // {
    //   std::cout << Path2[i].node.configuration[j] << ", ";
    // }
    // std::cout << std::endl;
  }

  return 1;
}

void iRRT_CUpId::Simplify(std::vector<RRT_Node>& input, std::vector<RRT_Node>& output, double maxDis, int flag_CCD, double length_total)
{
  // 路径简化
  int i_pre = 0, i_post = 2;
  bool flag_avai = false;
  std::vector<RRT_Node> TEMP_Path;
  TEMP_Path.push_back(input[i_pre]);
  if(input.size() < 3)
  {
    output = input;   //因为两个点的路径的可达性已经得到了验证
    return;    
  }
  else
  {
    auto sq_maxDis = maxDis * maxDis;
    while(i_post < input.size())
    {
      flag_avai = false;
      auto i_post_temp = i_post;
      for(int i = i_post_temp; i < input.size(); i++)
      {
        if(!flag_CCD && SED(&input[i_pre], &input[i]) <= sq_maxDis)  
        { 
          i_post = i + 1;
          flag_avai = true;
        }
        else if(flag_CCD && (SED(&input[i_pre], &input[i]) <= length_total / 10.0) && !getCCD(&input[i_pre], &input[i], 0.05, true))
        {
          i_post = i + 1;
          flag_avai = true;
        }
      }

 
      if(flag_avai)
      {
        TEMP_Path.push_back(input[i_post - 1]);
        i_pre = i_post - 1;
        i_post = i_pre + 2;
      }
      else
      {
        TEMP_Path.push_back(input[i_pre + 1]);
        i_pre = i_pre + 1;
        i_post = i_pre + 2;
      }

      if(i_post == input.size())
      {
        TEMP_Path.push_back(input.back());
        break;
      }
      else if(i_post > input.size())
      {
        break;
      }
    }

    output = TEMP_Path;
    return; 
  }
}

void iRRT_CUpId::Simplify_longest(std::vector<RRT_Node>& input, std::vector<RRT_Node>& output, double maxDis)
{
    // 路径简化思路2：查找最远距离的三个相邻点，若SED大于sq_maxDis，尝试线性插补至间隔小于sq_maxDis
    // 若插补点发生碰撞，则不对此处进行简化，否则移除原始点并插入插补点，如此重复直至所有点都无法进一步简化
  int i_pre = 0, i_post = 2;
  bool flag_avai = false;
  std::vector<RRT_Node> TEMP_Path;
  TEMP_Path.push_back(input[i_pre]);
  if(input.size() < 3)
  {
    output = input;   //因为两个点的路径的可达性已经得到了验证
    return;    
  }
  else
  {
    auto sq_maxDis = maxDis * maxDis;
    while(i_post < input.size())
    {
      flag_avai = false;
      auto i_post_temp = i_post;
      for(int i = i_post_temp; i < input.size(); i++)
      {
        if(SED(&input[i_pre], &input[i]) <= sq_maxDis)  
        { 
          i_post = i + 1;
          flag_avai = true;
        }
      }

      if(flag_avai)
      {
        TEMP_Path.push_back(input[i_post - 1]);
        i_pre = i_post - 1;
        i_post = i_pre + 2;
      }
      else
      {
        TEMP_Path.push_back(input[i_pre + 1]);
        i_pre = i_pre + 1;
        i_post = i_pre + 2;
      }

      if(i_post == input.size())
      {
        TEMP_Path.push_back(input.back());
        break;
      }
      else if(i_post > input.size())
      {
        break;
      }
    }

    output = TEMP_Path;
    return; 
  }
}


bool iRRT_CUpId::RRTConnectToTrajectory(std::vector<RRT_Node>& output, double& length)
{
  std::vector<double> ColiPath_start(num_joints_, 0);
  std::vector<double> ColiPath_end(num_joints_, 0);
  std::vector<RRT_Node> TEMP_Path, TEMP_Path2;
  RRT_Node LocalStart, LocalEnd, TEMP_Point;
  bool ori = false;
  bool init = true;

  for(int j = 0;j < num_joints_;j++)
    ColiPath_start[j] = group_trajectory_.getTrajectory().row(free_vars_start_)[j];

  for(int j = 0;j < num_joints_;j++)
    ColiPath_end[j] = group_trajectory_.getTrajectory().row(free_vars_end_-1)[j];

  LocalStart.node.configuration = ColiPath_start;
  LocalEnd.node.configuration = ColiPath_end;
  TEMP_Point.node.configuration = ColiPath_start;
  LocalStart.iParent = -1;
  LocalEnd.iParent = -1;
  TEMP_Point.iParent = -1;


  if(SED(&LocalStart,&LocalEnd) < 1e-3)
    return true;

  double maxDis = 0;
  double length_total = SED(&LocalStart, &LocalEnd);
  double maxDis_threshold = 0.2;
  int flag = 0;
  int Path_Size;
  bool flag_CCD = false, coll_last = false;
  double TEMP = SED(&LocalStart, &LocalEnd);
  double Stepsize = 0.05;
  int num = ceil(TEMP / Stepsize);
  TEMP_Path2.push_back(LocalStart);
  TEMP_Path2.push_back(LocalEnd);

  maxDis = 0;
  for(int i = 0;i < TEMP_Path2.size() - 1;i++)
  {
    TEMP = SED(&TEMP_Path2[i], &TEMP_Path2[i + 1]);
    if(maxDis < TEMP)
      maxDis = TEMP;
  }

  if(maxDis <= Stepsize)
  {
    length = 0;
    output = TEMP_Path2;
    for(int i = 0;i < output.size() - 1;i++)
    {
      double max_diff = 0;
      for(int j = 0;j < num_joints_;j++)
      {
        if(fabs(output[i].node.configuration[j]-output[i+1].node.configuration[j]) > max_diff)
        {
          max_diff = fabs(output[i].node.configuration[j]-output[i+1].node.configuration[j]);
        }
      }
      length += max_diff;
    }
    return true;
  }

  maxDis /= 2;

  if(upsampling_ == false)
    maxDis = maxDis_threshold;    //如果要切换为原始RRT-connect则使用这一行，否则注释掉这一行

  while(init || maxDis >= maxDis_threshold)
  {
    Path_Size = TEMP_Path2.size();
    output.clear();
    output.push_back(TEMP_Path2[0]);
    if(maxDis == maxDis_threshold)
    {
      flag_CCD = true;
    }

    for(int i = 0;i < Path_Size - 1;i++)
    {
      LocalStart = TEMP_Path2[i];
      LocalEnd = TEMP_Path2[i + 1];

      flag = RRTConnectOnce(TEMP_Path, &LocalStart, &LocalEnd, maxDis, flag_CCD, init);
      if(flag == 0)
      {
        break;
      }

      if(flag > 0)
      {
        for(int j = 1;j < TEMP_Path.size();j++)
        {
          output.push_back(TEMP_Path[j]);
        }
      }
      TEMP_Path.clear();
      if(init && flag == 2)
      {
        length = length_total;
        return true;
      }
      init = false;
    }

    if(!flag)
    {
      break;
    }

    if(DPS_ == true)
    {
      if(!flag_CCD)        //最后做连续碰撞检测的一轮时不做简化，以避免影响可达性
      {
        int size1, size2, size3;
        while(true)
        {
          // std::cout << "output_1.size(): " << output.size() << std::endl;
          size1 = output.size();
          std::vector<RRT_Node> output_reverse;
          for(int i = output.size() - 1; i > -1; i--)
          {
            output[i].isCCDfree = false;
            output_reverse.push_back(output[i]);
          }
          // std::cout << "output_2.size(): " << output.size() << std::endl;
          size2 = output.size();
          Simplify(output_reverse, output_reverse, maxDis, flag_CCD, length_total);
          output.clear();
          for(int i = output_reverse.size() - 1; i > -1; i--)
          {
            output.push_back(output_reverse[i]);
          }
          Simplify(output, output, maxDis, flag_CCD, length_total);
          // std::cout << "output_3.size(): " << output.size() << std::endl;
          size3 = output.size();

          if(size1 == size2 && size1 == size3)
            break;
        }
      }
      else
      {
        Simplify(output, output, maxDis, flag_CCD, length_total);
      }
    }

    TEMP_Path2 = output;

    if(flag_CCD)
    {
      break;
    }

    maxDis /= 2;
    if(maxDis < maxDis_threshold)
    {
      maxDis = maxDis_threshold;
    }
  }
  output = TEMP_Path2;

  if(!flag)
    return false;
  else
  {
    length = 0;
    // 在此处加入产生数据集的内容，并用bool变量do_export_dataset_控制是否产生数据集
    for(int i = 0;i < output.size() - 1;i++)
    {
      double max_diff = 0;
      max_diff = SED(&output[i], &output[i+1]);
      length += sqrt(max_diff);
    }

    return true;
  }
  
}

void iRRT_CUpId::append_data(bool is_collision, std::vector<double> parent_q, std::vector<double> goal_q, std::vector<double> start_q, std::vector<double> sampled_q)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> norm(0.0f, 0.1f);
  std::vector<Eigen::MatrixXd> obs_map;
  obs_map = createOccupancyMapVolume(planning_scene_, 0.05);
  
  // 计算奖励所需中间量
  double dist_to_start = (Eigen::Map<Eigen::VectorXd>(sampled_q.data(), sampled_q.size()) - 
                   Eigen::Map<Eigen::VectorXd>(start_q.data(), start_q.size())).norm();
  double dist_to_goal = (Eigen::Map<Eigen::VectorXd>(sampled_q.data(), sampled_q.size()) - 
                   Eigen::Map<Eigen::VectorXd>(goal_q.data(), goal_q.size())).norm();
  double dist_to_parent = (Eigen::Map<Eigen::VectorXd>(sampled_q.data(), sampled_q.size()) - 
                   Eigen::Map<Eigen::VectorXd>(parent_q.data(), parent_q.size())).norm();
  
  double path_length = dist_to_goal + dist_to_start;

  double rand_num = norm(gen);

  float reward = 1.0f
              - 0.1f * dist_to_goal
              - 0.05f * path_length
              - 5.0f * static_cast<float>(is_collision)
              + rand_num; // 噪声

  // 缓存用于后续扩展
  parent_q_list_.push_back(Eigen::Map<Eigen::VectorXd>(parent_q.data(), parent_q.size()));
  goal_q_list_.push_back(Eigen::Map<Eigen::VectorXd>(goal_q.data(), goal_q.size()));
  sampled_q_list_.push_back(Eigen::Map<Eigen::VectorXd>(sampled_q.data(), sampled_q.size()));
  obs_map_list_.push_back(obs_map);
  reward_list_.push_back(reward);

  // 展平并保存
  for (int i = 0; i < parent_q.size(); ++i) {
      parent_q_flat_.push_back(parent_q[i]);
      goal_q_flat_.push_back(goal_q[i]);
      sampled_q_flat_.push_back(sampled_q[i]);
      reward_list_flat_.push_back(reward);
  }
  for (int y = 0; y < obs_map[0].rows(); ++y)
      for (int x = 0; x < obs_map[0].cols(); ++x)
          obs_map_flat_.push_back(obs_map[0](y, x));

  dataset_.parent_q_list = parent_q_list_;
  dataset_.goal_q_list = goal_q_list_;
  dataset_.sampled_q_list = sampled_q_list_;
  dataset_.obs_map_list = obs_map_list_;
  dataset_.reward_list = reward_list_;
  dataset_.n_dof = num_joints_;
  dataset_.map_rows = obs_map[0].rows();
  dataset_.map_cols = obs_map[0].cols();
  dataset_.num_samples = parent_q_list_.size();
}


// === 保存为二进制文件 ===
void iRRT_CUpId::saveToBinary(const Dataset& data, const std::string& filename) {
  if (data.num_samples < 1) {
      std::cout << "数据集为空，跳过保存。" << std::endl;
      return;
  }

  // 新增：获取 map_channels (Z)
  int map_channels = data.obs_map_list[0].size();  // 假设所有样本 channel 数一致
  for (const auto& obs : data.obs_map_list) {
      if (obs.size() != map_channels) {
          throw std::runtime_error("所有 obs_map 的通道数必须一致！");
      }
  }

  std::fstream file(filename, std::ios::in | std::ios::out | std::ios::binary);
  bool file_exists = file.is_open();

  int total_samples = 0;
  int n_dof = 0, map_rows = 0, map_cols = 0, existing_channels = 0;

  if (!file_exists) {
      file.open(filename, std::ios::out | std::ios::binary);
      if (!file.is_open()) {
          throw std::runtime_error("无法创建文件: " + filename);
      }

      total_samples = data.num_samples;
      n_dof = data.n_dof;
      map_rows = data.map_rows;
      map_cols = data.map_cols;

      // ✅ 写入 5 个 int：total_samples, n_dof, map_rows, map_cols, map_channels
      file.write(reinterpret_cast<const char*>(&total_samples), sizeof(int));
      file.write(reinterpret_cast<const char*>(&n_dof),      sizeof(int));
      file.write(reinterpret_cast<const char*>(&map_rows),   sizeof(int));
      file.write(reinterpret_cast<const char*>(&map_cols),   sizeof(int));
      file.write(reinterpret_cast<const char*>(&map_channels), sizeof(int));  // 新增！

      std::cout << "📝 新建文件，写入元信息: " << filename
                << " (channels=" << map_channels << ")" << std::endl;
  } else {
      // 读取现有元信息（包括 map_channels）
      file.read(reinterpret_cast<char*>(&total_samples), sizeof(int));
      file.read(reinterpret_cast<char*>(&n_dof),      sizeof(int));
      file.read(reinterpret_cast<char*>(&map_rows),   sizeof(int));
      file.read(reinterpret_cast<char*>(&map_cols),   sizeof(int));
      file.read(reinterpret_cast<char*>(&existing_channels), sizeof(int));  // 新增！

      if (n_dof != data.n_dof || map_rows != data.map_rows || 
          map_cols != data.map_cols || map_channels != existing_channels) {
          throw std::runtime_error("数据维度不匹配！现有: (" + std::to_string(n_dof) +
                                  "," + std::to_string(map_rows) + "x" + std::to_string(map_cols) +
                                  "x" + std::to_string(existing_channels) +
                                  "), 新数据: (" + std::to_string(data.n_dof) +
                                  "," + std::to_string(data.map_rows) + "x" + std::to_string(data.map_cols) +
                                  "x" + std::to_string(map_channels) + ")");
      }

      int new_total = total_samples + data.num_samples;
      file.seekp(0, std::ios::beg);
      file.write(reinterpret_cast<const char*>(&new_total), sizeof(int));
      file.seekp(0, std::ios::end);
      total_samples = new_total;

      std::cout << "📎 文件已存在，更新总样本数: " << new_total << " (新增 " << data.num_samples << ")" << std::endl;
  }

  auto write = [&file](const void* ptr, size_t size) {
      file.write(static_cast<const char*>(ptr), size);
  };

  for (int i = 0; i < data.num_samples; ++i) {
      write(data.parent_q_list[i].data(), data.n_dof * sizeof(double));
      write(data.goal_q_list[i].data(), data.n_dof * sizeof(double));
      write(data.sampled_q_list[i].data(), data.n_dof * sizeof(double));

      // 写 obs_map: Z x H x W，按 z 优先顺序
      for (int z = 0; z < map_channels; ++z) {
          for (int y = 0; y < map_rows; ++y) {
              for (int x = 0; x < map_cols; ++x) {
                  float val = static_cast<float>(data.obs_map_list[i][z](y, x));
                  write(&val, sizeof(float));
              }
          }
      }

      write(&data.reward_list[i], sizeof(double));
  }

  file.close();
  std::cout << "✅ 数据已追加，当前总样本数: " << total_samples << std::endl;
}


std::vector<Eigen::MatrixXd> iRRT_CUpId::createOccupancyMapVolume(
    const planning_scene::PlanningScenePtr& scene,
    double resolution)
{
    // 工作空间边界
    const double min_x = -0.6, max_x = 0.6;
    const double min_y = -0.6, max_y = 0.6;
    const double min_z = -0.45, max_z = 1.1;

    // 计算网格数量
    int nx = static_cast<int>(std::ceil((max_x - min_x) / resolution)); // 24
    int ny = static_cast<int>(std::ceil((max_y - min_y) / resolution)); // 24
    int nz = static_cast<int>(std::floor((max_z - min_z) / resolution)); // 31

    // 创建输出容器：31 层，每层 24x24
    std::vector<Eigen::MatrixXd> obs_map_volume(nz, Eigen::MatrixXd::Zero(nx, ny));

    const collision_detection::World& world = *scene->getWorld();

    // 遍历所有障碍物
    for (const auto& pair : world) {
        const collision_detection::World::ObjectConstPtr& obj = pair.second;

        for (size_t i = 0; i < obj->shapes_.size(); ++i) {
            const shapes::ShapeConstPtr& shape = obj->shapes_[i];
            const Eigen::Isometry3d& pose = obj->global_shape_poses_[i];

            // if (shape->type != shapes::BOX) {
            //     RCLCPP_WARN(rclcpp::get_logger("occupancy_map"), 
            //                 "跳过非 Box 形状: %s", shapes::shapeString(shape->type).c_str());
            //     continue;
            // }

            const shapes::Box* box = static_cast<const shapes::Box*>(shape.get());
            Eigen::Vector3d size(box->size[0], box->size[1], box->size[2]);

            // Box 在世界坐标系中的 AABB（轴对齐包围盒）
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

            // 找出与该 box 相交的 Z 层索引范围
            int z_start_idx = std::max(0, static_cast<int>(std::floor((bz_min - min_z) / resolution)));
            int z_end_idx   = std::min(nz - 1, static_cast<int>(std::floor((bz_max - min_z) / resolution)));

            // 遍历所有相交的 Z 层
            for (int z_idx = z_start_idx; z_idx <= z_end_idx; ++z_idx) {
                double slice_z = min_z + (z_idx + 0.5) * resolution; // 当前层中心高度

                // 判断该层是否与 box 相交
                double layer_z_min = min_z + z_idx * resolution;
                double layer_z_max = layer_z_min + resolution;

                if (bz_max >= layer_z_min && bz_min <= layer_z_max) {
                    // 获取该层对应的 map 引用
                    Eigen::MatrixXd& map = obs_map_volume[z_idx];

                    // 计算 box 在 XY 平面的投影，并映射到网格
                    int x_start = std::max(0, static_cast<int>(std::floor((bx_min - min_x) / resolution)));
                    int x_end   = std::min(nx - 1, static_cast<int>(std::floor((bx_max - min_x) / resolution)));
                    int y_start = std::max(0, static_cast<int>(std::floor((by_min - min_y) / resolution)));
                    int y_end   = std::min(ny - 1, static_cast<int>(std::floor((by_max - min_y) / resolution)));

                    // 填充占据格子
                    for (int i = x_start; i <= x_end; ++i)
                        for (int j = y_start; j <= y_end; ++j)
                            map(i, j) = 1.0;
                }
            }
        }
    }

    return obs_map_volume;
}

void iRRT_CUpId::updateFullTrajectory()
{
  full_trajectory_->updateFromGroupTrajectory(group_trajectory_);
}

void iRRT_CUpId::setRobotState(std::vector<double>& joint_states)
{
  state_.setJointGroupPositions(planning_group_, joint_states);
  state_.update();
}


}  // namespace my_planner

