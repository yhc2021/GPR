#ifndef CLUSTERINGPROCESSOR_H_
#define CLUSTERINGPROCESSOR_H_

#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <map>
#include <math.h>
#include <numeric>
#include <vector>
#include <pcl/common/common.h>
#include <pcl/point_types.h>
// #include <pcl_ros/point_cloud.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>

#include "clusterers/image_based_clusterer.h"
#include "image_labelers/abstract_image_labeler.h"
#include "image_labelers/diff_helpers/diff_factory.h"
#include "projections/projection_params.h"
#include "utils/cloud.h"
#include "utils/mem_utils.h"
#include "utils/radians.h"
#include "utils/timer.h"
namespace depth_clustering
{
  using namespace std;
  using mem_utils::make_unique;

  struct Params
  {
    bool verbose;

    Radians angle_tollerance;
    int angle;
    int min_cluster_size;
    int max_cluster_size;
    float dist_thr;
    uint8_t dataset_flag;

    Params()
    {
      verbose = false;
      angle = 10;
      dist_thr = 5.0;
      // proj_params_ptr = ProjectionParams::HDL_64();
      angle_tollerance = Radians::FromDegrees(angle);
      min_cluster_size = 20;
      max_cluster_size = 100000;
      dataset_flag = 0;
    }
    // Params(const Params &p)
    // {
    //   verbose = p.verbose;
    //   // proj_params_ptr = ProjectionParams::HDL_64();
    //   angle_tollerance = p.angle_tollerance;
    //   min_cluster_size = p.min_cluster_size;
    //   max_cluster_size = p.max_cluster_size;
    // }
    // ~Params(){
    //     if (proj_params_ptr != nullptr)
    //     {proj_params_ptr.reset();
    //     proj_params_ptr =nullptr; }
    // }
  };

  class ClusteringProcessor
  {
  public:
    ClusteringProcessor(const depth_clustering::Params &_params)
        : cparams_(_params),
          clusterer_(_params.angle_tollerance, _params.min_cluster_size,
                     _params.max_cluster_size)
    {
      clusterer_.SetDiffType(DiffFactory::DiffType::ANGLES);
      if (_params.dataset_flag == 0){
        cloud_clu_.SetProjectionPtr(CloudProjection::Ptr(new SphericalProjection(*proj_kitti_ptr)));
      }
      else{
        cloud_clu_.SetProjectionPtr(CloudProjection::Ptr(new SphericalProjection(*proj_waymo_ptr)));
      }

          }
    // ~ClusteringProcessor(){}

    std::unique_ptr<ProjectionParams> proj_kitti_ptr = ProjectionParams::HDL_64();
    std::unique_ptr<ProjectionParams> proj_waymo_ptr = ProjectionParams::HDL_64_WAYMO();
    Params cparams_;
    ImageBasedClusterer<LinearImageLabeler<>> clusterer_;
    uint64_t time_taken_;
    Cloud cloud_clu_;

    vector<pcl::PointXYZ> cloud_cluster_, cloud_noncluster_;
    void process_cluster(const Eigen::MatrixXf &cloud_in);
    // void process_cluster(const Cloud& cloud_in);
    void EigenToCloud(const Eigen::MatrixXf &cloud_in, Cloud &cloud_clu);
    Eigen::MatrixX3f toEigenCloud3(vector<pcl::PointXYZ> &cloud);
    Eigen::MatrixXf getCluster() { return toEigenCloud3(cloud_cluster_); }
    Eigen::MatrixXf getNoncluster() { return toEigenCloud3(cloud_noncluster_); }
    uint64_t getTimeTakenUs() { return time_taken_; }
  };
}; // namespace depth_clustering
#endif
