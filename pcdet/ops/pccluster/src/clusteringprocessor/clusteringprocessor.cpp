#include "clusteringprocessor.h"
namespace depth_clustering {
using namespace std;
// using namespace clustering;

void ClusteringProcessor::EigenToCloud(const Eigen::MatrixXf& cloud_in,
                                       Cloud& cloud) {
  if (cloud_in.cols() != 3) {
    std::cout << "clouds only need xyz information !" << endl;
    return;
  }
  int cnt = 0;
  for (int i = 0; i < cloud_in.rows(); i++) {
    cloud.push_back(
        RichPoint(cloud_in.row(i)(0), cloud_in.row(i)(1), cloud_in.row(i)(2)));
    cnt++;
  }
  if (cparams_.verbose) std::cout << "the num of points is:  " << cnt << endl;
}

Eigen::MatrixX3f ClusteringProcessor::toEigenCloud3(vector<pcl::PointXYZ>& cloud)
{
    Eigen::MatrixX3f dst(cloud.size(), 3);
    int j=0;
    for (auto &p: cloud) {
        dst.row(j++) << p.x, p.y, p.z;
    }
    return dst;
}
void ClusteringProcessor::process_cluster(const Eigen::MatrixXf& cloud_in)
// void ClusteringProcessor::process_cluster(const Cloud& cloud_in) 
{
  time_utils::Timer timer;
  time_taken_ = 0;
  bool cloud_empty = cloud_cluster_.empty();
  if(!cloud_empty)
  {cloud_cluster_.clear();
  cloud_noncluster_.clear();
  // clusterer_pc.clear();
  cloud_clu_.clear_points();
  cloud_clu_.projection_ptr()->ClearDepthImage();}
  // pcl::PointCloud<pcl::PointXYZ> cloud_cluster_, cloud_noncluster_;
  // if(cv::countNonZero(cloud_clu_.projection_ptr()->depth_image()) <1)
  // {
  //   std::cout << "INFO: cloud_clu_.projection_ptr()->depth_image clear:" << endl;
  // }
  // else{
  //   std::cout << "ERROR: depth_image NOT ZERO:" << endl;
  // }
  std::unordered_map<uint16_t, Cloud> clusterer_pc;
  if (cparams_.verbose){
    std::cout << "INFO: It took  us to clear:"  << timer.measure() << endl;}
  EigenToCloud(cloud_in, cloud_clu_);
  if (cparams_.verbose){
    std::cout << "INFO: cloud_cluster_.size:"  << cloud_cluster_.size()<< "  cloud_nocluster_.size:"  << cloud_noncluster_.size() << "  cloud_clu_.size:"  << cloud_clu_.size() << endl;}
    if (cparams_.verbose){
    std::cout << "INFO: It took  us to EigenToCloud:"  << timer.measure() << endl;}
  // cloud_clu_.projection_ptr() = cloud_clu_.projection_ptr()->Clone();
  cloud_clu_.projection_ptr()->InitFromPoints(cloud_clu_.points());
  // cloud_clu_.InitProjection(*proj_params_ptr);
      if (cparams_.verbose){
    std::cout << "INFO: It took  us to InitProjection:"  << timer.measure() << endl;}
  clusterer_.cluster_nonground(cloud_clu_, clusterer_pc);
        if (cparams_.verbose){
    std::cout << "INFO: It took  us to cluster_nonground:"  << timer.measure() << endl;}
  int cluster_cnt = 0;
  int nocluster_cnt = 0;
  for (const auto& kv : clusterer_pc) {
    const auto& one_group_pc = kv.second;
    Eigen::Vector3f max_point(std::numeric_limits<float>::lowest(),
                              std::numeric_limits<float>::lowest(),
                              std::numeric_limits<float>::lowest());
    Eigen::Vector3f min_point(std::numeric_limits<float>::max(),
                              std::numeric_limits<float>::max(),
                              std::numeric_limits<float>::max());

    for (const auto& point : one_group_pc.points()) {
      min_point << std::min(min_point.x(), point.x()),
          std::min(min_point.y(), point.y()),
          std::min(min_point.z(), point.z());
      max_point << std::max(max_point.x(), point.x()),
          std::max(max_point.y(), point.y()),
          std::max(max_point.z(), point.z());
    }
    if ((std::abs(max_point.x() - min_point.x() > cparams_.dist_thr) ||
        std::abs(max_point.y() - min_point.y() > cparams_.dist_thr) ||
        ((pow((max_point.x() - min_point.x()), 2) +
          pow((max_point.y() - min_point.y()), 2)) >= pow(1.2f * cparams_.dist_thr, 2))) && std::abs(max_point.z() - min_point.z() > 0.8f)) {
      for (const auto& point : one_group_pc.points()) {
        pcl::PointXYZ pcl_point;
        pcl_point.x = point.x();
        pcl_point.y = point.y();
        pcl_point.z = point.z();
        cloud_cluster_.push_back(pcl_point);
        cluster_cnt++;
      }
    } else {
      for (const auto& point : one_group_pc.points()) {
        pcl::PointXYZ pcl_point;
        pcl_point.x = point.x();
        pcl_point.y = point.y();
        pcl_point.z = point.z();
        cloud_noncluster_.push_back(pcl_point);
        nocluster_cnt++;
      }
    }
  }
      if (cparams_.verbose){
    std::cout << "INFO: It took  us to houchuli:"  << timer.measure() << endl;}
  time_taken_ = timer.measure();
  if (cparams_.verbose){
    std::cout << "INFO: It took  us to process all:"  << time_taken_ << endl;
    std::cout << "source points num: " << cloud_in.size()/3 << endl;
    std::cout << "Cluster points num: " << cluster_cnt << "     noncluster points num: " << nocluster_cnt << endl;}
}

}