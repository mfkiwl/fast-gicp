#include <chrono>
#include <iostream>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/approximate_voxel_grid.h>

#include <pcl/registration/ndt.h>
#include <pcl/registration/gicp.h>
#include <fast_gicp/gicp/fast_gicp.hpp>
#include <fast_gicp/gicp/fast_gicp_st.hpp>
#include <fast_gicp/gicp/fast_vgicp.hpp>
#include <ros/ros.h>

#ifdef USE_VGICP_CUDA
#include <fast_gicp/ndt/ndt_cuda.hpp>
#include <fast_gicp/gicp/fast_vgicp_cuda.hpp>
#endif

// benchmark for PCL's registration methods
template <typename Registration>
void test_pcl(Registration& reg, const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& target, const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& source) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>);

  double fitness_score = 0.0;

  // single run
  auto t1 = std::chrono::high_resolution_clock::now();
  reg.setInputTarget(target);
  reg.setInputSource(source);
  reg.align(*aligned);
  auto t2 = std::chrono::high_resolution_clock::now();
  fitness_score = reg.getFitnessScore();
  double single = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6;
  std::cout << "align result: \n" << reg.getFinalTransformation() << std::endl;
  std::cout << "single:" << single << "[msec] " << std::endl;

  // // 100 times
  // t1 = std::chrono::high_resolution_clock::now();
  // for (int i = 0; i < 100; i++) {
  //   reg.setInputTarget(target);
  //   reg.setInputSource(source);
  //   reg.align(*aligned);
  // }
  // t2 = std::chrono::high_resolution_clock::now();
  // double multi = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6;
  // std::cout << "100times:" << multi << "[msec] fitness_score:" << fitness_score << std::endl;
}

// benchmark for fast_gicp registration methods
template <typename Registration>
void test(Registration& reg, const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& target, const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& source) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>);

  double fitness_score = 0.0;

  // single run
  auto t1 = std::chrono::high_resolution_clock::now();
  // fast_gicp reuses calculated covariances if an input cloud is the same as the previous one
  // to prevent this for benchmarking, force clear source and target clouds
  reg.clearTarget();
  reg.clearSource();
  reg.setInputTarget(target);
  reg.setInputSource(source);
  reg.align(*aligned);
  auto t2 = std::chrono::high_resolution_clock::now();
  fitness_score = reg.getFitnessScore();
  double single = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6;
  std::cout << "align result: \n" << reg.getFinalTransformation() << std::endl; 
  std::cout << "single:" << single << "[msec] " << std::endl;

  // // 100 times
  // t1 = std::chrono::high_resolution_clock::now();
  // for (int i = 0; i < 100; i++) {
  //   reg.clearTarget();
  //   reg.clearSource();
  //   reg.setInputTarget(target);
  //   reg.setInputSource(source);
  //   reg.align(*aligned);
  // }
  // t2 = std::chrono::high_resolution_clock::now();
  // double multi = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6;
  // std::cout << "100 times:" << multi << "[msec] " << std::flush;

  // // for some tasks like odometry calculation,
  // // you can reuse the covariances of a source point cloud in the next registration
  // t1 = std::chrono::high_resolution_clock::now();
  // pcl::PointCloud<pcl::PointXYZ>::ConstPtr target_ = target;
  // pcl::PointCloud<pcl::PointXYZ>::ConstPtr source_ = source;
  // for (int i = 0; i < 100; i++) {
  //   reg.swapSourceAndTarget();
  //   reg.clearSource();

  //   reg.setInputTarget(target_);
  //   reg.setInputSource(source_);
  //   reg.align(*aligned);

  //   target_.swap(source_);
  // }
  // t2 = std::chrono::high_resolution_clock::now();
  // double reuse = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6;

  // std::cout << "100 times_reuse:" << reuse << "[msec] fitness_score:" << fitness_score << std::endl;
}

/**
 * @brief main
 */
int main(int argc, char** argv) {
  ros::init(argc,argv,"align_node");
  ros::NodeHandle nh("~");
  std::string pcd1 = nh.param<std::string>("pcd_1","");
  std::string pcd2 = nh.param<std::string>("pcd_2","");
  std::string vgicp_direct_method = nh.param<std::string>("vgicp_direct_method","DIRECT1");
  double vgicp_resolution = nh.param<double>("vgicp_resolution",1.0);
  double transform_epsilon = nh.param<double>("transform_epsilon",1e-2);
  double MaxCorrespondenceDistance = nh.param<double>("MaxCorrespondenceDistance",0.5);
  int num_threads = nh.param<int>("num_threads",4);
  fast_gicp::NeighborSearchMethod nn_method;
  if(vgicp_direct_method == "DIRECT1"){
    nn_method = fast_gicp::NeighborSearchMethod::DIRECT1;
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>());

  if (pcl::io::loadPCDFile(pcd1, *target_cloud)) {
    std::cerr << "failed to open " << pcd1 << std::endl;
    return 1;
  }
  if (pcl::io::loadPCDFile(pcd2, *source_cloud)) {
    std::cerr << "failed to open " << pcd2 << std::endl;
    return 1;
  }
  int a = 0;
  // remove invalid points around origin
  source_cloud->erase(
    std::remove_if(source_cloud->begin(), source_cloud->end(), [=](const pcl::PointXYZ& pt) { return pt.getVector3fMap().squaredNorm() < 1e-3; }),
    source_cloud->end());
  target_cloud->erase(
    std::remove_if(target_cloud->begin(), target_cloud->end(), [=](const pcl::PointXYZ& pt) { return pt.getVector3fMap().squaredNorm() < 1e-3; }),
    target_cloud->end());

  // downsampling
  pcl::VoxelGrid<pcl::PointXYZ> voxelgrid;
  voxelgrid.setLeafSize(0.1f, 0.1f, 0.1f);

  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>());
  voxelgrid.setInputCloud(target_cloud);
  voxelgrid.filter(*filtered);
  target_cloud = filtered;

  filtered.reset(new pcl::PointCloud<pcl::PointXYZ>());
  voxelgrid.setInputCloud(source_cloud);
  voxelgrid.filter(*filtered);
  source_cloud = filtered;

  std::cout << "target:" << target_cloud->size() << "[pts] source:" << source_cloud->size() << "[pts]" << std::endl;

  // std::cout << "--- pcl_gicp ---" << std::endl;
  // pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> pcl_gicp;
  // pcl_gicp.setTransformationEpsilon(1e-2);
  // pcl_gicp.setMaxCorrespondenceDistance(0.5);
  // test_pcl(pcl_gicp, target_cloud, source_cloud);

  // std::cout << "--- pcl_ndt ---" << std::endl;
  // pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> pcl_ndt;
  // pcl_ndt.setResolution(1.0);
  // pcl_ndt.setTransformationEpsilon(1e-2);
  // pcl_ndt.setMaxCorrespondenceDistance(0.5);
  // test_pcl(pcl_ndt, target_cloud, source_cloud);

  // std::cout << "--- fgicp_st ---" << std::endl;
  // fast_gicp::FastGICPSingleThread<pcl::PointXYZ, pcl::PointXYZ> fgicp_st;
  // fgicp_st.setTransformationEpsilon(1e-2);
  // fgicp_st.setMaxCorrespondenceDistance(0.5);
  // test(fgicp_st, target_cloud, source_cloud);

  std::cout << "--- fgicp_mt ---" << std::endl;
  fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ> fgicp_mt;
  // fast_gicp uses all the CPU cores by default
  fgicp_mt.setNumThreads(num_threads);
  fgicp_mt.setMaxCorrespondenceDistance(MaxCorrespondenceDistance);
  fgicp_mt.setTransformationEpsilon(transform_epsilon);
  test(fgicp_mt, target_cloud, source_cloud);

  std::cout << "--- fgicp_ceres ---" << std::endl;
  fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ> fgicp_ceres;
  // fast_gicp uses all the CPU cores by default
  fgicp_ceres.setNumThreads(num_threads);
  fgicp_ceres.setTransformationEpsilon(transform_epsilon);
  fgicp_ceres.setMaxCorrespondenceDistance(MaxCorrespondenceDistance);
  fgicp_ceres.setLocalParameterization(true);
  fgicp_ceres.setLSQType(fast_gicp::LSQ_OPTIMIZER_TYPE::CeresDogleg);
  test(fgicp_ceres, target_cloud, source_cloud);

  // std::cout << "--- vgicp_st ---" << std::endl;
  fast_gicp::FastVGICP<pcl::PointXYZ, pcl::PointXYZ> vgicp;
  // vgicp.setResolution(1.0);
  // vgicp.setNumThreads(1);
  // vgicp.setMaxCorrespondenceDistance(0.5);
  // fvgicp.setTransformationEpsilon(1e-2);
  // test(vgicp, target_cloud, source_cloud);

  std::cout << "--- vgicp_mt ---" << std::endl;
  vgicp.setNumThreads(num_threads);
  vgicp.setTransformationEpsilon(transform_epsilon);
  vgicp.setMaxCorrespondenceDistance(MaxCorrespondenceDistance);
  vgicp.setResolution(vgicp_resolution);

  test(vgicp, target_cloud, source_cloud);

#ifdef USE_VGICP_CUDA
  std::cout << "--- ndt_cuda (P2D) ---" << std::endl;
  fast_gicp::NDTCuda<pcl::PointXYZ, pcl::PointXYZ> ndt_cuda;
  ndt_cuda.setResolution(1.0);
  ndt_cuda.setDistanceMode(fast_gicp::NDTDistanceMode::P2D);
  test(ndt_cuda, target_cloud, source_cloud);

  std::cout << "--- ndt_cuda (D2D) ---" << std::endl;
  ndt_cuda.setDistanceMode(fast_gicp::NDTDistanceMode::D2D);
  test(ndt_cuda, target_cloud, source_cloud);

  std::cout << "--- vgicp_cuda (parallel_kdtree) ---" << std::endl;
  fast_gicp::FastVGICPCuda<pcl::PointXYZ, pcl::PointXYZ> vgicp_cuda;
  vgicp_cuda.setResolution(1.0);
  // vgicp_cuda uses CPU-based parallel KDTree in covariance estimation by default
  // on a modern CPU, it is faster than GPU_BRUTEFORCE
  // vgicp_cuda.setNearestNeighborSearchMethod(fast_gicp::NearestNeighborMethod::CPU_PARALLEL_KDTREE);
  test(vgicp_cuda, target_cloud, source_cloud);

  std::cout << "--- vgicp_cuda (gpu_bruteforce) ---" << std::endl;
  // use GPU-based bruteforce nearest neighbor search for covariance estimation
  // this would be a good choice if your PC has a weak CPU and a strong GPU (e.g., NVIDIA Jetson)
  vgicp_cuda.setNearestNeighborSearchMethod(fast_gicp::NearestNeighborMethod::GPU_BRUTEFORCE);
  test(vgicp_cuda, target_cloud, source_cloud);

  std::cout << "--- vgicp_cuda (gpu_rbf_kernel) ---" << std::endl;
  // use RBF-kernel-based covariance estimation
  // extremely fast but maybe a bit inaccurate
  vgicp_cuda.setNearestNeighborSearchMethod(fast_gicp::NearestNeighborMethod::GPU_RBF_KERNEL);
  // kernel width (and distance threshold) need to be tuned
  vgicp_cuda.setKernelWidth(0.5);
  test(vgicp_cuda, target_cloud, source_cloud);
#endif

  return 0;
}
