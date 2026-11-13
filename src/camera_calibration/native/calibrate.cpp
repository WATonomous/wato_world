// File-only adapter for Koide's unchanged NIDCost and Ceres BFGS solver.
// Upstream:
// koide3/direct_visual_lidar_calibration@02a0dc039f5509708f384be4ff3228e0ae09352d.
// The upstream objective is MIT licensed (upstream README); Sophus is MIT.
// Unlike the upstream CLI, this runner needs no ROS, PCL, GTSAM or GL context.
// Upstream camera virtual overloads require a complete Jet type.
// clang-format off
#include <ceres/jet.h>
// clang-format on
#include <algorithm>
#include <camera/fisheye.hpp>
#include <camera/generic_camera.hpp>
#include <camera/pinhole.hpp>
#include <camera/rational_polynomial.hpp>
#include <ceres/autodiff_first_order_function.h>
#include <ceres/ceres.h>
#include <ceres/product_manifold.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <sophus/ceres_manifold.hpp>
#include <sophus/se3.hpp>
#include <vector>
#include <vlcal/costs/nid_cost.hpp>

using json = nlohmann::json;
constexpr const char *revision = "02a0dc039f5509708f384be4ff3228e0ae09352d";

Eigen::Isometry3d transform(const json &value) {
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      T.matrix()(i, j) = value.at(i).at(j);
  return T;
}
json matrix(const Eigen::Isometry3d &T) {
  json out = json::array();
  for (int i = 0; i < 4; ++i) {
    json row = json::array();
    for (int j = 0; j < 4; ++j)
      row.push_back(T.matrix()(i, j));
    out.push_back(row);
  }
  return out;
}
template <class Model>
camera::GenericCameraBase::ConstPtr camera_model(const std::vector<double> &k,
                                                 const std::vector<double> &d) {
  if (k.size() != 4 ||
      d.size() != camera::CameraModelTraits<Model>::num_distortion_params)
    throw std::runtime_error("camera coefficient count mismatch");
  return std::make_shared<camera::GenericCamera<Model>>(
      Eigen::Map<const Eigen::VectorXd>(k.data(), k.size()),
      Eigen::Map<const Eigen::VectorXd>(d.data(), d.size()));
}
struct Pair {
  cv::Mat image;
  std::vector<Eigen::Vector4d> xyz;
  std::vector<double> intensity;
  void load(const json &row) {
    int width = row.at("width"), height = row.at("height");
    if (width < 1 || height < 1 || width > 16384 || height > 16384)
      throw std::runtime_error("invalid image dimensions");
    cv::Mat bytes(height, width, CV_8UC1);
    std::ifstream image_in(row.at("image").get<std::string>(),
                           std::ios::binary);
    image_in.read(reinterpret_cast<char *>(bytes.data), bytes.total());
    if (!image_in || image_in.peek() != EOF)
      throw std::runtime_error("invalid grayscale image buffer");
    bytes.convertTo(image, CV_64FC1, 1.0 / 255.0);
    std::ifstream in(row.at("points").get<std::string>(), std::ios::binary);
    if (!in)
      throw std::runtime_error("cannot read point file");
    double p[4];
    while (in.read(reinterpret_cast<char *>(p), sizeof(p))) {
      if (!std::isfinite(p[0]) || !std::isfinite(p[1]) ||
          !std::isfinite(p[2]) || !std::isfinite(p[3]) || p[3] < 0 || p[3] > 1)
        throw std::runtime_error("invalid point/intensity");
      xyz.emplace_back(p[0], p[1], p[2], 1.0);
      intensity.push_back(p[3]);
    }
    if (in.gcount() != 0 || xyz.empty())
      throw std::runtime_error("truncated or empty point file");
  }
};
// Pixel depth culling, rebuilt each outer iteration, follows the upstream CLI's
// visible-point selection. Storage owns the buffers referenced by Frame.
struct Visible {
  std::vector<Eigen::Vector4d> xyz;
  std::vector<double> intensity;
  std::shared_ptr<vlcal::Frame> frame = std::make_shared<vlcal::Frame>();
  Visible(const Pair &pair, const camera::GenericCameraBase::ConstPtr &proj,
          const Eigen::Isometry3d &T) {
    std::vector<double> depth(pair.image.total(),
                              std::numeric_limits<double>::infinity());
    std::vector<int> pixels(pair.xyz.size(), -1);
    for (size_t i = 0; i < pair.xyz.size(); ++i) {
      Eigen::Vector3d p = (T * pair.xyz[i]).head<3>();
      if (p.z() < 0.5)
        continue;
      auto uv = proj->project(p);
      if (!uv.allFinite() || uv.x() < 1 || uv.y() < 1 ||
          uv.x() >= pair.image.cols - 2 || uv.y() >= pair.image.rows - 2)
        continue;
      int pixel = int(uv.y()) * pair.image.cols + int(uv.x());
      pixels[i] = pixel;
      depth[pixel] = std::min(depth[pixel], p.z());
    }
    for (size_t i = 0; i < pair.xyz.size(); ++i)
      if (pixels[i] >= 0 && (T * pair.xyz[i]).z() <= depth[pixels[i]] + 0.1) {
        xyz.push_back(pair.xyz[i]);
        intensity.push_back(pair.intensity[i]);
      }
    if (xyz.size() < 100)
      throw std::runtime_error("fewer than 100 visible points in a view");
    frame->num_points = xyz.size();
    frame->points = xyz.data();
    frame->intensities = intensity.data();
  }
};
struct Objective {
  Sophus::SE3d initial;
  double max_translation, max_rotation;
  std::vector<std::shared_ptr<vlcal::NIDCost>> costs;
  bool camera_to_ego;
  template <typename T> bool operator()(const T *params, T *residual) const {
    std::vector<double> real(7);
    std::transform(params, params + 7, real.begin(),
                   [](const auto &v) { return vlcal::get_real(v); });
    Eigen::Map<const Sophus::SE3d> pose(real.data());
    auto old_camera = initial.inverse();
    Sophus::SE3d camera = camera_to_ego ? Sophus::SE3d(pose) : pose.inverse();
    if ((camera.translation() - old_camera.translation()).norm() >
            max_translation ||
        (old_camera.so3().inverse() * camera.so3()).log().norm() > max_rotation)
      return false;
    *residual = T(0);
    Eigen::Map<const Sophus::SE3<T>> parameter_pose(params);
    Sophus::SE3<T> projection_pose(parameter_pose);
    if (camera_to_ego)
      projection_pose = projection_pose.inverse();
    for (const auto &cost : costs) {
      T term;
      if (!(*cost)(projection_pose.data(), &term))
        return false;
      *residual += term;
    }
    *residual /= double(costs.size());
    return true;
  }
};
int main(int argc, char **argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--version") {
      std::cout << "koide-nid-headless " << revision << "\n";
      return 0;
    }
    if (argc != 3)
      throw std::runtime_error(
          "usage: wato_koide_calibrate INPUT.json OUTPUT.json");
    std::ifstream in(argv[1]);
    json config;
    in >> config;
    std::vector<double> k = config.at("intrinsics"),
                        d = config.at("distortion");
    std::string model = config.at("model");
    camera::GenericCameraBase::ConstPtr proj;
    if (model == "plumb_bob")
      proj = camera_model<camera::PinholeProjection>(k, d);
    else if (model == "equidistant" || model == "fisheye")
      proj = camera_model<camera::FisheyeProjection>(k, d);
    else if (model == "rational_polynomial")
      proj = camera_model<camera::RationalPolynomialProjection>(k, d);
    else
      throw std::runtime_error("unsupported image model");
    std::vector<Pair> pairs;
    for (const auto &row : config.at("pairs")) {
      pairs.emplace_back();
      pairs.back().load(row);
    }
    if (pairs.empty())
      throw std::runtime_error("no image/cloud pairs");
    Sophus::SE3d pose(
        transform(config.at("initial_ego_T_cam")).inverse().matrix());
    Sophus::SE3d initial(
        transform(config.at("reference_ego_T_cam")).inverse().matrix());
    const std::string motion_model = config.value("motion_model", "rigid");
    if (motion_model != "rigid" && motion_model != "rotation_only")
      throw std::runtime_error("unsupported motion_model");
    const bool rotation_only = motion_model == "rotation_only";
    if (rotation_only)
      pose =
          pose.inverse(); // Optimize camera-to-ego so its centre stays fixed.
    double max_translation = config.at("max_translation_m"),
           max_rotation =
               config.at("max_rotation_deg").get<double>() * M_PI / 180;
    json history = json::array();
    for (int outer = 0; outer < config.value("outer_iterations", 3); ++outer) {
      std::vector<std::shared_ptr<Visible>> visible;
      auto objective = new Objective{
          initial, max_translation, max_rotation, {}, rotation_only};
      for (const auto &pair : pairs) {
        auto view = std::make_shared<Visible>(
            pair, proj,
            Eigen::Isometry3d(
                (rotation_only ? pose.inverse() : pose).matrix()));
        objective->costs.push_back(std::make_shared<vlcal::NIDCost>(
            proj, pair.image, view->frame, config.value("bins", 16)));
        visible.push_back(view);
      }
      auto function =
          new ceres::AutoDiffFirstOrderFunction<Objective, 7>(objective);
      ceres::Manifold *manifold;
      if (rotation_only) {
        // Sophus stores quaternion xyzw then translation xyz. This freezes
        // the physical camera centre during fitting, rather than discarding
        // translation from an unconstrained fit afterwards.
        manifold = new ceres::ProductManifold<ceres::EigenQuaternionManifold,
                                              ceres::SubsetManifold>(
            ceres::EigenQuaternionManifold{},
            ceres::SubsetManifold(3, {0, 1, 2}));
      } else {
        manifold = new Sophus::Manifold<Sophus::SE3>();
      }
      ceres::GradientProblem problem(function, manifold);
      ceres::GradientProblemSolver::Options options;
      options.line_search_direction_type = ceres::BFGS;
      options.max_num_iterations = config.value("max_iterations", 100);
      options.max_solver_time_in_seconds = config.value("timeout_s", 120.0) /
                                           config.value("outer_iterations", 3);
      options.logging_type = ceres::SILENT;
      ceres::GradientProblemSolver::Summary summary;
      ceres::Solve(options, problem, pose.data(), &summary);
      history.push_back({{"initial_cost", summary.initial_cost},
                         {"final_cost", summary.final_cost},
                         {"usable", summary.IsSolutionUsable()},
                         {"message", summary.message},
                         {"iterations", summary.iterations.size()}});
      if (!summary.IsSolutionUsable())
        throw std::runtime_error("Ceres: " + summary.message);
    }
    json result = {
        {"backend", "koide-nid-headless"},
        {"revision", revision},
        {"motion_model", motion_model},
        {"ego_T_cam", matrix(Eigen::Isometry3d(
                          (rotation_only ? pose : pose.inverse()).matrix()))},
        {"history", history}};
    std::ofstream out(argv[2]);
    out << result.dump(2) << "\n";
    if (!out)
      throw std::runtime_error("cannot write result");
  } catch (const std::exception &error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
