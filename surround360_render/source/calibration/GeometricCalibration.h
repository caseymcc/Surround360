#include "Camera.h"

using namespace surround360;

Camera makeCamera(
    const Camera& camera,
    const Camera::Vector3& position,
    const Camera::Vector3& rotation,
    const Camera::Vector2& principal,
    const Camera::Real& focal,
    const Camera::DistortionModel distortionModel,
    const std::vector<double> &distortion) {
  Camera result = camera;
  result.position = position;
  result.setRotation(rotation);
  result.principal = principal;
  result.setScalarFocal(focal);
  result.distortionModel = distortionModel;
  result.distortion = distortion;

  return result;
}

template<size_t _DistortionCount>
struct ReprojectionFunctor {
  static ceres::CostFunction* addResidual(
      ceres::Problem& problem,
      Camera::Vector3& position,
      Camera::Vector3& rotation,
      Camera::Vector2& principal,
      Camera::Real& focal,
      std::vector<Camera::Real> distortion,
      Camera::Vector3& world,
      const Camera& camera,
      const Camera::Vector2& pixel,
      bool robust = false) {
    auto* cost = new CostFunction(new ReprojectionFunctor<_DistortionCount>(camera, pixel));
    auto* loss = robust ? new ceres::HuberLoss(1.0) : nullptr;
    problem.AddResidualBlock(
      cost,
      loss,
      position.data(),
      rotation.data(),
      principal.data(),
      &focal,
      distortion.data(),
      world.data());
    return cost;
  }

  bool operator()(
      double const* const position,
      double const* const rotation,
      double const* const principal,
      double const* const focal,
      double const *distortion,
      double const* const world,
      double* residuals) const {
    // create a camera using parameters
    // TODO: maybe compute modified cameras once per iteration using
    //   vector<IterationCallback> Solver::Options::callbacks?
    size_t distortionCount=Camera::distortionModelCount(camera.distortionModel);

    Camera modified = makeCamera(
      camera,
      Eigen::Map<const Camera::Vector3>(position),
      Eigen::Map<const Camera::Vector3>(rotation),
      Eigen::Map<const Camera::Vector2>(principal),
      *focal,
      camera.distortionModel,
      std::vector<Camera::Real>(distortion, distortion+distortionCount));
    // transform world with that camera and compare to pixel
    Eigen::Map<const Camera::Vector3> w(world);
    Eigen::Map<Camera::Vector2> r(residuals);
    r = modified.pixel(w) - pixel;

    return true;
  }

 private:
  using CostFunction = ceres::NumericDiffCostFunction<
    ReprojectionFunctor<_DistortionCount>,
    ceres::CENTRAL,
    2, // residuals
    3, // position
    3, // rotation
    2, // principal
    1, // focal
    _DistortionCount, // distortion
    3>; // world

  ReprojectionFunctor(const Camera& camera, const Camera::Vector2& pixel) :
      camera(camera),
      pixel(pixel) {
  }

  const Camera& camera;
  const Camera::Vector2 pixel;
};

struct TriangulationFunctor {
  static ceres::CostFunction* addResidual(
      ceres::Problem& problem,
      Camera::Vector3& world,
      const Camera& camera,
      const Camera::Vector2& pixel,
      const bool robust = false) {
    auto* cost = new CostFunction(new TriangulationFunctor(camera, pixel));
    auto* loss = robust ? new ceres::HuberLoss(1.0) : nullptr;
    problem.AddResidualBlock(
      cost,
      loss,
      world.data());
    return cost;
  }

  bool operator()(
      double const* const world,
      double* residuals) const {
    Eigen::Map<const Camera::Vector3> w(world);
    Eigen::Map<Camera::Vector2> r(residuals);

    // transform world with camera and compare to pixel
    r = camera.pixel(w) - pixel;

    return true;
  }

 private:
  using CostFunction = ceres::NumericDiffCostFunction<
    TriangulationFunctor,
    ceres::CENTRAL,
    2, // residuals
    3>; // world

  TriangulationFunctor(const Camera& camera, const Camera::Vector2& pixel) :
      camera(camera),
      pixel(pixel) {
  }

  const Camera& camera;
  const Camera::Vector2 pixel;
};

using Observations = std::vector<std::pair<const Camera&, Camera::Vector2>>;

Camera::Vector3 averageAtDistance(
    const Observations& observations,
    const Camera::Real distance) {
  Camera::Vector3 sum = Camera::Vector3::Zero();
  for (const auto& obs : observations) {
    sum += obs.first.rig(obs.second).pointAt(distance);
  }
  return sum / observations.size();
}

Camera::Vector3 triangulateNonlinear(
    const Observations& observations,
    const bool forceInFront) {
  ceres::Solver::Options options;

  // initial value is average of distant points
  const Camera::Real kInitialDistance = 1000; // not hugely important
  Camera::Vector3 world = averageAtDistance(observations, kInitialDistance);
  ceres::Problem problem;
  for (const auto& obs : observations) {
    TriangulationFunctor::addResidual(problem, world, obs.first, obs.second);
  }

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  if (forceInFront) {
    for (const auto& obs : observations) {
      if (obs.first.isBehind(world)) {
        return averageAtDistance(observations, Camera::kNearInfinity);
      }
    }
  }
  
  return world;
}

double calcPercentile(std::vector<double> values, double percentile = 0.5) {
  if (values.empty()) {
   return NAN;
  }
  CHECK_LT(percentile, 1);
  size_t index(percentile * values.size());
  std::nth_element(values.begin(), values.begin() + index, values.end());
  return values[index];
}

Camera::Vector2 reprojectionError(
    const ceres::Problem& problem,
    ceres::ResidualBlockId id) {
  auto cost = problem.GetCostFunctionForResidualBlock(id);
  std::vector<double*> parameterBlocks;
  problem.GetParameterBlocksForResidualBlock(id, &parameterBlocks);
  Camera::Vector2 residual;
  cost->Evaluate(parameterBlocks.data(), residual.data(), nullptr);
  return residual;
}

std::vector<double> getReprojectionErrorNorms(const ceres::Problem& problem) {
  std::vector<double> result;
  std::vector<ceres::ResidualBlockId> ids;
  problem.GetResidualBlocks(&ids);
  for (auto& id : ids) {
    result.push_back(reprojectionError(problem, id).norm());
  }
  return result;
}

// remove if residual error is more than threshold
void removeOutliers(ceres::Problem& problem, double threshold) {
  std::vector<ceres::ResidualBlockId> ids;
  problem.GetResidualBlocks(&ids);
  for (auto & id: ids) {
    if (reprojectionError(problem, id).norm() > threshold) {
      problem.RemoveResidualBlock(id);
    }
  }
}

struct Keypoint
{
    Camera::Vector2 position;
    Camera::Real scale;
    Camera::Real orientation;
    int index;

    Keypoint(
        const Camera::Vector2& position,
        Camera::Real scale=0,
        Camera::Real orientation=0,
        int index=-1):
        position(position),
        scale(scale),
        orientation(orientation),
        index(index)
    {}
};

using KeypointMap=std::unordered_map<std::string, std::vector<Keypoint>>;

// a trace is a world coordinate and a list of observations that reference it
struct Trace
{
    Camera::Vector3 position;

    std::vector<std::pair<std::string, int>> references;

    void add(const std::string& image, const int index)
    {
        references.emplace_back(image, index);
    }

    // inherit trace's references
    void inherit(Trace& trace, KeypointMap& keypointMap, int index)
    {
        for(const auto& ref:trace.references)
        {
            keypointMap[ref.first][ref.second].index=index;
        }
        references.insert(
            references.end(),
            trace.references.begin(),
            trace.references.end());
        trace.references.clear();
    }
};

template<size_t _DistortionCount>
void reprojectTraces(
    ceres::Problem &problem,
    std::vector<Camera> &cameras,
    std::vector<Camera::Vector3> &positions,
    std::vector<Camera::Vector3> &rotations,
    std::vector<Camera::Vector2> &principals,
    std::vector<Camera::Real> &focals,
    std::vector<std::vector<double> > &distortions,
    KeypointMap &keypointMap,
    std::vector<Trace> &traces)
{
    if(FLAGS_shared_distortion)
    {
        for(const Trace& trace:traces)
        {
            for(const auto& ref:trace.references)
            {
                const std::string& image=ref.first;
                const auto& keypoint=keypointMap[image][ref.second];
                const int camera=getCameraIndex(image);
                const int group=cameraGroupToIndex[cameras[camera].group];
                ReprojectionFunctor<_DistortionCount>::addResidual(
                    problem,
                    positions[camera],
                    rotations[camera],
                    principals[camera],
                    focals[camera],
                    distortions[group],
                    traces[keypoint.index].position,
                    cameras[camera],
                    keypoint.position,
                    FLAGS_robust);
            }
        }
    }
    else
    {
        for(const Trace& trace:traces)
        {
            for(const auto& ref:trace.references)
            {
                const std::string& image=ref.first;
                const auto& keypoint=keypointMap[image][ref.second];
                const int camera=getCameraIndex(image);
                const int group=cameraGroupToIndex[cameras[camera].group];
                ReprojectionFunctor<_DistortionCount>::addResidual(
                    problem,
                    positions[camera],
                    rotations[camera],
                    principals[camera],
                    focals[camera],
                    distortions[camera],
                    traces[keypoint.index].position,
                    cameras[camera],
                    keypoint.position,
                    FLAGS_robust);
            }
        }
    }
}