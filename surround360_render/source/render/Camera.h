/**
* Copyright (c) 2016-present, Facebook, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-style license found in the
* LICENSE_render file in the root directory of this subproject. An additional grant
* of patent rights can be found in the PATENTS file in the same directory.
*/

#pragma once

#include <vector>

#include <Eigen/Geometry>
#include <json.h>
#include <glog/logging.h>

#ifdef _WINDOWS
#define _USE_MATH_DEFINES
#endif

#include <math.h>

namespace surround360 {

struct Camera {
  using Real = double;
  using Vector2 = Eigen::Matrix<Real, 2, 1>;
  using Vector3 = Eigen::Matrix<Real, 3, 1>;
  using Matrix3 = Eigen::Matrix<Real, 3, 3>;
  using Ray = Eigen::ParametrizedLine<Real, 3>;
  using Rig = std::vector<Camera>;
  static const Camera::Real kNearInfinity;

  // member variables
  enum struct Type{ FTHETA, RECTILINEAR} type;
  enum class DistortionModel
  {
    Simple=0,
    OpenCV=1, //OpenCV pinhole
    OpenCV_FE=2, //OpenCV FishEye
    OmniDir=3,
    Count=4
  } distortionModel;

  Vector3 position;
  Matrix3 rotation;

  Vector2 resolution;

  Vector2 principal;
  std::vector<Real> distortion;
  Vector2 focal;
  Real fovThreshold; // cos(fov) * abs(cos(fov))

  std::string id;
  std::string group;

  // construction and de/serialization
  Camera(const Type type, DistortionModel model, const Vector2& resolution, const Vector2& focal);
  Camera(const json::Value &json);
  json::Value serialize() const;

  static Rig loadRig(const std::string& filename);
  static void saveRig(const std::string& filename, const Rig& rig);
  static Camera createRescaledCamera(const Camera& cam, const float scale);

  // access rotation as forward/up/right vectors
  Vector3 forward() const { return -backward(); }
  Vector3 up() const { return rotation.row(1); }
  Vector3 right() const { return rotation.row(0); }
  void setRotation(
    const Vector3& forward,
    const Vector3& up,
    const Vector3& right);
  void setRotation(const Vector3& forward, const Vector3& up);

  // access rotation as angle * axis
  Vector3 getRotation() const;
  void setRotation(const Vector3& angleAxis);

  // access focal as a scalar (x right, y down, square pixels)
  void setScalarFocal(const Real& scalar);
  Real getScalarFocal() const;

  // access fov (measured in radians from optical axis)
  void setFov(const Real& radians);
  Real getFov() const;
  void setDefaultFov();
  bool isDefaultFov() const;

    // compute pixel coordinates
  Vector2 pixel(const Vector3& rig) const {
    // transform from rig to camera space
    Vector3 camera = rotation * (rig - position);
    // transform from camera to distorted sensor coordinates
    Vector2 sensor = cameraToSensor(camera);
    // transform from sensor coordinates to pixel coordinates
    return focal.cwiseProduct(sensor) + principal;
  }

  // compute rig coordinates, returns a ray, inverse of pixel()
  Ray rig(const Vector2& pixel) const {
    // transform from pixel to distorted sensor coordinates
    Vector2 sensor = (pixel - principal).cwiseQuotient(focal);
    // transform from distorted sensor coordinates to unit camera vector
    Vector3 unit = sensorToCamera(sensor);
    // transform from camera space to rig space
    return Ray(position, rotation.transpose() * unit);
  }

  // compute rig coordinates for point near infinity, inverse of pixel()
  Vector3 rigNearInfinity(const Vector2& pixel) const {
    return rig(pixel).pointAt(kNearInfinity);
  }

  bool isBehind(const Vector3& rig) const {
    return backward().dot(rig - position) >= 0;
  }

  bool isOutsideFov(const Vector3& rig) const {
    if (fovThreshold == -1) {
      return false;
    }
    if (fovThreshold == 0) {
      return isBehind(rig);
    }
    Vector3 v = rig - position;
    Real dot = -backward().dot(v);
    return dot * std::abs(dot) <= fovThreshold * v.squaredNorm();
  }

  bool sees(const Vector3& rig) const {
    if (isOutsideFov(rig)) {
      return false;
    }
    Vector2 p = pixel(rig);
    return
      0 <= p.x() && p.x() < resolution.x() &&
      0 <= p.y() && p.y() < resolution.y();
  }

  // estimate the fraction of the frame that is covered by the other camera
  Real overlap(const Camera& other) const {
    // just brute force probeCount x probeCount points
    const int kProbeCount = 10;
    int inside = 0;
    for (int y = 0; y < kProbeCount; ++y) {
      for (int x = 0; x < kProbeCount; ++x) {
        Vector2 p(x, y);
        p /= kProbeCount - 1;
        if (other.sees(rigNearInfinity(p.cwiseProduct(resolution)))) {
          ++inside;
        }
      }
    }
    return inside / Real(kProbeCount * kProbeCount);
  }

  static size_t distortionModelCount(DistortionModel model)
  {
      switch(model)
      {
      case DistortionModel::Simple:
          return 2;
          break;
      case DistortionModel::OpenCV:
          return 6;
          break;
      case DistortionModel::OpenCV_FE:
          return 4;
          break;
      case DistortionModel::OmniDir:
          return 3;
          break;
      }
      return 0;
  }

  // sample the camera's fov cone to find the closest point to the image center
  static float approximateUsablePixelsRadius(const Camera& camera) {
    const Camera::Real fov = camera.getFov();
    const Camera::Real kStep = 2 * M_PI / 10.0;
    Camera::Real result = camera.resolution.norm();
    for (Camera::Real a = 0; a < 2 * M_PI; a += kStep) {
      Camera::Vector3 ortho = cos(a) * camera.right() + sin(a) * camera.up();
      Camera::Vector3 direction = cos(fov) * camera.forward() + sin(fov) * ortho;
      Camera::Vector2 pixel = camera.pixel(camera.position + direction);
      result = std::min(result, (pixel - camera.resolution / 2.0).norm());
    }
    return result;
  }

  static void unitTest();

 private:
  Vector3 backward() const { return rotation.row(2); }

  // distortion is modeled in pixel space as:
  //   DistortionModel::Simple -distort(r) = r + d0 * r^3 + d1 * r^5
  //   DistortionModel::OpenCV -distort(r) = r + d0 * r^3 + d1 * r^5
  //   DistortionModel::OpenCV_FE -distort(r) = r + d0 * r^3 + d1 * r^5
  //   DistortionModel::OmniDir -distort(r) = r + d0 * r^3 + d1 * r^5
  Real distort(Real r) const {
    return distortFactor(r);
  }

  Real distortFactor(Real r) const
  {
    Real r2=r*r;

    switch(distortionModel)
    {
    case DistortionModel::Simple:
      {
        Real r4=r2 * r2;

        return 1 + r * (r2*distortion[0] + r4*distortion[1]);
      }
      break;
    case DistortionModel::OpenCV:
      {
        Real r4 = r2 * r2;
        Real r6 = r4 * r2;
        Real num = 1 + r2 * distortion[0] + r4 * distortion[1] + r6 * distortion[2];
        Real den = 1 + r2 * distortion[3] + r4 * distortion[4] + r6 * distortion[5];

        return num/den;
      }
      break;
    case DistortionModel::OpenCV_FE:
      {
        Real r4 = r2 * r2;
        Real r6 = r4 * r2;
        Real r8 = r4 * r4;
        return 1 + r2 * distortion[0] + r4 * distortion[1] + r6 * distortion[2] + r8 * distortion[3];
      }
      break;
    case DistortionModel::OmniDir:
      {
        Real r4 = r2 * r2;
        Real r6 = r4 * r2;

        Real cdist = 1 + distortion[1]*r2 + distortion[2]*r4;

        return cdist;
      }
      break;
    }
    return 0.0f;
  }

  static bool isDistortionZero(const std::vector<Real> distortion)
  {
    Real distortionSum=0.0;

    for(size_t i=0; i<distortion.size(); ++i)
        distortionSum+=distortion[i];
    return (distortionSum==0.0);
  }

  static void setsDistortionZero(std::vector<Real> &distortion)
  {
      for(size_t i=0; i<distortion.size(); ++i)
          distortion[i]=0.0;
  }

  Real undistort(Real d) const {
    if (isDistortionZero(distortion)) {
      return d; // short circuit common case
    }
    // solve d = distort(r) for r using newton's method
    Real r0 = d;
    const Real smidgen = 1.0 / kNearInfinity;
    const int kMaxSteps = 10;
    for (int step = 0; step < kMaxSteps; ++step) {
      Real d0 = distort(r0);
      if (std::abs(d0 - d) < smidgen)
        break; // close enough
      // probably ok to assume derivative == 1, but let's do the right thing
      Real r1 = r0 + smidgen;
      Real d1 = distort(r1);
      Real derivative = (d1 - d0) / smidgen;
      r0 -= (d0 - d) / derivative;
    }
    return r0;
  }

  Vector2 cameraToSensor(const Vector3& camera) const {
    if (type == Type::FTHETA) {
        if(distortionModel == DistortionModel::OmniDir){
            Vector3 point=camera.normalized();

            //surround360 expects Z into optical axis and Y up, omnidir is calculated with Z out from optical axis and Y down
            Vector2 xu=Vector2(point.x()/(-point.z()+distortion[0]), -point.y()/(-point.z()+distortion[0]));
            Real r=xu.norm();
            Real d=distort(r);
            
            return xu*d;
        }
        else
        {
            Real norm=camera.head<2>().norm();
            Real r=atan2(norm, -camera.z());
            return distort(r)/norm * camera.head<2>();
        }
    } else {
      CHECK(type == Type::RECTILINEAR) << "unexpected: " << int(type);
      // project onto z = -1 plane
      Vector2 planar = camera.head<2>() / -camera.z();
      return distortFactor(planar.squaredNorm()) * planar;
    }
  }

  // compute unit vector in camera coordinates
  Vector3 sensorToCamera(const Vector2& sensor) const {
    Real squaredNorm = sensor.squaredNorm();
    if (squaredNorm == 0) {
      // avoid divide-by-zero later
      return Vector3(0, 0, -1);
    }
    Real norm = sqrt(squaredNorm);
    Real r = undistort(norm);
    Vector3 unit;

    if(distortionModel==DistortionModel::OmniDir)
    {
        Real r2=r*r;
        Real xi=distortion[0];
        Real z=(sqrt(r2-xi*xi*r2+1)-xi*r2)/(r2+1);
        
        unit.x()=sensor.x()*(z+xi)/norm;
        unit.y()=-sensor.y()*(z+xi)/norm;
        unit.z()=-z;
    }
    else
    {
        Real angle;
        if(type==Type::FTHETA)
        {
            angle=r;
        }
        else
        {
            CHECK(type==Type::RECTILINEAR)<<"unexpected: "<<int(type);
            angle=atan(r);
        }
        
        unit.head<2>()=sin(angle)/norm * sensor;
        unit.z()=-cos(angle);
    }

    return unit;
  }

  Vector3 pixelToCamera(const Vector2& pixel) const {
    // transform from pixel to distorted sensor coordinates
    Vector2 sensor = (pixel - principal).cwiseQuotient(focal);
    // transform from distorted sensor coordinates to unit camera vector
    return sensorToCamera(sensor);
  }

  static json::Value serializeDistortion(const std::vector<Real> &distortion)
  {
      json::Array values;

      for(size_t i=0; i<distortion.size(); ++i)
          values.push_back(distortion[i]);
      return values;
  }

  static void deserializeDistortion(const json::Array &json, std::vector<Real> &distortion)
  {
      distortion.resize(json.size());

      for(size_t i=0; i < json.size(); ++i)
          distortion[i]=json[i].ToDouble();
  }


  template<typename V>
  static json::Value serializeVector(const V& v) {
      json::Array values;

      for(size_t i=0; i<v.size(); ++i)
          values.push_back(v[i]);
      return values;
  }

  template <int kSize>
  static Eigen::Matrix<Real, kSize, 1> deserializeVector(const json::Array &json) {
    CHECK_EQ(kSize, json.size()) << "bad vector";
    Eigen::Matrix<Real, kSize, 1> result;
    for (int i = 0; i < kSize; ++i) {
      result[i] = json[i].ToDouble();
    }
    return result;
  }

  static std::string serializeType(const Type& type) {
    if (type == Type::FTHETA) {
      return "FTHETA";
    } else {
      CHECK(type == Type::RECTILINEAR) << "unexpected: " << int(type);
      return "RECTILINEAR";
    }
  }

  static Type deserializeType(const json::Value &json) {
    for (int i = 0; ; ++i) {
      if (serializeType(Type(i)) == json.ToString()) {
        return Type(i);
      }
    }
  }

  static std::string serializeDistortionModel(const DistortionModel& model)
  {
      if(model==DistortionModel::OpenCV)
          return "OpenCV";
      else if(model==DistortionModel::OpenCV_FE)
          return "OpenCV_FE";
      else if(model==DistortionModel::OmniDir)
          return "OmniDir";
      else
          return "Simple";
  }

  static DistortionModel deserializeDistortionModel(const json::Value &json)
  {
      for(size_t i=0; i<(size_t)DistortionModel::Count; ++i)
      {
          if(serializeDistortionModel(DistortionModel(i))==json.ToString())
              return DistortionModel(i);
      }
      return DistortionModel::Simple;
  }
};

} // namespace surround360
