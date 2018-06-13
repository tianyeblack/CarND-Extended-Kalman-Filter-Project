#include "tools.h"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  // check the validity of the following inputs:
  //  * the estimation vector size should not be zero
  //  * the estimation vector size should equal ground truth vector size
  int est_size = estimations.size();
  int gnd_size = ground_truth.size();
  if (est_size == 0 || est_size != gnd_size) {
    cout << "Wrong estimation or ground_truth size" << endl;
    return rmse;
  }

  // accumulate squared residuals
  for (int i = 0; i < estimations.size(); ++i) {
    VectorXd residual = estimations[i] - ground_truth[i];
    VectorXd res_sqre = residual.array() * residual.array();
    rmse += res_sqre;
  }

  // calculate the mean
  rmse /= est_size;

  // calculate the squared root
  rmse = rmse.array().sqrt();

  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd &x_state) {
  MatrixXd Hj(3, 4);
  // recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  // pre-compute a set of terms to avoid repeated calculation
  float c1 = px * px + py * py;
  float c2 = sqrt(c1);
  float c3 = (c1 * c2);

  // check division by zero
  if (fabs(c1) < 0.0001) {
    cout << "CalculateJacobian () - Error - Division by Zero" << endl;
    return Hj;
  }

  // compute the Jacobian matrix
  // Hj(2, 0) and Hj(2, 1) seems to have common factor whose difference is only
  // negation It's repeated there for 0 and -0 are actually different in
  // floating point numbers
  Hj << (px / c2), (py / c2), 0, 0, -(py / c1), (px / c1), 0, 0,
      py * (vx * py - vy * px) / c3, px * (px * vy - py * vx) / c3, px / c2,
      py / c2;

  return Hj;
}
