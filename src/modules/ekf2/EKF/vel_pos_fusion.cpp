/****************************************************************************
 *
 *   Copyright (c) 2015-2023 PX4 Development Team. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 * 3. Neither the name PX4 nor the names of its contributors may be
 *    used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 * OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 * AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 ****************************************************************************/

/**
 * @file vel_pos_fusion.cpp
 *
 * @author Roman Bast <bapstroman@gmail.com>
 * @author Siddharth Bharat Purohit <siddharthbharatpurohit@gmail.com>
 * @author Paul Riseborough <p_riseborough@live.com.au>
 *
 */

#include <mathlib/mathlib.h>
#include "ekf.h"

// TODO: void Ekf::updateYawAidSrcStatus(const uint64_t &time_us, const float &obs, const float &obs_var,
//				     const float innov_gate, estimator_aid_source1d_s &aid_src) const

void Ekf::updateVelocityAidStatus(estimator_aid_source2d_s &aid_src, const uint64_t &time_us,
				  const Vector2f &observation, const Vector2f &observation_variance, float innovation_gate) const
{
	Vector2f innovation = Vector2f(_state.vel.xy()) - observation;
	Vector2f innovation_variance = Vector2f(getStateVariance<State::vel>()) + observation_variance;

	updateEstimatorAidStatus(aid_src, time_us,
				 observation, observation_variance,
				 innovation, innovation_variance,
				 innovation_gate);
}

void Ekf::updateVelocityAidStatus(estimator_aid_source3d_s &aid_src, const uint64_t &time_us,
				  const Vector3f &observation, const Vector3f &observation_variance, float innovation_gate) const
{
	Vector3f innovation = _state.vel - observation;
	Vector3f innovation_variance = getStateVariance<State::vel>() + observation_variance;

	updateEstimatorAidStatus(aid_src, time_us,
				 observation, observation_variance,
				 innovation, innovation_variance,
				 innovation_gate);

	// vz special case if there is bad vertical acceleration data, then don't reject measurement,
	// but limit innovation to prevent spikes that could destabilise the filter
	if (_fault_status.flags.bad_acc_vertical && aid_src.innovation_rejected) {
		const float innov_limit = innovation_gate * sqrtf(aid_src.innovation_variance[2]);
		aid_src.innovation[2] = math::constrain(aid_src.innovation[2], -innov_limit, innov_limit);
		aid_src.innovation_rejected = false;
	}
}

void Ekf::updateVerticalPositionAidStatus(estimator_aid_source1d_s &aid_src, const uint64_t &time_us,
		const float observation, const float observation_variance, const float innovation_gate) const
{
	float innovation = _state.pos(2) - observation;
	float innovation_variance = getStateVariance<State::pos>()(2) + observation_variance;

	updateEstimatorAidStatus(aid_src, time_us,
				 observation, observation_variance,
				 innovation, innovation_variance,
				 innovation_gate);

	// z special case if there is bad vertical acceleration data, then don't reject measurement,
	// but limit innovation to prevent spikes that could destabilise the filter
	if (_fault_status.flags.bad_acc_vertical && aid_src.innovation_rejected) {
		const float innov_limit = innovation_gate * sqrtf(aid_src.innovation_variance);
		aid_src.innovation = math::constrain(aid_src.innovation, -innov_limit, innov_limit);
		aid_src.innovation_rejected = false;
	}
}

void Ekf::updateHorizontalPositionAidStatus(estimator_aid_source2d_s &aid_src, const uint64_t &time_us,
		const Vector2f &observation, const Vector2f &observation_variance, const float innov_gate) const
{
	Vector2f innovation = Vector2f(_state.pos) - observation;
	Vector2f innovation_variance = Vector2f(getStateVariance<State::pos>()) + observation_variance;

	updateEstimatorAidStatus(aid_src, time_us,
				 observation, observation_variance,
				 innovation,  innovation_variance,
				 innov_gate);
}

bool Ekf::fuseVelocity(estimator_aid_source2d_s &aid_src)
{
	// vx, vy
	if (!aid_src.innovation_rejected
	    && fuseVelPosHeight(aid_src.innovation[0], aid_src.innovation_variance[0], State::vel.idx + 0)
	    && fuseVelPosHeight(aid_src.innovation[1], aid_src.innovation_variance[1], State::vel.idx + 1)
	   ) {
		aid_src.fused = true;
		aid_src.time_last_fuse = _time_delayed_us;

	} else {
		aid_src.fused = false;
	}

	return aid_src.fused;
}

bool Ekf::fuseVelocity(estimator_aid_source3d_s &aid_src)
{
	// vx, vy, vz
	if (!aid_src.innovation_rejected
	    && fuseVelPosHeight(aid_src.innovation[0], aid_src.innovation_variance[0], State::vel.idx + 0)
	    && fuseVelPosHeight(aid_src.innovation[1], aid_src.innovation_variance[1], State::vel.idx + 1)
	    && fuseVelPosHeight(aid_src.innovation[2], aid_src.innovation_variance[2], State::vel.idx + 2)
	   ) {
		aid_src.fused = true;
		aid_src.time_last_fuse = _time_delayed_us;

	} else {
		aid_src.fused = false;
	}

	return aid_src.fused;
}

bool Ekf::fuseHorizontalPosition(estimator_aid_source2d_s &aid_src)
{
	// x & y
	if (!aid_src.innovation_rejected
	    && fuseVelPosHeight(aid_src.innovation[0], aid_src.innovation_variance[0], State::pos.idx + 0)
	    && fuseVelPosHeight(aid_src.innovation[1], aid_src.innovation_variance[1], State::pos.idx + 1)
	   ) {
		aid_src.fused = true;
		aid_src.time_last_fuse = _time_delayed_us;

	} else {
		aid_src.fused = false;
	}

	return aid_src.fused;
}

bool Ekf::fuseVerticalPosition(estimator_aid_source1d_s &aid_src)
{
	// z
	if (!aid_src.innovation_rejected
	    && fuseVelPosHeight(aid_src.innovation, aid_src.innovation_variance, State::pos.idx + 2)
	   ) {
		aid_src.fused = true;
		aid_src.time_last_fuse = _time_delayed_us;

	} else {
		aid_src.fused = false;
	}

	return aid_src.fused;
}

// Helper function that fuses a single velocity or position measurement
bool Ekf::fuseVelPosHeight(const float innov, const float innov_var, const int state_index)
{
	VectorState Kfusion;  // Kalman gain vector for any single observation - sequential fusion is used.

	// calculate kalman gain K = PHS, where S = 1/innovation variance
	for (int row = 0; row < State::size; row++) {
		Kfusion(row) = P(row, state_index) / innov_var;
	}

	clearInhibitedStateKalmanGains(Kfusion);

	SquareMatrixState KHP;

	for (unsigned row = 0; row < State::size; row++) {
		for (unsigned column = 0; column < State::size; column++) {
			KHP(row, column) = Kfusion(row) * P(state_index, column);
		}
	}

	const bool healthy = checkAndFixCovarianceUpdate(KHP);

	setVelPosStatus(state_index, healthy);

	if (healthy) {
		// apply the covariance corrections
		P -= KHP;

		fixCovarianceErrors(true);

		// apply the state corrections
		fuse(Kfusion, innov);

		return true;
	}

	return false;
}

void Ekf::setVelPosStatus(const int state_index, const bool healthy)
{
	switch (state_index) {
	case State::vel.idx:
		if (healthy) {
			_fault_status.flags.bad_vel_N = false;
			_time_last_hor_vel_fuse = _time_delayed_us;

		} else {
			_fault_status.flags.bad_vel_N = true;
		}

		break;

	case State::vel.idx + 1:
		if (healthy) {
			_fault_status.flags.bad_vel_E = false;
			_time_last_hor_vel_fuse = _time_delayed_us;

		} else {
			_fault_status.flags.bad_vel_E = true;
		}

		break;

	case State::vel.idx + 2:
		if (healthy) {
			_fault_status.flags.bad_vel_D = false;
			_time_last_ver_vel_fuse = _time_delayed_us;

		} else {
			_fault_status.flags.bad_vel_D = true;
		}

		break;

	case State::pos.idx:
		if (healthy) {
			_fault_status.flags.bad_pos_N = false;
			_time_last_hor_pos_fuse = _time_delayed_us;

		} else {
			_fault_status.flags.bad_pos_N = true;
		}

		break;

	case State::pos.idx + 1:
		if (healthy) {
			_fault_status.flags.bad_pos_E = false;
			_time_last_hor_pos_fuse = _time_delayed_us;

		} else {
			_fault_status.flags.bad_pos_E = true;
		}

		break;

	case State::pos.idx + 2:
		if (healthy) {
			_fault_status.flags.bad_pos_D = false;
			_time_last_hgt_fuse = _time_delayed_us;

		} else {
			_fault_status.flags.bad_pos_D = true;
		}

		break;
	}
}
