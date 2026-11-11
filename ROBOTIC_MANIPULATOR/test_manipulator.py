"""
Automated Unit and Regression Test Suite
Project: 2-DOF Robotic Pick-and-Place Manipulator
------------------------------------------------
Validates:
1. Analytical kinematic benchmark configurations.
2. Closed-loop forward/inverse kinematic consistency.
3. Out-of-bounds reachability rejection.
4. Trajectory continuity and zero-velocity boundary conditions.
5. PID controller tracking and stability.
"""

import unittest
import numpy as np

from kinematics import (
    L1,
    L2,
    forward_kinematics,
    forward_kinematics_full,
    inverse_kinematics,
    is_reachable
)
from trajectory import cubic_trajectory, joint_trajectory
from controller import PIDController, ServoActuator, GripperActuator, GripperState


class TestKinematics(unittest.TestCase):
    """Test Suite for Forward & Inverse Kinematics."""

    def test_forward_kinematics_benchmarks(self):
        """Verify the 4 standard analytical benchmark poses."""
        cases = [
            # (name, theta1, theta2, expected_x, expected_y)
            ("Fully Extended", 0.0, 0.0, 160.0, 0.0),
            ("Folded", 0.0, 180.0, 0.0, 0.0),
            ("Bent Vertical", 90.0, 0.0, 0.0, 160.0),
            ("Right Angle", 0.0, 90.0, 80.0, 80.0),
        ]
        for name, th1, th2, exp_x, exp_y in cases:
            x, y = forward_kinematics(th1, th2)
            self.assertAlmostEqual(x, exp_x, places=2, msg=f"FK Failed for {name} on X")
            self.assertAlmostEqual(y, exp_y, places=2, msg=f"FK Failed for {name} on Y")

    def test_full_link_positions(self):
        """Verify intermediate joint coordinates."""
        (x0, y0), (x1, y1), (x2, y2) = forward_kinematics_full(0.0, 90.0)
        self.assertEqual((x0, y0), (0.0, 0.0))
        self.assertAlmostEqual(x1, 80.0, places=2)
        self.assertAlmostEqual(y1, 0.0, places=2)
        self.assertAlmostEqual(x2, 80.0, places=2)
        self.assertAlmostEqual(y2, 80.0, places=2)

    def test_inverse_kinematics_nominal_target(self):
        """Verify analytical IK solution for target (100, 80) mm."""
        target_x, target_y = 100.0, 80.0
        th1, th2 = inverse_kinematics(target_x, target_y, elbow_up=False)

        # Expected: ~1.83 deg and ~73.67 deg
        self.assertAlmostEqual(th1, 1.83, places=1)
        self.assertAlmostEqual(th2, 73.67, places=1)

        # Reconstructed forward kinematics must match target exactly
        rec_x, rec_y = forward_kinematics(th1, th2)
        self.assertAlmostEqual(rec_x, target_x, places=3)
        self.assertAlmostEqual(rec_y, target_y, places=3)

    def test_out_of_bounds_target_raises_error(self):
        """Verify out-of-reach positions raise ValueError."""
        unreachable_points = [
            (200.0, 100.0),   # Radius ~223.6 mm > 160 mm
            (150.0, 150.0),   # Radius ~212.1 mm > 160 mm
            (0.0, 180.0),     # Radius = 180 mm > 160 mm
        ]
        for ux, uy in unreachable_points:
            self.assertFalse(is_reachable(ux, uy))
            with self.assertRaises(ValueError):
                inverse_kinematics(ux, uy)


class TestTrajectoryPlanning(unittest.TestCase):
    """Test Suite for Cubic Trajectory Interpolation."""

    def test_boundary_conditions(self):
        """Verify zero velocity at start and end."""
        start_angle = 10.0
        end_angle = 85.0
        duration = 2.0
        dt = 0.01

        t, pos, vel, acc = cubic_trajectory(start_angle, end_angle, duration, dt)

        # Start conditions
        self.assertAlmostEqual(pos[0], start_angle, places=3)
        self.assertAlmostEqual(vel[0], 0.0, places=4)

        # End conditions
        self.assertAlmostEqual(pos[-1], end_angle, places=3)
        self.assertAlmostEqual(vel[-1], 0.0, places=4)

    def test_joint_trajectory_shape(self):
        """Verify coordinate dimensions of 2-joint trajectory."""
        t, pos, vel = joint_trajectory((0.0, 0.0), (45.0, 90.0), duration=1.0, dt=0.02)
        self.assertEqual(pos.shape[1], 2)
        self.assertEqual(vel.shape[1], 2)
        self.assertEqual(len(t), len(pos))


class TestControlAndActuation(unittest.TestCase):
    """Test Suite for PID Control and Actuators."""

    def test_pid_error_reduction(self):
        """Verify PID reduces large initial error and stabilizes."""
        pid = PIDController(kp=5.0, ki=0.2, kd=0.3)
        actuator = ServoActuator(initial_angle=0.0, max_velocity=180.0, response_lag=0.02)

        setpoint = 30.0
        dt = 0.02
        for _ in range(250):  # 5.0 seconds
            u = pid.update(setpoint, actuator.angle, dt)
            actuator.update(u, dt)

        # Must settle within 0.5 deg of setpoint following a 30 deg instantaneous step
        self.assertAlmostEqual(actuator.angle, setpoint, delta=0.5)

    def test_pid_trajectory_tracking(self):
        """Verify closed-loop tracking of smooth cubic trajectory."""
        pid = PIDController(kp=5.0, ki=0.8, kd=0.20)
        actuator = ServoActuator(initial_angle=0.0, max_velocity=180.0, response_lag=0.04)

        t, pos, vel, _ = cubic_trajectory(start=0.0, end=45.0, duration=2.0, dt=0.02)
        errors = []
        for i in range(len(t)):
            u = pid.update(pos[i], actuator.angle, 0.02)
            actuator.update(u, 0.02)
            errors.append(pos[i] - actuator.angle)

        rmse = np.sqrt(np.mean(np.array(errors)**2))
        self.assertLess(rmse, 5.0, "Trajectory tracking RMSE exceeded threshold")

    def test_gripper_state_machine(self):
        """Verify gripper transition states."""
        gripper = GripperActuator(stroke_mm=30.0, speed_mms=100.0)
        self.assertEqual(gripper.state, GripperState.OPEN)

        gripper.close()
        for _ in range(20):
            gripper.update(dt=0.02, near_object=True)

        self.assertEqual(gripper.state, GripperState.CLOSED_HOLDING)
        self.assertTrue(gripper.holding_object)

        gripper.open()
        for _ in range(25):
            gripper.update(dt=0.02, near_object=False)

        self.assertEqual(gripper.state, GripperState.OPEN)
        self.assertFalse(gripper.holding_object)


if __name__ == "__main__":
    unittest.main(verbosity=2)
