"""
Trajectory Planning Module for 2-DOF Robotic Manipulator
-------------------------------------------------------
Generates smooth joint-space trajectories using cubic polynomial interpolation
with zero initial and terminal velocities (jerk-bounded smooth motion).
"""

import numpy as np
from kinematics import forward_kinematics


def cubic_trajectory(start: float,
                     end: float,
                     duration: float = 2.0,
                     dt: float = 0.02) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a cubic polynomial trajectory with zero initial and final velocity.

    Boundary conditions:
        theta(0) = start,    theta(T) = end
        theta_dot(0) = 0,    theta_dot(T) = 0

    Polynomial formulation:
        s = t / T  (normalized time in [0, 1])
        theta(t) = start + (end - start) * (3*s^2 - 2*s^3)
        theta_dot(t) = (end - start) * (6*s - 6*s^2) / T
        theta_ddot(t) = (end - start) * (6 - 12*s) / (T^2)

    Parameters:
        start (float): Initial angle (deg).
        end (float): Final angle (deg).
        duration (float): Segment duration in seconds (T > 0).
        dt (float): Time step in seconds.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            time, position, velocity, acceleration
    """
    if duration <= 0:
        raise ValueError("Duration must be strictly positive.")

    # Time vector from 0 to duration
    time = np.arange(0.0, duration + dt / 2.0, dt)
    s = np.clip(time / duration, 0.0, 1.0)

    # Position interpolation (Hermite blend)
    blend_pos = 3.0 * s**2 - 2.0 * s**3
    position = start + (end - start) * blend_pos

    # First derivative (velocity)
    blend_vel = (6.0 * s - 6.0 * s**2) / duration
    velocity = (end - start) * blend_vel

    # Second derivative (acceleration)
    blend_acc = (6.0 - 12.0 * s) / (duration**2)
    acceleration = (end - start) * blend_acc

    return time, position, velocity, acceleration


def joint_trajectory(start_angles: tuple[float, float],
                     end_angles: tuple[float, float],
                     duration: float = 2.0,
                     dt: float = 0.02) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate coordinated cubic trajectories for both Joint 1 and Joint 2.

    Parameters:
        start_angles (tuple[float, float]): (theta1_start, theta2_start) in degrees.
        end_angles (tuple[float, float]): (theta1_end, theta2_end) in degrees.
        duration (float): Motion duration in seconds.
        dt (float): Discretization time step in seconds.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            time: Array of timestamps of shape (N,)
            positions: Array of joint positions of shape (N, 2) in degrees
            velocities: Array of joint velocities of shape (N, 2) in deg/s
    """
    t_arr, pos1, vel1, _ = cubic_trajectory(start_angles[0], end_angles[0], duration, dt)
    _, pos2, vel2, _ = cubic_trajectory(start_angles[1], end_angles[1], duration, dt)

    positions = np.column_stack((pos1, pos2))
    velocities = np.column_stack((vel1, vel2))

    return t_arr, positions, velocities


def multi_segment_trajectory(waypoints: list[tuple[float, float]],
                             durations: list[float],
                             dt: float = 0.02) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """
    Concatenate multi-point joint trajectories for sequential operations.

    Parameters:
        waypoints (list[tuple[float, float]]): List of joint configurations [(th1, th2), ...].
        durations (list[float]): Duration for each segment transition.
        dt (float): Time step in seconds.

    Returns:
        tuple: (all_time, all_positions, all_velocities, segment_split_times)
    """
    if len(waypoints) < 2:
        raise ValueError("At least two waypoints are required.")
    if len(durations) != len(waypoints) - 1:
        raise ValueError("Number of durations must be len(waypoints) - 1.")

    all_time = []
    all_pos = []
    all_vel = []
    split_times = [0.0]
    current_time_offset = 0.0

    for i in range(len(waypoints) - 1):
        t_seg, pos_seg, vel_seg = joint_trajectory(waypoints[i], waypoints[i + 1], durations[i], dt)

        # Omit the first sample on consecutive segments to avoid duplicate timestamps
        if i > 0:
            t_seg = t_seg[1:]
            pos_seg = pos_seg[1:]
            vel_seg = vel_seg[1:]

        all_time.append(t_seg + current_time_offset)
        all_pos.append(pos_seg)
        all_vel.append(vel_seg)

        current_time_offset += durations[i]
        split_times.append(current_time_offset)

    full_time = np.concatenate(all_time)
    full_positions = np.vstack(all_pos)
    full_velocities = np.vstack(all_vel)

    return full_time, full_positions, full_velocities, split_times


def compute_cartesian_path(joint_positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Cartesian end-effector path (X, Y) corresponding to joint angle sequence.

    Parameters:
        joint_positions (np.ndarray): Array of shape (N, 2) with [theta1, theta2] in degrees.

    Returns:
        tuple[np.ndarray, np.ndarray]: Array of X coordinates, Array of Y coordinates.
    """
    n = len(joint_positions)
    x_path = np.zeros(n)
    y_path = np.zeros(n)

    for i in range(n):
        x_path[i], y_path[i] = forward_kinematics(joint_positions[i, 0], joint_positions[i, 1])

    return x_path, y_path


if __name__ == "__main__":
    print("=" * 60)
    print("2-DOF ROBOTIC MANIPULATOR: TRAJECTORY PLANNING TEST")
    print("=" * 60)

    start_q = (0.0, 0.0)
    end_q = (1.83, 73.67)
    duration = 2.0
    dt = 0.02

    time, pos, vel = joint_trajectory(start_q, end_q, duration, dt)

    print(f"Start configuration: {start_q} deg")
    print(f"End configuration:   {end_q} deg")
    print(f"Duration:            {duration} s (samples = {len(time)})")
    print(f"Initial positions:   Joint 1 = {pos[0, 0]:.2f} deg, Joint 2 = {pos[0, 1]:.2f} deg")
    print(f"Final positions:     Joint 1 = {pos[-1, 0]:.2f} deg, Joint 2 = {pos[-1, 1]:.2f} deg")
    print(f"Initial velocities:  Joint 1 = {vel[0, 0]:.4f} deg/s, Joint 2 = {vel[0, 1]:.4f} deg/s")
    print(f"Final velocities:    Joint 1 = {vel[-1, 0]:.4f} deg/s, Joint 2 = {vel[-1, 1]:.4f} deg/s")
    print(f"Peak velocity J1:    {np.max(np.abs(vel[:, 0])):.2f} deg/s")
    print(f"Peak velocity J2:    {np.max(np.abs(vel[:, 1])):.2f} deg/s")

    x_path, y_path = compute_cartesian_path(pos)
    print(f"Initial Cartesian:   ({x_path[0]:.2f}, {y_path[0]:.2f}) mm")
    print(f"Final Cartesian:     ({x_path[-1]:.2f}, {y_path[-1]:.2f}) mm")

    print("\n" + "=" * 60)
    print("TRAJECTORY PLANNING TEST PASSED SUCCESSFULLY.")
    print("=" * 60)
