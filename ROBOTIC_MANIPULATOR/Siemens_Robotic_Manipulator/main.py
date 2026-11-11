"""
Main Simulation and Control Orchestrator
Project: Simulation and Control of a 2-DOF Robotic Pick-and-Place Manipulator
-----------------------------------------------------------------------------
Executes the full closed-loop simulation:
1. Inverse kinematics calculation for task waypoints (Home, Pick, Place).
2. Trajectory generation with cubic Hermite blending.
3. Closed-loop PID control and servo actuator dynamics simulation.
4. Parallel gripper mechanical state machine.
5. Export of engineering performance plots, CSV telemetry, and animated visualizer.
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from kinematics import (
    L1,
    L2,
    forward_kinematics,
    inverse_kinematics,
    is_reachable
)
from trajectory import (
    cubic_trajectory,
    compute_cartesian_path
)
from controller import (
    PIDController,
    ServoActuator,
    GripperActuator,
    GripperState
)
from robot_visualization import (
    plot_workspace,
    plot_tracking_performance,
    plot_cartesian_trajectory,
    animate_pick_and_place,
    setup_results_dir
)


# Configuration & Task Specifications
RESULTS_DIR = "results"
setup_results_dir(RESULTS_DIR)

# Cartesian Waypoints (in mm)
HOME_POS = (120.0, 40.0)      # Ready / standby position
PICK_POS = (100.0, 80.0)      # Target pickup location
PLACE_POS = (60.0, 120.0)     # Target drop-off location

# Motion Timing Parameters (seconds)
T_MOVE_PICK = 2.0             # Travel time: Home -> Pick
T_DWELL_GRASP = 0.8           # Gripper closure dwell
T_MOVE_PLACE = 2.5            # Travel time: Pick -> Place
T_DWELL_RELEASE = 0.8         # Gripper release dwell
T_MOVE_HOME = 2.0             # Travel time: Place -> Home
DT = 0.02                     # Simulation sampling step (50 Hz)


def validate_waypoints() -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """
    Check reachability and solve inverse kinematics for all task points.
    Returns joint configurations (deg) for (Home, Pick, Place).
    """
    print("=" * 68)
    print("PHASE 1 & 2: KINEMATICS VALIDATION & WAYPOINT CALCULATION")
    print("=" * 68)

    waypoints = [("Home", HOME_POS), ("Pick", PICK_POS), ("Place", PLACE_POS)]
    joint_angles = []

    for name, (x, y) in waypoints:
        if not is_reachable(x, y):
            raise ValueError(f"CRITICAL: Point {name} at ({x}, {y}) mm is outside arm reach!")
        th1, th2 = inverse_kinematics(x, y, elbow_up=False)
        x_rec, y_rec = forward_kinematics(th1, th2)
        joint_angles.append((th1, th2))
        print(f"[{name.upper():5s}] Target: ({x:6.2f}, {y:6.2f}) mm -> Joint Angles: J1 = {th1:6.2f}°, J2 = {th2:6.2f}° "
              f"| Verified: ({x_rec:6.2f}, {y_rec:6.2f}) mm")

    return joint_angles[0], joint_angles[1], joint_angles[2]


def build_mission_profile(q_home, q_pick, q_place, dt=0.02):
    """
    Synthesize complete multi-stage reference trajectory with dwell states.
    Returns:
        time_seq (np.ndarray): Master simulation timeline
        cmd_q (np.ndarray): Array of shape (N, 2) with desired joint angles
        gripper_cmd (list[str]): Desired gripper actions ('OPEN', 'CLOSE')
        phase_names (list[str]): Active task phase label for each step
    """
    time_list = []
    q_list = []
    gripper_cmd_list = []
    phase_list = []
    curr_time = 0.0

    def append_segment(start_q, end_q, duration, phase_name, grip_action):
        nonlocal curr_time
        t_seg, pos1, _, _ = cubic_trajectory(start_q[0], end_q[0], duration, dt)
        _, pos2, _, _ = cubic_trajectory(start_q[1], end_q[1], duration, dt)

        # Skip first timestamp if continuing from previous phase to avoid duplicate times
        start_idx = 1 if len(time_list) > 0 else 0
        for i in range(start_idx, len(t_seg)):
            time_list.append(curr_time + t_seg[i])
            q_list.append([pos1[i], pos2[i]])
            gripper_cmd_list.append(grip_action)
            phase_list.append(phase_name)
        curr_time += duration

    def append_dwell(hold_q, duration, phase_name, grip_action):
        nonlocal curr_time
        steps = int(duration / dt)
        for _ in range(steps):
            curr_time += dt
            time_list.append(curr_time)
            q_list.append(list(hold_q))
            gripper_cmd_list.append(grip_action)
            phase_list.append(phase_name)

    # Sequence Execution
    # Stage 1: Move from Home to Pick
    append_segment(q_home, q_pick, T_MOVE_PICK, "MOVE_TO_PICK", "OPEN")
    # Stage 2: Gripper Close & Dwell at Pick
    append_dwell(q_pick, T_DWELL_GRASP, "GRASPING_PAYLOAD", "CLOSE")
    # Stage 3: Transfer to Place
    append_segment(q_pick, q_place, T_MOVE_PLACE, "TRANSFER_TO_PLACE", "HOLD")
    # Stage 4: Gripper Open & Dwell at Place
    append_dwell(q_place, T_DWELL_RELEASE, "RELEASING_PAYLOAD", "OPEN")
    # Stage 5: Return to Home
    append_segment(q_place, q_home, T_MOVE_HOME, "RETURN_TO_HOME", "OPEN")

    return np.array(time_list), np.array(q_list), gripper_cmd_list, phase_list


def run_pick_and_place_simulation():
    """Execute complete closed-loop dynamics simulation and log telemetry."""
    # 1. Kinematics & Waypoints
    q_home, q_pick, q_place = validate_waypoints()

    # 2. Trajectory Profile Construction
    print("\n" + "=" * 68)
    print("PHASE 3: TRAJECTORY SYNTHESIS (CUBIC HERMITE INTERPOLATION)")
    print("=" * 68)
    time_sim, cmd_q, grip_cmd, phase_names = build_mission_profile(q_home, q_pick, q_place, DT)
    total_duration = time_sim[-1]
    n_steps = len(time_sim)
    print(f"Total Trajectory Duration: {total_duration:.2f} s | Total Simulation Steps: {n_steps} (dt = {DT}s)")

    # 3. Initialize Controllers & Actuators
    print("\n" + "=" * 68)
    print("PHASE 4 & 5: CLOSED-LOOP PID & ACTUATOR SIMULATION")
    print("=" * 68)
    # Joint 1 PID Controller & Servo Actuator
    pid_j1 = PIDController(kp=5.0, ki=0.8, kd=0.20, integral_limit=80.0, output_limit=180.0)
    actuator_j1 = ServoActuator(initial_angle=q_home[0], max_velocity=180.0, response_lag=0.04)

    # Joint 2 PID Controller & Servo Actuator
    pid_j2 = PIDController(kp=5.0, ki=0.8, kd=0.20, integral_limit=80.0, output_limit=180.0)
    actuator_j2 = ServoActuator(initial_angle=q_home[1], max_velocity=180.0, response_lag=0.04)

    # Parallel Jaw Gripper Model
    gripper = GripperActuator(stroke_mm=30.0, speed_mms=75.0)

    # Preallocate telemetry buffers
    act_q = np.zeros_like(cmd_q)
    err_q = np.zeros_like(cmd_q)
    cart_act = np.zeros((n_steps, 2))
    gripper_state_log = []
    gripper_width_log = []

    # 4. Step-by-Step Simulation Loop
    for k in range(n_steps):
        t = time_sim[k]
        q1_ref, q2_ref = cmd_q[k, 0], cmd_q[k, 1]
        action = grip_cmd[k]

        # Update Gripper State Machine
        if action == "CLOSE":
            gripper.close()
        elif action == "OPEN":
            gripper.open()

        # Check proximity to pick location for realistic grasping
        cur_x, cur_y = forward_kinematics(actuator_j1.angle, actuator_j2.angle)
        dist_to_pick = np.hypot(cur_x - PICK_POS[0], cur_y - PICK_POS[1])
        gripper.update(dt=DT, near_object=(dist_to_pick < 5.0))

        # PID Control Update (Compute commanded angular velocity)
        u1 = pid_j1.update(setpoint=q1_ref, measured=actuator_j1.angle, dt=DT)
        u2 = pid_j2.update(setpoint=q2_ref, measured=actuator_j2.angle, dt=DT)

        # Actuator Dynamics Simulation
        q1_actual = actuator_j1.update(control_signal=u1, dt=DT)
        q2_actual = actuator_j2.update(control_signal=u2, dt=DT)

        # Log States
        act_q[k, 0] = q1_actual
        act_q[k, 1] = q2_actual
        err_q[k, 0] = q1_ref - q1_actual
        err_q[k, 1] = q2_ref - q2_actual

        ee_x, ee_y = forward_kinematics(q1_actual, q2_actual)
        cart_act[k, 0] = ee_x
        cart_act[k, 1] = ee_y

        gripper_state_log.append(gripper.state.value)
        gripper_width_log.append(gripper.width)

    print("Dynamic loop simulation completed successfully without errors.")

    # 5. Export Telemetry to CSV
    csv_file = os.path.join(RESULTS_DIR, "joint_data.csv")
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Time_s", "Phase",
            "Joint1_Cmd_deg", "Joint1_Act_deg", "Joint1_Err_deg",
            "Joint2_Cmd_deg", "Joint2_Act_deg", "Joint2_Err_deg",
            "EE_X_act_mm", "EE_Y_act_mm",
            "Gripper_State", "Gripper_Width_mm"
        ])
        for k in range(n_steps):
            writer.writerow([
                f"{time_sim[k]:.3f}", phase_names[k],
                f"{cmd_q[k, 0]:.4f}", f"{act_q[k, 0]:.4f}", f"{err_q[k, 0]:.4f}",
                f"{cmd_q[k, 1]:.4f}", f"{act_q[k, 1]:.4f}", f"{err_q[k, 1]:.4f}",
                f"{cart_act[k, 0]:.4f}", f"{cart_act[k, 1]:.4f}",
                gripper_state_log[k], f"{gripper_width_log[k]:.2f}"
            ])
    print(f"\n[Telemetry Exported] {csv_file}")

    # 6. Performance Evaluation Metrics
    max_err_j1 = np.max(np.abs(err_q[:, 0]))
    max_err_j2 = np.max(np.abs(err_q[:, 1]))
    rmse_j1 = np.sqrt(np.mean(err_q[:, 0]**2))
    rmse_j2 = np.sqrt(np.mean(err_q[:, 1]**2))

    # Evaluate Placement Accuracy at completion of Stage 3
    # Find step where Place was reached
    place_indices = [i for i, p in enumerate(phase_names) if p == "RELEASING_PAYLOAD"]
    eval_idx = place_indices[0] if place_indices else -1
    actual_place_x = cart_act[eval_idx, 0]
    actual_place_y = cart_act[eval_idx, 1]
    place_error_mm = np.hypot(actual_place_x - PLACE_POS[0], actual_place_y - PLACE_POS[1])

    print("\n" + "=" * 68)
    print("PHASE 7: PERFORMANCE & ACCURACY METRICS")
    print("=" * 68)
    print(f"Joint 1 (Shoulder) Max Error:  {max_err_j1:6.3f}° | RMS Error: {rmse_j1:6.3f}°")
    print(f"Joint 2 (Elbow)    Max Error:  {max_err_j2:6.3f}° | RMS Error: {rmse_j2:6.3f}°")
    print(f"Nominal Place Target:         ({PLACE_POS[0]:.2f}, {PLACE_POS[1]:.2f}) mm")
    print(f"Achieved Place Position:      ({actual_place_x:.2f}, {actual_place_y:.2f}) mm")
    print(f"End-Effector Placement Error:  {place_error_mm:6.4f} mm")

    # 7. Generate Engineering Plots
    print("\n" + "=" * 68)
    print("GENERATING VERIFICATION PLOTS & ARTIFACTS")
    print("=" * 68)
    plot_workspace(
        target_pick=PICK_POS,
        target_place=PLACE_POS,
        home_pos=HOME_POS,
        save_path=os.path.join(RESULTS_DIR, "workspace.png")
    )

    plot_tracking_performance(
        time_log=time_sim,
        cmd_angles=cmd_q,
        act_angles=act_q,
        save_dir=RESULTS_DIR
    )

    plot_cartesian_trajectory(
        x_act=cart_act[:, 0],
        y_act=cart_act[:, 1],
        pick_pt=PICK_POS,
        place_pt=PLACE_POS,
        home_pt=HOME_POS,
        save_path=os.path.join(RESULTS_DIR, "end_effector_trajectory.png")
    )

    # 8. Render Animation to GIF
    gif_path = os.path.join(RESULTS_DIR, "pick_and_place_animation.gif")
    print(f"Rendering animated simulation to {gif_path} (step subsampling=3)...")
    animate_pick_and_place(
        time_array=time_sim,
        theta1_arr=act_q[:, 0],
        theta2_arr=act_q[:, 1],
        gripper_states=gripper_state_log,
        pick_target=PICK_POS,
        place_target=PLACE_POS,
        home_target=HOME_POS,
        step_interval=3,
        save_gif_path=gif_path,
        show=False
    )

    print("\n" + "=" * 68)
    print("ALL SIMULATION PHASES COMPLETED WITH DISTINCTION.")
    print("=" * 68)


if __name__ == "__main__":
    run_pick_and_place_simulation()
