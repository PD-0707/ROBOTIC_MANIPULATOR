"""
Visualization and Animation Module for 2-DOF Robotic Manipulator
----------------------------------------------------------------
Provides:
1. Static robot pose plotting with link and joint representation.
2. Reachable workspace plotting with target positions.
3. Commanded vs. actual joint tracking performance plots.
4. End-effector Cartesian trajectory visualization.
5. Interactive/GIF animation of the full pick-and-place sequence with gripper HUD.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

from kinematics import L1, L2, forward_kinematics_full, workspace_points


def setup_results_dir(directory: str = "results") -> str:
    """Ensure output results directory exists."""
    os.makedirs(directory, exist_ok=True)
    return directory


def plot_robot(theta1: float,
               theta2: float,
               target: tuple[float, float] = None,
               path: tuple[np.ndarray, np.ndarray] = None,
               title: str = "2-DOF Robotic Manipulator",
               save_path: str = None) -> plt.Figure:
    """
    Plot single kinematic configuration of the robotic arm.

    Parameters:
        theta1 (float): Joint 1 angle in degrees.
        theta2 (float): Joint 2 angle in degrees.
        target (tuple): Optional (x, y) target point in mm.
        path (tuple): Optional (x_array, y_array) path history.
        title (str): Figure title.
        save_path (str): Optional file path to save image.
    """
    (x0, y0), (x1, y1), (x2, y2) = forward_kinematics_full(theta1, theta2)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_xlabel("X Position (mm)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Y Position (mm)", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    # Draw maximum reach boundary circle
    max_reach = L1 + L2
    circle = plt.Circle((0, 0), max_reach, color="lightgray", fill=False, linestyle=":", linewidth=1.5, label="Max Workspace (R=160mm)")
    ax.add_patch(circle)

    # Plot base mount pedestal
    ax.plot([-25, 25], [-5, -5], color="#333333", linewidth=6)
    ax.plot([0, 0], [-5, 0], color="#555555", linewidth=8)

    # Plot trajectory path trail if provided
    if path is not None and len(path[0]) > 0:
        ax.plot(path[0], path[1], color="#E67E22", linestyle="--", linewidth=1.8, alpha=0.8, label="End-Effector Path")

    # Plot target marker if provided
    if target is not None:
        ax.scatter(target[0], target[1], color="#E74C3C", marker="X", s=140, zorder=5, label=f"Target ({target[0]:.1f}, {target[1]:.1f}) mm")

    # Plot Links
    # Link 1: Base to Elbow
    ax.plot([x0, x1], [y0, y1], color="#2980B9", linewidth=6, solid_capstyle="round", label=f"Link 1 (L1={L1:.0f}mm)", zorder=3)
    # Link 2: Elbow to End-Effector
    ax.plot([x1, x2], [y1, y2], color="#27AE60", linewidth=5, solid_capstyle="round", label=f"Link 2 (L2={L2:.0f}mm)", zorder=3)

    # Plot Joint Hubs
    ax.scatter([x0], [y0], s=180, color="#2C3E50", edgecolors="white", linewidths=2, zorder=4, label="Base Joint (J1)")
    ax.scatter([x1], [y1], s=140, color="#16A085", edgecolors="white", linewidths=2, zorder=4, label="Elbow Joint (J2)")
    ax.scatter([x2], [y2], s=120, color="#D35400", edgecolors="black", linewidths=1.5, zorder=4, label="End-Effector")

    # Annotation box
    info_text = (f"$\\theta_1$ (Shoulder): {theta1:.2f}°\n"
                 f"$\\theta_2$ (Elbow):    {theta2:.2f}°\n"
                 f"End-Effector: ({x2:.2f}, {y2:.2f}) mm")
    ax.text(0.03, 0.97, info_text, transform=ax.transAxes, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#BDC3C7", alpha=0.9),
            fontsize=10, family="monospace")

    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
    plt.tight_layout()

    if save_path:
        setup_results_dir(os.path.dirname(save_path) or "results")
        plt.savefig(save_path, dpi=300)
        print(f"[Saved Plot] {save_path}")

    return fig


def plot_workspace(target_pick: tuple[float, float] = None,
                   target_place: tuple[float, float] = None,
                   home_pos: tuple[float, float] = None,
                   save_path: str = "results/workspace.png") -> plt.Figure:
    """
    Generate and save the reachable workspace plot with operating waypoints.
    """
    setup_results_dir(os.path.dirname(save_path) or "results")

    # Generate workspace scatter points
    x_pts, y_pts = workspace_points(num_samples=3000)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlim(-185, 185)
    ax.set_ylim(-185, 185)

    ax.set_title("2-DOF Manipulator: Reachable Workspace Analysis", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("X (mm)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Y (mm)", fontsize=11, fontweight="bold")

    # Reachable points cloud
    ax.scatter(x_pts, y_pts, s=6, color="#AED6F1", alpha=0.35, label="Reachable Workspace Area")

    # Maximum boundary circle R = L1 + L2 = 160 mm
    r_max = L1 + L2
    outer_circle = plt.Circle((0, 0), r_max, color="#2980B9", fill=False, linestyle="-", linewidth=2.0, label=f"Outer Reach ($R_{{max}}={r_max:.0f}$ mm)")
    ax.add_patch(outer_circle)

    # Base hub
    ax.scatter([0], [0], s=120, color="#2C3E50", marker="o", label="Base Origin (0,0)")

    # Key task waypoints
    if home_pos:
        ax.scatter([home_pos[0]], [home_pos[1]], color="#8E44AD", s=140, marker="o", zorder=5, label=f"Home ({home_pos[0]:.0f}, {home_pos[1]:.0f})")
    if target_pick:
        ax.scatter([target_pick[0]], [target_pick[1]], color="#27AE60", s=160, marker="P", zorder=5, label=f"Pick Location ({target_pick[0]:.0f}, {target_pick[1]:.0f})")
    if target_place:
        ax.scatter([target_place[0]], [target_place[1]], color="#E74C3C", s=160, marker="X", zorder=5, label=f"Place Location ({target_place[0]:.0f}, {target_place[1]:.0f})")

    ax.legend(loc="lower left", framealpha=0.92, fontsize=9)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    print(f"[Saved Plot] {save_path}")
    return fig


def plot_tracking_performance(time_log: np.ndarray,
                              cmd_angles: np.ndarray,
                              act_angles: np.ndarray,
                              save_dir: str = "results"):
    """
    Generate professional tracking performance plots:
    1. joint1_tracking.png (Commanded vs Actual and Error)
    2. joint2_tracking.png (Commanded vs Actual and Error)
    3. tracking_error.png (Both joint errors on same timeline)
    """
    setup_results_dir(save_dir)

    err1 = cmd_angles[:, 0] - act_angles[:, 0]
    err2 = cmd_angles[:, 1] - act_angles[:, 1]

    # --- Plot Joint 1 Tracking ---
    fig1, (ax1_top, ax1_bot) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]})
    ax1_top.plot(time_log, cmd_angles[:, 0], "r--", linewidth=2.0, label="Commanded $\\theta_1$ (deg)")
    ax1_top.plot(time_log, act_angles[:, 0], "b-", linewidth=1.8, label="Actual $\\theta_1$ (deg)")
    ax1_top.set_ylabel("Joint 1 Angle (deg)", fontweight="bold")
    ax1_top.set_title("Joint 1 (Shoulder) Dynamic Tracking Performance", fontsize=12, fontweight="bold")
    ax1_top.grid(True, linestyle="--", alpha=0.6)
    ax1_top.legend(loc="upper right")

    ax1_bot.plot(time_log, err1, color="#C0392B", linewidth=1.5, label="Tracking Error ($e_1$)")
    ax1_bot.axhline(0, color="black", linestyle=":", linewidth=1.0)
    ax1_bot.set_xlabel("Time (seconds)", fontweight="bold")
    ax1_bot.set_ylabel("Error (deg)", fontweight="bold")
    ax1_bot.grid(True, linestyle="--", alpha=0.6)
    ax1_bot.legend(loc="upper right")
    plt.tight_layout()
    j1_path = os.path.join(save_dir, "joint1_tracking.png")
    fig1.savefig(j1_path, dpi=300)
    plt.close(fig1)
    print(f"[Saved Plot] {j1_path}")

    # --- Plot Joint 2 Tracking ---
    fig2, (ax2_top, ax2_bot) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]})
    ax2_top.plot(time_log, cmd_angles[:, 1], "r--", linewidth=2.0, label="Commanded $\\theta_2$ (deg)")
    ax2_top.plot(time_log, act_angles[:, 1], "g-", linewidth=1.8, label="Actual $\\theta_2$ (deg)")
    ax2_top.set_ylabel("Joint 2 Angle (deg)", fontweight="bold")
    ax2_top.set_title("Joint 2 (Elbow) Dynamic Tracking Performance", fontsize=12, fontweight="bold")
    ax2_top.grid(True, linestyle="--", alpha=0.6)
    ax2_top.legend(loc="upper right")

    ax2_bot.plot(time_log, err2, color="#27AE60", linewidth=1.5, label="Tracking Error ($e_2$)")
    ax2_bot.axhline(0, color="black", linestyle=":", linewidth=1.0)
    ax2_bot.set_xlabel("Time (seconds)", fontweight="bold")
    ax2_bot.set_ylabel("Error (deg)", fontweight="bold")
    ax2_bot.grid(True, linestyle="--", alpha=0.6)
    ax2_bot.legend(loc="upper right")
    plt.tight_layout()
    j2_path = os.path.join(save_dir, "joint2_tracking.png")
    fig2.savefig(j2_path, dpi=300)
    plt.close(fig2)
    print(f"[Saved Plot] {j2_path}")

    # --- Plot Combined Tracking Error ---
    fig3, ax3 = plt.subplots(figsize=(10, 4.5))
    ax3.plot(time_log, err1, label="Joint 1 Error (deg)", color="#2980B9", linewidth=1.8)
    ax3.plot(time_log, err2, label="Joint 2 Error (deg)", color="#27AE60", linewidth=1.8)
    ax3.axhline(0, color="gray", linestyle="--", alpha=0.7)
    ax3.set_title("PID Closed-Loop Joint Tracking Errors", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Time (seconds)", fontweight="bold")
    ax3.set_ylabel("Angular Error (deg)", fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend(loc="upper right")
    plt.tight_layout()
    err_path = os.path.join(save_dir, "tracking_error.png")
    fig3.savefig(err_path, dpi=300)
    plt.close(fig3)
    print(f"[Saved Plot] {err_path}")


def plot_cartesian_trajectory(x_act: np.ndarray,
                              y_act: np.ndarray,
                              pick_pt: tuple[float, float],
                              place_pt: tuple[float, float],
                              home_pt: tuple[float, float],
                              save_path: str = "results/end_effector_trajectory.png"):
    """
    Plot Cartesian path executed by the end-effector with target annotations.
    """
    setup_results_dir(os.path.dirname(save_path) or "results")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)

    # Workspace circle
    circle = plt.Circle((0, 0), L1 + L2, color="gray", fill=False, linestyle=":", linewidth=1.5, label="Workspace Limit")
    ax.add_patch(circle)

    # Executed path
    ax.plot(x_act, y_act, color="#2980B9", linewidth=2.5, label="End-Effector Trajectory")
    ax.scatter(x_act[0], y_act[0], color="#8E44AD", s=120, zorder=5, label=f"Start / Home ({home_pt[0]:.0f}, {home_pt[1]:.0f})")
    ax.scatter(pick_pt[0], pick_pt[1], color="#27AE60", marker="P", s=160, zorder=5, label=f"Pick Position ({pick_pt[0]:.0f}, {pick_pt[1]:.0f})")
    ax.scatter(place_pt[0], place_pt[1], color="#E74C3C", marker="X", s=160, zorder=5, label=f"Place Position ({place_pt[0]:.0f}, {place_pt[1]:.0f})")

    ax.set_title("End-Effector Cartesian Pick-and-Place Path", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("X Position (mm)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Y Position (mm)", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.92)
    plt.tight_layout()

    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"[Saved Plot] {save_path}")


def animate_pick_and_place(time_array: np.ndarray,
                           theta1_arr: np.ndarray,
                           theta2_arr: np.ndarray,
                           gripper_states: list[str],
                           pick_target: tuple[float, float],
                           place_target: tuple[float, float],
                           home_target: tuple[float, float],
                           step_interval: int = 2,
                           save_gif_path: str = None,
                           show: bool = False):
    """
    Create animated visualization of the robotic manipulator performing pick and place.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlabel("X (mm)", fontweight="bold")
    ax.set_ylabel("Y (mm)", fontweight="bold")
    ax.set_title("2-DOF Manipulator: Pick-and-Place Simulation", fontsize=12, fontweight="bold")

    # Static elements
    workspace_circle = plt.Circle((0, 0), L1 + L2, color="lightgray", fill=False, linestyle=":", linewidth=1.5)
    ax.add_patch(workspace_circle)
    ax.scatter([0], [0], s=120, color="#2C3E50", zorder=4)
    ax.scatter([home_target[0]], [home_target[1]], color="#8E44AD", marker="o", s=90, label="Home")
    ax.scatter([pick_target[0]], [pick_target[1]], color="#27AE60", marker="P", s=130, label="Pick")
    ax.scatter([place_target[0]], [place_target[1]], color="#E74C3C", marker="X", s=130, label="Place")

    # Dynamic line elements
    link1_line, = ax.plot([], [], color="#2980B9", linewidth=6, solid_capstyle="round")
    link2_line, = ax.plot([], [], color="#27AE60", linewidth=5, solid_capstyle="round")
    elbow_hub = ax.scatter([], [], s=100, color="#16A085", zorder=5)
    ee_hub = ax.scatter([], [], s=100, color="#D35400", zorder=5)
    trail_line, = ax.plot([], [], color="#E67E22", linestyle="--", linewidth=1.5, alpha=0.7)
    payload_marker = ax.scatter([], [], s=140, marker="s", color="#F39C12", edgecolors="black", zorder=6, label="Payload")

    hud_text = ax.text(0.03, 0.97, "", transform=ax.transAxes, verticalalignment="top",
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#BDC3C7", alpha=0.9),
                       fontsize=9, family="monospace")

    ax.legend(loc="lower right", fontsize=9)

    trail_x, trail_y = [], []
    frame_indices = list(range(0, len(time_array), step_interval))
    if frame_indices[-1] != len(time_array) - 1:
        frame_indices.append(len(time_array) - 1)

    def update_frame(idx):
        t = time_array[idx]
        th1 = theta1_arr[idx]
        th2 = theta2_arr[idx]
        state = gripper_states[idx]

        (x0, y0), (x1, y1), (x2, y2) = forward_kinematics_full(th1, th2)

        link1_line.set_data([x0, x1], [y0, y1])
        link2_line.set_data([x1, x2], [y1, y2])
        elbow_hub.set_offsets([[x1, y1]])
        ee_hub.set_offsets([[x2, y2]])

        trail_x.append(x2)
        trail_y.append(y2)
        trail_line.set_data(trail_x, trail_y)

        # Draw payload with gripper
        if state in ["CLOSED_HOLDING", "CLOSING"]:
            payload_marker.set_offsets([[x2, y2]])
        elif state == "OPEN" and len(trail_x) > len(frame_indices) * 0.7:
            payload_marker.set_offsets([[place_target[0], place_target[1]]])
        else:
            payload_marker.set_offsets([[pick_target[0], pick_target[1]]])

        hud_text.set_text(
            f"Time:        {t:5.2f} s\n"
            f"$\\theta_1$ (Shoulder): {th1:5.1f}°\n"
            f"$\\theta_2$ (Elbow):    {th2:5.1f}°\n"
            f"End-Effector: ({x2:5.1f}, {y2:5.1f}) mm\n"
            f"Gripper:     {state}"
        )
        return link1_line, link2_line, elbow_hub, ee_hub, trail_line, payload_marker, hud_text

    anim = FuncAnimation(fig, update_frame, frames=frame_indices, interval=30, blit=False)

    if save_gif_path:
        setup_results_dir(os.path.dirname(save_gif_path) or "results")
        try:
            from matplotlib.animation import PillowWriter
            writer = PillowWriter(fps=25)
            anim.save(save_gif_path, writer=writer)
            print(f"[Saved Animation GIF] {save_gif_path}")
        except Exception as ex:
            print(f"[Notice] Could not save GIF ({ex}), static plots are safely preserved.")

    if show:
        try:
            plt.show()
        except Exception:
            pass

    plt.close(fig)
    return anim


if __name__ == "__main__":
    print("=" * 60)
    print("2-DOF ROBOTIC MANIPULATOR: VISUALIZATION MODULE TEST")
    print("=" * 60)

    # Test static pose plot
    plot_robot(theta1=1.83, theta2=73.67, target=(100.0, 80.0), title="Test Pose (100, 80) mm", save_path="results/test_pose.png")

    # Test workspace plot
    plot_workspace(target_pick=(100.0, 80.0), target_place=(60.0, 120.0), home_pos=(120.0, 40.0), save_path="results/workspace.png")

    print("\n" + "=" * 60)
    print("VISUALIZATION TESTS COMPLETED SUCCESSFULLY.")
    print("=" * 60)
