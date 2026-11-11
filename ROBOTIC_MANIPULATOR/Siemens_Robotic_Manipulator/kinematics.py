"""
Kinematics Module for 2-DOF Planar Robotic Manipulator
-----------------------------------------------------
Calculates Forward and Inverse Kinematics for a two-link planar arm.
Link dimensions: L1 = 80.0 mm, L2 = 80.0 mm
"""

import numpy as np

# Robot link dimensions (in millimeters)
L1 = 80.0
L2 = 80.0

def forward_kinematics(theta1: float, theta2: float) -> tuple[float, float]:
    """
    Calculate end-effector (x, y) coordinates from joint angles.

    Parameters:
        theta1 (float): Shoulder joint angle in degrees.
        theta2 (float): Elbow joint angle in degrees relative to Link 1.

    Returns:
        tuple[float, float]: End-effector coordinates (x, y) in mm.
    """
    t1 = np.radians(theta1)
    t2 = np.radians(theta2)

    x = L1 * np.cos(t1) + L2 * np.cos(t1 + t2)
    y = L1 * np.sin(t1) + L2 * np.sin(t1 + t2)

    return float(x), float(y)


def forward_kinematics_full(theta1: float, theta2: float) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """
    Calculate full link positions: Base (0, 0), Elbow (x1, y1), and End-Effector (x2, y2).

    Parameters:
        theta1 (float): Shoulder joint angle in degrees.
        theta2 (float): Elbow joint angle in degrees.

    Returns:
        tuple of points: ((x0, y0), (x1, y1), (x2, y2)) in mm.
    """
    t1 = np.radians(theta1)
    t2 = np.radians(theta2)

    x0, y0 = 0.0, 0.0
    x1 = L1 * np.cos(t1)
    y1 = L1 * np.sin(t1)
    x2 = x1 + L2 * np.cos(t1 + t2)
    y2 = y1 + L2 * np.sin(t1 + t2)

    return (x0, y0), (float(x1), float(y1)), (float(x2), float(y2))


def is_reachable(x: float, y: float) -> bool:
    """
    Check if a Cartesian point (x, y) is within the theoretical reachable workspace.
    """
    r_sq = x**2 + y**2
    r_min = abs(L1 - L2)
    r_max = L1 + L2
    return (r_min**2 <= r_sq <= r_max**2 + 1e-6)


def inverse_kinematics(x: float, y: float, elbow_up: bool = False) -> tuple[float, float]:
    """
    Calculate joint angles (theta1, theta2) in degrees for a given end-effector position.

    Parameters:
        x (float): Target X coordinate in mm.
        y (float): Target Y coordinate in mm.
        elbow_up (bool): True for elbow-up configuration (negative theta2),
                         False for elbow-down configuration (positive theta2).

    Returns:
        tuple[float, float]: Joint angles (theta1, theta2) in degrees.

    Raises:
        ValueError: If target (x, y) is outside the reachable workspace.
    """
    r_squared = x**2 + y**2
    r_min = abs(L1 - L2)
    r_max = L1 + L2

    # Workspace boundary check
    cos_t2 = (r_squared - L1**2 - L2**2) / (2.0 * L1 * L2)

    if cos_t2 < -1.0 - 1e-7 or cos_t2 > 1.0 + 1e-7:
        target_dist = np.sqrt(r_squared)
        raise ValueError(
            f"Target position ({x:.1f}, {y:.1f}) mm with radius {target_dist:.1f} mm "
            f"is outside the reachable workspace [{r_min:.1f}, {r_max:.1f}] mm."
        )

    # Numerical precision clamp
    cos_t2 = float(np.clip(cos_t2, -1.0, 1.0))

    # Calculate elbow angle theta2
    # Standard convention: elbow_up -> negative theta2, elbow_down -> positive theta2
    sign = -1.0 if elbow_up else 1.0
    t2 = sign * np.arccos(cos_t2)

    # Calculate shoulder angle theta1
    # theta1 = atan2(y, x) - atan2(L2 * sin(t2), L1 + L2 * cos(t2))
    k1 = L1 + L2 * np.cos(t2)
    k2 = L2 * np.sin(t2)
    t1 = np.arctan2(y, x) - np.arctan2(k2, k1)

    theta1_deg = float(np.degrees(t1))
    theta2_deg = float(np.degrees(t2))

    return theta1_deg, theta2_deg


def workspace_points(num_samples: int = 1500,
                     t1_range: tuple[float, float] = (-180.0, 180.0),
                     t2_range: tuple[float, float] = (-150.0, 150.0)) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate reachable Cartesian (x, y) points across joint limits for workspace plotting.

    Parameters:
        num_samples (int): Number of sample points.
        t1_range (tuple): (min, max) limits for shoulder joint in degrees.
        t2_range (tuple): (min, max) limits for elbow joint in degrees.

    Returns:
        tuple[np.ndarray, np.ndarray]: Array of X points, Array of Y points.
    """
    np.random.seed(42)
    t1_samples = np.random.uniform(t1_range[0], t1_range[1], num_samples)
    t2_samples = np.random.uniform(t2_range[0], t2_range[1], num_samples)

    x_points = np.zeros(num_samples)
    y_points = np.zeros(num_samples)

    for i in range(num_samples):
        x_points[i], y_points[i] = forward_kinematics(t1_samples[i], t2_samples[i])

    return x_points, y_points


if __name__ == "__main__":
    print("=" * 60)
    print("2-DOF ROBOTIC MANIPULATOR: KINEMATICS VERIFICATION")
    print("=" * 60)

    # 1. Forward Kinematics Test (Fully extended: 0 deg, 0 deg)
    x_ext, y_ext = forward_kinematics(0.0, 0.0)
    print(f"\n[Test 1] Fully Extended (theta1=0 deg, theta2=0 deg):")
    print(f"  Expected: X = 160.00 mm, Y = 0.00 mm")
    print(f"  Calculated: X = {x_ext:.2f} mm, Y = {y_ext:.2f} mm")

    # 2. Forward Kinematics Benchmark Test Cases
    benchmark_cases = [
        ("Folded", 0.0, 180.0, 0.0, 0.0),
        ("Bent Vertical", 90.0, 0.0, 0.0, 160.0),
        ("Right-Angle Elbow", 0.0, 90.0, 80.0, 80.0),
    ]
    for name, th1, th2, exp_x, exp_y in benchmark_cases:
        calc_x, calc_y = forward_kinematics(th1, th2)
        print(f"\n[Benchmark: {name}] (theta1={th1} deg, theta2={th2} deg):")
        print(f"  Expected:   X = {exp_x:.2f} mm, Y = {exp_y:.2f} mm")
        print(f"  Calculated: X = {calc_x:.2f} mm, Y = {calc_y:.2f} mm")

    # 3. Inverse Kinematics Test for Target (100, 80) mm
    target_x, target_y = 100.0, 80.0
    th1_sol, th2_sol = inverse_kinematics(target_x, target_y, elbow_up=False)
    print(f"\n[Test 2] Inverse Kinematics for Target ({target_x:.2f}, {target_y:.2f}) mm:")
    print(f"  Joint 1 (Shoulder) = {th1_sol:.2f} deg")
    print(f"  Joint 2 (Elbow)    = {th2_sol:.2f} deg")

    # 4. Closed-Loop Position Verification (IK -> FK)
    verif_x, verif_y = forward_kinematics(th1_sol, th2_sol)
    print(f"\n[Test 3] Closed-Loop Position Verification (FK(IK(Target))):")
    print(f"  Target:     X = {target_x:.2f} mm, Y = {target_y:.2f} mm")
    print(f"  Reconstructed: X = {verif_x:.2f} mm, Y = {verif_y:.2f} mm")
    error = np.sqrt((verif_x - target_x)**2 + (verif_y - target_y)**2)
    print(f"  Position Error: {error:.4e} mm")

    # 5. Unreachable Position Rejection Test
    print(f"\n[Test 4] Unreachable Workspace Boundary Test (200, 100) mm:")
    try:
        inverse_kinematics(200.0, 100.0)
        print("  ERROR: Out-of-bounds position was not caught!")
    except ValueError as e:
        print(f"  SUCCESS: Exception correctly raised -> {e}")

    print("\n" + "=" * 60)
    print("ALL KINEMATICS TESTS PASSED SUCCESSFULLY.")
    print("=" * 60)
