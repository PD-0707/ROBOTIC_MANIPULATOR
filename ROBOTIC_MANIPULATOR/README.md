# 2-DOF Robotic Pick-and-Place Manipulator Simulation & Control

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete%20%26%20Validated-brightgreen.svg)]()

A comprehensive software simulation of a planar two-degree-of-freedom (2-DOF) robotic manipulator performing closed-loop pick-and-place operations. This project models the complete robotics stack: analytical forward/inverse kinematics, workspace boundary verification, cubic Hermite joint trajectory generation, discrete PID control, first-order servo actuator dynamics, and a parallel gripper finite-state machine.

---

## 1. Executive Summary & Engineering Architecture

In modern industrial automation (such as Siemens SIMATIC S7-1500 / Sinamics motion control workflows), physical robots are programmed and validated through digital simulations before deploying code to physical hardware. This project replicates an industrial planar manipulator in pure Python without requiring heavyweight CAD or proprietary simulation licenses.

```
                          SYSTEM CONTROL ARCHITECTURE
                          
 +-------------------------------------------------------------------------+
 |                            CARTESIAN GOALS                             |
 |           Home (120, 40) mm  --->  Pick (100, 80) mm  ---> Place (60, 120) mm          |
 +-------------------------------------------------------------------------+
                                      |
                                      v
 +-------------------------------------------------------------------------+
 |                     INVERSE KINEMATICS ENGINE (IK)                      |
 |               r^2 = x^2 + y^2  |  Workspace Radius Check                 |
 |        theta2 = arccos((r^2 - L1^2 - L2^2)/(2*L1*L2))                   |
 |        theta1 = atan2(y, x) - atan2(L2*sin(t2), L1 + L2*cos(t2))        |
 +-------------------------------------------------------------------------+
                                      | Joint Target Angles
                                      v
 +-------------------------------------------------------------------------+
 |                      CUBIC TRAJECTORY GENERATOR                         |
 |           s(t) = t/T  |  theta_cmd(t) = q0 + (qf - q0)*(3s^2 - 2s^3)   |
 |           Zero velocity boundary conditions: v(0) = 0, v(T) = 0         |
 +-------------------------------------------------------------------------+
                                      | Desired q_cmd(t)
                                      v
 +-------------------------------------------------------------------------+
 |                     DISCRETE-TIME PID CONTROLLER                        |
 |              e(t) = q_cmd(t) - q_act(t)                                 |
 |              u(t) = Kp*e(t) + Ki*integral(e dt) + Kd*(de/dt)            |
 |              Anti-windup clamping + output velocity saturation          |
 +-------------------------------------------------------------------------+
                                      | Commanded Joint Velocity u(t)
                                      v
 +-------------------------------------------------------------------------+
 |               ACTUATOR DYNAMICS & FORWARD KINEMATICS (FK)               |
 |         dq_act/dt = (u - v_act)/tau  -->  q_act(t)                      |
 |         x_ee = L1*cos(t1) + L2*cos(t1 + t2)                             |
 |         y_ee = L1*sin(t1) + L2*sin(t1 + t2)                             |
 +-------------------------------------------------------------------------+
                                      |
                                      v
 +-------------------------------------------------------------------------+
 |             GRIPPER STATE MACHINE & TELEMETRY LOGGING                   |
 |         OPEN <---> CLOSING <---> CLOSED_HOLDING <---> OPENING           |
 |         Saved Artifacts: PNG Plots, Telemetry CSV, Animated GIF         |
 +-------------------------------------------------------------------------+
```

---

## 2. Technical Specifications

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Manipulator Architecture** | Planar 2-DOF Serial Chain | Two rotary joints (Shoulder $J_1$, Elbow $J_2$) |
| **Link 1 Length ($L_1$)** | 80.0 mm | Base-to-elbow mechanical link |
| **Link 2 Length ($L_2$)** | 80.0 mm | Elbow-to-end-effector link |
| **Max Reach Radius ($R_{max}$)** | 160.0 mm | $L_1 + L_2$ |
| **Min Reach Radius ($R_{min}$)** | 0.0 mm | $|L_1 - L_2|$ (full folding capability) |
| **Actuator Model** | Velocity-saturated Servo | First-order response with inertial lag ($\tau = 40$ ms) |
| **Max Actuator Velocity** | $\pm 180.0^\circ/\text{s}$ | Realistic servo motor physical limit |
| **Control Frequency** | 50 Hz ($\Delta t = 0.02$ s) | Standard industrial motion cycle |
| **PID Gains** | $K_p = 5.0, K_i = 0.8, K_d = 0.20$ | Tuned for stable tracking and zero overshoot |
| **Gripper Type** | Parallel Jaw Mechanism | 30.0 mm max stroke, $75\text{ mm/s}$ travel speed |

---

## 3. Mathematical Foundations

### 3.1 Forward Kinematics (FK)
Forward kinematics maps joint angles $(\theta_1, \theta_2)$ to the Cartesian end-effector coordinates $(x, y)$.

Given base origin $(x_0, y_0) = (0, 0)$:
- **Elbow Position:**
  $$x_1 = L_1 \cos(\theta_1)$$
  $$y_1 = L_1 \sin(\theta_1)$$
- **End-Effector Position:**
  $$x_2 = x_1 + L_2 \cos(\theta_1 + \theta_2) = L_1 \cos(\theta_1) + L_2 \cos(\theta_1 + \theta_2)$$
  $$y_2 = y_1 + L_2 \sin(\theta_1 + \theta_2) = L_1 \sin(\theta_1) + L_2 \sin(\theta_1 + \theta_2)$$

### 3.2 Inverse Kinematics (IK)
Inverse kinematics determines the necessary joint angles $(\theta_1, \theta_2)$ to reach a designated target $(x, y)$.

1. **Radial Distance:**
   $$r^2 = x^2 + y^2$$
2. **Law of Cosines for Elbow Joint ($\theta_2$):**
   $$r^2 = L_1^2 + L_2^2 - 2 L_1 L_2 \cos(180^\circ - \theta_2) = L_1^2 + L_2^2 + 2 L_1 L_2 \cos(\theta_2)$$
   $$\cos(\theta_2) = \frac{r^2 - L_1^2 - L_2^2}{2 L_1 L_2}$$
   - **Reachability Condition:** $-1 \le \cos(\theta_2) \le 1$. If violated, the target is out of reachable workspace and rejected immediately.
   $$\theta_2 = \text{sign} \cdot \arccos(\cos(\theta_2))$$
   *(where $\text{sign} = +1$ denotes elbow-down and $-1$ denotes elbow-up)*.
3. **Shoulder Joint ($\theta_1$):**
   $$\theta_1 = \text{atan2}(y, x) - \text{atan2}(L_2 \sin(\theta_2), L_1 + L_2 \cos(\theta_2))$$

### 3.3 Trajectory Planning (Cubic Hermite Polynomial)
To prevent jerky step changes and infinite acceleration spikes, trajectories are interpolated using cubic polynomials with zero initial and terminal velocities:
$$s(t) = \frac{t}{T}, \quad t \in [0, T]$$
$$\theta(t) = \theta_0 + (\theta_f - \theta_0) (3s^2 - 2s^3)$$
$$\dot{\theta}(t) = (\theta_f - \theta_0) \frac{6s - 6s^2}{T}$$
$$\ddot{\theta}(t) = (\theta_f - \theta_0) \frac{6 - 12s}{T^2}$$

Boundary conditions satisfied:
- $\theta(0) = \theta_0, \quad \theta(T) = \theta_f$
- $\dot{\theta}(0) = 0, \quad \dot{\theta}(T) = 0$

### 3.4 Closed-Loop PID Control Law
At discrete step $k$ with sample time $\Delta t$:
$$e_k = \theta_{\text{ref}, k} - \theta_{\text{act}, k}$$
$$I_k = \text{clip}\left(I_{k-1} + e_k \Delta t, -I_{\max}, I_{\max}\right) \quad (\text{Anti-Windup})$$
$$D_k = \frac{e_k - e_{k-1}}{\Delta t}$$
$$u_k = \text{clip}\left(K_p e_k + K_i I_k + K_d D_k, -v_{\max}, v_{\max}\right)$$

---

## 4. Project Structure

```
Siemens_Robotic_Manipulator/
│
├── main.py                   # Master simulation loop, mission profile & report generator
├── kinematics.py             # Forward, inverse kinematics & workspace validation
├── trajectory.py             # Cubic Hermite trajectory generator & path functions
├── controller.py             # Discrete PID controller, ServoActuator & Gripper models
├── robot_visualization.py    # Matplotlib plots, workspace diagrams & GIF animator
├── test_manipulator.py       # Automated unit test suite (9 test cases)
├── README.md                 # Complete documentation & interview technical guide
│
└── results/                  # Generated engineering output artifacts
    ├── workspace.png                # Reachable workspace and task waypoints
    ├── joint1_tracking.png          # Joint 1 commanded vs actual & error
    ├── joint2_tracking.png          # Joint 2 commanded vs actual & error
    ├── tracking_error.png           # Combined joint tracking error timeline
    ├── end_effector_trajectory.png  # Cartesian X-Y path of end-effector
    ├── pick_and_place_animation.gif # Frame-by-frame animated visualizer
    └── joint_data.csv               # 50 Hz telemetry log (angles, error, positions)
```

---

## 5. Step-by-Step Execution Guide

### Prerequisites
Ensure Python 3.10+ is installed. Install the required dependencies:
```bash
python -m pip install numpy matplotlib
```

### Running the Complete Simulation
```bash
python main.py
```
This executes the 5-phase pick-and-place sequence, calculates dynamic tracking performance, logs 50 Hz telemetry to `results/joint_data.csv`, and generates all engineering plots and the GIF animation in `results/`.

### Running the Test Suite
```bash
python test_manipulator.py
```
Executes all 9 automated unit tests verifying kinematic benchmarks, closed-loop reconstruction, error rejection, trajectory boundary conditions, and actuator control.

---

## 6. Simulation Results & Performance Metrics

### Kinematics Benchmark Verification
| Test Case | Shoulder ($\theta_1$) | Elbow ($\theta_2$) | Expected Position $(X, Y)$ | Calculated Position $(X, Y)$ | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Fully Extended** | $0.0^\circ$ | $0.0^\circ$ | $(160.00, 0.00)$ mm | $(160.00, 0.00)$ mm | **PASS** |
| **Folded** | $0.0^\circ$ | $180.0^\circ$ | $(0.00, 0.00)$ mm | $(0.00, 0.00)$ mm | **PASS** |
| **Bent Vertical** | $90.0^\circ$ | $0.0^\circ$ | $(0.00, 160.00)$ mm | $(0.00, 160.00)$ mm | **PASS** |
| **Right Angle** | $0.0^\circ$ | $90.0^\circ$ | $(80.00, 80.00)$ mm | $(80.00, 80.00)$ mm | **PASS** |

### Pick-and-Place Mission Waypoints
- **Home:** $(120.00, 40.00)\text{ mm} \implies \theta_1 = -19.33^\circ, \theta_2 = 75.52^\circ$
- **Pick:** $(100.00, 80.00)\text{ mm} \implies \theta_1 = 1.83^\circ, \theta_2 = 73.67^\circ$
- **Place:** $(60.00, 120.00)\text{ mm} \implies \theta_1 = 30.42^\circ, \theta_2 = 66.03^\circ$

### Dynamic Closed-Loop Tracking Performance
- **Joint 1 (Shoulder) Max Error:** $6.833^\circ$
- **Joint 1 (Shoulder) RMS Error:** $2.807^\circ$
- **Joint 2 (Elbow) Max Error:** $1.322^\circ$
- **Joint 2 (Elbow) RMS Error:** $0.562^\circ$
- **Nominal Place Target:** $(60.00, 120.00)$ mm
- **Achieved Place Position:** $(59.23, 120.42)$ mm
- **Final End-Effector Placement Error:** **$0.8780$ mm (Sub-millimeter accuracy)**

---

## 7. Generated Visual Deliverables

| Deliverable | File Path | Description |
| :--- | :--- | :--- |
| **Reachable Workspace** | `results/workspace.png` | Scatter cloud of reachable Cartesian space with $R=160\text{ mm}$ boundary and task targets |
| **Joint 1 Tracking** | `results/joint1_tracking.png` | Commanded angle vs actual servo angle with instantaneous error subplot |
| **Joint 2 Tracking** | `results/joint2_tracking.png` | Commanded angle vs actual servo angle with instantaneous error subplot |
| **Tracking Error Profile** | `results/tracking_error.png` | Combined error timeline demonstrating convergence to zero during dwell |
| **Cartesian Path** | `results/end_effector_trajectory.png` | End-effector trajectory connecting Home $\to$ Pick $\to$ Place $\to$ Home |
| **Animated Simulation** | `results/pick_and_place_animation.gif` | 25 FPS animation showing link movement, end-effector trail, gripper state, and payload transport |
| **Telemetry Log** | `results/joint_data.csv` | 406 rows of time-series data at 50 Hz |

---
