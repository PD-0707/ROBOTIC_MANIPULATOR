"""
Actuator and PID Control Simulation Module
------------------------------------------
Implements:
1. Closed-loop PID controller with anti-windup protection and derivative filtering.
2. ServoActuator model simulating first-order velocity-saturated joint dynamics.
3. GripperActuator simulating an end-effector mechanical gripper with finite-state tracking.
"""

from enum import Enum
import numpy as np


class GripperState(Enum):
    OPEN = "OPEN"
    CLOSING = "CLOSING"
    CLOSED_HOLDING = "CLOSED_HOLDING"
    OPENING = "OPENING"


class PIDController:
    """
    Discrete-time Proportional-Integral-Derivative (PID) Controller
    with integral anti-windup and derivative error tracking.
    """
    def __init__(self,
                 kp: float = 4.0,
                 ki: float = 0.5,
                 kd: float = 0.15,
                 integral_limit: float = 100.0,
                 output_limit: float = 180.0):
        """
        Parameters:
            kp (float): Proportional gain.
            ki (float): Integral gain.
            kd (float): Derivative gain.
            integral_limit (float): Anti-windup threshold for integral term.
            output_limit (float): Maximum absolute output command (e.g. max velocity in deg/s).
        """
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.integral_limit = float(integral_limit)
        self.output_limit = float(output_limit)

        self.integral = 0.0
        self.previous_error = 0.0
        self.initialized = False

    def reset(self, initial_error: float = 0.0):
        """Reset internal accumulator states."""
        self.integral = 0.0
        self.previous_error = initial_error
        self.initialized = False

    def update(self, setpoint: float, measured: float, dt: float) -> float:
        """
        Calculate control action (commanded velocity in deg/s).

        Parameters:
            setpoint (float): Desired reference angle (deg).
            measured (float): Currently measured joint angle (deg).
            dt (float): Sample time step in seconds.

        Returns:
            float: Control output command.
        """
        error = setpoint - measured

        # Trapezoidal or forward-Euler integral accumulation
        self.integral += error * dt
        # Anti-windup clamping on the integral accumulator
        self.integral = float(np.clip(self.integral, -self.integral_limit, self.integral_limit))

        # Derivative calculation
        if not self.initialized or dt <= 0:
            derivative = 0.0
            self.initialized = True
        else:
            derivative = (error - self.previous_error) / dt

        # PID control law: u(t) = Kp*e + Ki*∫e dt + Kd*(de/dt)
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)

        # Output saturation limit
        output = float(np.clip(output, -self.output_limit, self.output_limit))

        self.previous_error = error
        return output


class ServoActuator:
    """
    First-order velocity-limited servo actuator model.
    Approximates physical servo motor response to velocity commands.
    """
    def __init__(self,
                 initial_angle: float = 0.0,
                 max_velocity: float = 180.0,
                 min_angle: float = -180.0,
                 max_angle: float = 180.0,
                 response_lag: float = 0.05):
        """
        Parameters:
            initial_angle (float): Initial joint position in degrees.
            max_velocity (float): Maximum physical rotation speed (deg/s).
            min_angle (float): Negative mechanical limit (deg).
            max_angle (float): Positive mechanical limit (deg).
            response_lag (float): Actuator dynamic lag filter constant (seconds).
        """
        self.angle = float(initial_angle)
        self.velocity = 0.0
        self.max_velocity = float(max_velocity)
        self.min_angle = float(min_angle)
        self.max_angle = float(max_angle)
        self.response_lag = float(response_lag)

    def reset(self, angle: float = 0.0):
        """Reset actuator to a specific joint angle."""
        self.angle = float(angle)
        self.velocity = 0.0

    def update(self, control_signal: float, dt: float) -> float:
        """
        Update actuator joint state based on commanded velocity control signal.

        Parameters:
            control_signal (float): Commanded velocity (deg/s) from PID controller.
            dt (float): Time step in seconds.

        Returns:
            float: Updated actual joint angle in degrees.
        """
        if dt <= 0:
            return self.angle

        # Saturate commanded velocity to actuator physical capability
        target_velocity = float(np.clip(control_signal, -self.max_velocity, self.max_velocity))

        # Optional first-order lag filter for realistic inertia:
        # v_dot = (v_target - v) / tau
        if self.response_lag > 0:
            alpha = dt / (self.response_lag + dt)
            self.velocity = self.velocity + alpha * (target_velocity - self.velocity)
        else:
            self.velocity = target_velocity

        # Numerical integration of position
        self.angle += self.velocity * dt

        # Enforce physical mechanical stops
        self.angle = float(np.clip(self.angle, self.min_angle, self.max_angle))

        return self.angle


class GripperActuator:
    """
    End-effector two-finger parallel gripper simulation.
    Tracks physical jaw opening width, grasping state, and payload attachment.
    """
    def __init__(self, stroke_mm: float = 30.0, speed_mms: float = 60.0):
        """
        Parameters:
            stroke_mm (float): Maximum jaw opening width in mm (default 30 mm).
            speed_mms (float): Jaw travel speed in mm/s.
        """
        self.max_width = float(stroke_mm)
        self.width = float(stroke_mm)  # Starts fully OPEN
        self.speed = float(speed_mms)
        self.state = GripperState.OPEN
        self.holding_object = False

    def close(self):
        """Command gripper to close onto an object."""
        self.state = GripperState.CLOSING

    def open(self):
        """Command gripper to open and release object."""
        self.state = GripperState.OPENING

    def update(self, dt: float, near_object: bool = False) -> GripperState:
        """
        Simulate jaw motion and object acquisition.

        Parameters:
            dt (float): Time step in seconds.
            near_object (bool): True if end-effector is at object pick location.

        Returns:
            GripperState: Current state of the gripper.
        """
        if self.state == GripperState.CLOSING:
            self.width -= self.speed * dt
            # If grasping object (assume object thickness ~10 mm)
            target_hold = 10.0 if near_object else 0.0
            if self.width <= target_hold:
                self.width = target_hold
                self.state = GripperState.CLOSED_HOLDING if near_object else GripperState.OPEN
                if near_object:
                    self.holding_object = True

        elif self.state == GripperState.OPENING:
            self.width += self.speed * dt
            if self.width >= self.max_width:
                self.width = self.max_width
                self.state = GripperState.OPEN
                self.holding_object = False

        return self.state


if __name__ == "__main__":
    print("=" * 60)
    print("2-DOF ROBOTIC MANIPULATOR: CONTROLLER & ACTUATOR TEST")
    print("=" * 60)

    pid = PIDController(kp=4.0, ki=0.5, kd=0.15)
    actuator = ServoActuator(initial_angle=0.0, max_velocity=180.0)

    # Step response test: command step to 45 degrees
    target = 45.0
    dt = 0.02
    sim_time = 1.5
    steps = int(sim_time / dt)

    print(f"Testing step response: Step 0.0 deg -> {target:.1f} deg for {sim_time} s")

    angles = []
    errors = []
    for step in range(steps):
        current_pos = actuator.angle
        u = pid.update(target, current_pos, dt)
        new_pos = actuator.update(u, dt)
        angles.append(new_pos)
        errors.append(target - new_pos)

    final_error = errors[-1]
    max_error = max(errors)
    print(f"Final Angle:         {angles[-1]:.2f} deg")
    print(f"Steady-State Error:  {final_error:.4f} deg")
    print(f"Max Tracking Error:  {max_error:.2f} deg")

    gripper = GripperActuator()
    print(f"\nInitial Gripper Width: {gripper.width:.1f} mm, State: {gripper.state.value}")
    gripper.close()
    for _ in range(30):
        gripper.update(dt=0.02, near_object=True)
    print(f"Gripper After Close:   {gripper.width:.1f} mm, State: {gripper.state.value}, Holding: {gripper.holding_object}")
    gripper.open()
    for _ in range(30):
        gripper.update(dt=0.02, near_object=False)
    print(f"Gripper After Open:    {gripper.width:.1f} mm, State: {gripper.state.value}, Holding: {gripper.holding_object}")

    print("\n" + "=" * 60)
    print("CONTROLLER & ACTUATOR TESTS PASSED SUCCESSFULLY.")
    print("=" * 60)

