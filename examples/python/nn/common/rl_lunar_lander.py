"""A simplified LunarLander (examples/nn/common/rl/lunar_lander.hpp).

A rigid body with a main engine and two side thrusters must touch down gently on a pad; no
Box2D, a point mass with orientation integrated with semi-implicit Euler. Observation (8):
[x, y, vx, vy, angle, angular_velocity, left_leg_contact, right_leg_contact]. Actions: 0 noop,
1 left thruster, 2 main engine, 3 right thruster. float32 arithmetic as in the C++.
"""
from pydatamunge.nn_workflows import helper

rl_env = helper("rl_env")
f32, fma, sincosf, sqrtf, cpp_round = rl_env.f32, rl_env.fma, rl_env.sincosf, rl_env.sqrtf, rl_env.cpp_round
ActionSpace, EnvBase, StepResult = rl_env.ActionSpace, rl_env.EnvBase, rl_env.StepResult


class LanderState:
    __slots__ = ("x", "y", "vx", "vy", "angle", "angular_velocity", "landed", "crashed")

    def __init__(self, x=0.0, y=0.0, vx=0.0, vy=0.0, angle=0.0, angular_velocity=0.0):
        self.x, self.y, self.vx, self.vy = x, y, vx, vy
        self.angle, self.angular_velocity = angle, angular_velocity
        self.landed = self.crashed = False


class LunarLander(EnvBase):
    DT, GRAVITY, MAIN_THRUST, SIDE_THRUST = f32(0.02), f32(1.6), 4.0, f32(0.4)
    SIDE_TORQUE, PAD_HALF_WIDTH = 3.0, 0.25

    def __init__(self, max_steps=1000):
        super().__init__(max_steps)
        self.s = LanderState()
        self._last_action = 0
        self.prev_shaping = 0.0

    def name(self):
        return "LunarLander"

    def obs_size(self):
        return 8

    def action_space(self):
        return ActionSpace.discrete(4)

    def action_names(self):
        return ["no-op", "left thruster", "main engine", "right thruster"]

    def state(self):
        return self.s

    def last_action(self):
        return self._last_action

    def _reset(self):
        self.begin_episode()
        u = self.uniform
        x = u(-0.3, 0.3)
        vx = u(-0.3, 0.3)
        vy = u(-0.3, 0.0)
        angle = u(-0.2, 0.2)
        self.s = LanderState(x, f32(1.4), vx, vy, angle, 0.0)
        self._last_action = 0
        self.prev_shaping = self.shaping()
        return self.obs_now()

    def step(self, action):
        c, s = LunarLander, self.s
        a = action.discrete
        self._last_action = a
        fx, fy, torque, fuel = 0.0, -c.GRAVITY, 0.0, 0.0
        if a == 2:  # the main engine pushes along the body's up axis
            cos_a, sin_a = sincosf(s.angle)
            fx = fma(-sin_a, c.MAIN_THRUST, fx)
            fy = fma(cos_a, c.MAIN_THRUST, fy)
            fuel = f32(0.3)
        elif a in (1, 3):
            direction = 1.0 if a == 1 else -1.0  # left thruster pushes right / rotates ccw
            cos_a, sin_a = sincosf(s.angle)
            fx = fma(direction * cos_a, c.SIDE_THRUST, fx)
            fy = fma(direction * sin_a, c.SIDE_THRUST, fy)
            torque = fma(direction, c.SIDE_TORQUE, torque)
            fuel = f32(0.03)
        s.vx = fma(fx, c.DT, s.vx)
        s.vy = fma(fy, c.DT, s.vy)
        s.angular_velocity = fma(torque, c.DT, s.angular_velocity)
        s.x = fma(s.vx, c.DT, s.x)
        s.y = fma(s.vy, c.DT, s.y)
        s.angle = fma(s.angular_velocity, c.DT, s.angle)
        if s.y <= 0.0:  # touched the ground
            s.y = 0.0
            gentle = abs(s.vy) < 0.5 and abs(s.vx) < 0.5 and abs(s.angle) < f32(0.3)
            on_pad = abs(s.x) <= c.PAD_HALF_WIDTH
            if gentle and on_pad:
                s.landed = True
            else:
                s.crashed = True
            s.vx = s.vy = s.angular_velocity = 0.0
        if abs(s.x) > 1.0 or s.y > 2.0:
            s.crashed = True
        shaping = self.shaping()
        reward = f32(f32(shaping - self.prev_shaping) - fuel)
        self.prev_shaping = shaping
        if s.landed:
            reward = f32(reward + 100.0)
        if s.crashed:
            reward = f32(reward - 100.0)
        terminated = s.landed or s.crashed
        truncated = not terminated and self.count_step()
        return StepResult(self.obs_now(), reward, terminated, truncated)

    def render(self):
        """21 columns x 11 rows; the pad is drawn on the bottom row."""
        grid = [[" "] * 21 for _ in range(11)]
        for col in range(21):
            grid[10][col] = "=" if abs(col - 10) <= 2 else "_"
        col = cpp_round(fma(self.s.x, 10.0, 10.0))
        row = cpp_round(fma(-f32(self.s.y / f32(1.5)), 9.0, 9.0))
        if 0 <= col < 21 and 0 <= row < 11:
            grid[row][col] = "*" if self.s.crashed else "V"
        return "".join("".join(r) + "\n" for r in grid)

    def obs_now(self):
        s = self.s
        contact = 1.0 if s.y <= 0.0 else 0.0
        return [s.x, s.y, s.vx, s.vy, s.angle, s.angular_velocity, contact, contact]

    def shaping(self):
        """Potential-based shaping: higher when close, slow and upright."""
        s = self.s
        distance = sqrtf(fma(s.x, s.x, f32(s.y * s.y)))
        speed = sqrtf(fma(s.vx, s.vx, f32(s.vy * s.vy)))
        value = fma(-100.0, distance, -f32(100.0 * speed))
        return fma(-100.0, abs(s.angle), value)
