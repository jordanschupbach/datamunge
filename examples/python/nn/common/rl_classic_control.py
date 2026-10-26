"""Classic control environments (examples/nn/common/rl/classic_control.hpp).

Same dynamics and constants as Gymnasium's CartPole-v1, MountainCar-v0,
MountainCarContinuous-v0, Pendulum-v1 and Acrobot-v1. Arithmetic is float32 as in the C++:
f32() rounds one operation, fma() is clang's contraction of `c + a * b` in one expression.
"""
import math

from pydatamunge.nn_workflows import helper

rl_env = helper("rl_env")
f32, fma, cosf, sinf, cpp_round = rl_env.f32, rl_env.fma, rl_env.cosf, rl_env.sinf, rl_env.cpp_round
sincosf = rl_env.sincosf
ActionSpace, EnvBase, StepResult = rl_env.ActionSpace, rl_env.EnvBase, rl_env.StepResult

PI = f32(math.pi)
TWO_PI = f32(2 * PI)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def wrap_angle(a):
    """To [-pi, pi) with fmodf."""
    a = math.fmod(f32(a + PI), TWO_PI)
    if a < 0:
        a = f32(a + TWO_PI)
    return f32(a - PI)


class CartPole(EnvBase):
    """Balance a pole by pushing left (0) or right (1); +1 per step; 500-step limit."""

    GRAVITY, MASSCART, MASSPOLE = f32(9.8), 1.0, f32(0.1)
    TOTAL_MASS = f32(1.0 + f32(0.1))
    LENGTH = 0.5
    POLEMASS_LENGTH = f32(f32(0.1) * 0.5)
    FORCE_MAG, TAU = 10.0, f32(0.02)
    FOUR_THIRDS = f32(4.0 / 3.0)
    THETA_LIMIT = f32(f32(24 * PI) / 360)
    X_LIMIT = f32(2.4)

    def __init__(self, max_steps=500):
        super().__init__(max_steps)
        self.s = [0.0, 0.0, 0.0, 0.0]

    def name(self):
        return "CartPole"

    def obs_size(self):
        return 4

    def action_space(self):
        return ActionSpace.discrete(2)

    def action_names(self):
        return ["push left", "push right"]

    def state(self):
        return list(self.s)

    def _reset(self):
        self.begin_episode()
        self.s = [self.uniform(-0.05, 0.05) for _ in range(4)]
        return list(self.s)

    def step(self, action):
        c = CartPole
        x, x_dot, theta, theta_dot = self.s
        force = c.FORCE_MAG if action.discrete == 1 else -c.FORCE_MAG
        costheta, sintheta = sincosf(theta)
        temp = f32(fma(f32(f32(c.POLEMASS_LENGTH * theta_dot) * theta_dot), sintheta, force) / c.TOTAL_MASS)
        numerator = fma(c.GRAVITY, sintheta, -f32(costheta * temp))
        denominator = f32(c.LENGTH * f32(c.FOUR_THIRDS - f32(f32(f32(c.MASSPOLE * costheta) * costheta) / c.TOTAL_MASS)))
        thetaacc = f32(numerator / denominator)
        xacc = f32(temp - f32(f32(f32(c.POLEMASS_LENGTH * thetaacc) * costheta) / c.TOTAL_MASS))
        x = fma(c.TAU, x_dot, x)
        x_dot = fma(c.TAU, xacc, x_dot)
        theta = fma(c.TAU, theta_dot, theta)
        theta_dot = fma(c.TAU, thetaacc, theta_dot)
        self.s = [x, x_dot, theta, theta_dot]
        terminated = abs(x) > c.X_LIMIT or abs(theta) > c.THETA_LIMIT
        truncated = not terminated and self.count_step()
        return StepResult(list(self.s), 1.0, terminated, truncated)

    def render(self):
        """A 41-column strip; the cart is at the centre when x = 0."""
        line = ["-"] * 41
        cart = cpp_round(fma(f32(self.s[0] / CartPole.X_LIMIT), 20.0, 20.0))
        cart = max(0, min(40, cart))
        line[cart] = "="
        tip = cpp_round(fma(sinf(self.s[2]), 20.0, float(cart)))
        pole = [" "] * 41
        pole[max(0, min(40, tip))] = "|"
        return "".join(pole) + "\n" + "".join(line) + "\n"


class MountainCar(EnvBase):
    """Rock an underpowered car to the flag at x = 0.5; push left/none/right; -1 per step."""

    def __init__(self, max_steps=200):
        super().__init__(max_steps)
        self.pos = 0.0
        self.vel = 0.0

    def name(self):
        return "MountainCar"

    def obs_size(self):
        return 2

    def action_space(self):
        return ActionSpace.discrete(3)

    def action_names(self):
        return ["push left", "coast", "push right"]

    def position(self):
        return self.pos

    def velocity(self):
        return self.vel

    def _reset(self):
        self.begin_episode()
        self.pos = self.uniform(-0.6, -0.4)
        self.vel = 0.0
        return [self.pos, self.vel]

    def integrate(self, force):
        self.vel = clamp(fma(cosf(f32(3 * self.pos)), f32(-0.0025), f32(self.vel + force)), f32(-0.07), f32(0.07))
        self.pos = clamp(f32(self.pos + self.vel), f32(-1.2), f32(0.6))
        if self.pos <= f32(-1.2) and self.vel < 0:
            self.vel = 0.0

    def step(self, action):
        self.integrate(f32((action.discrete - 1) * f32(0.001)))
        terminated = self.pos >= 0.5
        truncated = not terminated and self.count_step()
        return StepResult([self.pos, self.vel], -1.0, terminated, truncated)


class MountainCarContinuous(MountainCar):
    """Throttle in [-1, 1] scaled by 0.0015; reward -0.1 a^2 per step, +100 at x = 0.45."""

    def __init__(self, max_steps=999):
        super().__init__(max_steps)

    def name(self):
        return "MountainCarContinuous"

    def action_space(self):
        return ActionSpace.continuous([-1.0], [1.0])

    def action_names(self):
        return ["throttle"]

    def step(self, action):
        a = clamp(action.continuous[0], -1.0, 1.0)
        self.integrate(f32(a * f32(0.0015)))
        terminated = self.pos >= f32(0.45)
        reward = fma(f32(f32(-0.1) * a), a, 100.0 if terminated else 0.0)
        truncated = not terminated and self.count_step()
        return StepResult([self.pos, self.vel], reward, terminated, truncated)


class Pendulum(EnvBase):
    """Swing a pendulum upright with a torque in [-2, 2]; reward -(theta^2 + 0.1 thdot^2 + 0.001 u^2)."""

    def __init__(self, max_steps=200):
        super().__init__(max_steps)
        self.th = 0.0
        self.thdot = 0.0

    def name(self):
        return "Pendulum"

    def obs_size(self):
        return 3

    def action_space(self):
        return ActionSpace.continuous([-2.0], [2.0])

    def action_names(self):
        return ["torque"]

    def theta(self):
        return self.th

    def theta_dot(self):
        return self.thdot

    def _reset(self):
        self.begin_episode()
        self.th = self.uniform(-PI, PI)
        self.thdot = self.uniform(-1.0, 1.0)
        return self.obs_now()

    def step(self, action):
        dt, max_speed, max_torque = f32(0.05), 8.0, 2.0
        u = clamp(action.continuous[0], -max_torque, max_torque)
        wrapped = wrap_angle(self.th)
        cost = fma(wrapped, wrapped, f32(f32(f32(0.1) * self.thdot) * self.thdot))
        cost = fma(f32(f32(0.001) * u), u, cost)
        # 3 g / (2 l) = 15 and 3 / (m l^2) = 3 for g = 10, m = l = 1.
        accel = fma(15.0, sinf(self.th), f32(3.0 * u))
        self.thdot = clamp(fma(accel, dt, self.thdot), -max_speed, max_speed)
        self.th = fma(self.thdot, dt, self.th)
        truncated = self.count_step()
        return StepResult(self.obs_now(), -cost, False, truncated)

    def render(self):
        """9x9 picture of the rod; theta = 0 is upright."""
        grid = [[" "] * 9 for _ in range(9)]
        grid[4][4] = "o"
        for k in range(1, 5):
            r = cpp_round(fma(-float(k), cosf(self.th), 4.0))
            c = cpp_round(fma(float(k), sinf(self.th), 4.0))
            if 0 <= r < 9 and 0 <= c < 9:
                grid[r][c] = "#"
        return "".join("".join(row) + "\n" for row in grid)

    def obs_now(self):
        cos_th, sin_th = sincosf(self.th)
        return [cos_th, sin_th, self.thdot]


class Acrobot(EnvBase):
    """Two-link pendulum actuated at the elbow; swing the tip above height 1; -1 per step."""

    HALF_PI = f32(PI / 2)
    PHI2_SCALE = f32(0.5 * f32(9.8))           # m2 lc2 g
    PHI1_SCALE = f32(1.5 * f32(9.8))           # (m1 lc1 + m2 l1) g
    DT = f32(0.2)
    DT6 = f32(f32(0.2) / 6)
    DT1_LIMIT = f32(4 * PI)
    DT2_LIMIT = f32(9 * PI)

    def __init__(self, max_steps=500):
        super().__init__(max_steps)
        self.s = [0.0, 0.0, 0.0, 0.0]  # theta1, theta2, dtheta1, dtheta2

    def name(self):
        return "Acrobot"

    def obs_size(self):
        return 6

    def action_space(self):
        return ActionSpace.discrete(3)

    def action_names(self):
        return ["torque -1", "torque 0", "torque +1"]

    def tip_height(self):
        return f32(-cosf(self.s[0]) - cosf(f32(self.s[1] + self.s[0])))

    def theta1(self):
        return self.s[0]

    def theta2(self):
        return self.s[1]

    def _reset(self):
        self.begin_episode()
        self.s = [self.uniform(-0.1, 0.1) for _ in range(4)]
        return self.obs_now()

    def step(self, action):
        self.rk4(float(action.discrete - 1), Acrobot.DT)
        s = self.s
        s[0] = wrap_angle(s[0])
        s[1] = wrap_angle(s[1])
        s[2] = clamp(s[2], -Acrobot.DT1_LIMIT, Acrobot.DT1_LIMIT)
        s[3] = clamp(s[3], -Acrobot.DT2_LIMIT, Acrobot.DT2_LIMIT)
        terminated = self.tip_height() > 1.0
        truncated = not terminated and self.count_step()
        return StepResult(self.obs_now(), 0.0 if terminated else -1.0, terminated, truncated)

    def obs_now(self):
        s = self.s
        cos0, sin0 = sincosf(s[0])
        cos1, sin1 = sincosf(s[1])
        return [cos0, sin0, cos1, sin1, s[2], s[3]]

    @staticmethod
    def derivs(s, torque):
        """'Book' dynamics (Sutton & Barto) with m = l = I = 1, lc = 0.5, g = 9.8."""
        t1, t2, dt1, dt2 = s
        cos_t2, sin_t2 = sincosf(t2)
        d1 = f32(f32(f32(0.25 + f32(1.25 + cos_t2)) + 1.0) + 1.0)
        d2 = f32(f32(0.25 + 0.5 * cos_t2) + 1.0)
        phi2 = f32(Acrobot.PHI2_SCALE * cosf(f32(f32(t1 + t2) - Acrobot.HALF_PI)))
        a = f32(f32(-0.5 * dt2) * dt2)
        b = f32(f32(dt2 * dt1) * sin_t2)
        phi1 = fma(a, sin_t2, -b)
        phi1 = fma(Acrobot.PHI1_SCALE, cosf(f32(t1 - Acrobot.HALF_PI)), phi1)
        phi1 = f32(phi1 + phi2)
        numerator = fma(f32(d2 / d1), phi1, torque)
        numerator = fma(-f32(f32(0.5 * dt1) * dt1), sin_t2, numerator)
        numerator = f32(numerator - phi2)
        ddt2 = f32(numerator / f32(1.25 - f32(f32(d2 * d2) / d1)))
        ddt1 = f32(-fma(d2, ddt2, phi1) / d1)
        return [dt1, dt2, ddt1, ddt2]

    def rk4(self, torque, dt):
        def axpy(a, h, k):
            return [fma(h, k[i], a[i]) for i in range(4)]

        half = f32(dt / 2)
        k1 = Acrobot.derivs(self.s, torque)
        k2 = Acrobot.derivs(axpy(self.s, half, k1), torque)
        k3 = Acrobot.derivs(axpy(self.s, half, k2), torque)
        k4 = Acrobot.derivs(axpy(self.s, dt, k3), torque)
        dt6 = f32(dt / 6)
        for i in range(4):
            inner = f32(f32(f32(k1[i] + 2 * k2[i]) + 2 * k3[i]) + k4[i])
            self.s[i] = fma(dt6, inner, self.s[i])
