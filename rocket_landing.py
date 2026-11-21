"""
Residual reinforcement learning on top of model predictive control, for fuel-efficient
simulated rocket landings.

Three controllers are built and compared on Gymnasium's LunarLander-v3 (Box2D):

  MPC       a planner that looks ahead in a model of the lander identified from flight
            data, and picks the cheapest sequence of engine firings.
  PPO       a neural network trained by reinforcement learning, flying on its own.
  Residual  the planner proposes, and a trained network nudges its proposal. This is the
            method under test.

All three are measured on the same held-out episodes at five wind strengths, and the
result is one table of landing rates and one of fuel use.

Sections, in order:

  1  Progress reporting
  2  Settings
  3  Environment
  4  Lander model, and fitting it to flight data
  5  Wind estimator
  6  Planner
  7  Flying episodes and measuring them
  8  Standalone PPO
  9  Residual controller
  10 Experiments
  11 Command line

Usage:

    python rocket_landing.py --check                 self-tests
    python rocket_landing.py --sysid                 refit the lander model
    python rocket_landing.py --watch 5               watch five landings
    python rocket_landing.py --train-ppo             train the PPO baselines
    python rocket_landing.py --train-residual        train the residual grid
    python rocket_landing.py --measure               fly everything, write the results CSV
    python rocket_landing.py --report                print the result tables
    python rocket_landing.py --ablations             run the ablation sweep
    python rocket_landing.py --refly-floor           re-fly the floor arm only
"""

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import gymnasium as gym
from gymnasium.envs.box2d import lunar_lander as ll
from gymnasium.envs.box2d.lunar_lander import LunarLander, heuristic

HERE = os.path.dirname(os.path.abspath(__file__))


def hms(seconds):
    """Turn a number of seconds into a short readable string like 45s, 12m30s or 3h07m."""
    s = int(max(0.0, seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


class Progress:
    """A progress indicator that works both on screen and in a log file.

    Give it the total number of items and a label, then call advance() as work finishes.
    On a terminal it redraws a single line. When output is redirected to a file it prints
    a fresh line every so often instead, so a long run leaves a readable record. Only
    plain ASCII is printed, so it cannot fail on a console with a limited character set.
    """

    WIDTH = 30

    def __init__(self, total, label, every=None, unit="", show_eta=True):
        self.total = max(1, int(total))
        self.label, self.unit, self.show_eta = label, unit, show_eta
        self.every = every or max(1, self.total // (100 if sys.stdout.isatty() else 20))
        self.tty = sys.stdout.isatty()
        self.n, self.marked, self.t0 = 0, 0, time.time()
        self.note = ""
        print(f"  {label}: {self.total:,}{(' ' + unit) if unit else ''}", flush=True)

    def advance(self, k=1, note=None):
        """Record that k more items are done, optionally with a short note to display."""
        self.n += k
        if note is not None:
            self.note = note
        if self.n < self.total and self.n - self.marked < self.every:
            return
        self.marked = self.n
        self._draw()

    def _draw(self):
        frac = min(1.0, self.n / self.total)
        elapsed = time.time() - self.t0
        rate = self.n / elapsed if elapsed > 0 else 0.0
        eta = (self.total - self.n) / rate if rate > 0 else 0.0
        tail = f"  {self.note}" if self.note else ""
        if self.show_eta:
            tail = f"  {hms(elapsed)} elapsed, {hms(eta)} left" + tail
        if self.tty:
            fill = int(self.WIDTH * frac)
            bar = "=" * fill + ">" + "." * max(0, self.WIDTH - fill - 1)
            sys.stdout.write(f"\r  [{bar}] {100 * frac:5.1f}%  "
                             f"{self.n:,}/{self.total:,}{tail}   ")
            sys.stdout.flush()
            if self.n >= self.total:
                sys.stdout.write("\n")
                sys.stdout.flush()
        else:
            print(f"  {100 * frac:5.1f}%  {self.n:,}/{self.total:,}{tail}", flush=True)

    def close(self, note=None):
        """Finish the indicator and print how long the whole job took."""
        if note is not None:
            self.note = note
        if self.n < self.total:
            self.n = self.total
        self._draw()
        print(f"  {self.label}: done in {hms(time.time() - self.t0)}", flush=True)


_ROOT = 42

EPISODE_SETS = {"sysid": 0, "tune": 1, "train": 3, "eval": 4, "final": 5}

MAX_STEPS = 1000
FPS = ll.FPS
_S = ll.SCALE

WINDS = (0.0, 5.0, 10.0, 15.0, 20.0)
WIND_TRAIN = (0.0, 20.0)

PLAN_HORIZON = 48
PLAN_BLOCK = 4
PLAN_POP = 96
PLAN_POP_DECAY = 1.25
PLAN_ITERS = 4
PLAN_ELITES = 10
PLAN_KEEP = 5
PLAN_SMOOTH = 0.6
PLAN_EXPLORE = 0.2

GROUND_MARGIN = 0.01
W_TIME = 0.004
GS_SLOPE = 1.0
W_GS = 20.0
TILT_MAX = 0.4
W_TILT = 20.0
BRAKE_ETA = 0.7
W_BRAKE = 20.0
W_TD_V = 10.0
W_TD_VX = 10.0
W_TD_ANG = 5.0
W_TD_PAD = 50.0
W_T_X = 5.0
W_T_Y = 0.05
W_T_ANG = 1.0

X_PAD = 0.2
LEG_SPAN = ll.LEG_AWAY / (ll.VIEWPORT_W / 2)
X_SAFE = X_PAD - LEG_SPAN

OBSERVER_GAIN = 0.10

PROB_FLOOR = 0.15

FUEL_COST = {0: 0.0, 1: 0.1, 2: 1.0, 3: 0.1}

PPO_STEPS = 2_000_000
PPO_ENVS = 32
PPO_ROLLOUT = 256
PPO_EPOCHS = 4
PPO_MINIBATCH = 512
PPO_LR = 2.5e-4
PPO_ENTROPY = (0.01, 0.01)
RESIDUAL_ENTROPY = (0.03, 0.005)
PPO_GAMMA = 0.99
PPO_GAE = 0.95
PPO_CLIP = 0.2
PPO_VF = 0.5
PPO_MAX_GRAD = 0.5
PPO_SEEDS = 5

CAUTION_SCALES = (0.0, 0.5, 1.0, 2.0, 4.0)
CAUTION_BASE = 25.0
CAUTION_ALT0 = 0.1

RESIDUAL_TAU = 0.4181448499376904
RESIDUAL_SQUASH = 3.0
RESIDUAL_TRUNK = (256, 256, 128)
RESIDUAL_LAMBDAS = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)
RESIDUAL_LAMBDA = 3.0
RESIDUAL_SEEDS = 15
ABLATION_SEEDS = 5
SHAPING_SCALE = 0.3133279875684444
SHAPING_BASE = 25.0
SHAPING_ALT0 = 0.1
SHAPING_TILT = 1.0

EVAL_RUNS = 3
EVAL_PER_RUN = 100
EVAL_BATCH = 16

PPO_BUNDLE = os.path.join(HERE, "ppo_agents.pt")
RESIDUAL_BUNDLE = os.path.join(HERE, "residual_agents.pt")
ABLATION_BUNDLE = os.path.join(HERE, "ablation_agents.pt")
RESULTS_CSV = os.path.join(HERE, "final_results.csv")
SPREAD_CSV = os.path.join(HERE, "final_spread.csv")


def episode_seeds(which, run, n):
    """Return n environment seeds for one of the named episode sets.

    `which` is "sysid", "tune", "train" or "eval"; `run` numbers repeat measurements
    within a set. The seeds are derived from a fixed starting number, so the sets are
    reproducible and no seed ever appears in two different sets. check_episode_sets()
    verifies that.
    """
    rng = np.random.default_rng([_ROOT, EPISODE_SETS[which], run])
    seeds = np.unique(rng.integers(0, 2 ** 31 - 1, size=3 * n))
    return rng.permutation(seeds)[:n].tolist()


def check_episode_sets(runs=50, n=100):
    """Check that the four episode sets never share a seed. Prints the sizes it checked."""
    sets = {k: {x for r in range(runs) for x in episode_seeds(k, r, n)} for k in EPISODE_SETS}
    for a in EPISODE_SETS:
        for b in EPISODE_SETS:
            if a < b:
                assert not sets[a] & sets[b], (a, b)
    print("episode sets: " + ", ".join(f"{k}={len(v)}" for k, v in sets.items())
          + " distinct seeds, no overlap between any pair")


class _DispersionRNG:
    """Records the small random engine misalignments the environment draws each step.

    LunarLander draws two uniform(-1, 1) numbers per step to jitter the engine thrust.
    This wrapper passes those draws through unchanged and appends each one to the
    environment's _disp_log, which is what lets the fuel measurement reconstruct the
    exact impulse that was applied. Every other random draw is passed straight through.
    """

    def __init__(self, gen, env):
        self._gen = gen
        self._env = env

    def uniform(self, low=0.0, high=1.0, size=None):
        if size is None and low == -1.0 and high == 1.0:
            v = float(self._gen.uniform(-1.0, 1.0))
            self._env._disp_log.append(v)
            return v
        return self._gen.uniform(low, high, size)

    def __getattr__(self, name):
        return getattr(self._gen, name)


_original_step = LunarLander.step


def _patched_step(self, action):
    if not isinstance(self._np_random, _DispersionRNG):
        self._np_random = _DispersionRNG(self._np_random, self)
        self._disp_log = []
    return _original_step(self, action)


LunarLander.step = _patched_step


def make_env(render_mode=None, wind=0.0):
    """Build one evaluation environment at a fixed wind strength from 0 to 20.

    Turbulence is set to a tenth of the wind strength, as the environment intends. With
    wind 0 the wind machinery is switched off entirely.
    """
    if wind > 0:
        return gym.make("LunarLander-v3", render_mode=render_mode, enable_wind=True,
                        wind_power=float(wind), turbulence_power=float(wind) / 10.0)
    return gym.make("LunarLander-v3", render_mode=render_mode, enable_wind=False)


def make_train_env(wind_train=WIND_TRAIN):
    """Build one training environment whose wind strength is re-drawn every episode.

    Pass wind_train=None for an environment that is always calm.
    """
    if wind_train is None:
        return make_env()
    return gym.make("LunarLander-v3", enable_wind=True, wind_power=0.0,
                    turbulence_power=0.0)


def reset_train_env(env, rng, wind_train=WIND_TRAIN, seed=None):
    """Start a new training episode, first drawing this episode's wind strength.

    The strength is drawn uniformly from wind_train, so an agent meets the whole range
    instead of one setting. Returns whatever env.reset returns.
    """
    if wind_train is not None:
        w = float(rng.uniform(*wind_train))
        env.unwrapped.wind_power = w
        env.unwrapped.turbulence_power = w / 10.0
    return env.reset(seed=seed)


def true_wind(u):
    """Return the wind force and turbulence torque the environment is about to apply.

    Used only to check the wind estimator against the truth. No controller ever sees it.
    """
    if not u.enable_wind or u.legs[0].ground_contact or u.legs[1].ground_contact:
        return 0.0, 0.0
    f = math.tanh(math.sin(0.02 * u.wind_idx) + math.sin(math.pi * 0.01 * u.wind_idx))
    t = math.tanh(math.sin(0.02 * u.torque_idx) + math.sin(math.pi * 0.01 * u.torque_idx))
    return f * u.wind_power, t * u.turbulence_power


def engine_impulse(a, d0, d1):
    """Return the impulse in newton-seconds that firing action `a` produced this step.

    d0 and d1 are the two misalignment draws the environment made for that step. Action
    0 is coast, 2 is the main engine, 1 and 3 are the side engines.
    """
    d0, d1 = d0 / _S, d1 / _S
    if a == 2:
        return ll.MAIN_ENGINE_POWER * math.hypot(
            ll.MAIN_ENGINE_Y_LOCATION / _S + 2 * d0, d1)
    if a in (1, 3):
        return ll.SIDE_ENGINE_POWER * math.hypot(
            d0, 3 * d1 + (a - 2) * ll.SIDE_ENGINE_AWAY / _S)
    return 0.0


def episode_outcome(u, terminated):
    """Classify a finished episode as landed, crashed, out_of_bounds or timeout."""
    if terminated and not u.lander.awake:
        return "landed"
    if terminated and u.game_over:
        return "crashed"
    return "out_of_bounds" if terminated else "timeout"


MODEL_PARAMS = {
    "kin": (0.0101, 0.0225, 0.0500),
    "vx": (0.0694, 0.0100, 0.0001),
    "vy": (0.0466, 0.0071, -0.0266),
    "w": (-0.0398, -0.0112, 0.0004, -0.0000),
}

DIRS = np.array([0.0, -1.0, 0.0, 1.0])
MAIN = np.array([0.0, 0.0, 1.0, 0.0])


class LanderModel:
    """A simple model of how the lander moves, with coefficients fitted to flight data.

    The shape of the model comes from rigid-body mechanics: each engine changes the
    velocities and spin by a fixed amount, rotated by the current tilt, and position
    follows from velocity. The numbers multiplying those terms are measured, not assumed.
    The planner uses this model and never touches the simulator itself.
    """

    def __init__(self, p=None):
        p = p or MODEL_PARAMS
        self.kx, self.ky, self.kth = p["kin"]
        self.vx_m, self.vx_s, self.vx_0 = p["vx"]
        self.vy_m, self.vy_s, self.vy_0 = p["vy"]
        self.w_s, self.w_s2, self.w_m, self.w_0 = p["w"]
        self.ratio = self.vx_m / self.vy_m
        self.g = -self.vy_0
        self.dv_action = np.array([0.0, abs(self.vx_s) / self.ratio, self.vy_m,
                                   abs(self.vx_s) / self.ratio])
        self.a_net = self.vy_m - self.g

    def step(self, x, y, vx, vy, th, w, a, dist=(0.0, 0.0, 0.0)):
        """Advance one step. Every argument may be an array, and all must share a shape.

        `a` is the action, `dist` an optional estimate of the unmodelled disturbance.
        Returns the new x, y, vx, vy, tilt and spin.
        """
        s, c = np.sin(th), np.cos(th)
        m, d = MAIN[a], DIRS[a]
        vx = vx + self.vx_m * m * -s + self.vx_s * d * c + self.vx_0 + dist[0]
        vy = vy + self.vy_m * m * c + self.vy_s * d * s + self.vy_0 + dist[1]
        w = w + self.w_s * d + self.w_s2 * d * s * s + self.w_m * m + self.w_0 + dist[2]
        return x + self.kx * vx, y + self.ky * vy, vx, vy, th + self.kth * w, w

    def fuel_to_go(self, y, vx, vy):
        """Estimate the speed change still needed to land from a given state.

        The vertical part is the speed an unpowered fall would reach, which is what the
        engine has to cancel. Sideways speed has to be cancelled separately because one
        engine cannot brake both directions at full efficiency. Used as the planner's
        estimate of the cost remaining beyond its look-ahead.
        """
        vxe = np.abs(vx) / self.ratio
        return vxe + np.sqrt(vy * vy + 2.0 * self.g / self.ky * np.maximum(y, 0.0))


def _flight_mask(O, O2):
    return (O[:, 6] == 0) & (O[:, 7] == 0) & (O2[:, 6] == 0) & (O2[:, 7] == 0)


def collect_flights(seeds, explore=0.3, rng_seed=0, wind=0.0):
    """Fly episodes to gather data for fitting the lander model.

    Flies the environment's own built-in heuristic with a fraction `explore` of random
    actions mixed in, so that every engine gets used in every attitude. Returns the
    observations, actions, next observations and an episode index for each step.
    """
    rng = np.random.default_rng(rng_seed)
    env = make_env(wind=wind)
    O, A, O2, E = [], [], [], []
    for ep, s in enumerate(seeds):
        o, _ = env.reset(seed=int(s))
        for _ in range(MAX_STEPS):
            a = (int(rng.integers(0, 4)) if rng.random() < explore
                 else int(heuristic(env.unwrapped, o)))
            o2, _, term, trunc, _ = env.step(a)
            O.append(o)
            A.append(a)
            O2.append(o2)
            E.append(ep)
            o = o2
            if term or trunc:
                break
    env.close()
    return (np.array(O, dtype=np.float64), np.array(A),
            np.array(O2, dtype=np.float64), np.array(E))


def _regressors(O, A):
    th = O[:, 4]
    s, c = np.sin(th), np.cos(th)
    m, d = MAIN[A], DIRS[A]
    one = np.ones_like(m)
    return {"vx": (np.c_[m * -s, d * c, one], 2),
            "vy": (np.c_[m * c, d * s, one], 3),
            "w": (np.c_[d, d * s * s, m, one], 5)}


def identify(O, A, O2, trim=0.01):
    """Fit the lander model's coefficients to flight data by least squares.

    Fits once, discards the worst 1 per cent of residuals, then refits. Those outliers
    are ground impacts, which the flight model is not meant to describe. Only steps with
    both legs off the ground are used. Returns the coefficients and how many steps were
    used.
    """
    fl = _flight_mask(O, O2)
    O, A, O2 = O[fl], A[fl], O2[fl]
    params = {}
    for name, (X, j) in _regressors(O, A).items():
        y = O2[:, j] - O[:, j]
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        e = y - X @ coef
        keep = np.abs(e) <= np.quantile(np.abs(e), 1 - trim)
        coef = np.linalg.lstsq(X[keep], y[keep], rcond=None)[0]
        params[name] = tuple(float(v) for v in coef)
    kin = []
    for i, j in ((0, 2), (1, 3), (4, 5)):
        kin.append(float(np.linalg.lstsq(O2[:, j:j + 1], O2[:, i] - O[:, i],
                                        rcond=None)[0][0]))
    params["kin"] = tuple(kin)
    return params, int(fl.sum())


def validate_model(model, O, A, O2, E, horizons=(1, 10, 25, 50)):
    """Measure how well a fitted model predicts flight data it was not fitted on.

    Reports the fraction of variance explained for each velocity channel one step ahead,
    and the position error after running the model forward for several step counts with
    no feedback. Returns a dictionary of those figures.
    """
    out = {}
    fl = _flight_mask(O, O2)
    X = O[fl]
    nx = model.step(X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5], A[fl])
    for name, j, k in (("vx", 2, 2), ("vy", 3, 3), ("w", 5, 5)):
        y = O2[fl, j] - X[:, j]
        e = y - (nx[k] - X[:, j])
        out[f"R2_{name}"] = 1 - e.var() / y.var()
    for H in horizons:
        errs = []
        for ep in np.unique(E):
            idx = np.where(E == ep)[0]
            for start in range(0, len(idx) - H, 10):
                seg = idx[start:start + H]
                if not _flight_mask(O[seg], O2[seg]).all():
                    continue
                s = [np.array([v]) for v in O[seg[0], :6]]
                for t in seg:
                    s = list(model.step(*s, np.array([A[t]])))
                true = O2[seg[-1]]
                errs.append(math.hypot(s[0][0] - true[0], s[1][0] - true[1]))
        out[f"pos_err_{H}"] = (float(np.median(errs)),
                              float(np.percentile(errs, 90)), len(errs))
    return out


class WindEstimator:
    """Estimates the part of the motion the lander model does not explain.

    Whatever is left over after the model's prediction is treated as a slowly changing
    push on the two velocities and the spin, and is tracked with a fixed gain. In
    practice that leftover is the wind. The planner then predicts with the estimate
    included, which is what lets it hold a course in wind. Handles many environments at
    once; one row per environment.
    """

    def __init__(self, model, n, gain=OBSERVER_GAIN, enabled=True):
        self.model, self.gain, self.enabled = model, gain, enabled
        self.d = np.zeros((n, 3))

    def update(self, o_prev, a, o_new, mask=None):
        """Fold one step of evidence into the estimate.

        o_prev and o_new are the observations before and after the step, `a` the actions.
        Steps where a leg is touching the ground are skipped, because contact forces are
        not a disturbance worth learning. `mask` can switch off rows with no episode in
        them.
        """
        if not self.enabled:
            return
        live = ((o_prev[:, 6] == 0) & (o_prev[:, 7] == 0)
                & (o_new[:, 6] == 0) & (o_new[:, 7] == 0))
        if mask is not None:
            live &= mask
        if not live.any():
            return
        p = self.model.step(*(o_prev[:, i] for i in range(6)), a)
        resid = np.stack([o_new[:, 2] - p[2], o_new[:, 3] - p[3], o_new[:, 5] - p[5]], 1)
        self.d += self.gain * (resid - self.d) * live[:, None]

    def reset(self, mask):
        """Forget the estimate for the rows in `mask`, which have started new episodes."""
        self.d[mask] = 0.0

    def compact(self, idx):
        """Keep only the rows listed in `idx`, discarding the rest."""
        self.d = self.d[idx].copy()

    @property
    def estimate(self):
        """The current estimate, one row of three numbers per environment."""
        return self.d if self.enabled else np.zeros_like(self.d)


def heuristic_actions(x, y, vx, vy, th, w, leg1=None, leg2=None):
    """The environment's built-in landing heuristic, applied to many states at once.

    Takes six state arrays of equal shape and optionally the two leg-contact flags, and
    returns an array of actions. Once a leg is touching, the heuristic stops steering and
    only damps the remaining descent; leaving that out makes it keep correcting its angle
    on the ground and tip over, which costs about twenty points of landing rate.

    The planner also uses this on states rolled out inside the lander model, which has no
    legs and so no contact flags; with the flags left out the override never fires and the
    planner's behaviour is unchanged.
    """
    angle_targ = np.clip(x * 0.5 + vx * 1.0, -0.4, 0.4)
    hover_targ = 0.55 * np.abs(x)
    angle_todo = (angle_targ - th) * 0.5 - w * 1.0
    hover_todo = (hover_targ - y) * 0.5 - vy * 0.5
    if leg1 is not None:
        down = (leg1 != 0) | (leg2 != 0)
        angle_todo = np.where(down, 0.0, angle_todo)
        hover_todo = np.where(down, -vy * 0.5, hover_todo)
    a = np.zeros(np.shape(x), np.int64)
    np.copyto(a, 1, where=angle_todo > 0.05)
    np.copyto(a, 3, where=angle_todo < -0.05)
    np.copyto(a, 2, where=(hover_todo > np.abs(angle_todo)) & (hover_todo > 0.05))
    return a


def action_distribution(votes, floor=PROB_FLOOR):
    """Turn the planner's preferred first actions into a probability over the four actions.

    `votes` is one row of candidate first actions per environment. Every action keeps a
    floor of probability so the residual network always sees a usable distribution: its
    input is the logarithm of these numbers, which a zero would send to minus infinity.
    Returns one row of four probabilities per environment, checked to be a valid
    distribution before returning.
    """
    L = votes.shape[1]
    counts = np.zeros((votes.shape[0], 4))
    for a in range(4):
        counts[:, a] = (votes == a).sum(axis=1)
    p = floor + (1.0 - floor) * counts / L
    p = p / (4 * floor + 1.0 - floor)
    assert np.allclose(p.sum(axis=1), 1.0) and (p > 0).all(), p
    return p


def sharpen(p, tau):
    """Make a probability distribution more or less decisive.

    tau below 1 sharpens it towards its most likely action, above 1 flattens it. Applied
    to the planner's distribution before the residual network adjusts it.
    """
    z = np.log(p) / tau
    z = np.exp(z - z.max(axis=-1, keepdims=True))
    return z / z.sum(axis=-1, keepdims=True)


class Planner:
    """Chooses engine firings by searching in the lander model, for many environments at once.

    The search is a cross-entropy method: sample a population of action sequences, score
    every one by simulating it in the model, keep the best few, re-aim the sampling
    distribution at them, and repeat. Actions are held for four steps at a time, which
    shortens the search without losing much. Each call also carries the previous
    solution forward as a starting candidate, along with the environment's heuristic.

    Every environment is advanced through the same search in lockstep on shared arrays.
    That is purely a speed measure: the arrays get longer while the work stays the same,
    and it makes a solve around four times cheaper per environment.
    """

    def __init__(self, model, rng, n, iters=PLAN_ITERS, pop=PLAN_POP,
                 horizon=PLAN_HORIZON, block=PLAN_BLOCK, terminal=True,
                 floor=PROB_FLOOR, time_it=False):
        self.m, self.rng, self.n = model, rng, n
        self.iters, self.pop, self.H, self.B = iters, pop, horizon, block
        self.nb = horizon // block
        self.terminal = terminal
        self.floor = floor
        self.prev = np.zeros((n, horizon), np.int64)
        self.has_prev = np.zeros(n, bool)
        self.time_it = time_it
        self.solve_ms = []

    def reset(self, mask):
        """Forget the carried-forward solution for rows in `mask`, which have restarted."""
        self.has_prev[np.asarray(mask)] = False

    def compact(self, idx):
        """Keep only the rows listed in `idx`.

        An environment whose episode has ended still costs a full share of the search, so
        dropping empty rows matters when a batch drains.
        """
        self.prev = self.prev[idx].copy()
        self.has_prev = self.has_prev[idx].copy()
        self.n = len(idx)

    def cost(self, O, seqs, D):
        """Score candidate action sequences by simulating them in the lander model.

        O holds one state per environment, seqs holds candidate sequences shaped
        (environments, candidates, steps), D holds the wind estimate per environment.
        The score adds up fuel, a small charge per step in flight, penalties for leaving
        the glide slope, over-tilting or exceeding a safe braking envelope, then a
        touchdown charge on speed, tilt and distance from the pad, or an estimate of the
        cost still to come if the sequence ends in mid-air. Returns one score per
        candidate.
        """
        m = self.m
        E, M, H = seqs.shape
        sq = np.ascontiguousarray(seqs.transpose(2, 0, 1))
        cx = m.vx_0 + D[:, 0, None]
        cy = m.vy_0 + D[:, 1, None]
        cw = m.w_0 + D[:, 2, None]
        ground = np.minimum(0.0, O[:, 1:2] - GROUND_MARGIN)
        brake = m.ky / (2.0 * BRAKE_ETA * m.a_net)
        x, y, vx, vy, th, w = (np.repeat(O[:, i:i + 1], M, axis=1) for i in range(6))
        total = np.zeros((E, M))
        live = np.ones((E, M), bool)
        down = np.zeros((E, M), bool)
        xf, yf, vxf, vyf, thf, wf = (np.empty((E, M)) for _ in range(6))
        for t in range(H):
            a = sq[t]
            mm, dd = MAIN[a], DIRS[a]
            s, c = np.sin(th), np.cos(th)
            vx = vx - m.vx_m * mm * s + m.vx_s * dd * c + cx
            vy = vy + m.vy_m * mm * c + m.vy_s * dd * s + cy
            w = w + (m.w_s * dd + m.w_m * mm + cw) + (m.w_s2 * dd) * s * s
            x = x + m.kx * vx
            y = y + m.ky * vy
            th = th + m.kth * w
            gs = np.maximum(GS_SLOPE * (np.abs(x) - X_PAD) - y, 0.0)
            tilt = np.maximum(np.abs(th) - TILT_MAX, 0.0)
            envl = np.maximum(brake * np.minimum(vy, 0.0) ** 2 - np.maximum(y, 0.0), 0.0)
            stage = (m.dv_action[a] + W_TIME + W_GS * gs * gs
                     + W_TILT * tilt * tilt + W_BRAKE * envl)
            total += stage * live
            hit = live & (y <= ground)
            if hit.any():
                for dst, src in ((xf, x), (yf, y), (vxf, vx), (vyf, vy), (thf, th), (wf, w)):
                    np.copyto(dst, src, where=hit)
                down |= hit
                live &= ~hit
            if t == H - 1:
                rest = ~down
                for dst, src in ((xf, x), (yf, y), (vxf, vx), (vyf, vy), (thf, th), (wf, w)):
                    np.copyto(dst, src, where=rest)
        wf = wf * (m.kth * FPS / 10)
        vxe = vxf / m.ratio
        pad = np.maximum(np.abs(xf) - X_SAFE, 0.0)
        td = (W_TD_VX * vxe * vxe + W_TD_V * vyf * vyf
              + W_TD_ANG * (thf * thf + wf * wf) + W_TD_PAD * pad * pad)
        term = np.zeros((E, M))
        if self.terminal:
            term = (m.fuel_to_go(yf, vxf, vyf) + W_T_X * xf * xf
                    + W_T_Y * np.maximum(yf, 0.0) + W_T_ANG * (thf * thf + wf * wf))
        return total + np.where(down, td, term)

    def heuristic_plans(self, O, D):
        """Fly the environment's heuristic forward inside the model.

        Returns one action sequence per environment, used as a starting candidate so the
        search never does worse than the heuristic by accident.
        """
        m = self.m
        x, y, vx, vy, th, w = (O[:, i].astype(np.float64).copy() for i in range(6))
        d0, d1, d2 = D[:, 0], D[:, 1], D[:, 2]
        seq = np.zeros((O.shape[0], self.H), np.int64)
        for t in range(self.H):
            a = heuristic_actions(x, y, vx, vy, th, w)
            seq[:, t] = a
            s, c = np.sin(th), np.cos(th)
            mm, dd = MAIN[a], DIRS[a]
            vx = vx + m.vx_m * mm * -s + m.vx_s * dd * c + m.vx_0 + d0
            vy = vy + m.vy_m * mm * c + m.vy_s * dd * s + m.vy_0 + d1
            w = w + m.w_s * dd + m.w_s2 * dd * s * s + m.w_m * mm + m.w_0 + d2
            x, y, th = x + m.kx * vx, y + m.ky * vy, th + m.kth * w
        return seq

    def _block_freq(self, seqs):
        r = seqs.reshape(seqs.shape[0], seqs.shape[1], self.nb, self.B)
        out = np.empty((seqs.shape[0], self.nb, 4))
        for a in range(4):
            out[:, :, a] = (r == a).mean(axis=(1, 3))
        return out

    def plan(self, O, D, elapsed=1):
        """Run the search and return a decision for every environment.

        O holds one state per environment, D the wind estimate. Returns the best action
        sequence found for each environment and a probability over the next action, taken
        as a vote among the best candidates. The residual controller uses that
        probability; the plain planner just flies the first action of its best sequence.
        """
        E, H, nb = self.n, self.H, self.nb
        t_start = time.perf_counter() if self.time_it else None
        heur = self.heuristic_plans(O, D)
        e = min(max(int(elapsed), 1), H - 1)
        shifted = np.concatenate(
            [self.prev[:, e:], np.repeat(self.prev[:, -1:], e, 1)], 1)
        shifted = np.where(self.has_prev[:, None], shifted, heur)
        P = np.where(self.has_prev[:, None, None],
                     (1 - PLAN_EXPLORE) * self._block_freq(shifted[:, None])
                     + PLAN_EXPLORE / 4, 0.25)
        keep = np.stack([shifted, heur], axis=1)
        dead = ~self.has_prev
        best_cost = np.full(E, np.inf)
        best_seq = np.zeros((E, H), np.int64)
        rows = np.arange(E)
        elites = None
        for i in range(self.iters):
            n_i = max(int(self.pop / PLAN_POP_DECAY ** i), 2 * PLAN_ELITES)
            cum = P.cumsum(axis=2)
            u01 = self.rng.random((E, n_i, nb, 1))
            blocks = np.minimum((u01 > cum[:, None, :, :]).sum(axis=3), 3)
            seqs = np.concatenate([keep, np.repeat(blocks, self.B, axis=2)], axis=1)
            costs = self.cost(O, seqs, D)
            if i == 0 and dead.any():
                costs[dead, 0] = np.inf
            order = np.argsort(costs, axis=1)
            top = order[:, 0]
            c0 = costs[rows, top]
            upd = c0 < best_cost
            if upd.any():
                best_cost = np.where(upd, c0, best_cost)
                best_seq[upd] = seqs[rows[upd], top[upd]]
            elites = np.take_along_axis(seqs, order[:, :PLAN_ELITES, None], axis=1)
            P = PLAN_SMOOTH * self._block_freq(elites) + (1 - PLAN_SMOOTH) * P
            keep = elites[:, :PLAN_KEEP]
        self.prev = best_seq
        self.has_prev[:] = True
        if t_start is not None:
            self.solve_ms.append(1000 * (time.perf_counter() - t_start) / max(1, E))
        votes = np.concatenate([best_seq[:, :1], elites[:, :, 0]], axis=1)
        return best_seq, action_distribution(votes, self.floor)


class PlannerBase:
    """The planner packaged as something the residual controller can sit on top of.

    Holds a planner and a wind estimator for a fixed number of environments, and exposes
    the planner's action distribution and the current wind estimate.
    """

    def __init__(self, seed, n, iters=PLAN_ITERS, horizon=PLAN_HORIZON, terminal=True,
                 floor=PROB_FLOOR, replan=1, estimator=True, time_it=False):
        self.model = LanderModel()
        self.rng = np.random.default_rng(seed)
        self.n = n
        self.planner = Planner(self.model, self.rng, n, iters=iters, horizon=horizon,
                               terminal=terminal, floor=floor, time_it=time_it)
        self.estimator = WindEstimator(self.model, n, enabled=estimator)
        self.replan = max(1, int(replan))
        self.k = None
        self.since = 1
        self.held = None

    def pi(self, O, tau):
        """Return the planner's sharpened probability over the next action, per environment."""
        _, pi = self.planner.plan(O, self.estimator.estimate)
        return sharpen(pi, tau)

    def decide(self, O):
        """Return the action to fly, per environment, from the planner alone.

        With `replan` at 1 the planner re-solves every step and the first action of its
        best sequence is flown. With a larger value it keeps the sequence it found and
        flies the next action along, re-solving only every `replan` steps.

        Each environment carries its own step counter, so one environment starting a new
        episode does not disturb the cadence of the others sharing the batch.
        """
        if self.held is None:
            self.k = np.full(self.n, self.replan, np.int64)
        if self.held is None or bool((self.k >= self.replan).any()):
            seq, _ = self.planner.plan(O, self.estimator.estimate,
                                       elapsed=max(1, self.since))
            due = self.k >= self.replan
            if self.held is None:
                self.held, self.k = seq, np.zeros(self.n, np.int64)
            else:
                self.held[due] = seq[due]
                self.k[due] = 0
            self.since = 0
        self.since += 1
        a = self.held[np.arange(self.held.shape[0]),
                      np.minimum(self.k, self.held.shape[1] - 1)]
        self.k = self.k + 1
        return a

    def update(self, o_prev, a, o_new, mask=None):
        """Pass one step of experience to the wind estimator."""
        self.estimator.update(o_prev, a, o_new, mask)

    def reset(self, mask):
        """Forget planner and wind state for rows that have started new episodes.

        Rows named by the mask are made due for a fresh solve. Rows not named keep the
        sequence they are part way through.
        """
        self.planner.reset(mask)
        self.estimator.reset(mask)
        if self.k is not None:
            self.k = np.where(mask, self.replan, self.k)

    def compact(self, idx):
        """Keep only the rows listed in `idx`."""
        self.planner.compact(idx)
        self.estimator.compact(idx)
        if self.held is not None:
            self.held = self.held[idx]
        if self.k is not None:
            self.k = self.k[idx]
        self.n = len(idx)


def caution_risk(o):
    """How dangerous a state is: speed that matters because the ground is close, plus tilt.

    Takes one or many observations and returns one number each.
    """
    y = np.maximum(o[..., 1], 0.0)
    vx, vy = o[..., 2], np.minimum(o[..., 3], 0.0)
    return (vx * vx + vy * vy) / (y + CAUTION_ALT0) + o[..., 4] ** 2


def caution_potential(o, func):
    """The shaping term for standalone PPO agent number `func`, from 1 to 5.

    Agent 1 gets no shaping and agents 2 to 5 get increasing amounts of caution. The term
    is a function of the state alone, which means it changes how quickly an agent learns
    but not which behaviour is finally best.
    """
    return -CAUTION_BASE * CAUTION_SCALES[func - 1] * caution_risk(o)


def residual_risk(o, alt0=SHAPING_ALT0, tilt=SHAPING_TILT):
    """How dangerous a state is, for the residual controller's shaping term."""
    y = np.maximum(o[..., 1], 0.0)
    vx, vy = o[..., 2], np.minimum(o[..., 3], 0.0)
    return (vx * vx + vy * vy) / (y + alt0) + tilt * o[..., 4] ** 2


def residual_potential(o, scale=SHAPING_SCALE, alt0=SHAPING_ALT0, tilt=SHAPING_TILT):
    """The shaping term the residual controller is trained against.

    Like the standalone agents' caution term, it depends on the state alone, so it
    changes the speed of learning and not the behaviour that is finally best. The three
    constants have defaults and are recorded on every agent that was trained with them.
    """
    return -SHAPING_BASE * scale * residual_risk(o, alt0, tilt)


class RunningMeanStd:
    """Tracks a running mean and variance, used to normalise observations during training.

    Call update() with a batch of observations, then normalize() to rescale them.
    """

    def __init__(self, shape=()):
        self.mean, self.var, self.count = np.zeros(shape), np.ones(shape), 1e-4

    def update(self, x):
        """Fold a batch of rows into the running statistics."""
        bm, bv, bc = x.mean(axis=0), x.var(axis=0), x.shape[0]
        d, tot = bm - self.mean, self.count + bc
        self.mean = self.mean + d * bc / tot
        self.var = (self.var * self.count + bv * bc
                    + d * d * self.count * bc / tot) / tot
        self.count = tot

    def normalize(self, x):
        """Rescale rows to roughly zero mean and unit spread, clipped to plus or minus ten."""
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -10.0, 10.0)


def build_actor_critic():
    """Build the network a standalone PPO agent uses.

    Two shared layers of 256 units feed a head that scores the four actions and a head
    that estimates how good the state is. Returns the network.
    """
    import torch.nn as nn

    def lin(i, o, gain):
        layer = nn.Linear(i, o)
        nn.init.orthogonal_(layer.weight, gain)
        nn.init.zeros_(layer.bias)
        return layer

    class ActorCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = nn.Sequential(
                lin(8, 256, math.sqrt(2)), nn.LayerNorm(256), nn.Tanh(),
                lin(256, 256, math.sqrt(2)), nn.LayerNorm(256), nn.Tanh())
            self.actor = lin(256, 4, 0.01)
            self.critic = lin(256, 1, 1.0)

        def forward(self, x):
            h = self.trunk(x)
            return self.actor(h), self.critic(h).squeeze(-1)

    return ActorCritic()


def build_residual_net(lam, squash=RESIDUAL_SQUASH):
    """Build the network the residual controller uses.

    It reads the eight observations plus the planner's four action probabilities, and
    outputs one adjustment per action, limited to `squash` either side of zero. The last
    layer starts small and is scaled down further for larger `lam`, so training always
    begins from "leave the planner's proposal alone" whatever the setting of lam.
    """
    import torch
    import torch.nn as nn

    def lin(i, o, gain):
        layer = nn.Linear(i, o)
        nn.init.orthogonal_(layer.weight, gain)
        nn.init.zeros_(layer.bias)
        return layer

    class ResidualNet(nn.Module):
        def __init__(self):
            super().__init__()
            sizes = (12,) + tuple(RESIDUAL_TRUNK)
            layers = []
            for a, b in zip(sizes, sizes[1:]):
                layers += [lin(a, b, math.sqrt(2)), nn.LayerNorm(b), nn.Tanh()]
            self.trunk = nn.Sequential(*layers)
            self.actor = lin(sizes[-1], 4, 0.01 / max(1.0, lam))
            self.critic = lin(sizes[-1], 1, 1.0)

        def forward(self, x):
            h = self.trunk(x)
            return squash * torch.tanh(self.actor(h)), self.critic(h).squeeze(-1)

    return ResidualNet()


def combine(pi_tau, delta, lam, mode="log", squash=RESIDUAL_SQUASH):
    """Blend the planner's proposal with the network's adjustment.

    pi_tau is the planner's sharpened probability over actions, delta the network's
    adjustment, lam how much weight the adjustment carries. Returns the probability the
    residual controller actually acts on. With lam at zero this is exactly the planner.

    Two ways of blending are implemented. "log" shifts the log-probabilities and takes a
    softmax, so the result is always a valid distribution and its relation to the
    planner's is smooth. "add" adds the adjustment to the probabilities directly and
    renormalises, which needs clipping to stay positive; it is the alternative the review
    asked to see measured.
    """
    if mode == "log":
        z = np.log(pi_tau) + lam * delta
        z = z - z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)
    q = np.clip(pi_tau + lam * delta / (4 * squash), 1e-4, None)
    return q / q.sum(axis=-1, keepdims=True)


def sample_rows(probs, rng):
    """Draw one action per row from a batch of probability rows."""
    c = probs.cumsum(axis=1)
    c[:, -1] = 1.0
    return np.minimum((c < rng.random((probs.shape[0], 1))).sum(axis=1), 3)


def ppo_train(job):
    """Train one standalone PPO agent.

    job is (tag, shaping function 1 to 5, seed number, overrides). Trains for the shared
    step budget on environments whose wind is re-drawn each episode, and returns the
    weights, the observation statistics and a learning curve.
    """
    tag, func, seed_idx, over = job
    import torch
    import zlib
    torch.set_num_threads(1)
    hp = dict(steps=PPO_STEPS, envs=PPO_ENVS, rollout=PPO_ROLLOUT, epochs=PPO_EPOCHS,
              minibatch=PPO_MINIBATCH, lr=PPO_LR, ent=PPO_ENTROPY, gamma=PPO_GAMMA,
              lam=PPO_GAE, clip=PPO_CLIP, vf=PPO_VF, max_grad=PPO_MAX_GRAD,
              wind_train=WIND_TRAIN)
    hp.update(over or {})
    torch.manual_seed(int(np.random.SeedSequence(
        [_ROOT, EPISODE_SETS["train"], func, seed_idx,
         zlib.crc32(tag.encode())]).generate_state(1)[0]))
    rng = np.random.default_rng([_ROOT, EPISODE_SETS["train"], func, seed_idx])
    net = build_actor_critic()
    opt = torch.optim.Adam(net.parameters(), lr=hp["lr"], eps=1e-5)
    N, T, gamma, lam = hp["envs"], hp["rollout"], hp["gamma"], hp["lam"]
    wind_train = hp["wind_train"]
    envs = [make_train_env(wind_train) for _ in range(N)]
    first = episode_seeds("train", 1000 * func + seed_idx, N)
    raw = np.stack([reset_train_env(e, rng, wind_train, seed=int(s))[0]
                    for e, s in zip(envs, first)]).astype(np.float64)
    obs_rms, ret_rms = RunningMeanStd((8,)), RunningMeanStd(())
    ret_acc = np.zeros(N)
    n_updates = math.ceil(hp["steps"] / (N * T))
    ep_ret = np.zeros(N)
    curve, t0 = [], time.time()
    buf_obs = np.zeros((T, N, 8), np.float32)
    buf_act = np.zeros((T, N), np.int64)
    buf_logp, buf_val, buf_rew = (np.zeros((T, N), np.float32) for _ in range(3))
    buf_done = np.zeros((T, N), np.float32)
    for upd in range(n_updates):
        frac = upd / max(1, n_updates - 1)
        ent_coef = hp["ent"][0] + frac * (hp["ent"][1] - hp["ent"][0])
        finished = []
        for t in range(T):
            obs_rms.update(raw)
            on = obs_rms.normalize(raw).astype(np.float32)
            with torch.no_grad():
                logits, val = net(torch.from_numpy(on))
                dist = torch.distributions.Categorical(logits=logits)
                act = dist.sample()
                logp = dist.log_prob(act)
            a_np = act.numpy()
            rew = np.zeros(N)
            nxt = np.empty_like(raw)
            term = np.zeros(N, bool)
            trunc_final = {}
            for i, e in enumerate(envs):
                o2, r, te, tr, _ = e.step(int(a_np[i]))
                o2 = o2.astype(np.float64)
                phi_next = 0.0 if te else caution_potential(o2, func)
                rew[i] = r + gamma * phi_next - caution_potential(raw[i], func)
                ep_ret[i] += r
                term[i] = te
                if te or tr:
                    finished.append((ep_ret[i],
                                     episode_outcome(e.unwrapped, te) == "landed"))
                    if tr and not te:
                        trunc_final[i] = o2
                    ep_ret[i] = 0.0
                    o2 = reset_train_env(e, rng, wind_train)[0].astype(np.float64)
                nxt[i] = o2
            if trunc_final:
                fo = obs_rms.normalize(
                    np.stack(list(trunc_final.values()))).astype(np.float32)
                with torch.no_grad():
                    fv = net(torch.from_numpy(fo))[1].numpy()
            ret_acc = ret_acc * gamma + rew
            ret_rms.update(ret_acc)
            rew = rew / np.sqrt(ret_rms.var + 1e-8)
            ret_acc[term] = 0.0
            if trunc_final:
                for j, i in enumerate(trunc_final):
                    rew[i] += gamma * fv[j]
                    ret_acc[i] = 0.0
            done = term | np.array([i in trunc_final for i in range(N)])
            buf_obs[t], buf_act[t] = on, a_np
            buf_logp[t], buf_val[t] = logp.numpy(), val.numpy()
            buf_rew[t], buf_done[t] = rew, done
            raw = nxt
        for e in envs:
            e.unwrapped._disp_log.clear()
        with torch.no_grad():
            next_val = net(torch.from_numpy(
                obs_rms.normalize(raw).astype(np.float32)))[1].numpy()
        adv = np.zeros((T, N), np.float32)
        last = np.zeros(N, np.float32)
        for t in reversed(range(T)):
            nv = next_val if t == T - 1 else buf_val[t + 1]
            nonterm = 1.0 - buf_done[t]
            delta = buf_rew[t] + gamma * nv * nonterm - buf_val[t]
            last = delta + gamma * lam * nonterm * last
            adv[t] = last
        ret = adv + buf_val
        b_obs = torch.from_numpy(buf_obs.reshape(-1, 8))
        b_act = torch.from_numpy(buf_act.reshape(-1))
        b_logp = torch.from_numpy(buf_logp.reshape(-1))
        b_adv = torch.from_numpy(adv.reshape(-1))
        b_ret = torch.from_numpy(ret.reshape(-1))
        idx = np.arange(N * T)
        stats = []
        for _ in range(hp["epochs"]):
            rng.shuffle(idx)
            for s in range(0, N * T, hp["minibatch"]):
                mb = torch.from_numpy(idx[s:s + hp["minibatch"]])
                logits, v = net(b_obs[mb])
                dist = torch.distributions.Categorical(logits=logits)
                lp = dist.log_prob(b_act[mb])
                ratio = torch.exp(lp - b_logp[mb])
                a_mb = b_adv[mb]
                a_mb = (a_mb - a_mb.mean()) / (a_mb.std() + 1e-8)
                pg = -torch.min(ratio * a_mb,
                                torch.clamp(ratio, 1 - hp["clip"],
                                            1 + hp["clip"]) * a_mb).mean()
                vl = 0.5 * ((v - b_ret[mb]) ** 2).mean()
                ent = dist.entropy().mean()
                opt.zero_grad()
                (pg + hp["vf"] * vl - ent_coef * ent).backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), hp["max_grad"])
                opt.step()
                stats.append((pg.item(), vl.item(), ent.item()))
        st = np.mean(stats, axis=0)
        curve.append({"steps": (upd + 1) * N * T,
                      "landed_frac": float(np.mean([f[1] for f in finished]))
                      if finished else float("nan"),
                      "entropy": float(st[2]), "ent_coef": ent_coef})
    for e in envs:
        e.close()
    return {"tag": tag, "kind": "ppo", "function": func, "seed_idx": seed_idx,
            "weights": {k: v.detach().numpy().copy() for k, v in net.state_dict().items()},
            "obs_mean": obs_rms.mean, "obs_var": obs_rms.var, "curve": curve,
            "hp": dict(hp, caution_scale=CAUTION_SCALES[func - 1],
                       caution_base=CAUTION_BASE),
            "minutes": (time.time() - t0) / 60}


def residual_train(job):
    """Train one residual agent.

    job is (tag, lam, seed number, options). The planner runs underneath at every step and
    the network learns only how to adjust its proposal.

    Options override the settings that tie the planner and the network together, each
    defaulting to the module constant: "mode" ("add" for the additive blend instead of the
    log-space one), "floor" for the planner's probability floor, "tau" for the sharpening
    temperature, "squash" for the limit on the adjustment, "entropy" for the pair of
    exploration-bonus endpoints, "shaping_scale", "shaping_alt0" and "shaping_tilt" for the
    shaping term, and "steps" for the training budget. Every one of them is recorded on the
    returned agent, so a saved agent describes itself without reference to this file.

    "seed_tag" replaces the name used to derive the network's starting weights. Two agents
    given the same seed number and the same seed_tag start from identical weights and see
    identical episodes, so a comparison between them is paired and differences between them
    are due to the settings and not to luck.

    Returns the weights, observation statistics, the settings used and a learning curve.
    """
    tag, lam, seed_idx = job[:3]
    opts = job[3] if len(job) > 3 else {}
    mode = opts.get("mode", "log")
    floor = opts.get("floor", PROB_FLOOR)
    tau = float(opts.get("tau", RESIDUAL_TAU))
    squash = float(opts.get("squash", RESIDUAL_SQUASH))
    entropy = tuple(opts.get("entropy", RESIDUAL_ENTROPY))
    shape = (float(opts.get("shaping_scale", SHAPING_SCALE)),
             float(opts.get("shaping_alt0", SHAPING_ALT0)),
             float(opts.get("shaping_tilt", SHAPING_TILT)))
    budget = int(opts.get("steps", PPO_STEPS))
    seed_tag = opts.get("seed_tag", tag)
    import torch
    import zlib
    torch.set_num_threads(1)
    torch.manual_seed(int(np.random.SeedSequence(
        [_ROOT, EPISODE_SETS["train"], 7, seed_idx,
         zlib.crc32(seed_tag.encode())]).generate_state(1)[0]))
    rng = np.random.default_rng([_ROOT, EPISODE_SETS["train"], 77, seed_idx])
    N, T = PPO_ENVS, PPO_ROLLOUT
    gamma, gae = PPO_GAMMA, PPO_GAE
    net = build_residual_net(lam, squash=squash)
    opt = torch.optim.Adam(net.parameters(), lr=PPO_LR, eps=1e-5)
    envs = [make_train_env(WIND_TRAIN) for _ in range(N)]
    base = PlannerBase([_ROOT, 7, seed_idx], N, floor=floor)
    first = episode_seeds("train", 9000 + seed_idx, N)
    raw = np.stack([reset_train_env(e, rng, WIND_TRAIN, seed=int(s))[0]
                    for e, s in zip(envs, first)]).astype(np.float64)
    base.reset(np.ones(N, bool))
    obs_rms, ret_rms = RunningMeanStd((8,)), RunningMeanStd(())
    ret_acc = np.zeros(N)
    n_updates = math.ceil(budget / (N * T))
    ep_ret, curve, t0 = np.zeros(N), [], time.time()
    buf_in = np.zeros((T, N, 12), np.float32)
    buf_pi = np.zeros((T, N, 4), np.float32)
    buf_act = np.zeros((T, N), np.int64)
    buf_logp, buf_val, buf_rew = (np.zeros((T, N), np.float32) for _ in range(3))
    buf_done = np.zeros((T, N), np.float32)
    for upd in range(n_updates):
        frac = upd / max(1, n_updates - 1)
        ent_coef = entropy[0] + frac * (entropy[1] - entropy[0])
        finished = []
        for t in range(T):
            pi_tau = base.pi(raw, tau)
            obs_rms.update(raw)
            x = np.concatenate([obs_rms.normalize(raw), np.log(pi_tau)],
                               axis=1).astype(np.float32)
            with torch.no_grad():
                delta, val = net(torch.from_numpy(x))
            d = delta.numpy().astype(np.float64)
            probs = combine(pi_tau, d, lam, mode)
            act = sample_rows(probs, rng)
            logp = np.log(probs[np.arange(N), act] + 1e-12)
            rew = np.zeros(N)
            nxt_raw = np.empty_like(raw)
            nxt = np.empty_like(raw)
            term = np.zeros(N, bool)
            restart = np.zeros(N, bool)
            for i, e in enumerate(envs):
                o2, r, te, tr, _ = e.step(int(act[i]))
                o2 = o2.astype(np.float64)
                nxt_raw[i] = o2
                phi_next = 0.0 if te else residual_potential(o2, *shape)
                rew[i] = r + gamma * phi_next - residual_potential(raw[i], *shape)
                ep_ret[i] += r
                term[i] = te
                if te or tr:
                    finished.append((ep_ret[i],
                                     episode_outcome(e.unwrapped, te) == "landed"))
                    ep_ret[i] = 0.0
                    o2 = reset_train_env(e, rng, WIND_TRAIN)[0].astype(np.float64)
                    restart[i] = True
                nxt[i] = o2
            base.update(raw, act, nxt_raw)
            base.reset(restart)
            ret_acc = ret_acc * gamma + rew
            ret_rms.update(ret_acc)
            buf_in[t], buf_pi[t] = x, pi_tau
            buf_act[t], buf_logp[t], buf_val[t] = act, logp, val.numpy()
            buf_rew[t] = rew / math.sqrt(float(ret_rms.var) + 1e-8)
            buf_done[t] = term
            raw = nxt
        with torch.no_grad():
            pi_tau = base.pi(raw, tau)
            last_in = np.concatenate([obs_rms.normalize(raw), np.log(pi_tau)],
                                     axis=1).astype(np.float32)
            last_v = net(torch.from_numpy(last_in))[1].numpy()
        adv = np.zeros((T, N), np.float32)
        run = np.zeros(N, np.float32)
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else buf_val[t + 1]
            nonterm = 1.0 - buf_done[t]
            dlt = buf_rew[t] + gamma * nv * nonterm - buf_val[t]
            run = dlt + gamma * gae * nonterm * run
            adv[t] = run
        ret = adv + buf_val
        b_in = torch.from_numpy(buf_in.reshape(T * N, 12))
        b_pi = torch.from_numpy(buf_pi.reshape(T * N, 4))
        b_act = torch.from_numpy(buf_act.reshape(T * N))
        b_logp = torch.from_numpy(buf_logp.reshape(T * N))
        b_adv = torch.from_numpy(adv.reshape(T * N))
        b_ret = torch.from_numpy(ret.reshape(T * N))
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        stats = []
        for _ in range(PPO_EPOCHS):
            idx = np.random.permutation(T * N)
            for s in range(0, T * N, PPO_MINIBATCH):
                mb = torch.from_numpy(idx[s:s + PPO_MINIBATCH])
                delta, v = net(b_in[mb])
                if mode == "log":
                    dist = torch.distributions.Categorical(
                        logits=torch.log(b_pi[mb]) + lam * delta)
                else:
                    q = torch.clamp(b_pi[mb] + lam * delta / (4 * squash),
                                    min=1e-4)
                    dist = torch.distributions.Categorical(probs=q / q.sum(-1, keepdim=True))
                lp = dist.log_prob(b_act[mb])
                ratio = torch.exp(lp - b_logp[mb])
                a_mb = b_adv[mb]
                pg = -torch.min(ratio * a_mb,
                                torch.clamp(ratio, 1 - PPO_CLIP,
                                            1 + PPO_CLIP) * a_mb).mean()
                vl = ((v - b_ret[mb]) ** 2).mean()
                ent = dist.entropy().mean()
                (pg + PPO_VF * vl - ent_coef * ent).backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), PPO_MAX_GRAD)
                opt.step()
                opt.zero_grad()
                stats.append((ent.item(), delta.detach().abs().mean().item()))
        st = np.mean(stats, axis=0)
        curve.append({"steps": (upd + 1) * N * T,
                      "landed_frac": float(np.mean([f[1] for f in finished]))
                      if finished else float("nan"),
                      "entropy": float(st[0]), "abs_delta": float(st[1])})
    for e in envs:
        e.close()
    return {"tag": tag, "kind": "residual", "lam": lam, "seed_idx": seed_idx,
            "weights": {k: v.detach().numpy().copy() for k, v in net.state_dict().items()},
            "obs_mean": obs_rms.mean, "obs_var": obs_rms.var, "curve": curve,
            "hp": {"steps": budget, "envs": N, "rollout": T, "lr": PPO_LR,
                   "entropy": entropy, "gamma": PPO_GAMMA, "gae": PPO_GAE,
                   "clip": PPO_CLIP, "tau": tau, "squash": squash,
                   "trunk": RESIDUAL_TRUNK, "lam": lam, "mode": mode, "floor": floor,
                   "shaping_scale": shape[0], "shaping_alt0": shape[1],
                   "shaping_tilt": shape[2], "shaping_base": SHAPING_BASE,
                   "wind_train": WIND_TRAIN},
            "mode": mode, "floor": floor, "tau": tau, "squash": squash,
            "minutes": (time.time() - t0) / 60}


def train_many(jobs, trainer, save_to, workers, label):
    """Train a list of agents, a batch at a time, saving after every one.

    `trainer` is ppo_train or residual_train. Agents already present in the saved bundle
    are skipped, so an interrupted run resumes and costs at most the agents in flight.
    Returns the bundle, a dictionary from tag to agent.
    """
    import torch
    bundle = torch.load(save_to, weights_only=False) if os.path.exists(save_to) else {}
    todo = [j for j in jobs if j[0] not in bundle]
    if not todo:
        print(f"{label}: all {len(bundle)} already trained")
        return bundle
    if bundle:
        print(f"{label}: resuming, {len(bundle)} already done")
    bar = Progress(len(todo), label, every=1, unit="agents")
    for start in range(0, len(todo), workers):
        batch = todo[start:start + workers]
        with mp.Pool(len(batch), maxtasksperchild=1) as pool:
            for agent in pool.imap_unordered(trainer, batch):
                bundle[agent["tag"]] = agent
                torch.save(bundle, save_to)
                bar.advance(note=f"{agent['tag']} {agent['minutes']:.0f} min")
    bar.close(note="")
    return bundle


class PPOPolicy:
    """A trained standalone agent, ready to fly, as plain array arithmetic.

    Call it with a batch of observations to get one action per row.
    """

    def __init__(self, agent):
        w = agent["weights"]
        self.mean = agent["obs_mean"]
        self.std = np.sqrt(agent["obs_var"] + 1e-8)
        self.layers = [(w["trunk.0.weight"], w["trunk.0.bias"],
                        w["trunk.1.weight"], w["trunk.1.bias"]),
                       (w["trunk.3.weight"], w["trunk.3.bias"],
                        w["trunk.4.weight"], w["trunk.4.bias"])]
        self.wa, self.ba = w["actor.weight"], w["actor.bias"]

    def __call__(self, obs, pi_tau=None, est=None):
        h = np.clip((np.asarray(obs, float) - self.mean) / self.std, -10, 10)
        for W, b, g, beta in self.layers:
            h = h @ W.T + b
            mu = h.mean(axis=-1, keepdims=True)
            var = h.var(axis=-1, keepdims=True)
            h = np.tanh((h - mu) / np.sqrt(var + 1e-5) * g + beta)
        return np.argmax(h @ self.wa.T + self.ba, axis=-1)


class ResidualPolicy:
    """A trained residual agent, ready to fly, as plain array arithmetic.

    Call it with a batch of observations and the planner's probabilities to get one
    action per row.
    """

    def __init__(self, agent):
        self.lam = agent["lam"]
        self.mode = agent.get("mode", "log")
        self.tau = float(agent.get("tau", RESIDUAL_TAU))
        self.squash = float(agent.get("squash", RESIDUAL_SQUASH))
        w = agent["weights"]
        self.mean = agent["obs_mean"]
        self.std = np.sqrt(agent["obs_var"] + 1e-8)
        n = len(RESIDUAL_TRUNK)
        self.layers = [(w[f"trunk.{3 * i}.weight"], w[f"trunk.{3 * i}.bias"],
                        w[f"trunk.{3 * i + 1}.weight"], w[f"trunk.{3 * i + 1}.bias"])
                       for i in range(n)]
        self.wa, self.ba = w["actor.weight"], w["actor.bias"]

    def delta(self, obs, pi_tau):
        """Return the adjustment the network wants to make to the planner's proposal."""
        h = np.concatenate(
            [np.clip((np.asarray(obs, float) - self.mean) / self.std, -10, 10),
             np.log(pi_tau)], axis=-1)
        for W, b, g, beta in self.layers:
            h = h @ W.T + b
            mu = h.mean(axis=-1, keepdims=True)
            var = h.var(axis=-1, keepdims=True)
            h = np.tanh((h - mu) / np.sqrt(var + 1e-5) * g + beta)
        return self.squash * np.tanh(h @ self.wa.T + self.ba)

    def __call__(self, obs, pi_tau, est=None):
        return np.argmax(
            combine(pi_tau, self.delta(obs, pi_tau), self.lam, self.mode,
                    self.squash), axis=-1)


def spec(agent=None, kind=None, cutoff=True, **planner):
    """Describe one controller to fly.

    `kind` is "planner", "heuristic", "ppo" or "residual", inferred from `agent` when not
    given. `cutoff` controls whether the engines are cut once both legs are down, which
    only the planner does. Anything else is passed to the planner: iters, horizon,
    terminal, floor, replan, estimator, time_it. Returns a dictionary the flight worker
    understands.
    """
    if kind is None:
        kind = "planner" if agent is None else agent.get("kind", "residual")
    return {"kind": kind, "agent": agent, "cutoff": cutoff, "planner": planner}


_FLY = {}


def _fly_init(payload):
    built = {}
    for tag, s in payload.items():
        s = spec() if s is None else (s if isinstance(s, dict) and "kind" in s
                                     else spec(agent=s))
        pol = None
        if s["kind"] == "ppo":
            pol = PPOPolicy(s["agent"])
        elif s["kind"] == "residual":
            pol = ResidualPolicy(s["agent"])
            s = dict(s, planner=dict(s["planner"]))
            s["planner"].setdefault("floor", s["agent"].get("floor", PROB_FLOOR))
        built[tag] = (pol, s)
    _FLY["pol"] = built


def _fly_worker(job):
    tag, wind, seeds, chunk = job
    pol, s = _FLY["pol"][tag]
    kind = s["kind"]
    planner_needed = kind in ("planner", "residual")
    E = min(EVAL_BATCH, len(seeds))
    envs = [make_env(wind=wind) for _ in range(E)]
    base = (PlannerBase([_ROOT, 7, int(wind * 10), chunk], E, **s["planner"])
            if planner_needed else None)
    queue = list(seeds)
    obs = np.zeros((E, 8))
    live = np.zeros(E, bool)
    steps = np.zeros(E, np.int64)
    main = np.zeros(E, np.int64)
    side = np.zeros(E, np.int64)
    impulse = np.zeros(E)
    reward = np.zeros(E)
    mass = np.zeros(E)
    dv_min = np.zeros(E)
    ep_seed = np.zeros(E, np.int64)
    td = [None] * E

    def _start(i, seed):
        u = envs[i].unwrapped
        o = envs[i].reset(seed=int(seed))[0]
        mass[i] = u.lander.mass + sum(leg.mass for leg in u.legs)
        h = u.lander.position.y - (u.helipad_y + ll.LEG_DOWN / _S)
        v = u.lander.linearVelocity
        dv_min[i] = math.sqrt(v.x ** 2 + v.y ** 2
                              + 2 * abs(u.gravity) * max(h, 0.0))
        steps[i] = main[i] = side[i] = 0
        impulse[i] = reward[i] = 0.0
        ep_seed[i] = int(seed)
        td[i] = None
        return o

    for i in range(E):
        obs[i] = _start(i, queue.pop(0))
        live[i] = True
    if base is not None:
        base.reset(np.ones(E, bool))
    out = []
    while live.any():
        if kind == "planner":
            act = base.decide(obs)
            if s["cutoff"]:
                act = np.where((obs[:, 6] > 0) & (obs[:, 7] > 0), 0, act)
        elif kind == "residual":
            act = pol(obs, base.pi(obs, pol.tau))
        elif kind == "heuristic":
            act = heuristic_actions(*(obs[:, i] for i in range(6)),
                                    leg1=obs[:, 6], leg2=obs[:, 7])
        else:
            act = pol(obs)
        nxt_raw = obs.copy()
        nxt = obs.copy()
        restart = np.zeros(E, bool)
        for i in range(E):
            if not live[i]:
                continue
            o2, r, term, trunc, _ = envs[i].step(int(act[i]))
            nxt_raw[i] = nxt[i] = o2
            steps[i] += 1
            main[i] += act[i] == 2
            side[i] += act[i] in (1, 3)
            reward[i] += r
            d0, d1 = envs[i].unwrapped._disp_log[-2:]
            impulse[i] += engine_impulse(int(act[i]), d0, d1)
            if td[i] is None and (o2[6] or o2[7]):
                td[i] = (float(obs[i][2]), float(obs[i][3]), float(abs(obs[i][4])))
            if term or trunc:
                u = envs[i].unwrapped
                outcome = episode_outcome(u, term)
                on_pad = (outcome == "landed"
                          and u.helipad_x1 <= u.lander.position.x <= u.helipad_x2)
                t = td[i] or (float("nan"),) * 3
                out.append({
                    "seed": int(ep_seed[i]), "outcome": outcome,
                    "on_pad": int(bool(on_pad)), "reward": float(reward[i]),
                    "steps": int(steps[i]), "n_main": int(main[i]),
                    "n_side": int(side[i]),
                    "fuel_legacy": float(main[i] * FUEL_COST[2]
                                         + side[i] * FUEL_COST[1]),
                    "impulse": float(impulse[i]),
                    "dv": float(impulse[i] / mass[i]),
                    "dv_min": float(dv_min[i]),
                    "td_vx": t[0], "td_vy": t[1], "td_tilt": t[2]})
                restart[i] = True
                if queue:
                    nxt[i] = _start(i, queue.pop(0))
                else:
                    live[i] = False
        if base is not None:
            base.update(obs, act, nxt_raw, mask=live | restart)
            base.reset(restart)
        obs = nxt
        n_live = int(live.sum())
        if 0 < n_live <= len(live) // 2:
            idx = np.flatnonzero(live)
            for i in np.flatnonzero(~live):
                envs[i].close()
            envs = [envs[i] for i in idx]
            obs, live = obs[idx], live[idx]
            steps, main, side = steps[idx], main[idx], side[idx]
            impulse, reward, mass = impulse[idx], reward[idx], mass[idx]
            dv_min, ep_seed = dv_min[idx], ep_seed[idx]
            td = [td[i] for i in idx]
            if base is not None:
                base.compact(idx)
            E = len(idx)
    for e in envs:
        e.close()
    ms = list(base.planner.solve_ms) if base is not None else []
    return tag, wind, out, ms


def fly(payload, winds, episode_set, runs, per_run, workers, bar_label="flying",
        sink=None, skip=()):
    """Fly a set of controllers over the same episodes at every wind strength.

    `payload` maps a name to a trained agent, or to None for the plain planner. Episodes
    come from the named set, `runs` times `per_run` of them. Every controller sees
    exactly the same episodes, so any comparison between them is paired.

    `sink`, if given, is called with each finished chunk of rows as it arrives, so a long
    run is written to disk as it goes instead of held in memory. `skip` lists
    (controller, wind, run) triples already done, which is what makes a restart cheap.
    Returns the rows when there is no sink, and an empty list when there is.
    """
    jobs = []
    for run in range(runs):
        seeds = episode_seeds(episode_set, run, per_run)
        size = max(EVAL_BATCH, -(-len(seeds) // max(1, (2 * workers)
                                                    // max(1, len(payload) * len(winds)))))
        for tag in payload:
            for w in winds:
                if (tag, w, run) in skip:
                    continue
                for i in range(0, len(seeds), size):
                    jobs.append((tag, w, seeds[i:i + size], run * 1000 + i // size))
    if not jobs:
        print("nothing left to fly")
        return []
    rows, timing = [], []
    bar = Progress(len(jobs), bar_label, every=max(1, len(jobs) // 60), unit="chunks")
    with mp.Pool(min(workers, max(1, len(jobs))), initializer=_fly_init,
                 initargs=(payload,), maxtasksperchild=40) as pool:
        for tag, wind, res, ms in pool.imap_unordered(_fly_worker, jobs):
            for r in res:
                r["controller"] = tag
                r["wind"] = wind
            timing.extend(ms)
            if sink is not None:
                sink(res)
            else:
                rows.extend(res)
            bar.advance(note=f"{tag} wind {wind:g}")
    bar.close()
    if timing:
        t = np.array(timing)
        print(f"  planner solve time per environment step: mean {t.mean():.2f} ms, "
              f"p99 {np.percentile(t, 99):.2f} ms, max {t.max():.2f} ms "
              f"(control period {1000 / FPS:.0f} ms)")
    return rows


def wilson(k, n, z=1.96):
    """Return a 95 per cent confidence interval for a proportion, as percentages."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * max(0.0, centre - half), 100 * min(1.0, centre + half)


def mcnemar(b, c):
    """Return the two-sided exact p-value for a paired win/loss count."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, k) * 0.5 ** n for k in range(min(b, c) + 1)))


def ppo_jobs():
    """The list of standalone agents to train: five shaping functions, several seeds each."""
    return [(f"PPO F{k} s{i}", k, i, {}) for k in range(1, 6)
            for i in range(PPO_SEEDS)]


def residual_jobs():
    """The list of residual agents to train: five settings of lam, several seeds each."""
    return [(f"RES lam{lam:g} s{i}", lam, i, {}) for lam in RESIDUAL_LAMBDAS
            for i in range(RESIDUAL_SEEDS)]


def controller_payload():
    """Load every controller to be compared, ready to fly.

    That is the environment's built-in heuristic, the planner on its own, every trained
    standalone agent and every trained residual agent. Returns a dictionary from name to
    a description the flight worker understands.
    """
    import torch
    payload = {"Heuristic": spec(kind="heuristic"), "MPC": spec()}
    for bundle in (PPO_BUNDLE, RESIDUAL_BUNDLE):
        if os.path.exists(bundle):
            for tag, a in torch.load(bundle, weights_only=False).items():
                payload[tag] = spec(agent=a)
    return payload


CSV_COLUMNS = ["controller", "wind", "seed", "outcome", "on_pad", "reward", "steps",
               "n_main", "n_side", "fuel_legacy", "impulse", "dv", "dv_min",
               "td_vx", "td_vy", "td_tilt"]

_EVAL_SEED_SETS = [set(episode_seeds("eval", r, EVAL_PER_RUN)) for r in range(EVAL_RUNS)]


PLANNER_RUNS = 10


def extend_planner(workers):
    """Fly the plain planner over more held-out episodes, to sharpen the bar it sets.

    The planner holds the bar at the low wind strengths, and unlike a learned cell it has
    no training seeds to average over, so its only precision comes from episode count. The
    learned cells each pool several thousand episodes; this brings the planner to the same
    order. The first EVAL_RUNS runs are the episodes every controller flew, so paired
    comparisons remain available on that subset; the extra runs only tighten the interval.
    Appends to the results CSV.
    """
    seen = set()
    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, newline="") as fh:
            for r in csv.DictReader(fh):
                if r["controller"] == "MPC":
                    seen.add(int(r["seed"]))
    todo = [run for run in range(PLANNER_RUNS)
            if not set(episode_seeds("eval", run, EVAL_PER_RUN)) <= seen]
    if not todo:
        print(f"planner already flown over {PLANNER_RUNS} runs")
        return
    print(f"planner: {len(todo)} more runs of {EVAL_PER_RUN} episodes at each wind",
          flush=True)
    fh = open(RESULTS_CSV, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    written = [0]

    def sink(rows):
        for r in rows:
            w.writerow(r)
        written[0] += len(rows)
        fh.flush()

    jobs = []
    for run in todo:
        seeds = episode_seeds("eval", run, EVAL_PER_RUN)
        for wind in WINDS:
            for i in range(0, len(seeds), EVAL_BATCH):
                jobs.append(("MPC", wind, seeds[i:i + EVAL_BATCH],
                             run * 1000 + i // EVAL_BATCH))
    bar = Progress(len(jobs), "planner", every=max(1, len(jobs) // 30), unit="chunks")
    try:
        with mp.Pool(min(workers, len(jobs)), initializer=_fly_init,
                     initargs=({"MPC": None},), maxtasksperchild=40) as pool:
            for tag, wind, res, _ in pool.imap_unordered(_fly_worker, jobs):
                for r in res:
                    r["controller"] = tag
                    r["wind"] = wind
                sink(res)
                bar.advance(note=f"wind {wind:g}")
    finally:
        fh.close()
    bar.close()
    print(f"{written[0]:,} planner rows added")
    write_spread()


def _done_already():
    """Return the (controller, wind, run) triples the results CSV already covers.

    A run is complete for a triple when it holds the full number of episodes for it, so a
    partly written chunk is redone instead of trusted.
    """
    if not os.path.exists(RESULTS_CSV):
        return set(), 0
    counts = {}
    n = 0
    with open(RESULTS_CSV, newline="") as fh:
        for r in csv.DictReader(fh):
            n += 1
            for run in range(EVAL_RUNS):
                if int(r["seed"]) in _EVAL_SEED_SETS[run]:
                    key = (r["controller"], float(r["wind"]), run)
                    counts[key] = counts.get(key, 0) + 1
                    break
    return {k for k, c in counts.items() if c >= EVAL_PER_RUN}, n


def measure(workers):
    """Fly every controller over the held-out episodes and write one row per episode.

    Appends to RESULTS_CSV as each chunk finishes, so an interrupted run loses at most
    one chunk and restarting skips whatever is already complete. Then writes SPREAD_CSV,
    one row per controller, wind and training seed. Returns nothing.
    """
    payload = controller_payload()
    skip, have = _done_already()
    fresh = not os.path.exists(RESULTS_CSV)
    print(f"{len(payload)} controllers x {len(WINDS)} winds x "
          f"{EVAL_RUNS * EVAL_PER_RUN} held-out episodes", flush=True)
    if skip:
        print(f"resuming: {len(skip)} controller-wind-run blocks already done "
              f"({have:,} rows)", flush=True)
    fh = open(RESULTS_CSV, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    if fresh:
        w.writeheader()
    written = [0]

    def sink(rows):
        for r in rows:
            w.writerow(r)
        written[0] += len(rows)
        fh.flush()

    try:
        fly(payload, WINDS, "eval", EVAL_RUNS, EVAL_PER_RUN, workers, sink=sink,
            skip=skip)
    finally:
        fh.close()
    print(f"{written[0]:,} rows added -> {os.path.basename(RESULTS_CSV)}")
    write_spread()


def cell_of(controller):
    """Group a controller name into the cell it belongs to, dropping the seed number."""
    return controller.rsplit(" s", 1)[0] if " s" in controller else controller


def load_rows(path=None):
    """Read a results CSV back, with numbers converted. Returns a list of dictionaries."""
    out = []
    with open(path or RESULTS_CSV, newline="") as fh:
        for r in csv.DictReader(fh):
            r["wind"] = float(r["wind"])
            for k in ("reward", "dv", "dv_min", "impulse", "fuel_legacy"):
                r[k] = float(r[k])
            for k in ("steps", "n_main", "n_side", "on_pad"):
                r[k] = int(r[k])
            out.append(r)
    return out


def write_spread():
    """Write one row per cell, wind and training seed, for computing spreads and errors.

    Columns are the cell name, the wind, the seed number, how many episodes it flew, its
    landing rate and its mean fuel. This is the file to use for standard deviations
    across training seeds.
    """
    rows = load_rows()
    agg = {}
    for r in rows:
        key = (cell_of(r["controller"]), r["wind"], r["controller"])
        a = agg.setdefault(key, {"n": 0, "landed": 0, "dv": []})
        a["n"] += 1
        if r["outcome"] == "landed":
            a["landed"] += 1
            a["dv"].append(r["dv"])
    with open(SPREAD_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cell", "wind", "agent", "episodes", "landed", "landed_pct", "dv"])
        for (cell, wind, agent), a in sorted(agg.items()):
            w.writerow([cell, f"{wind:g}", agent, a["n"], a["landed"],
                        f"{100 * a['landed'] / a['n']:.4f}",
                        f"{np.mean(a['dv']):.4f}" if a["dv"] else ""])
    print(f"per-seed spread -> {os.path.basename(SPREAD_CSV)}")


def report(path=None):
    """Print the landing-rate and fuel tables from a results CSV.

    The bar at each wind is the best landing rate any non-residual controller achieves
    there. Every figure carries a spread across training seeds so the reader can see how
    much of a difference is real.
    """
    rows = load_rows(path)
    cells = {}
    for r in rows:
        c = cell_of(r["controller"])
        d = cells.setdefault(c, {})
        a = d.setdefault(r["wind"], {"n": 0, "landed": 0, "dv": [], "per": {}})
        a["n"] += 1
        p = a["per"].setdefault(r["controller"], [0, 0])
        p[1] += 1
        if r["outcome"] == "landed":
            a["landed"] += 1
            p[0] += 1
            a["dv"].append(r["dv"])
    names = sorted(cells)
    pct = lambda c, w: 100 * cells[c][w]["landed"] / cells[c][w]["n"]
    dv = lambda c, w: float(np.mean(cells[c][w]["dv"])) if cells[c][w]["dv"] else float("nan")

    def seed_sd(c, w):
        v = [100 * k / n for k, n in cells[c][w]["per"].values() if n]
        return float(np.std(v, ddof=1)) if len(v) > 1 else 0.0

    baseline = [c for c in names if not c.startswith("RES")]
    residual = [c for c in names if c.startswith("RES")]
    bar = {w: max(pct(c, w) for c in baseline) for w in WINDS}
    holder = {w: max(baseline, key=lambda c: pct(c, w)) for w in WINDS}

    print("\nLANDING RATE (%), held-out episodes, one action per step chosen greedily")
    print("| Controller | " + " | ".join(f"{w:g}" for w in WINDS) + " | worst margin |")
    print("|---" * (len(WINDS) + 2) + "|")
    for c in baseline:
        print(f"| {c} | " + " | ".join(f"{pct(c, w):.2f}" for w in WINDS) + " | |")
    print("| **the bar** | " + " | ".join(f"**{bar[w]:.2f}**" for w in WINDS) + " | |")
    print("| *held by* | " + " | ".join(holder[w] for w in WINDS) + " | |")
    for c in sorted(residual, key=lambda c: -min(pct(c, w) - bar[w] for w in WINDS)):
        worst = min(pct(c, w) - bar[w] for w in WINDS)
        print(f"| {c} | "
              + " | ".join(f"{pct(c, w):.2f} ({pct(c, w) - bar[w]:+.2f})" for w in WINDS)
              + f" | {worst:+.2f} |")

    print("\nSPREAD ACROSS TRAINING SEEDS (standard deviation of landing rate)")
    print("| Controller | " + " | ".join(f"{w:g}" for w in WINDS) + " |")
    print("|---" * (len(WINDS) + 1) + "|")
    for c in names:
        print(f"| {c} | " + " | ".join(f"{seed_sd(c, w):.2f}" for w in WINDS) + " |")

    print("\nFUEL: speed change spent per landing (lower is better)")
    print("| Controller | " + " | ".join(f"{w:g}" for w in WINDS) + " | mean |")
    print("|---" * (len(WINDS) + 2) + "|")
    for c in names:
        m = float(np.mean([dv(c, w) for w in WINDS]))
        print(f"| {c} | " + " | ".join(f"{dv(c, w):.2f}" for w in WINDS) + f" | {m:.2f} |")


MATCHED_CSV = os.path.join(HERE, "matched_fuel.csv")
MATCHED_SHARE = 0.8


def matched_fuel(path=None, share=MATCHED_SHARE):
    """Compare fuel three ways and write matched_fuel.csv.

    Fuel is only defined for an episode that was landed, so a plain average over successes
    charges a controller for the hard episodes it managed to land and its rivals crashed.
    That makes the plain average unusable for deciding whether one controller is thirstier
    than another.

    Three views are written. "mean over landed" is the plain average of the episodes a
    controller landed. "matched" restricts to episodes the planner landed and that both the
    standalone agent and the residual cell landed on at least `share` of their training
    seeds, so all three are averaged over the same episodes. "per success" divides the total
    speed change spent across every attempt by the number of successes, which charges a
    controller for fuel burnt on episodes it went on to crash.

    Takes a results CSV and returns nothing. Prints a table and writes one row per cell,
    wind, view and controller.
    """
    rows = load_rows(path)
    by = {}
    for r in rows:
        by.setdefault((cell_of(r["controller"]), r["wind"]), []).append(r)
    cells = sorted({c for c, _ in by})
    residual = [c for c in cells if c.startswith("RES")]
    learned = [c for c in cells if not c.startswith("RES") and c not in ("MPC", "Heuristic")]
    plan = "MPC"
    best = max(learned, key=lambda c: min(
        100 * sum(x["outcome"] == "landed" for x in by[(c, w)]) / len(by[(c, w)])
        for w in WINDS))

    def share_landed(c, w):
        num, den = {}, {}
        for x in by[(c, w)]:
            s = int(x["seed"])
            den[s] = den.get(s, 0) + 1
            num[s] = num.get(s, 0) + (x["outcome"] == "landed")
        return {s: num[s] / den[s] for s in den}

    frac = {(c, w): share_landed(c, w) for c in cells for w in WINDS}

    def mean_dv(c, w, keep=None):
        d = [x["dv"] for x in by[(c, w)] if x["outcome"] == "landed"
             and (keep is None or int(x["seed"]) in keep)]
        return float(np.mean(d)) if d else float("nan")

    def per_success(c, w):
        v = by[(c, w)]
        k = sum(x["outcome"] == "landed" for x in v)
        return float(sum(x["dv"] for x in v) / k) if k else float("nan")

    out = []
    for c in cells:
        for w in WINDS:
            n = len(by[(c, w)])
            out.append([c, w, "mean over landed", c, n, mean_dv(c, w)])
            out.append([c, w, "per success", c, n, per_success(c, w)])
    summary = {}
    for c in residual:
        kept, trio = {}, {}
        for w in WINDS:
            keep = {s for s, f in frac[(plan, w)].items() if f >= 1.0
                    and frac[(best, w)].get(s, 0.0) >= share
                    and frac[(c, w)].get(s, 0.0) >= share}
            kept[w] = keep
            for other in (plan, best, c):
                trio[(other, w)] = mean_dv(other, w, keep)
                out.append([c, w, "matched", other, len(keep), trio[(other, w)]])
        wt = np.array([len(kept[w]) for w in WINDS], float)
        summary[c] = {other: float(np.average([trio[(other, w)] for w in WINDS], weights=wt))
                      for other in (plan, best, c)}
        summary[c]["kept"] = wt

    with open(MATCHED_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cell", "wind", "view", "controller", "episodes", "dv"])
        for r in out:
            w.writerow([r[0], f"{r[1]:g}", r[2], r[3], r[4],
                        "" if r[5] != r[5] else f"{r[5]:.4f}"])

    print(f"\nFUEL ON MATCHED EPISODES (planner {plan}, best standalone {best}, "
          f"episodes landed on at least {share:.0%} of seeds)")
    print("| Residual cell | episodes kept per wind | planner | standalone | residual "
          "| vs planner | vs standalone |")
    print("|---" * 7 + "|")
    for c in residual:
        s = summary[c]
        print(f"| {c} | " + " ".join(f"{int(n)}" for n in s["kept"])
              + f" | {s[plan]:.2f} | {s[best]:.2f} | {s[c]:.2f} "
              f"| {s[c] - s[plan]:+.2f} | {s[c] - s[best]:+.2f} |")
    print(f"\nwritten to {os.path.basename(MATCHED_CSV)}")


FINAL_CSV = os.path.join(HERE, "final_confirmation.csv")
FINAL_EPISODES = 300


def final_check(workers):
    """Measure the reported residual cell once, on episodes nothing has ever touched.

    The five values of lam were compared on the `eval` episodes, so the strongest of them
    is a best-of-five on the same episodes its margins are quoted against. This flies that
    cell, and the two baselines that set the bar, over the `final` episode set, which no
    training, no development and no earlier measurement has used. It is one measurement of
    one cell and is not repeated. Writes final_confirmation.csv.
    """
    import torch
    payload = {"Heuristic": spec(kind="heuristic"), "MPC": spec()}
    for bundle in (PPO_BUNDLE, RESIDUAL_BUNDLE):
        if not os.path.exists(bundle):
            continue
        for tag, a in torch.load(bundle, weights_only=False).items():
            keep = (a.get("kind") == "ppo"
                    or (a.get("lam") == RESIDUAL_LAMBDA))
            if keep:
                payload[tag] = spec(agent=a)
    print(f"{len(payload)} controllers on the `final` episodes, "
          f"{FINAL_EPISODES} per wind", flush=True)
    rows = fly(payload, WINDS, "final", 1, FINAL_EPISODES, workers,
               bar_label="final confirmation")
    with open(FINAL_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["controller"], r["wind"], r["seed"])):
            w.writerow(r)
    print(f"{len(rows):,} rows -> {os.path.basename(FINAL_CSV)}")
    report(FINAL_CSV)


ABLATION_CSV = os.path.join(HERE, "ablations.csv")
ABLATION_EPISODES = 200


def ablation_arms():
    """The controller variants behind the design questions the review asked about.

    Each entry is a name and a description the flight worker understands. The planner
    arms vary one setting at a time from the reported controller; the residual arms need
    their own trained agents and are handled by --train-ablations.
    """
    arms = {"planner as reported": spec(time_it=True)}
    for it in (1, 2, 6, 8):
        arms[f"planner iterations {it}"] = spec(iters=it)
    for h in (24, 72):
        arms[f"planner horizon {h}"] = spec(horizon=h)
    for r in (2, 6):
        arms[f"planner replan every {r}"] = spec(replan=r)
    arms["planner without engine cut-off"] = spec(cutoff=False)
    arms["planner without terminal cost"] = spec(terminal=False)
    arms["planner without wind estimator"] = spec(estimator=False)
    return arms


def ablation_jobs():
    """The residual agents needed for the blend-rule and probability-floor questions."""
    jobs = []
    for i in range(ABLATION_SEEDS):
        jobs.append((f"ABL add s{i}", RESIDUAL_LAMBDA, i, {"mode": "add"}))
        jobs.append((f"ABL floor0.02 s{i}", RESIDUAL_LAMBDA, i, {"floor": 0.02}))
    return jobs


def ablations(workers):
    """Fly every ablation arm and write ablations.csv.

    Covers the planner's iteration count, look-ahead length, replanning cadence, engine
    cut-off, terminal cost and wind estimator, plus the residual blend rule and
    probability floor when those agents have been trained. One row per arm and wind, with
    landing rate and fuel. Also reports the planner's solve time against the control
    period.
    """
    import torch
    payload = dict(ablation_arms())
    if os.path.exists(RESIDUAL_BUNDLE):
        res = torch.load(RESIDUAL_BUNDLE, weights_only=False)
        ref = [a for t, a in res.items()
               if a["lam"] == RESIDUAL_LAMBDA and a["seed_idx"] < ABLATION_SEEDS]
        for a in ref:
            payload[f"residual as reported s{a['seed_idx']}"] = spec(agent=a)
    if os.path.exists(ABLATION_BUNDLE):
        for tag, a in torch.load(ABLATION_BUNDLE, weights_only=False).items():
            payload[f"residual {tag[4:]}"] = spec(agent=a)
    print(f"{len(payload)} arms x {len(WINDS)} winds x {ABLATION_EPISODES} episodes",
          flush=True)
    rows = fly(payload, WINDS, "eval", 1, ABLATION_EPISODES, workers,
               bar_label="ablations")
    agg = {}
    for r in rows:
        key = (cell_of(r["controller"]), r["wind"])
        a = agg.setdefault(key, {"n": 0, "landed": 0, "dv": []})
        a["n"] += 1
        if r["outcome"] == "landed":
            a["landed"] += 1
            a["dv"].append(r["dv"])
    with open(ABLATION_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "wind", "episodes", "landed", "landed_pct", "dv"])
        for (arm, wind), a in sorted(agg.items()):
            w.writerow([arm, f"{wind:g}", a["n"], a["landed"],
                        f"{100 * a['landed'] / a['n']:.2f}",
                        f"{np.mean(a['dv']):.2f}" if a["dv"] else ""])
    print(f"\n{'arm':34} " + " ".join(f"{w:g}".rjust(7) for w in WINDS) + "   mean dv")
    for arm in sorted({k for k, _ in agg}):
        pcts = [100 * agg[(arm, w)]["landed"] / agg[(arm, w)]["n"] for w in WINDS]
        dvs = [np.mean(agg[(arm, w)]["dv"]) for w in WINDS if agg[(arm, w)]["dv"]]
        print(f"{arm:34} " + " ".join(f"{p:7.2f}" for p in pcts)
              + f"   {np.mean(dvs):7.2f}" if dvs else "")
    print(f"\nwritten to {os.path.basename(ABLATION_CSV)}")


def watch(n, wind, lam):
    """Open a window and watch the residual controller land n times."""
    import torch
    bundle = torch.load(RESIDUAL_BUNDLE, weights_only=False)
    tag = next(t for t in sorted(bundle) if t.startswith(f"RES lam{lam:g} "))
    pol = ResidualPolicy(bundle[tag])
    env = make_env(render_mode="human", wind=wind)
    base = PlannerBase([_ROOT, 7, 0], 1)
    for s in episode_seeds("eval", 0, n):
        obs = env.reset(seed=int(s))[0][None, :].astype(np.float64)
        base.reset(np.ones(1, bool))
        while True:
            pi_tau = base.pi(obs, pol.tau)
            a = int(pol(obs, pi_tau)[0])
            o2, _, term, trunc, _ = env.step(a)
            nxt = o2[None, :].astype(np.float64)
            base.update(obs, np.array([a]), nxt)
            obs = nxt
            if term or trunc:
                print(f"seed {s}: {episode_outcome(env.unwrapped, term)}")
                break
    env.close()


def sysid(n_flights, workers):
    """Refit the lander model from fresh flight data and print how well it predicts."""
    seeds = episode_seeds("sysid", 0, n_flights)
    half = len(seeds) // 2
    O, A, O2, E = collect_flights(seeds[:half])
    params, used = identify(O, A, O2)
    print(f"fitted on {used:,} in-flight steps from {half} flights")
    print(json.dumps({k: [round(x, 6) for x in v] for k, v in params.items()}, indent=2))
    Ov, Av, O2v, Ev = collect_flights(seeds[half:], rng_seed=1)
    stats = validate_model(LanderModel(params), Ov, Av, O2v, Ev)
    print("\nheld-out flights:")
    for k, v in stats.items():
        print(f"  {k}: {v if not isinstance(v, tuple) else tuple(round(x, 4) for x in v)}")


def check(workers):
    """Run the self-tests: episode sets, planner sanity, shaping sanity, policy agreement."""
    check_episode_sets()
    m = LanderModel()
    p = Planner(m, np.random.default_rng(0), 2)
    O = np.tile(np.array([0.0, 1.3, 0.0, -0.4, 0.0, 0.0, 0.0, 0.0]), (2, 1))
    seq, pi = p.plan(O, np.zeros((2, 3)))
    assert seq.shape == (2, PLAN_HORIZON)
    assert np.allclose(pi.sum(axis=1), 1.0)
    assert (pi > 0).all()
    print(f"planner: shapes fine, action probabilities sum to 1, "
          f"smallest probability {pi.min():.3f}")
    rest = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    fast = np.array([0.0, 0.05, 0.0, -0.6, 0.1, 0.0, 0.0, 0.0])
    assert residual_risk(rest) == 0.0
    assert residual_risk(fast) > 0.0
    print(f"shaping: risk is 0 at rest, {residual_risk(fast):.3f} when fast and low")
    pi_tau = sharpen(pi, RESIDUAL_TAU)
    q = combine(pi_tau, np.zeros((2, 4)), RESIDUAL_LAMBDA)
    assert np.allclose(q, pi_tau)
    print("residual: a zero adjustment leaves the planner's choice untouched")
    streams = {}
    for replan in (1, 2, 6):
        env = make_env(wind=15.0)
        b = PlannerBase([_ROOT, 7, 150, 0], 1, replan=replan)
        o = env.reset(seed=int(episode_seeds("tune", 0, 1)[0]))[0]
        b.reset(np.ones(1, bool))
        acts = []
        for _ in range(40):
            a = int(b.decide(o[None])[0])
            acts.append(a)
            o, _, term, trunc, _ = env.step(a)
            b.reset(np.zeros(1, bool))
            if term or trunc:
                break
        env.close()
        streams[replan] = acts
    assert streams[1] != streams[2] and streams[1] != streams[6], (
        "replanning cadence has no effect on the actions flown", streams)
    print("planner cadence: re-solving every 1, 2 and 6 steps fly different actions")
    if os.path.exists(PPO_BUNDLE) and os.path.exists(RESIDUAL_BUNDLE):
        import torch
        steps = {}
        for bundle, name in ((PPO_BUNDLE, "standalone"), (RESIDUAL_BUNDLE, "residual")):
            b = torch.load(bundle, weights_only=False)
            for tag, a in b.items():
                assert a["curve"], f"{tag} has no learning curve"
                steps.setdefault(name, set()).add(a["curve"][-1]["steps"])
            print(f"bundles: {len(b)} {name} agents, "
                  f"{sorted(steps[name])} steps each")
        allsteps = steps["standalone"] | steps["residual"]
        assert len(allsteps) == 1, f"training budgets differ: {sorted(allsteps)}"
        print(f"budgets: both families trained for exactly "
              f"{allsteps.pop():,} steps per agent")
        res = torch.load(RESIDUAL_BUNDLE, weights_only=False)
        missing = [t for t, a in res.items() if not a.get("hp")]
        print(f"settings recorded on {len(res) - len(missing)} of {len(res)} "
              f"residual agents")
    if os.path.exists(RESULTS_CSV):
        rows = load_rows()
        seeds = {r["seed"] for r in rows}
        evals = set(episode_seeds("eval", 0, EVAL_PER_RUN))
        tunes = {s for r in range(EVAL_RUNS) for s in episode_seeds("tune", r, 500)}
        assert not (seeds & tunes), "held-out episodes must not appear in the tuning set"
        print(f"results: {len(rows):,} rows, {len(seeds)} distinct episodes, "
              f"none shared with the tuning set")
    print("\nall checks passed")



def refly_floor(workers):
    """Re-fly the probability-floor ablation arm on its own.

    The floor-0.02 agents were trained against a 0.02 floor but originally flown against
    the 0.15 one, because a saved agent's settings were not reaching the planner it flies
    on. That defect is fixed above (the planner now inherits the agent's floor), so this
    re-flies only that arm instead of the whole ablation sweep. Same protocol as
    ablations(): five seeds, 200 episodes per seed per wind, on the eval episode set.
    """
    import torch
    if not os.path.exists(ABLATION_BUNDLE):
        sys.exit(f"missing {ABLATION_BUNDLE}; train the ablation agents first")

    payload = {}
    for tag, agent in torch.load(ABLATION_BUNDLE, weights_only=False).items():
        if tag.startswith("ABL floor0.02"):
            payload[f"residual {tag[4:]}"] = spec(agent=agent)
    if not payload:
        sys.exit("no 'ABL floor0.02' agents found in the bundle")

    print(f"{len(payload)} seeds x {len(WINDS)} winds x {ABLATION_EPISODES} episodes"
          f"  ({len(payload) * len(WINDS) * ABLATION_EPISODES} episodes total)", flush=True)

    rows = fly(payload, WINDS, "eval", 1, ABLATION_EPISODES, workers,
               bar_label="floor re-flight")

    agg = {}
    for r in rows:
        g = agg.setdefault((cell_of(r["controller"]), r["wind"]),
                           {"n": 0, "landed": 0, "dv": []})
        g["n"] += 1
        if r["outcome"] == "landed":
            g["landed"] += 1
            g["dv"].append(r["dv"])

    out = os.path.join(HERE, "ablations_floor_reflown.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "wind", "episodes", "landed", "landed_pct", "dv"])
        for (arm, wind), g in sorted(agg.items()):
            w.writerow([arm, f"{wind:g}", g["n"], g["landed"],
                        f"{100 * g['landed'] / g['n']:.2f}",
                        f"{np.mean(g['dv']):.2f}" if g["dv"] else ""])

    print(f"\nwritten to {os.path.basename(out)}\n")
    for arm in sorted({k for k, _ in agg}):
        pcts = [100 * agg[(arm, w)]["landed"] / agg[(arm, w)]["n"] for w in WINDS]
        print(f"{arm:30} " + ", ".join(f"{p:.2f}" for p in pcts))
    print("\nthe old confounded row was: 83.90, 81.00, 73.50, 54.10, 33.80")
    print("the reported controller is:  99.10, 99.00, 97.40, 90.80, 71.60")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--sysid", action="store_true")
    ap.add_argument("--flights", type=int, default=200)
    ap.add_argument("--watch", type=int, default=0)
    ap.add_argument("--wind", type=float, default=0.0)
    ap.add_argument("--lam", type=float, default=RESIDUAL_LAMBDA)
    ap.add_argument("--train-ppo", action="store_true")
    ap.add_argument("--train-residual", action="store_true")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--extend-planner", action="store_true")
    ap.add_argument("--train-ablations", action="store_true")
    ap.add_argument("--ablations", action="store_true")
    ap.add_argument("--refly-floor", action="store_true")
    ap.add_argument("--final-check", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--spread", action="store_true")
    ap.add_argument("--matched-fuel", action="store_true")
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args()
    if a.check:
        return check(a.workers)
    if a.sysid:
        return sysid(a.flights, a.workers)
    if a.watch:
        return watch(a.watch, a.wind, a.lam)
    if a.train_ppo:
        return train_many(ppo_jobs(), ppo_train, PPO_BUNDLE, a.workers,
                          "standalone agents")
    if a.train_residual:
        return train_many(residual_jobs(), residual_train, RESIDUAL_BUNDLE, a.workers,
                          "residual agents")
    if a.measure:
        return measure(a.workers)
    if a.extend_planner:
        return extend_planner(a.workers)
    if a.train_ablations:
        return train_many(ablation_jobs(), residual_train, ABLATION_BUNDLE, a.workers,
                          "ablation agents")
    if a.ablations:
        return ablations(a.workers)
    if a.refly_floor:
        return refly_floor(a.workers)
    if a.final_check:
        return final_check(a.workers)
    if a.spread:
        return write_spread()
    if a.matched_fuel:
        return matched_fuel()
    if a.report:
        return report()
    ap.print_help()


if __name__ == "__main__":
    main()
