import sys
import time
import os
import collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import gymnasium.envs.box2d.lunar_lander as lunar_lander_module

output_dir = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

plt.rcParams.update({
    'font.family': 'serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.35,
    'grid.color': '#cccccc',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

velocity_damping = 0.9005
gravity = -0.027
main_thrust_y = 0.0646
side_thrust_x = 0.024
side_torque = 0.040

mpc_num_episodes = 30
ppo_total_steps = 150_000
residual_total_steps = 150_000

ppo_parallel_envs = 16
ppo_rollout_length = 128
ppo_minibatch_size = 256
ppo_learning_rate = 3e-3
ppo_clip_ratio = 0.30
ppo_entropy_coef = 0.05

residual_parallel_envs = 16
residual_rollout_length = 128
residual_minibatch_size = 256
residual_learning_rate = 2e-3
residual_clip_ratio = 0.25
residual_entropy_coef = 0.02
fuel_penalty_coef = 0.05
cem_replan_every = 2
kl_penalty_coef = 0.01
mpc_prob_temperature = 1.5

discount_factor = 0.99
gae_lambda = 0.95
value_loss_coef = 0.5
max_grad_norm = 0.5
update_epochs = 4

residual_scales = [0.3, 1.0, 2.5]

slate = '#2c5f8a'
gray = '#888888'
muted_blue = '#4a7fb5'
muted_red = '#c0392b'
muted_orange = '#d4823a'
muted_green = '#4a9e6b'
muted_purple = '#7b5ea7'

function_colors = [muted_blue, muted_red, muted_orange, muted_green, muted_purple]
scale_linewidths = {0.3: 1.2, 1.0: 2.2, 2.5: 3.5}


class ZeroDispersionRNG:
    '''Wraps numpy RNG to suppress stochastic wind dispersion in LunarLander.
    Args: rng — numpy RandomState from the environment.
    Returns: wrapped RNG that returns 0.0 for the dispersion uniform draws.'''

    __slots__ = ('_rng',)

    def __init__(self, rng):
        '''Wraps the provided numpy RNG.
        Args: rng — numpy RandomState instance from the environment.
        Returns: None.'''
        self._rng = rng

    def uniform(self, low=-1.0, high=1.0, size=None):
        '''Returns 0.0 for the dispersion draw, delegates everything else.'''
        if size is None and low == -1.0 and high == 1.0:
            return 0.0
        return self._rng.uniform(low, high, size)

    def __getattr__(self, name):
        '''Delegates all other attribute lookups to the wrapped RNG.
        Args: name — attribute name string.
        Returns: attribute from the underlying RNG.'''
        return getattr(self._rng, name)


def patched_step(self, action):
    '''LunarLander.step replacement that zeroes wind dispersion.
    Args: action — integer action.
    Returns: standard gymnasium step tuple.'''
    real_rng = self.np_random
    self.np_random = ZeroDispersionRNG(real_rng)
    result = original_step(self, action)
    self.np_random = real_rng
    return result


def patched_reset(self, **kwargs):
    '''LunarLander.reset replacement that wraps RNG after reset.
    Args: kwargs — passed through to original reset.
    Returns: standard gymnasium reset tuple.'''
    result = original_reset(self, **kwargs)
    self.np_random = ZeroDispersionRNG(self.np_random)
    return result


original_step = lunar_lander_module.LunarLander.step
original_reset = lunar_lander_module.LunarLander.reset
lunar_lander_module.LunarLander.step = patched_step
lunar_lander_module.LunarLander.reset = patched_reset


def step_physics(states, actions):
    '''Vectorised one-step physics model for CEM rollouts.
    Args: states — (N, 8) float32 array of lander states. actions — (N,) integer actions.
    Returns: (N, 8) float32 next states.'''
    x_pos = states[:, 0]
    y_pos = states[:, 1]
    x_vel = states[:, 2]
    y_vel = states[:, 3]
    tilt_angle = states[:, 4]
    spin_rate = states[:, 5]

    sine = np.sin(tilt_angle)
    cosine = np.cos(tilt_angle)
    main_on = (actions == 2).astype(np.float32)
    side_direction = (actions == 1).astype(np.float32) - (actions == 3).astype(np.float32)

    next_x_vel = velocity_damping * x_vel + main_on * (-sine * main_thrust_y) + side_direction * side_thrust_x * cosine
    next_y_vel = velocity_damping * y_vel + gravity + main_on * (cosine * main_thrust_y) + side_direction * side_thrust_x * sine
    next_spin = velocity_damping * spin_rate + side_direction * side_torque
    next_x = x_pos + next_x_vel
    next_y = y_pos + next_y_vel
    next_angle = tilt_angle + next_spin
    leg_contact = ((next_y <= 0.05) & (np.abs(next_x) < 0.3)).astype(np.float32)

    return np.stack(
        [next_x, next_y, next_x_vel, next_y_vel, next_angle, next_spin, leg_contact, leg_contact],
        axis=1
    ).astype(np.float32)


def score_action_sequences(initial_states, action_sequences, horizon):
    '''Scores batched action sequences under the surrogate physics model.
    Args: initial_states — (N, 8) states. action_sequences — (N, horizon) integer actions. horizon — planning horizon.
    Returns: (N,) float64 total rewards.'''
    states = initial_states
    prev_shaping = (
        -100.0 * np.sqrt(states[:, 0] ** 2 + states[:, 1] ** 2)
        - 100.0 * np.sqrt(states[:, 2] ** 2 + states[:, 3] ** 2)
        - 100.0 * np.abs(states[:, 4])
        + 10.0 * states[:, 6]
        + 10.0 * states[:, 7]
    )
    total_reward = np.zeros(len(states), dtype=np.float64)

    for timestep in range(horizon):
        step_actions = action_sequences[:, timestep]
        states = step_physics(states, step_actions)

        shaping = (
            -100.0 * np.sqrt(states[:, 0] ** 2 + states[:, 1] ** 2)
            - 100.0 * np.sqrt(states[:, 2] ** 2 + states[:, 3] ** 2)
            - 100.0 * np.abs(states[:, 4])
            + 10.0 * states[:, 6]
            + 10.0 * states[:, 7]
        )
        reward = shaping - prev_shaping
        prev_shaping = shaping

        main_fired = step_actions == 2
        side_fired = (step_actions == 1) | (step_actions == 3)
        reward -= main_fired.astype(np.float64) * 0.30
        reward -= side_fired.astype(np.float64) * 0.03

        crashed = (states[:, 1] <= 0) & ((np.abs(states[:, 2]) > 0.40) | (np.abs(states[:, 3]) > 0.40))
        reward -= crashed.astype(np.float64) * 80.0

        landed = (
            (states[:, 1] <= 0.05)
            & (np.abs(states[:, 2]) < 0.25)
            & (np.abs(states[:, 3]) < 0.25)
            & (np.abs(states[:, 4]) < 0.25)
        )
        reward += landed.astype(np.float64) * 80.0
        reward -= (np.abs(states[:, 0]) >= 1.0).astype(np.float64) * 100.0

        total_reward += reward

    return total_reward


class CEMPlanner:
    '''Single-environment Cross-Entropy Method planner.
    Args: horizon — steps to plan ahead. num_samples — trajectory samples per iteration.
          elite_frac — fraction kept as elite. num_iters — CEM refinement iterations.
    Returns: instance with a plan(observation) method.'''

    def __init__(self, horizon=15, num_samples=150, elite_frac=0.10, num_iters=2):
        '''Initialises planner state and warm-start action probability table.
        Args: horizon — steps to plan. num_samples — trajectory samples. elite_frac — top fraction kept. num_iters — CEM iterations.
        Returns: None.'''
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_elites = max(1, int(num_samples * elite_frac))
        self.num_iters = num_iters
        self.action_probs = np.ones((horizon, 4), dtype=np.float64) / 4

    def plan(self, observation):
        '''Plans one step from the current observation.
        Args: observation — (8,) state array.
        Returns: best_action integer, first_step_probs (4,) float32 array.'''
        probs = self.action_probs.copy()
        sequences = None
        costs = None

        for _ in range(self.num_iters):
            cumulative_dist = np.cumsum(probs, axis=1)
            sequences = (np.random.rand(self.num_samples, self.horizon, 1) > cumulative_dist[None]).sum(axis=2)
            tiled_states = np.tile(observation, (self.num_samples, 1)).astype(np.float32)
            costs = -score_action_sequences(tiled_states, sequences, self.horizon)

            elite_indices = np.argpartition(costs, self.num_elites)[:self.num_elites]
            elite_seqs = sequences[elite_indices]
            updated_probs = np.stack([(elite_seqs == action).mean(axis=0) for action in range(4)], axis=1)
            probs = 0.7 * updated_probs + 0.3 / 4.0
            probs /= probs.sum(axis=1, keepdims=True)

        self.action_probs[:-1] = probs[1:]
        self.action_probs[-1] = np.ones(4) / 4

        best_action = int(sequences[np.argmin(costs)][0])
        first_step_probs = probs[0].astype(np.float32)
        return best_action, first_step_probs


class BatchCEMPlanner:
    '''Vectorised CEM planner for a fixed number of parallel environments.
    Args: num_envs — number of parallel environments. horizon, num_samples, elite_frac, num_iters — same as CEMPlanner.
    Returns: instance with plan(observations) and reset_env(index) methods.'''

    def __init__(self, num_envs, horizon=15, num_samples=80, elite_frac=0.10, num_iters=2):
        '''Initialises per-environment warm-start action probability tables.
        Args: num_envs — number of parallel environments. horizon, num_samples, elite_frac, num_iters — same as CEMPlanner.
        Returns: None.'''
        self.num_envs = num_envs
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_elites = max(1, int(num_samples * elite_frac))
        self.num_iters = max(1, num_iters)
        self.action_probs = np.ones((num_envs, horizon, 4), dtype=np.float64) / 4

    def plan(self, observations):
        '''Plans one step for all environments simultaneously.
        Args: observations — (num_envs, 8) state array.
        Returns: best_actions (num_envs,) int64, first_step_probs (num_envs, 4) float32.'''
        num_envs = self.num_envs
        num_samples = self.num_samples
        horizon = self.horizon
        probs = self.action_probs.copy()
        sequences = None
        costs = None

        for _ in range(self.num_iters):
            cumulative_dist = np.cumsum(probs, axis=2)
            random_draws = np.random.rand(num_envs, num_samples, horizon, 1)
            sequences = (random_draws > cumulative_dist[:, np.newaxis]).sum(axis=3)

            flat_sequences = sequences.reshape(num_envs * num_samples, horizon)
            repeated_states = np.repeat(observations, num_samples, axis=0).astype(np.float32)
            total_rewards = score_action_sequences(repeated_states, flat_sequences, horizon)

            costs = (-total_rewards).reshape(num_envs, num_samples)
            elite_indices = np.argpartition(costs, self.num_elites, axis=1)[:, :self.num_elites]
            elite_seqs = sequences[np.arange(num_envs)[:, None], elite_indices]

            updated_probs = (elite_seqs[:, :, :, np.newaxis] == np.arange(4)).mean(axis=1)
            probs = 0.7 * updated_probs + 0.3 / 4.0
            probs /= probs.sum(axis=2, keepdims=True)

        best_indices = np.argmin(costs, axis=1)
        best_actions = sequences[np.arange(num_envs), best_indices, 0].astype(np.int64)
        first_step_probs = probs[:, 0, :].astype(np.float32)

        self.action_probs[:, :-1] = probs[:, 1:]
        self.action_probs[:, -1] = 1.0 / 4.0

        return best_actions, first_step_probs

    def reset_env(self, env_index):
        '''Resets the warm-started action probs for a single environment.
        Args: env_index — integer index of the environment to reset.
        Returns: None.'''
        self.action_probs[env_index] = 1.0 / 4.0


class PPONetwork(nn.Module):
    '''Two-head actor-critic network for vanilla PPO.
    Args: obs_dim — observation dimensionality. num_actions — discrete action count. hidden_size — layer width.
    Returns: instance with get_action and evaluate methods.'''

    def __init__(self, obs_dim=8, num_actions=4, hidden_size=128):
        '''Builds shared trunk, actor head and critic head with orthogonal initialisation.
        Args: obs_dim — observation size. num_actions — number of discrete actions. hidden_size — hidden layer width.
        Returns: None.'''
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
        )
        self.actor_head = nn.Linear(hidden_size, num_actions)
        self.critic_head = nn.Linear(hidden_size, 1)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)

    def forward(self, obs_tensor):
        '''Computes logits and value from observation.
        Args: obs_tensor — (N, obs_dim) float tensor.
        Returns: logits (N, num_actions), value (N,).'''
        hidden = self.trunk(obs_tensor)
        return self.actor_head(hidden), self.critic_head(hidden).squeeze(-1)

    def get_action(self, obs_tensor):
        '''Samples an action and returns log prob and value.
        Args: obs_tensor — (N, obs_dim) float tensor.
        Returns: action, log_prob, value — all (N,) tensors.'''
        logits, value = self(obs_tensor)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value

    def evaluate(self, obs_tensor, actions_tensor):
        '''Evaluates stored actions for PPO loss computation.
        Args: obs_tensor — (N, obs_dim). actions_tensor — (N,) integer actions.
        Returns: log_probs (N,), values (N,), entropy (N,).'''
        logits, value = self(obs_tensor)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions_tensor), value, dist.entropy()


class ResidualNetwork(nn.Module):
    '''Residual actor-critic that adds corrections on top of MPC action probabilities.
    Args: correction_scale — scalar weight on the learned correction logits. hidden_size — layer width.
    Returns: instance with get_action and evaluate methods.'''

    def __init__(self, correction_scale=1.0, hidden_size=128):
        '''Builds trunk, correction actor head and critic head; small init on actor to preserve MPC prior early.
        Args: correction_scale — scalar weight applied to learned correction logits. hidden_size — hidden layer width.
        Returns: None.'''
        super().__init__()
        self.correction_scale = correction_scale
        self.trunk = nn.Sequential(
            nn.Linear(12, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
        )
        self.actor_head = nn.Linear(hidden_size, 4)
        self.critic_head = nn.Linear(hidden_size, 1)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.005)

    def forward(self, combined_input):
        '''Computes correction logits and value from concatenated obs+mpc input.
        Args: combined_input — (N, 12) float tensor (8 obs + 4 mpc probs).
        Returns: correction_logits (N, 4), value (N,).'''
        hidden = self.trunk(combined_input)
        return self.actor_head(hidden), self.critic_head(hidden).squeeze(-1)

    def get_action(self, obs_tensor, mpc_probs_tensor):
        '''Samples action from MPC prior + learned correction.
        Args: obs_tensor — (N, 8). mpc_probs_tensor — (N, 4) raw MPC probabilities.
        Returns: action, log_prob, value, entropy, policy_probs — all (N,) or (N,4).'''
        sharpened_mpc = torch.softmax(torch.log(mpc_probs_tensor + 1e-8) / mpc_prob_temperature, dim=-1)
        combined = torch.cat([obs_tensor, sharpened_mpc], dim=-1)
        correction, value = self(combined)
        logits = torch.log(sharpened_mpc + 1e-8) + self.correction_scale * correction
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value, dist.entropy(), dist.probs

    def evaluate(self, obs_tensor, mpc_probs_tensor, actions_tensor):
        '''Evaluates stored actions for residual PPO loss.
        Args: obs_tensor — (N, 8). mpc_probs_tensor — (N, 4). actions_tensor — (N,) integers.
        Returns: log_probs, values, entropy, policy_probs, sharpened_mpc — all (N,) or (N,4).'''
        sharpened_mpc = torch.softmax(torch.log(mpc_probs_tensor + 1e-8) / mpc_prob_temperature, dim=-1)
        combined = torch.cat([obs_tensor, sharpened_mpc], dim=-1)
        correction, value = self(combined)
        logits = torch.log(sharpened_mpc + 1e-8) + self.correction_scale * correction
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions_tensor), value, dist.entropy(), dist.probs, sharpened_mpc


def compute_gae_and_update(network, optimizer, obs_buffer, action_buffer, logprob_buffer,
                           reward_buffer, done_buffer, value_buffer,
                           num_envs, rollout_length, minibatch_size,
                           clip_ratio, entropy_coef, mpc_buffer=None,
                           next_obs=None, next_mpc_probs=None):
    '''Runs GAE advantage estimation and PPO update epochs over a collected rollout.
    Args: network — PPONetwork or ResidualNetwork. optimizer — Adam optimizer.
          obs_buffer, action_buffer, logprob_buffer, reward_buffer, done_buffer, value_buffer —
              (rollout_length, num_envs, ...) rollout arrays.
          num_envs, rollout_length, minibatch_size — training dimensions.
          clip_ratio, entropy_coef — PPO hyperparameters.
          mpc_buffer — optional (rollout_length, num_envs, 4) MPC prob array for residual networks.
          next_obs — (num_envs, 8) observation AFTER the last rollout step (FIX 4: correct bootstrap).
          next_mpc_probs — (num_envs, 4) MPC probs corresponding to next_obs (residual only).
    Returns: mean policy loss, mean value loss, mean entropy as floats.'''

    bootstrap_obs = next_obs if next_obs is not None else obs_buffer[-1]

    if mpc_buffer is not None:
        bootstrap_mpc = next_mpc_probs if next_mpc_probs is not None else mpc_buffer[-1]
        sharpened_last = torch.softmax(
            torch.log(torch.from_numpy(bootstrap_mpc).to(device) + 1e-8) / mpc_prob_temperature, dim=-1
        )
        last_combined = torch.cat([torch.from_numpy(bootstrap_obs).to(device), sharpened_last], dim=-1)
        with torch.no_grad():
            _, last_values = network(last_combined)
    else:
        with torch.no_grad():
            _, last_values = network(torch.from_numpy(bootstrap_obs).to(device))
    last_values = last_values.cpu().numpy()

    advantages = np.zeros_like(reward_buffer)
    running_gae = np.zeros(num_envs, np.float32)
    for step in reversed(range(rollout_length)):
        next_val = last_values if step == rollout_length - 1 else value_buffer[step + 1]
        td_error = reward_buffer[step] + discount_factor * next_val * (1 - done_buffer[step]) - value_buffer[step]
        running_gae = td_error + discount_factor * gae_lambda * (1 - done_buffer[step]) * running_gae
        advantages[step] = running_gae
    returns = advantages + value_buffer

    flat_advantages = advantages.reshape(-1)
    flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)

    obs_flat = torch.from_numpy(obs_buffer.reshape(-1, obs_buffer.shape[-1])).to(device)
    acts_flat = torch.from_numpy(action_buffer.reshape(-1)).to(device)
    old_logprobs_flat = torch.from_numpy(logprob_buffer.reshape(-1)).to(device)
    advantages_flat = torch.from_numpy(flat_advantages).to(device)
    returns_flat = torch.from_numpy(returns.reshape(-1)).to(device)
    mpc_flat = torch.from_numpy(mpc_buffer.reshape(-1, 4)).to(device) if mpc_buffer is not None else None

    total_samples = obs_flat.shape[0]
    policy_losses, value_losses, entropies = [], [], []

    for _ in range(update_epochs):
        shuffle_indices = torch.randperm(total_samples, device=device)
        for start in range(0, total_samples, minibatch_size):
            batch_indices = shuffle_indices[start: start + minibatch_size]

            if mpc_flat is not None:
                new_logprobs, values, entropy, policy_probs, sharp_mpc_batch = network.evaluate(
                    obs_flat[batch_indices], mpc_flat[batch_indices], acts_flat[batch_indices]
                )
                kl_divergence = (sharp_mpc_batch * (
                    torch.log(sharp_mpc_batch + 1e-8) - torch.log(policy_probs + 1e-8)
                )).sum(dim=-1).mean()
            else:
                new_logprobs, values, entropy = network.evaluate(obs_flat[batch_indices], acts_flat[batch_indices])
                kl_divergence = None

            prob_ratio = (new_logprobs - old_logprobs_flat[batch_indices]).exp()
            batch_advantages = advantages_flat[batch_indices]
            policy_loss = torch.max(
                -batch_advantages * prob_ratio,
                -batch_advantages * prob_ratio.clamp(1 - clip_ratio, 1 + clip_ratio)
            ).mean()
            value_loss = 0.5 * (values - returns_flat[batch_indices]).pow(2).mean()

            total_loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy.mean()
            if kl_divergence is not None:
                total_loss = total_loss + kl_penalty_coef * kl_divergence

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), max_grad_norm)
            optimizer.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(entropy.mean().item())

    return float(np.mean(policy_losses)), float(np.mean(value_losses)), float(np.mean(entropies))


def ppo_cost_1(observations, rewards, dones):
    '''Raw environment reward with no shaping.
    Args: observations — (N, 8). rewards — (N,). dones — (N,).
    Returns: (N,) shaped rewards.'''
    return rewards.copy()


def ppo_cost_2(observations, rewards, dones):
    '''Penalises horizontal drift and speed, rewards downward descent.
    Args: observations — (N, 8). rewards — (N,). dones — (N,).
    Returns: (N,) shaped rewards.'''
    shaped = rewards.copy()
    shaped -= 2.0 * np.abs(observations[:, 2])
    shaped -= 1.5 * np.abs(observations[:, 0])
    shaped += np.where(observations[:, 3] < 0, 0.5 * np.abs(observations[:, 3]), 0.0)
    return shaped


def ppo_cost_3(observations, rewards, dones):
    '''Penalises excess speed near the ground to discourage hard landings.
    Args: observations — (N, 8). rewards — (N,). dones — (N,).
    Returns: (N,) shaped rewards.'''
    shaped = rewards.copy()
    altitude = observations[:, 1]
    speed = np.sqrt(observations[:, 2] ** 2 + observations[:, 3] ** 2)
    near_ground = altitude < 0.15
    shaped += np.where(near_ground & (speed > 0.5), 30.0, 0.0)
    shaped += np.where(near_ground, 0.5 * speed, 0.0)
    return shaped


def ppo_cost_4(observations, rewards, dones):
    '''Penalises tilt and spin rate, rewards vertical alignment.
    Args: observations — (N, 8). rewards — (N,). dones — (N,).
    Returns: (N,) shaped rewards.'''
    shaped = rewards.copy()
    shaped -= 1.0 * np.abs(observations[:, 5])
    shaped -= 0.5 * np.abs(observations[:, 4])
    shaped += np.where((np.abs(observations[:, 4]) < 0.1) & (np.abs(observations[:, 5]) < 0.1), 2.0, 0.0)
    return shaped


def ppo_cost_5(observations, rewards, dones):
    '''Guidance-law shaping: penalises speed relative to altitude and horizontal offset.
    Args: observations — (N, 8). rewards — (N,). dones — (N,).
    Returns: (N,) shaped rewards.'''
    shaped = rewards.copy()
    altitude = observations[:, 1]
    speed = np.sqrt(observations[:, 2] ** 2 + observations[:, 3] ** 2)
    excess_speed = np.maximum(speed - np.clip(altitude, 0.0, 1.0), 0.0)
    shaped -= 2.5 * excess_speed
    shaped -= 0.8 * np.abs(observations[:, 0])
    shaped += np.where((altitude < 0.2) & (speed < 0.25), 3.0, 0.0)
    return shaped


ppo_strategies = [
    ('Function 1', ppo_cost_1, function_colors[0]),
    ('Function 2', ppo_cost_2, function_colors[1]),
    ('Function 3', ppo_cost_3, function_colors[2]),
    ('Function 4', ppo_cost_4, function_colors[3]),
    ('Function 5', ppo_cost_5, function_colors[4]),
]


def residual_cost_1(observations, rewards, dones, actions):
    '''Guidance-law shaping matching ppo_cost_5 for a competitive unshaped baseline.
    Args: observations — (N, 8). rewards — (N,). dones — (N,). actions — (N,) integers.
    Returns: (N,) shaped rewards.'''
    shaped = rewards.copy()
    altitude = observations[:, 1]
    speed = np.sqrt(observations[:, 2] ** 2 + observations[:, 3] ** 2)
    excess_speed = np.maximum(speed - np.clip(altitude, 0.0, 1.0), 0.0)
    shaped -= 2.5 * excess_speed
    shaped -= 0.8 * np.abs(observations[:, 0])
    shaped += np.where((altitude < 0.2) & (speed < 0.25), 3.0, 0.0)
    return shaped


def residual_cost_2(observations, rewards, dones, actions):
    '''Penalises horizontal drift and velocity with fuel consumption penalty.
    Args: observations — (N, 8). rewards — (N,). dones — (N,). actions — (N,) integers.
    Returns: (N,) shaped rewards.'''
    shaped = rewards.copy()
    shaped -= 2.0 * np.abs(observations[:, 2])
    shaped -= 1.5 * np.abs(observations[:, 0])
    main_fired = (actions == 2).astype(np.float32)
    side_fired = ((actions == 1) | (actions == 3)).astype(np.float32)
    shaped -= fuel_penalty_coef * (main_fired + side_fired * 0.1)
    return shaped


def residual_cost_3(observations, rewards, dones, actions):
    '''Penalises tilt and spin with fuel consumption penalty.
    Args: observations — (N, 8). rewards — (N,). dones — (N,). actions — (N,) integers.
    Returns: (N,) shaped rewards.'''
    shaped = rewards.copy()
    shaped -= 1.0 * np.abs(observations[:, 5])
    shaped -= 0.5 * np.abs(observations[:, 4])
    shaped += np.where((np.abs(observations[:, 4]) < 0.1) & (np.abs(observations[:, 5]) < 0.1), 2.0, 0.0)
    main_fired = (actions == 2).astype(np.float32)
    side_fired = ((actions == 1) | (actions == 3)).astype(np.float32)
    shaped -= fuel_penalty_coef * (main_fired + side_fired * 0.1)
    return shaped


def residual_cost_4(observations, rewards, dones, actions):
    '''Strong tilt-angle penalty with fuel cost weight, rescaled to stay within env reward range.
    Args: observations — (N, 8). rewards — (N,). dones — (N,). actions — (N,) integers.
    Returns: (N,) shaped rewards.'''
    shaped = rewards.copy()
    shaped -= 5.0 * observations[:, 4] ** 2
    shaped -= 1.0 * observations[:, 5] ** 2
    shaped += np.where(np.abs(observations[:, 4]) < 0.05, 3.0, 0.0)
    main_fired = (actions == 2).astype(np.float32)
    side_fired = ((actions == 1) | (actions == 3)).astype(np.float32)
    shaped -= fuel_penalty_coef * 2.0 * (main_fired + side_fired * 0.1)
    return shaped


def residual_cost_5(observations, rewards, dones, actions):
    '''Balanced shaping across position, velocity, angle and fuel use.
    Args: observations — (N, 8). rewards — (N,). dones — (N,). actions — (N,) integers.
    Returns: (N,) shaped rewards.'''
    shaped = rewards.copy()
    shaped -= 1.0 * np.abs(observations[:, 2])
    shaped -= 0.8 * np.abs(observations[:, 0])
    shaped -= 0.8 * np.abs(observations[:, 4])
    shaped -= 0.4 * np.abs(observations[:, 5])
    main_fired = (actions == 2).astype(np.float32)
    side_fired = ((actions == 1) | (actions == 3)).astype(np.float32)
    shaped -= fuel_penalty_coef * 1.5 * (main_fired + side_fired * 0.1)
    return shaped


residual_strategies = [
    ('Function 1', residual_cost_1, function_colors[0]),
    ('Function 2', residual_cost_2, function_colors[1]),
    ('Function 3', residual_cost_3, function_colors[2]),
    ('Function 4', residual_cost_4, function_colors[3]),
    ('Function 5', residual_cost_5, function_colors[4]),
]


def run_mpc(num_episodes=mpc_num_episodes, seed=42):
    '''Runs the CEM-MPC planner for a fixed number of episodes and collects diagnostics.
    Args: num_episodes — number of evaluation episodes. seed — random seed.
    Returns: dict with rewards, fuel, efficiency, action_freq and summary statistics.'''
    print(f"MPC: running {num_episodes} episodes")
    env = gym.make('LunarLander-v3', render_mode=None)

    all_rewards, all_steps, all_fuel, all_fuel_efficiency = [], [], [], []
    action_counts = np.zeros(4)

    for episode in range(1, num_episodes + 1):
        observation, _ = env.reset(seed=seed + episode)
        planner = CEMPlanner()
        total_reward = 0.0
        fuel_used = 0.0

        for step in range(1000):
            action, _ = planner.plan(observation)
            if action == 2:
                fuel_used += 1.0
            elif action in (1, 3):
                fuel_used += 0.1
            action_counts[action] += 1
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        fuel_efficiency = total_reward / (fuel_used + 1e-6)
        all_rewards.append(total_reward)
        all_steps.append(step + 1)
        all_fuel.append(fuel_used)
        all_fuel_efficiency.append(fuel_efficiency)
        outcome = 'pass' if total_reward > 0 else 'fail'
        print(f"episode {episode} reward {total_reward:.1f} fuel {fuel_used:.1f} {outcome}")

    env.close()
    rewards_array = np.array(all_rewards)
    positive_rate = np.mean(rewards_array > 0) * 100
    print(f"MPC finished positive {positive_rate:.0f}% mean reward {np.mean(rewards_array):.1f}")

    mpc_save_path = os.path.join(output_dir, 'mpc_results.npz')
    np.savez(
        mpc_save_path,
        rewards         = rewards_array,
        steps           = np.array(all_steps),
        fuel            = np.array(all_fuel),
        fuel_efficiency = np.array(all_fuel_efficiency),
        positive_rate   = np.array([positive_rate]),
        mean_reward     = np.array([float(np.mean(rewards_array))]),
        mean_fuel       = np.array([float(np.mean(all_fuel))]),
        mean_efficiency = np.array([float(np.mean(all_fuel_efficiency))]),
    )
    print(f"saved MPC results to {mpc_save_path}")

    return {
        'label': 'MPC',
        'color': slate,
        'rewards': rewards_array,
        'steps': np.array(all_steps),
        'fuel': np.array(all_fuel),
        'fuel_efficiency': np.array(all_fuel_efficiency),
        'action_freq': action_counts / action_counts.sum(),
        'positive_rate': positive_rate,
        'mean_reward': float(np.mean(rewards_array)),
        'mean_fuel': float(np.mean(all_fuel)),
        'mean_efficiency': float(np.mean(all_fuel_efficiency)),
    }


def run_ppo_agent(name, color, shape_fn, total_steps=ppo_total_steps, seed=42):
    '''Trains a vanilla PPO agent with a given reward shaping function.
    Args: name — string label. color — plot color. shape_fn — reward shaping callable.
          total_steps — environment steps to train for. seed — random seed.
    Returns: dict with training histories and final performance statistics.'''
    torch.manual_seed(seed)
    np.random.seed(seed)

    envs = gym.vector.SyncVectorEnv([lambda: gym.make('LunarLander-v3') for _ in range(ppo_parallel_envs)])
    network = PPONetwork().to(device)
    optimizer = optim.Adam(network.parameters(), lr=ppo_learning_rate, eps=1e-5)

    current_obs, _ = envs.reset()
    current_obs = current_obs.astype(np.float32)

    episode_rewards, episode_fuel, episode_fuel_efficiency = [], [], []
    recent_rewards = collections.deque(maxlen=100)
    recent_efficiency = collections.deque(maxlen=100)
    running_reward = np.zeros(ppo_parallel_envs, np.float32)
    running_fuel = np.zeros(ppo_parallel_envs, np.float32)

    positive_rate_history, mean_reward_history, efficiency_history = [], [], []

    obs_buffer = np.zeros((ppo_rollout_length, ppo_parallel_envs, 8), np.float32)
    action_buffer = np.zeros((ppo_rollout_length, ppo_parallel_envs), np.int64)
    logprob_buffer = np.zeros((ppo_rollout_length, ppo_parallel_envs), np.float32)
    reward_buffer = np.zeros((ppo_rollout_length, ppo_parallel_envs), np.float32)
    done_buffer = np.zeros((ppo_rollout_length, ppo_parallel_envs), np.float32)
    value_buffer = np.zeros((ppo_rollout_length, ppo_parallel_envs), np.float32)

    total_env_steps = 0
    start_time = time.time()
    network.eval()

    while total_env_steps < total_steps:
        with torch.inference_mode():
            for timestep in range(ppo_rollout_length):
                obs_tensor = torch.from_numpy(current_obs).to(device)
                actions, log_probs, values = network.get_action(obs_tensor)
                actions_np = actions.cpu().numpy()

                next_obs, rewards, terminated, truncated, _ = envs.step(actions_np)
                dones = terminated | truncated
                shaped_rewards = shape_fn(current_obs, rewards, dones).astype(np.float32)

                obs_buffer[timestep] = current_obs
                action_buffer[timestep] = actions_np
                logprob_buffer[timestep] = log_probs.cpu().numpy()
                reward_buffer[timestep] = shaped_rewards
                done_buffer[timestep] = dones.astype(np.float32)
                value_buffer[timestep] = values.cpu().numpy()

                fuel_step = (actions_np == 2).astype(np.float32) + ((actions_np != 0) & (actions_np != 2)).astype(np.float32) * 0.1
                running_reward += rewards
                running_fuel += fuel_step

                for env_index, done in enumerate(dones):
                    if done:
                        ep_reward = float(running_reward[env_index])
                        ep_fuel = float(running_fuel[env_index])
                        ep_efficiency = ep_reward / (ep_fuel + 1e-6)
                        recent_rewards.append(ep_reward)
                        recent_efficiency.append(ep_efficiency)
                        episode_rewards.append(ep_reward)
                        episode_fuel.append(ep_fuel)
                        episode_fuel_efficiency.append(ep_efficiency)
                        running_reward[env_index] = 0.0
                        running_fuel[env_index] = 0.0

                current_obs = next_obs.astype(np.float32)

        network.train()
        compute_gae_and_update(
            network, optimizer,
            obs_buffer, action_buffer, logprob_buffer, reward_buffer, done_buffer, value_buffer,
            ppo_parallel_envs, ppo_rollout_length, ppo_minibatch_size, ppo_clip_ratio, ppo_entropy_coef,
            next_obs=current_obs,  # FIX 4: pass post-rollout obs for correct bootstrap
        )
        network.eval()
        total_env_steps += ppo_rollout_length * ppo_parallel_envs

        if len(recent_rewards) >= 20:
            positive_rate = float(np.mean(np.array(list(recent_rewards)) > 0)) * 100
            mean_reward = float(np.mean(list(recent_rewards)))
            mean_efficiency = float(np.mean(list(recent_efficiency)))
            positive_rate_history.append((total_env_steps, positive_rate))
            mean_reward_history.append((total_env_steps, mean_reward))
            efficiency_history.append((total_env_steps, mean_efficiency))

    envs.close()
    elapsed = time.time() - start_time
    final_positive_rate = float(np.mean(np.array(list(recent_rewards)) > 0)) * 100 if recent_rewards else 0.0
    final_mean_reward = float(np.mean(list(recent_rewards))) if recent_rewards else 0.0
    final_mean_efficiency = float(np.mean(list(recent_efficiency))) if recent_efficiency else 0.0
    print(f"{name} done in {elapsed:.0f}s positive {final_positive_rate:.0f}% mean {final_mean_reward:.1f} episodes {len(episode_rewards)}")

    ppo_slug = name.lower().replace(' ', '_')
    torch.save({'state_dict': network.state_dict(), 'name': name, 'obs_dim': 8, 'num_actions': 4, 'hidden_size': 128},
               os.path.join(output_dir, f'ppo_{ppo_slug}.pt'))
    np.savez(
        os.path.join(output_dir, f'ppo_{ppo_slug}_stats.npz'),
        episode_rewards        = np.array(episode_rewards),
        episode_fuel           = np.array(episode_fuel),
        episode_fuel_efficiency = np.array(episode_fuel_efficiency),
        positive_rate_history  = np.array(positive_rate_history) if positive_rate_history else np.array([]),
        mean_reward_history    = np.array(mean_reward_history) if mean_reward_history else np.array([]),
    )
    print(f"saved PPO model ppo_{ppo_slug}.pt")

    return {
        'label': name,
        'color': color,
        'episode_rewards': np.array(episode_rewards),
        'episode_fuel': np.array(episode_fuel),
        'episode_fuel_efficiency': np.array(episode_fuel_efficiency),
        'positive_rate_history': positive_rate_history,
        'mean_reward_history': mean_reward_history,
        'efficiency_history': efficiency_history,
        'positive_rate': final_positive_rate,
        'mean_reward': final_mean_reward,
        'mean_efficiency': final_mean_efficiency,
        'mean_fuel': float(np.mean(episode_fuel[-500:])) if episode_fuel else 0.0,
    }


def run_residual_agent(fn_name, fn_color, shape_fn, correction_scale,
                       total_steps=residual_total_steps, seed=42):
    '''Trains a residual RL agent combining CEM-MPC with a learned correction network.
    Args: fn_name — cost function label string. fn_color — plot color. shape_fn — reward shaping callable.
          correction_scale — scalar weight on learned corrections. total_steps — training budget. seed — random seed.
    Returns: dict with training histories, kl_history and final performance statistics.'''
    torch.manual_seed(seed)
    np.random.seed(seed)

    envs = gym.vector.SyncVectorEnv([lambda: gym.make('LunarLander-v3') for _ in range(residual_parallel_envs)])
    network = ResidualNetwork(correction_scale=correction_scale).to(device)
    optimizer = optim.Adam(network.parameters(), lr=residual_learning_rate, eps=1e-5)
    batch_planner = BatchCEMPlanner(num_envs=residual_parallel_envs)

    current_obs, _ = envs.reset()
    current_obs = current_obs.astype(np.float32)

    episode_rewards, episode_fuel, episode_fuel_efficiency = [], [], []
    recent_rewards = collections.deque(maxlen=100)
    recent_efficiency = collections.deque(maxlen=100)
    running_reward = np.zeros(residual_parallel_envs, np.float32)
    running_fuel = np.zeros(residual_parallel_envs, np.float32)

    positive_rate_history, mean_reward_history, efficiency_history, kl_history = [], [], [], []

    obs_buffer = np.zeros((residual_rollout_length, residual_parallel_envs, 8), np.float32)
    mpc_probs_buffer = np.zeros((residual_rollout_length, residual_parallel_envs, 4), np.float32)
    action_buffer = np.zeros((residual_rollout_length, residual_parallel_envs), np.int64)
    logprob_buffer = np.zeros((residual_rollout_length, residual_parallel_envs), np.float32)
    reward_buffer = np.zeros((residual_rollout_length, residual_parallel_envs), np.float32)
    done_buffer = np.zeros((residual_rollout_length, residual_parallel_envs), np.float32)
    value_buffer = np.zeros((residual_rollout_length, residual_parallel_envs), np.float32)

    total_env_steps = 0
    network.eval()
    cached_mpc_probs = np.ones((residual_parallel_envs, 4), np.float32) / 4

    while total_env_steps < total_steps:
        with torch.inference_mode():
            rollout_kl_values = []
            for timestep in range(residual_rollout_length):
                if timestep % cem_replan_every == 0:
                    _, cached_mpc_probs = batch_planner.plan(current_obs)
                mpc_probs = cached_mpc_probs

                obs_tensor = torch.from_numpy(current_obs).to(device)
                mpc_tensor = torch.from_numpy(mpc_probs).to(device)

                actions, log_probs, values, _, combined_probs = network.get_action(obs_tensor, mpc_tensor)
                actions_np = actions.cpu().numpy()

                kl = (mpc_tensor * (torch.log(mpc_tensor + 1e-8) - torch.log(combined_probs + 1e-8))).sum(dim=-1).mean().item()
                rollout_kl_values.append(kl)

                next_obs, rewards, terminated, truncated, _ = envs.step(actions_np)
                dones = terminated | truncated
                shaped_rewards = shape_fn(current_obs, rewards, dones, actions_np).astype(np.float32)

                obs_buffer[timestep] = current_obs
                mpc_probs_buffer[timestep] = mpc_probs
                action_buffer[timestep] = actions_np
                logprob_buffer[timestep] = log_probs.cpu().numpy()
                reward_buffer[timestep] = shaped_rewards
                done_buffer[timestep] = dones.astype(np.float32)
                value_buffer[timestep] = values.cpu().numpy()

                fuel_step = (actions_np == 2).astype(np.float32) + ((actions_np == 1) | (actions_np == 3)).astype(np.float32) * 0.1
                running_reward += rewards
                running_fuel += fuel_step

                for env_index, done in enumerate(dones):
                    if done:
                        ep_reward = float(running_reward[env_index])
                        ep_fuel = float(running_fuel[env_index])
                        ep_efficiency = ep_reward / (ep_fuel + 1e-6)
                        recent_rewards.append(ep_reward)
                        recent_efficiency.append(ep_efficiency)
                        episode_rewards.append(ep_reward)
                        episode_fuel.append(ep_fuel)
                        episode_fuel_efficiency.append(ep_efficiency)
                        running_reward[env_index] = 0.0
                        running_fuel[env_index] = 0.0
                        batch_planner.reset_env(env_index)

                current_obs = next_obs.astype(np.float32)

        network.train()
        compute_gae_and_update(
            network, optimizer,
            obs_buffer, action_buffer, logprob_buffer, reward_buffer, done_buffer, value_buffer,
            residual_parallel_envs, residual_rollout_length, residual_minibatch_size,
            residual_clip_ratio, residual_entropy_coef,
            mpc_buffer=mpc_probs_buffer,
            next_obs=current_obs,             # FIX 4: correct bootstrap state
            next_mpc_probs=cached_mpc_probs,  # FIX 4: correct bootstrap MPC probs
        )
        network.eval()
        total_env_steps += residual_rollout_length * residual_parallel_envs

        if len(recent_rewards) >= 20:
            positive_rate = float(np.mean(np.array(list(recent_rewards)) > 0)) * 100
            mean_reward = float(np.mean(list(recent_rewards)))
            mean_efficiency = float(np.mean(list(recent_efficiency)))
            positive_rate_history.append((total_env_steps, positive_rate))
            mean_reward_history.append((total_env_steps, mean_reward))
            efficiency_history.append((total_env_steps, mean_efficiency))
            kl_history.append((total_env_steps, float(np.mean(rollout_kl_values))))

    envs.close()
    final_positive_rate = float(np.mean(np.array(list(recent_rewards)) > 0)) * 100 if recent_rewards else 0.0
    final_mean_reward = float(np.mean(list(recent_rewards))) if recent_rewards else 0.0
    final_mean_efficiency = float(np.mean(list(recent_efficiency))) if recent_efficiency else 0.0
    label = f'{fn_name} scale={correction_scale}'
    print(f"{label} done positive {final_positive_rate:.0f}% mean {final_mean_reward:.1f} episodes {len(episode_rewards)}")

    residual_slug = fn_name.lower().replace(' ', '_')
    torch.save(
        {'state_dict': network.state_dict(), 'fn_name': fn_name, 'scale': correction_scale,
         'obs_dim': 8, 'num_actions': 4, 'hidden_size': 128, 'mpc_prob_temperature': mpc_prob_temperature},
        os.path.join(output_dir, f'residual_{residual_slug}_scale{correction_scale}.pt')
    )
    np.savez(
        os.path.join(output_dir, f'residual_{residual_slug}_scale{correction_scale}_stats.npz'),
        episode_rewards         = np.array(episode_rewards),
        episode_fuel            = np.array(episode_fuel),
        episode_fuel_efficiency = np.array(episode_fuel_efficiency),
        positive_rate_history   = np.array(positive_rate_history) if positive_rate_history else np.array([]),
        mean_reward_history     = np.array(mean_reward_history) if mean_reward_history else np.array([]),
        kl_history              = np.array(kl_history) if kl_history else np.array([]),
    )
    print(f"saved residual model residual_{residual_slug}_scale{correction_scale}.pt")

    return {
        'label': label,
        'fn_name': fn_name,
        'scale': correction_scale,
        'color': fn_color,
        'episode_rewards': np.array(episode_rewards),
        'episode_fuel': np.array(episode_fuel),
        'episode_fuel_efficiency': np.array(episode_fuel_efficiency),
        'positive_rate_history': positive_rate_history,
        'mean_reward_history': mean_reward_history,
        'efficiency_history': efficiency_history,
        'kl_history': kl_history,
        'positive_rate': final_positive_rate,
        'mean_reward': final_mean_reward,
        'mean_efficiency': final_mean_efficiency,
        'mean_fuel': float(np.mean(episode_fuel[-500:])) if episode_fuel else 0.0,
    }


def unzip_history(history):
    '''Splits a list of (step, value) tuples into two separate lists.
    Args: history — list of (step, value) tuples.
    Returns: steps list, values list (both empty if history is empty).'''
    if not history:
        return [], []
    steps, values = zip(*history)
    return list(steps), list(values)


def smooth_series(values, window_fraction=0.08):
    '''Applies a uniform moving average to a 1D array.
    Args: values — 1D array-like. window_fraction — fraction of length to use as window.
    Returns: smoothed numpy array (shorter than input by window-1).'''
    if len(values) < 4:
        return values
    window = max(1, int(len(values) * window_fraction))
    return np.convolve(values, np.ones(window) / window, mode='valid')


def save_figure(filename):
    '''Saves the current matplotlib figure to the output directory.
    Args: filename — output filename string.
    Returns: None.'''
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"saved {filename}")


def generate_plots(mpc_results, ppo_agents, residual_agents):
    '''Generates and saves all analysis plots.
    Args: mpc_results — dict from run_mpc. ppo_agents — list of dicts from run_ppo_agent.
          residual_agents — list of dicts from run_residual_agent.
    Returns: None.'''

    function_names = ['Function 1', 'Function 2', 'Function 3', 'Function 4', 'Function 5']

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Residual RL — Mean Reward over Training (by cost function)', fontsize=12)
    axes = axes.flatten()

    for function_index, fn_name in enumerate(function_names):
        ax = axes[function_index]
        agents_this_fn = [agent for agent in residual_agents if agent['fn_name'] == fn_name]
        fn_color = agents_this_fn[0]['color'] if agents_this_fn else gray
        for agent in agents_this_fn:
            steps, values = unzip_history(agent['mean_reward_history'])
            if steps:
                ax.plot(
                    steps, values,
                    color=fn_color,
                    linewidth=scale_linewidths[agent['scale']],
                    label=f'scale {agent["scale"]}',
                )
        ax.axhline(mpc_results['mean_reward'], color=gray, linewidth=1.2,
                   linestyle='--', label='MPC baseline')
        ax.axhline(0, color='#cccccc', linewidth=0.7, linestyle=':')
        ax.set_title(fn_name, fontsize=10)
        ax.set_xlabel('Training steps', fontsize=8)
        ax.set_ylabel('Mean reward (100 ep)', fontsize=8)
        ax.legend(frameon=False, fontsize=7)

    axes[-1].set_visible(False)
    save_figure('plot1_residual_mean_reward_by_function.png')

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Residual RL — Positive Rate over Training (by cost function)', fontsize=12)
    axes = axes.flatten()

    for function_index, fn_name in enumerate(function_names):
        ax = axes[function_index]
        agents_this_fn = [agent for agent in residual_agents if agent['fn_name'] == fn_name]
        fn_color = agents_this_fn[0]['color'] if agents_this_fn else gray
        for agent in agents_this_fn:
            steps, values = unzip_history(agent['positive_rate_history'])
            if steps:
                ax.plot(
                    steps, values,
                    color=fn_color,
                    linewidth=scale_linewidths[agent['scale']],
                    label=f'scale {agent["scale"]}',
                )
        ax.axhline(mpc_results['positive_rate'], color=gray, linewidth=1.2,
                   linestyle='--', label='MPC baseline')
        ax.axhline(50, color='#cccccc', linewidth=0.7, linestyle=':')
        ax.set_ylim(0, 105)
        ax.set_title(fn_name, fontsize=10)
        ax.set_xlabel('Training steps', fontsize=8)
        ax.set_ylabel('% episodes positive reward', fontsize=8)
        ax.legend(frameon=False, fontsize=7)

    axes[-1].set_visible(False)
    save_figure('plot2_residual_positive_rate_by_function.png')

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Residual RL — KL Divergence from MPC Prior over Training', fontsize=12)
    axes = axes.flatten()

    for function_index, fn_name in enumerate(function_names):
        ax = axes[function_index]
        agents_this_fn = [agent for agent in residual_agents if agent['fn_name'] == fn_name]
        fn_color = agents_this_fn[0]['color'] if agents_this_fn else gray
        for agent in agents_this_fn:
            steps, values = unzip_history(agent['kl_history'])
            if steps:
                ax.plot(
                    steps, values,
                    color=fn_color,
                    linewidth=scale_linewidths[agent['scale']],
                    label=f'scale {agent["scale"]}',
                )
        ax.set_title(fn_name, fontsize=10)
        ax.set_xlabel('Training steps', fontsize=8)
        ax.set_ylabel('KL(MPC || policy)', fontsize=8)
        ax.legend(frameon=False, fontsize=7)

    axes[-1].set_visible(False)
    save_figure('plot3_residual_kl_divergence_by_function.png')

    fig, (ax_reward, ax_positive) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Residual RL — Final Performance Heatmap (5 functions × 3 scales)', fontsize=12)

    reward_grid = np.zeros((5, 3))
    positive_grid = np.zeros((5, 3))
    scale_list = residual_scales

    for agent in residual_agents:
        fn_index = function_names.index(agent['fn_name'])
        scale_index = scale_list.index(agent['scale'])
        reward_grid[fn_index, scale_index] = agent['mean_reward']
        positive_grid[fn_index, scale_index] = agent['positive_rate']

    scale_labels = [f'scale {scale}' for scale in scale_list]

    reward_im = ax_reward.imshow(reward_grid, aspect='auto', cmap='RdYlGn', vmin=-50, vmax=200)
    ax_reward.set_xticks(range(3))
    ax_reward.set_xticklabels(scale_labels, fontsize=9)
    ax_reward.set_yticks(range(5))
    ax_reward.set_yticklabels(function_names, fontsize=9)
    ax_reward.set_title('Mean Reward', fontsize=10)
    for row in range(5):
        for col in range(3):
            ax_reward.text(col, row, f'{reward_grid[row, col]:.0f}', ha='center', va='center', fontsize=9, fontweight='bold')
    plt.colorbar(reward_im, ax=ax_reward, shrink=0.8)

    positive_im = ax_positive.imshow(positive_grid, aspect='auto', cmap='RdYlGn', vmin=0, vmax=100)
    ax_positive.set_xticks(range(3))
    ax_positive.set_xticklabels(scale_labels, fontsize=9)
    ax_positive.set_yticks(range(5))
    ax_positive.set_yticklabels(function_names, fontsize=9)
    ax_positive.set_title('Positive Rate %', fontsize=10)
    for row in range(5):
        for col in range(3):
            ax_positive.text(col, row, f'{positive_grid[row, col]:.0f}%', ha='center', va='center', fontsize=9, fontweight='bold')
    plt.colorbar(positive_im, ax=ax_positive, shrink=0.8)

    save_figure('plot4_residual_performance_heatmap.png')

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Residual RL — Reward vs Fuel (all 15 agents)', fontsize=12)

    ax.axhline(mpc_results['mean_reward'], color=slate, linewidth=1.2, linestyle='--', alpha=0.7,
               label=f'MPC reward ({mpc_results["mean_reward"]:.0f})')
    ax.axvline(mpc_results['mean_fuel'], color=slate, linewidth=1.0, linestyle=':', alpha=0.5,
               label=f'MPC fuel ({mpc_results["mean_fuel"]:.1f})')

    for agent in residual_agents:
        is_best_scale = agent['scale'] == 1.0
        ax.scatter(agent['mean_fuel'], agent['mean_reward'],
                   color=agent['color'],
                   marker='o',
                   s=110 if is_best_scale else 65,
                   facecolors=agent['color'] if is_best_scale else 'none',
                   edgecolors=agent['color'],
                   linewidths=1.8,
                   alpha=0.9, zorder=4)
        ax.annotate(
            f'{agent["fn_name"]}\ns={agent["scale"]}',
            (agent['mean_fuel'], agent['mean_reward']),
            textcoords='offset points', xytext=(6, 3), fontsize=7,
        )

    for fn_name, fn_color in zip(function_names, function_colors):
        ax.scatter([], [], color=fn_color, marker='o', s=60, label=fn_name)
    ax.scatter([], [], color='black', marker='o', s=80, facecolors='black', label='scale 1.0 (filled)')
    ax.scatter([], [], color='black', marker='o', s=50, facecolors='none', edgecolors='black', label='scale 0.3 / 2.5 (hollow)')

    ax.axhline(0, color=gray, linewidth=0.7, linestyle=':')
    ax.set_xlabel('Mean fuel consumed per episode', fontsize=10)
    ax.set_ylabel('Mean reward', fontsize=10)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    save_figure('plot5_residual_reward_vs_fuel.png')

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Residual RL vs MPC vs PPO — Final Mean Reward by scale', fontsize=12)

    best_ppo_reward = max(agent['mean_reward'] for agent in ppo_agents)
    best_ppo_label = max(ppo_agents, key=lambda agent: agent['mean_reward'])['label']

    for col_index, scale in enumerate(residual_scales):
        ax = axes[col_index]
        agents_this_scale = [agent for agent in residual_agents if agent['scale'] == scale]
        fn_labels = [agent['fn_name'] for agent in agents_this_scale]
        fn_rewards = [agent['mean_reward'] for agent in agents_this_scale]
        fn_colors = [agent['color'] for agent in agents_this_scale]

        bar_positions = np.arange(len(fn_labels))
        ax.bar(bar_positions, fn_rewards, color=fn_colors, alpha=0.85, width=0.6)
        ax.axhline(mpc_results['mean_reward'], color=slate, linewidth=1.4, linestyle='--',
                   label=f'MPC ({mpc_results["mean_reward"]:.0f})')
        ax.axhline(best_ppo_reward, color=muted_purple, linewidth=1.2, linestyle='-.',
                   label=f'Best PPO ({best_ppo_label}, {best_ppo_reward:.0f})')
        ax.axhline(0, color=gray, linewidth=0.7, linestyle=':')
        ax.set_xticks(bar_positions)
        ax.set_xticklabels(fn_labels, rotation=30, ha='right', fontsize=8)
        ax.set_title(f'Correction scale = {scale}', fontsize=10)
        ax.set_ylabel('Mean reward', fontsize=9)
        ax.legend(frameon=False, fontsize=7)

    save_figure('plot6_residual_vs_baselines_by_scale.png')

    print(f"All plots saved to {os.path.abspath(output_dir)}")


def run_all(mpc_episodes=mpc_num_episodes, ppo_steps=ppo_total_steps,
            residual_steps=residual_total_steps, seed=42):
    '''Runs the full experiment: MPC, all PPO agents, all residual agents, then saves plots.
    Args: mpc_episodes — evaluation episodes for MPC. ppo_steps — training budget per PPO agent.
          residual_steps — training budget per residual agent. seed — global random seed.
    Returns: None.'''
    print(f"Device: {device}")
    print(f"MPC episodes: {mpc_episodes}")
    print(f"PPO steps per agent: {ppo_steps} x5")
    print(f"Residual steps per agent: {residual_steps} x15  CEM replan every {cem_replan_every} steps")

    total_start = time.time()
    mpc_results = run_mpc(num_episodes=mpc_episodes, seed=seed)

    print(f"Training 5 PPO agents")
    ppo_agents = []
    for name, shape_fn, color in ppo_strategies:
        ppo_agents.append(run_ppo_agent(name, color, shape_fn, total_steps=ppo_steps, seed=seed))

    print(f"Training 15 residual RL agents")
    residual_agents = []
    for fn_name, shape_fn, fn_color in residual_strategies:
        for scale in residual_scales:
            residual_agents.append(run_residual_agent(fn_name, fn_color, shape_fn, scale,
                                                      total_steps=residual_steps, seed=seed))

    total_elapsed = time.time() - total_start
    print(f"All training done in {total_elapsed:.0f}s ({total_elapsed / 60:.1f} min)")

    all_results = (
        [{'label': 'MPC', 'positive_rate': mpc_results['positive_rate'], 'mean_reward': mpc_results['mean_reward']}]
        + [{'label': agent['label'], 'positive_rate': agent['positive_rate'], 'mean_reward': agent['mean_reward']} for agent in ppo_agents]
        + [{'label': agent['label'], 'positive_rate': agent['positive_rate'], 'mean_reward': agent['mean_reward']} for agent in residual_agents]
    )
    print("Final rankings by positive rate:")
    for rank, result in enumerate(sorted(all_results, key=lambda result_item: -result_item['positive_rate']), 1):
        print(f"{rank}. {result['label']} positive {result['positive_rate']:.0f}% mean {result['mean_reward']:.1f}")

    generate_plots(mpc_results, ppo_agents, residual_agents)


if __name__ == '__main__':
    mpc_episodes = mpc_num_episodes
    ppo_steps = ppo_total_steps
    residual_steps = residual_total_steps
    seed = 42

    for arg in sys.argv[1:]:
        if arg.startswith('--mpc-episodes='):
            mpc_episodes = int(arg.split('=')[1])
        elif arg.startswith('--ppo-steps='):
            ppo_steps = int(arg.split('=')[1])
        elif arg.startswith('--residual-steps='):
            residual_steps = int(arg.split('=')[1])
        elif arg.startswith('--seed='):
            seed = int(arg.split('=')[1])
        elif arg.startswith('--cem-replan-every='):
            cem_replan_every = int(arg.split('=')[1])

    run_all(mpc_episodes=mpc_episodes, ppo_steps=ppo_steps, residual_steps=residual_steps, seed=seed)