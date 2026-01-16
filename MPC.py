import gymnasium as gym
import numpy as np
import casadi as ca
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
from collections import deque


NUM_TRAINING_EPISODES = 20
NUM_EVAL_EPISODES = 5


class ResidualActor(Model):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(128, activation='relu')
        self.out = layers.Dense(action_dim, activation='tanh')

    def call(self, state):
        '''
        Compute residual action from state.
        
        :param state: Current state tensor
        :return: Tanh-activated action correction
        '''
        x = self.dense1(state)
        x = self.dense2(x)
        return self.out(x)



# Hyperparameters
state_dim = 6
action_dim = 2
learning_rate = 1e-3
gamma = 0.99
max_steps = 500
residual_scale = 0.1


residual_actor = ResidualActor(state_dim, action_dim)
optimizer = optimizers.Adam(learning_rate=learning_rate)

# MPC Parameters
dt = 0.05
N = 20
g = -10.0

u_min = np.array([0.0, -1.0])
u_max = np.array([1.0, 1.0])

# Landing pad
x_limit = 0.5
y_min = 0.0
vx_max = 2.0
vy_max = 2.0
theta_max = 0.5
omega_max = 2.0

# Weights for numerical values
Q = np.diag([50.0, 50.0, 1.0, 5.0, 100.0, 10.0])
R = np.diag([0.1, 0.1])
target_state = np.zeros(6)

# MPC
x = ca.SX.sym('x')
y = ca.SX.sym('y')
vx = ca.SX.sym('vx')
vy = ca.SX.sym('vy')
theta = ca.SX.sym('theta')
omega = ca.SX.sym('omega')
state_sym = ca.vertcat(x, y, vx, vy, theta, omega)

u1 = ca.SX.sym('u1')
u2 = ca.SX.sym('u2')
control_sym = ca.vertcat(u1, u2)

vx_dot = u1 * ca.sin(theta)
vy_dot = u1 * ca.cos(theta) + g
omega_dot = u2

xdot = ca.vertcat(vx, vy, vx_dot, vy_dot, omega, omega_dot)
state_next = state_sym + dt * xdot
f_dynamics = ca.Function('f_dynamics', [state_sym, control_sym], [state_next])


def compute_mpc_action(state):
    '''
    Solve MPC optimization for given state.
    
    :param state: Current 6D state vector
    :return: Optimal control action (2D array)
    '''
    try:
        X = ca.SX.sym('X', 6, N + 1)
        U = ca.SX.sym('U', 2, N)

        cost = 0
        g_cons = []

        for k in range(N):
            state_k = X[:, k]
            control_k = U[:, k]

            dif = state_k - target_state
            cost += ca.mtimes([dif.T, Q, dif]) + ca.mtimes([control_k.T, R, control_k])

            x_next = f_dynamics(state_k, control_k)
            g_cons.append(X[:, k + 1] - x_next)

            g_cons.append(x_limit - ca.fabs(state_k[0]))
            g_cons.append(state_k[1] - y_min)
            g_cons.append(vx_max - ca.fabs(state_k[2]))
            g_cons.append(vy_max - ca.fabs(state_k[3]))
            g_cons.append(theta_max - ca.fabs(state_k[4]))
            g_cons.append(omega_max - ca.fabs(state_k[5]))

        dif_N = X[:, N] - target_state
        cost += ca.mtimes([dif_N.T, Q, dif_N])

        g_cons = ca.vertcat(*g_cons)
        opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        nlp = {'x': opt_vars, 'f': cost, 'g': g_cons}

        solver = ca.nlpsol('solver', 'ipopt', nlp, {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.max_iter': 500,
            'ipopt.acceptable_tol': 1e-6
        })

        X_init = np.tile(state.reshape(6, 1), (1, N + 1))
        U_init = np.zeros((2, N))
        init_vals = np.concatenate((X_init.flatten(), U_init.flatten()))

        ng = g_cons.size1()
        lbg = np.zeros(ng)
        ubg = np.inf * np.ones(ng)

        nx = 6 * (N + 1)
        lbx = np.concatenate([
            -np.inf * np.ones(nx),
            np.tile(u_min, N)
        ])
        ubx = np.concatenate([
            np.inf * np.ones(nx),
            np.tile(u_max, N)
        ])

        sol = solver(x0=init_vals, lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)

        if solver.stats()['success']:
            nv = sol["x"].full().flatten()
            U_seq = nv[nx:].reshape((2, N))
            return np.clip(U_seq[:, 0], u_min, u_max)
        else:
            return np.array([0.5, 0.0])

    except Exception as e:
        return np.array([0.5, 0.0])



def compute_returns(rewards, gamma):
    '''
    Calculate normalized discounted returns.
    
    :param rewards: List of episode rewards
    :param gamma: Discount factor
    :return: Normalized returns array
    '''
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = np.array(returns)
    if len(returns) > 1:
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
    return returns


def run_episode(env, residual_actor, residual_scale, max_steps, train=True):
    '''
    Execute one episode with MPC + residual policy.
    
    :param env: Gymnasium environment
    :param residual_actor: Neural network for residual actions
    :param residual_scale: Scaling factor for residual
    :param max_steps: Maximum episode length
    :param train: Whether to collect training data
    :return: states, rewards, total_reward, steps
    '''
    state, _ = env.reset()
    state = np.array(state[:6], dtype=np.float32)
    done = False
    step = 0
    total_reward = 0

    states = []
    rewards = []

    while not done and step < max_steps:
        u_mpc = compute_mpc_action(state)

        state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
        u_res_raw = residual_actor(state_tensor).numpy().flatten()
        u_res = u_res_raw * residual_scale

        u_total = u_mpc + u_res
        u_total = np.clip(u_total, u_min, u_max)

        next_state, reward, terminated, truncated, _ = env.step(u_total)
        total_reward += reward

        if train:
            states.append(state)
            rewards.append(reward)

        state = np.array(next_state[:6], dtype=np.float32)
        done = terminated or truncated
        step += 1

    return states, rewards, total_reward, step



# TRAINING
print("TRAINING PHASE -", NUM_TRAINING_EPISODES, "Episodes (No Rendering - Fast)")

train_env = gym.make("LunarLanderContinuous-v3")
recent_rewards = deque(maxlen=10)

for ep in range(NUM_TRAINING_EPISODES):
    states, rewards, total_reward, steps = run_episode(
        train_env, residual_actor, residual_scale, max_steps, train=True
    )

    if len(states) > 0:
        returns = compute_returns(rewards, gamma)

        states_batch = tf.convert_to_tensor(np.array(states), dtype=tf.float32)
        returns_batch = tf.convert_to_tensor(returns, dtype=tf.float32)

        with tf.GradientTape() as tape:
            residual_preds = residual_actor(states_batch)
            loss = -tf.reduce_mean(
                tf.reduce_sum(residual_preds * tf.expand_dims(returns_batch, 1), axis=1)
            )

        grads = tape.gradient(loss, residual_actor.trainable_variables)
        grads = [tf.clip_by_norm(g, 1.0) for g in grads]
        optimizer.apply_gradients(zip(grads, residual_actor.trainable_variables))

    recent_rewards.append(total_reward)
    avg_reward = np.mean(recent_rewards)

    # Logs
    status = "✓ SUCCESS" if total_reward > 200 else "Training"
    print("Episode", str(ep + 1).rjust(3), "/", NUM_TRAINING_EPISODES, "COMPLETE",
          "Reward:", str(round(total_reward, 2)).rjust(8),
          "Steps:", str(steps).rjust(3),
          status)

train_env.close()

print("TRAINING OVER")

# TESTING
print(NUM_EVAL_EPISODES, "Episodes (With Rendering)")
print("Watch the trained model in action!\n")

#rendering
eval_env = gym.make("LunarLanderContinuous-v3", render_mode="human")

eval_rewards = []
successes = 0

for ep in range(NUM_EVAL_EPISODES):
    _, _, total_reward, steps = run_episode(
        eval_env, residual_actor, residual_scale, max_steps, train=False
    )

    eval_rewards.append(total_reward)

    is_success = total_reward > 200
    if is_success:
        successes += 1

    status = "YESSS" if is_success else "NOOOOO"
    print("Eval Episode", ep + 1, "/", NUM_EVAL_EPISODES, "COMPLETE",
          "Reward:", str(round(total_reward, 2)).rjust(8),
          status)

eval_env.close()

print("FINAL EVALUATION RESULTS")
print("Average Reward:    ", str(round(np.mean(eval_rewards), 2)).rjust(8))
print("Std Deviation:     ", str(round(np.std(eval_rewards), 2)).rjust(8))
print("Success Rate:      ", successes, "/", NUM_EVAL_EPISODES, "(", round(100 * successes / NUM_EVAL_EPISODES, 1), "%)")
