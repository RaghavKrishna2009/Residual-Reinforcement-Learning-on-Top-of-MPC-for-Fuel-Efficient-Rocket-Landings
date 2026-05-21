import sys, time, os, collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import gymnasium.envs.box2d.lunar_lander as lunar_lander_module

output_dir = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

color_blue   = '#4a7fb5'; color_red    = '#c0392b'; color_orange = '#d4823a'
color_green  = '#4a9e6b'; color_purple = '#7b5ea7'; color_slate  = '#2c5f8a'
color_grey   = '#888888'
function_colors = [color_blue, color_red, color_orange, color_green, color_purple]
function_names  = ['Function 1','Function 2','Function 3','Function 4','Function 5']
function_slugs  = ['function_1','function_2','function_3','function_4','function_5']

velocity_x_damping = 0.99865; velocity_y_damping = 0.99865; spin_damping = 0.98837
gravity_per_step = -0.02567; position_x_scale = 0.009873; position_y_scale = 0.022594
angle_scale = 0.049414; main_thrust_dvy = 0.047008; side_thrust_dvx = 0.009858
side_thrust_dspin = 0.039546

mpc_episodes = 150
mpc_config = dict(horizon=22, num_samples=18, num_iterations=1, replan_every=6, gamma=0.995)

ppo_parallel_envs   = 32; ppo_rollout_length = 256; ppo_minibatch_size = 512
ppo_learning_rate   = 2.5e-4; ppo_total_steps = 500_000

residual_parallel_envs  = 8
residual_rollout_length = 128
residual_minibatch_size = 256
residual_learning_rate  = 3e-4
residual_total_steps    = 300_000
residual_scales = [1.0]

discount_factor = 0.99; gae_lambda = 0.95; value_loss_coef = 0.5; max_grad_norm = 0.5
update_epochs = 4

EFFICIENCY_REWARD_LO = -200.0; EFFICIENCY_REWARD_HI = 250.0; EFFICIENCY_MAX_FUEL = 100.0

def compute_efficiency(rewards, fuel):
    """
    Calculate landing efficiency from rewards and fuel consumption.
    
    Arguments:
        rewards: array of episode rewards
        fuel: array of fuel consumption values
    
    Returns:
        efficiency: array of efficiency scores between 0 and 1
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    fuel    = np.asarray(fuel,    dtype=np.float64)
    quality      = np.clip((rewards-EFFICIENCY_REWARD_LO)/(EFFICIENCY_REWARD_HI-EFFICIENCY_REWARD_LO),0.,1.)
    fuel_economy = np.clip(1.-fuel/EFFICIENCY_MAX_FUEL, 0., 1.)
    denom = quality + fuel_economy
    safe  = np.where(denom>1e-9, denom, 1.0)
    eff   = np.where(denom>1e-9, 2.*quality*fuel_economy/safe, 0.0)
    return eff.astype(np.float32)

class ZeroWindRNG:
    """Random number generator that always returns zero for wind."""
    __slots__ = ('_rng',)
    def __init__(self, rng): self._rng = rng
    def uniform(self, low=-1.0, high=1.0, size=None):
        if size is None and low==-1.0 and high==1.0: return 0.0
        return self._rng.uniform(low, high, size)
    def __getattr__(self, name): return getattr(self._rng, name)

def _patched_step(self, action):
    """Remove wind from environment step."""
    real = self.np_random; self.np_random = ZeroWindRNG(real)
    r = original_step(self, action); self.np_random = real; return r

def _patched_reset(self, **kw):
    """Remove wind from environment reset."""
    r = original_reset(self, **kw); self.np_random = ZeroWindRNG(self.np_random); return r

original_step  = lunar_lander_module.LunarLander.step
original_reset = lunar_lander_module.LunarLander.reset
lunar_lander_module.LunarLander.step  = _patched_step
lunar_lander_module.LunarLander.reset = _patched_reset

def forward_model(states, actions):
    """
    Predict next state given current state and action.
    
    Arguments:
        states: array of shape (N, 8) containing current states
        actions: array of shape (N,) containing actions
    
    Returns:
        next_states: array of shape (N, 8) containing predicted next states
    """
    x_pos,y_pos,vel_x,vel_y,tilt_angle,spin_rate,leg1,leg2 = (states[:,i] for i in range(8))
    sin_angle=np.sin(tilt_angle); cos_angle=np.cos(tilt_angle)
    main_firing=(actions==2).astype(np.float32)
    side_direction=(actions==3).astype(np.float32)-(actions==1).astype(np.float32)
    new_vel_x=velocity_x_damping*vel_x+main_firing*(-sin_angle*main_thrust_dvy)+side_direction*side_thrust_dvx*cos_angle
    new_vel_y=velocity_y_damping*vel_y+gravity_per_step+main_firing*(cos_angle*main_thrust_dvy)+side_direction*side_thrust_dvx*sin_angle
    new_spin=spin_damping*spin_rate+side_direction*(-side_thrust_dspin)
    new_x=x_pos+new_vel_x*position_x_scale; new_y=y_pos+new_vel_y*position_y_scale
    new_angle=tilt_angle+spin_rate*angle_scale
    on_pad=new_y<=0.0; new_y=np.maximum(new_y,0.0)
    new_vel_y=np.where(on_pad&(new_vel_y<0.0),new_vel_y*-0.05,new_vel_y)
    new_vel_x=np.where(on_pad,new_vel_x*0.65,new_vel_x)
    new_spin=np.where(on_pad,new_spin*0.85,new_spin)
    leg_contact=((new_y<=0.04)&(np.abs(new_x)<0.38)).astype(np.float32)
    return np.stack([new_x,new_y,new_vel_x,new_vel_y,new_angle,new_spin,leg_contact,leg_contact],axis=1).astype(np.float32)

def shaping_potential(states):
    """
    Compute potential value for reward shaping.
    
    Arguments:
        states: array of shape (N, 8) containing states
    
    Returns:
        potential: array of shape (N,) containing potential values
    """
    return (-100.*np.sqrt(states[:,0]**2+states[:,1]**2)
            -100.*np.sqrt(states[:,2]**2+states[:,3]**2)
            -100.*np.abs(states[:,4])-50.*np.abs(states[:,5])
            +10.*states[:,6]+10.*states[:,7])

def terminal_value(states):
    """
    Estimate value of terminal states.
    
    Arguments:
        states: array of shape (N, 8) containing states
    
    Returns:
        value: array of shape (N,) containing terminal state values
    """
    net_decel=main_thrust_dvy+gravity_per_step
    vel_y=states[:,3].astype(np.float64); altitude=states[:,1].astype(np.float64)
    vel_x=states[:,2].astype(np.float64); tilt=states[:,4].astype(np.float64)
    spin_rate=states[:,5].astype(np.float64); still_flying=altitude>0.05
    downward_vy=np.minimum(vel_y,0.0)
    min_brake_alt=(downward_vy**2)*position_y_scale/(2.*max(net_decel,1e-6))
    altitude_deficit=np.maximum(min_brake_alt-altitude,0.0)
    penalty=-400.*altitude_deficit**2-12.*spin_rate**2-6.*tilt**2-3.*vel_x**2
    return np.where(still_flying, penalty, 0.0)

def score_action_sequences(initial_states, sequences, horizon, main_penalty=0.0, side_penalty=0.0):
    """
    Evaluate action sequences using the forward model.
    
    Arguments:
        initial_states: array of shape (N, 8) containing initial states
        sequences: array of shape (N, horizon) containing action sequences
        horizon: number of steps to simulate
        main_penalty: penalty for using main engine
        side_penalty: penalty for using side engines
    
    Returns:
        scores: array of shape (N,) containing total discounted rewards
    """
    states=initial_states.copy().astype(np.float32)
    previous_potential=shaping_potential(states)
    total_score=np.zeros(len(states),dtype=np.float64)
    gamma=0.995; terminated=np.zeros(len(states),dtype=bool)
    for step_index in range(horizon):
        if np.all(terminated): break
        actions=sequences[:,step_index]
        prev_vel_x=states[:,2].copy(); prev_vel_y=states[:,3].copy(); prev_y=states[:,1].copy()
        states=forward_model(states,actions)
        current_potential=shaping_potential(states)
        pot_diff=(current_potential-previous_potential).astype(np.float64)
        reward=pot_diff
        previous_potential=current_potential
        reward-=(actions==2).astype(np.float64)*main_penalty
        reward-=((actions==1)|(actions==3)).astype(np.float64)*side_penalty
        impact_speed=np.sqrt(prev_vel_x**2+prev_vel_y**2)
        post_speed=np.sqrt(states[:,2]**2+states[:,3]**2)
        spin=np.abs(states[:,5]); tilt=np.abs(states[:,4])
        out_of_bounds=np.abs(states[:,0])>=1.0
        hull_crash=((prev_y>0.06)&(states[:,1]<=0.02)&((impact_speed>0.40)|(spin>0.35)|(tilt>0.35)))
        soft_landed=((states[:,1]<=0.05)&(impact_speed<0.30)&(post_speed<0.22)&(spin<0.14)&(tilt<0.18)&(np.abs(states[:,0])<0.40))
        terminal_bonus=np.zeros(len(states),dtype=np.float64)
        terminal_bonus[out_of_bounds|hull_crash]=-100.0; terminal_bonus[soft_landed]=100.0
        active=~terminated
        total_score+=active*(gamma**step_index)*reward
        total_score+=active*terminal_bonus
        terminated|=out_of_bounds|hull_crash|soft_landed
    still_flying=~terminated
    total_score+=still_flying*(gamma**horizon)*terminal_value(states)
    return total_score

def lunar_heuristic_action(obs):
    """
    Compute heuristic action for lunar lander.
    
    Arguments:
        obs: array of shape (8,) containing observation
    
    Returns:
        action: integer from 0 to 3 (0: none, 1: left, 2: main, 3: right)
    """
    s=obs; angle_targ=float(np.clip(s[0]*0.5+s[2]*1.0,-0.4,0.4))
    hover_targ=0.55*abs(float(s[0])); angle_todo=(angle_targ-s[4])*0.5-s[5]*1.0
    hover_todo=(hover_targ-s[1])*0.5-s[3]*0.5
    if s[6] or s[7]: angle_todo=0.0; hover_todo=-s[3]*0.5
    if hover_todo>abs(angle_todo) and hover_todo>0.05: return 2
    if angle_todo<-0.05: return 3
    if angle_todo>0.05: return 1
    return 0

def action_to_probs(action, certainty=0.94):
    """
    Convert action to probability distribution.
    
    Arguments:
        action: integer action
        certainty: probability mass for the chosen action
    
    Returns:
        probs: array of shape (4,) containing action probabilities
    """
    p=np.full(4,(1.-certainty)/3.,dtype=np.float32); p[int(action)]=certainty; return p

def _lunar_core(env):
    """Get the underlying LunarLander environment object."""
    env=env.unwrapped
    while hasattr(env,'env'): env=env.env.unwrapped
    return env

def read_lunar_obs(env):
    """
    Read observation from lunar lander environment.
    
    Arguments:
        env: lunar lander environment
    
    Returns:
        obs: array of shape (8,) containing normalized observation
    """
    u=_lunar_core(env); pos=u.lander.position; vel=u.lander.linearVelocity
    vw=lunar_lander_module.VIEWPORT_W; vh=lunar_lander_module.VIEWPORT_H
    scale=lunar_lander_module.SCALE; fps=lunar_lander_module.FPS
    leg_down=lunar_lander_module.LEG_DOWN
    return np.array([(pos.x-vw/scale/2)/(vw/scale/2),(pos.y-(u.helipad_y+leg_down/scale))/(vh/scale/2),
                     vel.x*(vw/scale/2)/fps,vel.y*(vh/scale/2)/fps,u.lander.angle,
                     20.*u.lander.angularVelocity/fps,
                     1. if u.legs[0].ground_contact else 0.,
                     1. if u.legs[1].ground_contact else 0.],dtype=np.float32)

def snapshot_lunar_env(env):
    """
    Capture environment state for later restoration.
    
    Arguments:
        env: lunar lander environment
    
    Returns:
        snap: dictionary containing environment state
    """
    u=_lunar_core(env); lander=u.lander
    snap=dict(lander_pos=(lander.position.x,lander.position.y),lander_angle=lander.angle,
              lander_vel=(lander.linearVelocity.x,lander.linearVelocity.y),
              lander_angvel=lander.angularVelocity,legs=[],prev_shaping=u.prev_shaping,
              game_over=u.game_over,wind_idx=getattr(u,'wind_idx',0),torque_idx=getattr(u,'torque_idx',0))
    for leg in u.legs:
        snap['legs'].append(dict(pos=(leg.position.x,leg.position.y),angle=leg.angle,
                                  vel=(leg.linearVelocity.x,leg.linearVelocity.y),
                                  angvel=leg.angularVelocity,ground_contact=leg.ground_contact))
    return snap

def lunar_env_step(env,action):
    """Take a step in the lunar lander environment."""
    return _lunar_core(env).step(int(action))

def restore_lunar_env(env,snap):
    """Restore environment state from snapshot."""
    u=_lunar_core(env); lander=u.lander
    lander.position=snap['lander_pos']; lander.angle=snap['lander_angle']
    lander.linearVelocity=snap['lander_vel']; lander.angularVelocity=snap['lander_angvel']
    lander.awake=True
    for leg,ls in zip(u.legs,snap['legs']):
        leg.position=ls['pos']; leg.angle=ls['angle']; leg.linearVelocity=ls['vel']
        leg.angularVelocity=ls['angvel']; leg.ground_contact=ls['ground_contact']; leg.awake=True
    u.prev_shaping=snap['prev_shaping']; u.game_over=snap['game_over']
    if hasattr(u,'wind_idx'): u.wind_idx=snap['wind_idx']
    if hasattr(u,'torque_idx'): u.torque_idx=snap['torque_idx']
    u.world.ClearForces()

def rollout_sequence_return(env,snap,sequence,gamma=0.99):
    """
    Execute action sequence and return total discounted reward.
    
    Arguments:
        env: lunar lander environment
        snap: environment snapshot to restore from
        sequence: array of actions
        gamma: discount factor
    
    Returns:
        total: total discounted reward
    """
    restore_lunar_env(env,snap); total=0.; discount=1.
    for action in sequence:
        _,reward,terminated,truncated,_=lunar_env_step(env,action)
        total+=discount*float(reward); discount*=gamma
        if terminated or truncated: break
    restore_lunar_env(env,snap); return total

def build_heuristic_sequence(env,snap,horizon,gamma=0.99):
    """
    Build action sequence using heuristic policy.
    
    Arguments:
        env: lunar lander environment
        snap: environment snapshot
        horizon: sequence length
        gamma: discount factor
    
    Returns:
        sequence: array of actions
        total: total discounted reward
    """
    restore_lunar_env(env,snap); sequence=np.zeros(horizon,dtype=np.int32)
    total=0.; discount=1.
    for t in range(horizon):
        obs=read_lunar_obs(env); action=int(lunar_heuristic_action(obs))
        sequence[t]=action; _,reward,terminated,truncated,_=lunar_env_step(env,action)
        total+=discount*float(reward); discount*=gamma
        if terminated or truncated: break
    restore_lunar_env(env,snap); return sequence,total

def sample_sequence_candidates(ref_seq,num_samples,horizon,rng):
    """
    Generate candidate sequences by mutating reference sequence.
    
    Arguments:
        ref_seq: reference action sequence
        num_samples: number of candidates to generate
        horizon: sequence length
        rng: random number generator
    
    Returns:
        candidates: array of shape (num_samples, horizon) containing action sequences
    """
    candidates=np.tile(ref_seq,(num_samples,1)); gentle_cutoff=max(2,num_samples//2)
    for i in range(1,num_samples):
        n_mut=1 if i<gentle_cutoff else int(rng.integers(2,max(3,horizon//5)+1))
        times=rng.integers(0,horizon,size=n_mut); candidates[i,times]=rng.integers(0,4,size=n_mut)
    return candidates

class EnvMPCPlanner:
    """MPC planner that uses the actual environment for rollouts."""
    
    def __init__(self,env,horizon=22,num_samples=44,num_iterations=2,
                 replan_every=2,gamma=0.99,seed=0):
        self.env=env; self.horizon=horizon; self.num_samples=num_samples
        self.num_iterations=num_iterations; self.replan_every=replan_every
        self.gamma=gamma; self.rng=np.random.default_rng(seed)
        self.cached_plan=None; self.plan_cursor=0

    def reset(self): self.cached_plan=None; self.plan_cursor=0

    def _effective_horizon(self,obs):
        alt=float(obs[1]); return min(10,self.horizon) if alt>0.55 else int(np.clip(10+alt*28,10,self.horizon))

    def _optimize(self,snap,horizon):
        best_seq,best_score=build_heuristic_sequence(self.env,snap,horizon,self.gamma)
        for _ in range(self.num_iterations):
            candidates=sample_sequence_candidates(best_seq,self.num_samples,horizon,self.rng)
            for seq in candidates[1:]:
                score=rollout_sequence_return(self.env,snap,seq,self.gamma)
                if score>best_score: best_score=score; best_seq=seq.copy()
        counts=np.stack([(best_seq==a).mean() for a in range(4)],axis=0)
        probs=0.15+0.85*counts; probs/=probs.sum()
        return best_seq,probs.astype(np.float32)

    def select_action(self,obs):
        """
        Select action using MPC planning.
        
        Arguments:
            obs: observation array
        
        Returns:
            action: chosen action
            probs: action probabilities
        """
        if (self.cached_plan is not None and
                self.plan_cursor<min(self.replan_every,len(self.cached_plan))):
            action=int(self.cached_plan[self.plan_cursor]); self.plan_cursor+=1
            return action,action_to_probs(action)
        snap=snapshot_lunar_env(self.env); horizon=self._effective_horizon(obs)
        plan,probs=self._optimize(snap,horizon)
        heuristic=int(lunar_heuristic_action(obs))
        if plan[0]!=heuristic:
            mpc_score=rollout_sequence_return(self.env,snap,plan[:1],1.0)
            h_score=rollout_sequence_return(self.env,snap,np.array([heuristic],dtype=np.int32),1.0)
            if h_score>mpc_score: plan[0]=heuristic; probs=action_to_probs(heuristic)
        self.cached_plan=plan; self.plan_cursor=1
        return int(plan[0]),probs

class VectorizedCEMPlanner:
    """Vectorized CEM planner that processes all environments together."""
    
    def __init__(self, num_envs, n_samples_per_env=200, horizon=22,
                 n_iterations=2, main_penalty=0.0, side_penalty=0.0):
        self.num_envs         = num_envs
        self.n_samples        = n_samples_per_env
        self.horizon          = horizon
        self.n_iterations     = n_iterations
        self.main_penalty     = main_penalty
        self.side_penalty     = side_penalty
        self.priors = np.ones((num_envs, horizon, 4), dtype=np.float32) / 4.0

    def plan(self, raw_obs_batch):
        """
        Plan actions for multiple environments.
        
        Arguments:
            raw_obs_batch: array of shape (N, 8) containing raw observations
        
        Returns:
            actions: array of shape (N,) containing chosen actions
            probs: array of shape (N, 4) containing action probabilities
        """
        N = self.num_envs
        S = self.n_samples
        H = self.horizon

        best_actions = np.zeros(N, dtype=np.int64)
        best_probs   = np.zeros((N, 4), dtype=np.float32)

        heuristic_actions = np.array(
            [lunar_heuristic_action(raw_obs_batch[i]) for i in range(N)],
            dtype=np.int64
        )

        for env_idx in range(N):
            obs      = raw_obs_batch[env_idx]
            prior    = self.priors[env_idx].copy()
            best_seq = None
            best_score = -np.inf

            for iteration in range(self.n_iterations):
                cumprob = np.cumsum(prior, axis=1)
                draws   = np.random.rand(S, H, 1)
                seqs    = (draws > cumprob[None]).sum(axis=2).astype(np.int32)

                seqs[0, :] = heuristic_actions[env_idx]

                states = np.tile(obs, (S, 1)).astype(np.float32)
                scores = score_action_sequences(
                    states, seqs, H,
                    self.main_penalty, self.side_penalty
                )

                top_idx = int(np.argmax(scores))
                if scores[top_idx] > best_score:
                    best_score = scores[top_idx]
                    best_seq   = seqs[top_idx].copy()

                k = max(2, S // 5)
                elite_idx = np.argpartition(scores, -k)[-k:]
                elite_seqs = seqs[elite_idx]
                new_prior = np.stack(
                    [(elite_seqs == a).mean(axis=0) for a in range(4)], axis=1
                ).astype(np.float32)
                prior = 0.7 * new_prior + 0.3 / 4.0
                prior /= prior.sum(axis=1, keepdims=True)

            self.priors[env_idx, :-1] = prior[1:]
            self.priors[env_idx, -1]  = 0.25

            best_actions[env_idx] = int(best_seq[0]) if best_seq is not None else heuristic_actions[env_idx]
            first_step_prior = prior[0].copy()
            first_step_prior = np.clip(first_step_prior, 0.02, None)
            first_step_prior /= first_step_prior.sum()
            best_probs[env_idx] = first_step_prior

        return best_actions, best_probs

    def reset_env(self, env_idx):
        """Reset prior for environment that just terminated."""
        self.priors[env_idx] = 0.25

class RunningNormaliser:
    """Normalize observations using running statistics."""
    
    def __init__(self, shape, clip=10.0, epsilon=1e-8):
        self.mean=np.zeros(shape,dtype=np.float64); self.var=np.ones(shape,dtype=np.float64)
        self.count=epsilon; self.clip=clip; self.epsilon=epsilon

    def update(self,data):
        """Update running statistics with new data."""
        bm=data.mean(axis=0); bv=data.var(axis=0); bc=data.shape[0]
        tc=self.count+bc; delta=bm-self.mean; self.mean+=delta*bc/tc
        self.var=(self.var*self.count+bv*bc+delta**2*self.count*bc/tc)/tc
        self.count=tc

    def normalise(self,data):
        """Normalize data using current statistics."""
        n=(data-self.mean)/np.sqrt(self.var+self.epsilon)
        return np.clip(n,-self.clip,self.clip).astype(np.float32)

class PPONetwork(nn.Module):
    """Neural network for PPO agent."""
    
    def __init__(self,obs_dim=8,num_actions=4,hidden_size=256):
        super().__init__()
        self.trunk=nn.Sequential(nn.Linear(obs_dim,hidden_size),nn.LayerNorm(hidden_size),nn.Tanh(),
                                  nn.Linear(hidden_size,hidden_size),nn.LayerNorm(hidden_size),nn.Tanh())
        self.actor_head=nn.Linear(hidden_size,num_actions)
        self.critic_head=nn.Linear(hidden_size,1)
        for layer in self.trunk:
            if isinstance(layer,nn.Linear):
                nn.init.orthogonal_(layer.weight,gain=np.sqrt(2)); nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor_head.weight,gain=0.01); nn.init.zeros_(self.actor_head.bias)
        nn.init.orthogonal_(self.critic_head.weight,gain=1.0); nn.init.zeros_(self.critic_head.bias)

    def forward(self,x):
        """Forward pass through the network."""
        h=self.trunk(x); return self.actor_head(h),self.critic_head(h).squeeze(-1)

    def get_action(self,x):
        """Sample an action from the policy."""
        logits,val=self(x); dist=torch.distributions.Categorical(logits=logits)
        a=dist.sample(); return a,dist.log_prob(a),val

    def evaluate(self,x,a):
        """Evaluate log probability, value, and entropy for given actions."""
        logits,val=self(x); dist=torch.distributions.Categorical(logits=logits)
        return dist.log_prob(a),val,dist.entropy()

class FixedResidualNetwork(nn.Module):
    """Residual network that blends MPC policy with learned correction."""
    
    def __init__(self, hidden_size=128):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(8, hidden_size), nn.LayerNorm(hidden_size), nn.Tanh(),
        )
        self.mpc_embed = nn.Sequential(
            nn.Linear(4, hidden_size // 2), nn.LayerNorm(hidden_size // 2), nn.Tanh(),
        )
        self.trunk = nn.Sequential(
            nn.Linear(hidden_size + hidden_size // 2, hidden_size),
            nn.LayerNorm(hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size), nn.Tanh(),
        )
        self.actor_head  = nn.Linear(hidden_size, 4)
        self.critic_head = nn.Linear(hidden_size, 1)
        self.gate_head   = nn.Linear(hidden_size, 1)

        for seq in [self.obs_encoder, self.mpc_embed, self.trunk]:
            for layer in seq:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    nn.init.zeros_(layer.bias)

        nn.init.zeros_(self.actor_head.weight)
        nn.init.zeros_(self.actor_head.bias)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.zeros_(self.critic_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)

    def _dist_and_value(self, obs_t, mpc_probs_t):
        """Compute action distribution and value from observations and MPC probs."""
        obs_feat = self.obs_encoder(obs_t)
        mpc_action = mpc_probs_t.argmax(dim=-1)
        mpc_onehot = torch.zeros_like(mpc_probs_t)
        mpc_onehot.scatter_(1, mpc_action.unsqueeze(1), 1.0)
        mpc_feat = self.mpc_embed(mpc_onehot)
        joint = torch.cat([obs_feat, mpc_feat], dim=-1)
        hidden = self.trunk(joint)
        correction = self.actor_head(hidden)
        value      = self.critic_head(hidden).squeeze(-1)
        gate       = torch.sigmoid(self.gate_head(hidden))
        mpc_logits = torch.log(mpc_probs_t.clamp(1e-8))
        logits     = gate * mpc_logits + (1. - gate) * correction
        dist = torch.distributions.Categorical(logits=logits)
        return dist, value

    def get_action(self, obs_t, mpc_probs_t):
        """Sample an action from the residual policy."""
        dist, value = self._dist_and_value(obs_t, mpc_probs_t)
        action = dist.sample()
        return action, dist.log_prob(action), value

    def evaluate(self, obs_t, mpc_probs_t, actions_t):
        """Evaluate log probability, value, and entropy for given actions."""
        dist, value = self._dist_and_value(obs_t, mpc_probs_t)
        return dist.log_prob(actions_t), value, dist.entropy()

    def forward(self, obs_t, mpc_probs_t):
        """Forward pass returning logits and value."""
        dist, value = self._dist_and_value(obs_t, mpc_probs_t)
        return dist.logits, value

def run_ppo_update(network, optimiser, obs_buf, action_buf, log_prob_buf,
                   reward_buf, done_buf, value_buf, num_envs, rollout_length,
                   minibatch_size, clip_ratio, entropy_coef,
                   mpc_buf=None, next_obs=None, next_mpc=None):
    """Perform PPO update on the policy network."""
    with torch.no_grad():
        bootstrap_obs = torch.from_numpy(
            next_obs if next_obs is not None else obs_buf[-1]
        ).to(device)
        if mpc_buf is not None:
            bootstrap_mpc = torch.from_numpy(
                next_mpc if next_mpc is not None else mpc_buf[-1]
            ).to(device)
            _, last_values = network(bootstrap_obs, bootstrap_mpc)
        else:
            _, last_values = network(bootstrap_obs)

    last_values_np = last_values.cpu().numpy()
    advantages  = np.zeros_like(reward_buf)
    running_gae = np.zeros(num_envs, np.float32)
    for step in reversed(range(rollout_length)):
        next_value = last_values_np if step == rollout_length-1 else value_buf[step+1]
        not_done   = 1.0 - done_buf[step]
        td_error   = reward_buf[step] + discount_factor*next_value*not_done - value_buf[step]
        running_gae = td_error + discount_factor*gae_lambda*not_done*running_gae
        advantages[step] = running_gae
    returns = advantages + value_buf

    obs_flat     = torch.from_numpy(obs_buf.reshape(-1, obs_buf.shape[-1])).to(device)
    actions_flat = torch.from_numpy(action_buf.reshape(-1)).to(device)
    old_lp_flat  = torch.from_numpy(log_prob_buf.reshape(-1)).to(device)
    adv_flat     = torch.from_numpy(advantages.reshape(-1)).to(device)
    ret_flat     = torch.from_numpy(returns.reshape(-1)).to(device)
    old_val_flat = torch.from_numpy(value_buf.reshape(-1)).to(device)
    mpc_flat     = torch.from_numpy(mpc_buf.reshape(-1,4)).to(device) if mpc_buf is not None else None

    total_samples = obs_flat.shape[0]
    for _ in range(update_epochs):
        indices = torch.randperm(total_samples, device=device)
        for start in range(0, total_samples, minibatch_size):
            bidx     = indices[start:start+minibatch_size]
            batch_adv = adv_flat[bidx]
            batch_adv = (batch_adv - batch_adv.mean()) / (batch_adv.std() + 1e-8)

            if mpc_flat is not None:
                new_lp, vals, entropy = network.evaluate(obs_flat[bidx], mpc_flat[bidx], actions_flat[bidx])
            else:
                new_lp, vals, entropy = network.evaluate(obs_flat[bidx], actions_flat[bidx])

            ratio = (new_lp - old_lp_flat[bidx]).exp()
            actor_loss = torch.max(-batch_adv*ratio,
                                   
