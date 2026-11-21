# Methods notes

Long-form companion to `README.md`. The first half describes how the three controllers work.
The second records why each design decision was made, including the ones that are not settled.
Numbers quoted here are read back from the result files, not retyped.

---

## How the three controllers work

### 1. MPC, planning in a model fitted to flight data

**The model.** The planner needs to predict what the lander will do, and it is not allowed to
ask the simulator. So a model is fitted to recorded flights. Its *shape* comes from rigid-body
mechanics: firing an engine changes the two velocities and the spin by a fixed amount, rotated by
however the lander is currently tilted, and position follows from velocity. Written out, with `m`
meaning the main engine is firing and `d` being −1 or +1 for the two side engines:

```
vx' = vx + a1·m·(−sin θ) + a2·d·cos θ + a3
vy' = vy + b1·m·cos θ    + b2·d·sin θ + b3        (b3 is gravity)
w'  = w  + c1·d + c2·d·sin²θ + c3·m + c4
x'  = x + kx·vx'      y' = y + ky·vy'      θ' = θ + kθ·w'
```

The eleven coefficients are *measured*, by least squares on 200 flights of the environment's own
heuristic with 30 % random actions mixed in so every engine gets used at every attitude. The fit is
run twice: once on everything, then again after discarding the worst 1 % of residuals, which are
ground impacts the flight model is not meant to describe. Only steps with both legs off the ground
are used. `--sysid` reproduces this and reports how well the result predicts held-out flights.

**The objective.** A sequence of firings is scored by adding up, at each step it is still in the
air: the speed change the firing costs, a small charge per step so hovering is not free, and
penalties for drifting below a glide slope towards the pad, tilting past 0.4 radians, or descending
faster than the engine could still arrest. When the sequence reaches the ground it is charged for
its touchdown speed, its tilt and spin, and how far outside the pad it landed. If it is still
airborne at the end of the look-ahead it is charged an estimate of the speed change still needed, the speed an unpowered fall would build up, plus the sideways speed, counted separately because one
engine cannot cancel both at full efficiency.

**The search.** Cross-entropy method. Sample a population of action sequences, score every one in
the model, keep the best ten, re-aim the sampling distribution at them, repeat four times. Actions
are held for four steps at a time, which cuts the search space from 4⁴⁸ to 4¹². Each call starts
from two free candidates: the previous solution shifted forward one step, and the environment's own
heuristic flown inside the model. The planner re-solves at every step.

**Wind.** The planner cannot see the wind, so it infers it. Whatever the model fails to predict
about the two velocities and the spin is treated as a slowly changing push and tracked with a fixed
gain. That leftover is, in practice, the wind. The planner then predicts *with* the estimate
included. This is the difference between holding a course in wind and being blown off it, and it is
worth nothing in calm air and a great deal at wind 15 and 20.

### 2. PPO, a network flying on its own

A standard PPO agent: two shared layers of 256 units with layer normalisation and tanh, one head
scoring the four actions and one estimating how good the state is. Observations are normalised by a
running mean and spread. 32 environments in parallel, 256-step rollouts, Adam at 2.5 × 10⁻⁴, four
epochs per batch, minibatches of 512, generalised advantage estimation with γ = 0.99 and λ = 0.95,
clipping at 0.2, and an entropy bonus falling from 0.03 to 0.005 across training so the agent keeps
exploring instead of committing early.

**Shaping.** Five agents are trained, differing only in how much *caution* is added to the reward:

```
risk(s)   = (vx² + min(vy, 0)²) / (max(y, 0) + 0.1) + θ²
Φ_k(s)    = −25 · c_k · risk(s)            c_k = 0, 0.5, 1, 2, 4
shaping   = γ·Φ_k(s') · (1 − terminal) − Φ_k(s)
```

In words: being fast is dangerous in proportion to how close the ground is, and being tilted is
dangerous. Agent 1 gets no shaping at all; agent 5 gets four times the middle amount. Because the
term depends only on the state, adding it changes *how quickly* an agent learns but not *which*
behaviour is finally best, so the five agents are five different learning signals aimed at the same
target. **No shaping term mentions fuel.** Fuel is measured, never rewarded.

Rewards are divided by the running spread of returns before use. Without this the value head, which
shares its layers with the policy, is swamped by the environment's ±100 terminal rewards and the
agent learns nothing.

### 3. Residual, the planner proposes, the network adjusts

The planner produces a probability over the four actions instead of a single choice: the first
actions of its best candidates vote, and every action keeps a floor of probability so the
distribution is always usable. That is sharpened, and the network's output is added in log space:

```
π_MPC(a|s)  = (f + (1 − f) · votes(a)/11) / (4f + 1 − f)       f = PROB_FLOOR
π_τ(a|s)    = softmax(log π_MPC / τ)                           τ = RESIDUAL_TAU
π(a|s)      = softmax(log π_τ + λ · δ_θ(s)[a])                  δ ∈ (−3, 3)
```

The constants `PROB_FLOOR` and `RESIDUAL_TAU` are defined in `rocket_landing.py`, and each trained
agent records the values it used in its own `hp` field, so an agent can be interpreted without
reference to the current source.

The network sees the eight observations plus the planner's four log-probabilities, and outputs one
adjustment per action, limited by a tanh. λ sets how much weight the adjustment carries. Nine values
of λ are reported, from 1 to 5 in steps of 0.5, at fifteen training seeds each.

Two properties matter. First, **at λ = 0 this is exactly the planner**, the architecture cannot do
worse than its own base except by learning to. Second, the network's last layer is initialised with
gain 0.01/max(1, λ), so training always begins from "leave the planner alone" whatever λ is, and the
comparison across λ starts from the same place.

Training is PPO again, on the combined policy, with actions sampled from it and that sampled
log-probability entering the ratio. The entropy bonus falls from 0.03 to 0.005, a slower decay than
the standalone agents use, because the residual policy starts already competent and needs to keep
exploring longer to find a correction worth making.

The shaping term has the same *form* as the standalone agents' caution term, with its own weight:

```
Φ(s) = −B · c · [ (vx² + min(vy, 0)²)/(max(y, 0) + a₀) + t · θ² ]
```

with `B = SHAPING_BASE`, `c = SHAPING_SCALE`, `a₀ = SHAPING_ALT0` and `t = SHAPING_TILT`, all defined
in `rocket_landing.py` and recorded on every trained agent. Like the standalone shaping it is a
function of the state alone, so it changes how quickly the agent learns and not which behaviour is
finally best.

Both learned families train for **2,000,000 steps** (2,007,040 exactly, being a whole number of
update batches), on environments whose wind strength is re-drawn uniformly from 0 to 20 every
episode. `--check` reads both bundles and asserts the budgets are equal.

---


---

## Design decisions

Every constant in this project has a reason, and in a few cases the reason is only that it was
fixed early and never revisited. Each entry below gives a **short** answer and a **longer** one.
The longer answer is where the shaky parts are recorded, for anyone picking the project up again.

### The exact action-distribution transformation

**Short.** The planner's distribution is now written exactly as implemented, and `action_distribution`
asserts on every call that it sums to one and is strictly positive: each action keeps a floor of 0.15 before normalising by 1.45, and
the temperature step is a softmax of the log-probabilities divided by τ. The combined policy is
`softmax(log π_τ + λ·δ)`. During training actions are **sampled** from that combined policy and it
is that sampled log-probability which enters the PPO ratio. At evaluation the action is the
**greedy** one. Both are stated wherever a number is reported.

**Longer.** The original text could be read three ways and the three read differently in code. The
resolution that works: the distribution describes the *next action only*, it is a vote among the
first actions of the best candidate plans, not a tally over the whole look-ahead. Counting over the
horizon instead produces a distribution that describes the next second of flight, and a controller
flown from it fails almost completely. This was found the hard way and is the single most important
implementation detail in the residual stage. Sampling from the *unsharpened* distribution also
fails (about 7 % landings); sharpening first is what makes sampling viable at all, and greedy
execution is better again. The floor of 0.15 exists to bound `log π_MPC`, which is a network input;
see Recommended 8.

### Why fuel is measured as impulse, and the shaping functions written out

**Short.** The shaping functions are written out explicitly in this README and in the code, and
**none of them contains a fuel term**, fuel is measured, never rewarded, so it cannot be counted
twice. Control effort is now a physical quantity: the impulse the engines actually produced,
reconstructed from the environment's own thrust equations including the random misalignment it
draws each step, divided by lander mass to give a speed change in metres per second. The
unshaped agent (c = 0) is the ablation and is reported alongside the rest, it is cell `PPO F1`, which
lands 3 to 7 points less often than the best shaped cell at every wind and is the most expensive
standalone cell on fuel. The design ablations are collected in
**Ablations** above and in `ablations.csv`.

**Longer.** The double-counting came from the original scoring both a fuel penalty inside the reward
and a fuel term in the headline metric. Both are gone. `dv` in the results CSV is computed from
`impulse/mass` where impulse comes from `engine_impulse()`, which reads the two misalignment draws
the environment logged for that step, so it is the impulse actually applied, not a nominal figure.
`dv_min` is recorded per episode as well, being the least speed change that could have landed from
that starting state, so propulsive efficiency can be computed if wanted. `fuel_legacy` (engine-time)
is kept in the CSV purely so numbers can be lined up against the earlier version.

### Why evaluation is stochastic

**Short.** Every number reported here comes from the standard stochastic environment, with its random
engine misalignment active and episode seeds drawn at random. Wind is evaluated at five strengths
from 0 to 20 and not in still air only.

**Longer.** The environment draws two uniform(−1, 1) numbers per step to jitter engine thrust. The
earlier version could switch these off, which made results reproducible and also made them
optimistic. They are now always on; the switch remains only as the mechanism that logs the draws so
impulse can be reconstructed. Wind in this environment is not a constant push, it is
`tanh(sin(0.02k) + sin(0.01πk)) × strength`, a smooth quasi-periodic signal whose phase is drawn per
episode, so an episode at wind 20 can begin in a lull or a gust. That is why a wind estimator helps
and a fixed bias correction would not.

### Equal training budgets

**Short.** Both learned families train for exactly the same number of environment steps per agent, 2,007,040, a whole number of update batches, on the same environment with the same wind distribution.
`--check` reads the step count from every agent's learning curve in both bundles and asserts they are
all equal.

**Longer.** The residual agent is roughly 25 times more expensive per step than a standalone agent,
because the planner runs underneath at every step. The budget is therefore set by what is
affordable for the residual family, and the standalone family is given the same number of steps
instead of the same wall-clock time. Steps are the fair currency here because the comparison is
about sample efficiency and final quality, not compute. Anyone repeating this should know the
practical consequence: the standalone arm finishes in minutes and the residual arm in hours.

### Seeds, spread and confidence intervals

**Short.** Every cell is trained on multiple independent seeds, five for each standalone
cell and fifteen for every residual cell, and `spread_summary.csv` reports each seed
separately. Both kinds of uncertainty are given: a Wilson interval for the landing rate over
episodes, and a standard deviation across training seeds.

**Longer.** This turned out to be the most important point in the whole project, and the reason is
worth recording. **Spread across training seeds, not across episodes, is the dominant source of
uncertainty here**, and it is large: at wind 20 the same configuration retrained on different seeds
varies by about 4 percentage points in landing rate, and individual seeds have ranged over 22 points
in one cell. Five seeds give a standard error of roughly 2 points on a cell mean, which is bigger
than most differences worth arguing about. Concretely: a cell measured on five seeds read +1.6 above
the bar at wind 20 and −0.9 on ten seeds, the same configuration, opposite conclusions. Anyone
extending this work should budget seeds before anything else, and should treat any margin under
about 2 points at high wind as unresolved. The episode-level interval is the easy half of the
problem and the one that looks reassuring; it is not the one that matters.

### What counts as a landing

**Short.** Outcomes are read from the environment's terminal state, not from the sign of the return:
an episode counts as landed only if the lander came to rest without the failure flag. Every rate is
an integer count out of an integer total with a Wilson interval, and the per-episode rows are in the
CSV so any figure can be recounted.

**Longer.** Return sign and landing success genuinely disagree, in the earlier archive they
disagreed on over two thousand episodes, because a landing that burns a lot of fuel can end with a
negative return and a crash after a long glide can end positive. Four outcomes are recorded rather
than two: `landed`, `crashed`, `out_of_bounds` and `timeout`. The distinction matters for reading the
results: in calm air a wind-trained controller's failures are almost all `timeout`, meaning it hovers
instead of committing to touchdown, while at wind 20 nearly all failures are `crashed`. Those are
different problems and a single success rate hides which one you have.

### Episode sets and paired comparison

**Short.** Every learned controller and the heuristic are flown on exactly the same held-out episodes
at every wind strength, so those comparisons are paired. The planner is flown on those episodes and on
seven hundred more, because it is the only controller with no training seeds to average over; the
shared episodes remain available for paired tests against it.

**Longer.** The episode seeds are derived from a fixed starting number, which is what guarantees both
that the sets are reproducible and that no episode used for development ever appears in the held-out
set; `--check` asserts the second. One trap worth flagging: asking for a different *number* of
episodes yields a different *set*, not a prefix of the larger one. Two measurements of the same
agents at 150 and 300 episodes per wind are therefore independent samples, and in this project they
differed by 2 points at wind 20, enough to move a cell from clearing the bar to missing it. Fix the
episode count before measuring and do not compare across counts.

### The planning horizon

**Short.** There is no adaptation rule: the look-ahead is a fixed 48 steps, just under a second, with
a terminal cost estimating what remains beyond it. The horizon is ablated at 24 and 72 steps.
**48 is a real-time choice, not an optimum**, 72 steps lands more often at every wind, by 13 points
at wind 20, and costs proportionately more solve time. The reported controller is the longest
look-ahead that fits the control period on this hardware.

**Longer.** Adapting the horizon to altitude was tried and dropped; the terminal cost does the same
job more cheaply, charging the speed change still needed to land, which grows with remaining
altitude, so a plan ending high in the air is already penalised without lengthening the search.

The ablation is uncomfortable and should be reported and not buried: of everything varied, the
horizon is the only continuous setting with a large monotone effect, and the reported value is not at
its best. That bounds the claim the paper can make about the planner. It is a planner operating under
a stated compute budget, not a well-tuned planner, and the residual's margin over it is therefore a
margin over *that* controller. It is fair to ask whether a residual on top of the
72-step planner would still improve on it; that experiment has not been run, and saying so is
cheaper than implying it has.

### Four cross-entropy iterations, and what they buy

**Short.** Four iterations are used, not one. A careful reader's underlying point is conceded in full:
**the iteration count makes no measurable difference.** Every value from 1 to 8 lands within sampling
noise of every other at every wind, one included. The four is a real-time budget, and the work the
search does is done by the first sampling pass, the move blocking and the warm start, and not by
iterative refinement.

**Longer.** This is the ablation that most changed what the paper can say. The earlier draft claimed
quality saturates between six and eight iterations; measured over 200 episodes per wind, it does not
saturate because it barely varies, one iteration reaches 99.50, 96.50, 87.00, 70.00, 44.00 against
the reported 98.00, 96.00, 87.50, 71.50, 47.00, and the largest gap is inside one standard error.

Two honest readings. Either the warm start is carrying the search, so that the first population is
already centred on a good sequence and refinement has little left to find; or 200 episodes per arm is
too coarse to resolve a real 2-to-3-point effect. Both are consistent with the data and the second
cannot be excluded without more episodes. What cannot be claimed either way is that four iterations
are *needed*. If the method is presented as CEM-MPC, the reader should be told that on this problem
it performs like random shooting with a warm start.

### The entropy schedule

**Short.** The two families use different settings, and both keep the bonus non-zero throughout so
exploration never stops. The standalone agents use a **constant 0.01**, chosen by a sweep on
development episodes which preferred it to a decaying schedule. The residual agents decay from 0.03
to 0.005, because a residual policy begins already competent, it starts as the planner, and needs
to keep exploring for longer before a correction worth making appears. Every trained agent records
the setting it used.

**Longer.** The honest account is that the standalone value was measured and the residual schedule was
not. The sweep behind the constant 0.01 was run at a shorter training budget than the one reported
here, so it justifies the *choice* but has not been repeated at 2,000,000 steps, and the residual
schedule was frozen early and never swept at all. Neither is a free parameter that was tuned to
improve the reported numbers, which is the point the real concern is, but neither should be
presented as an optimised value. If the project is revisited, a sweep of both at the reported budget
is cheap for the standalone family and expensive for the residual one. Note also the interaction with
Major 2: the residual policy is sampled during training and greedy at evaluation, so entropy affects
what is learned but not what is finally flown.

### The built-in heuristic as a floor

**Short.** The environment's built-in heuristic is flown on the same held-out episodes as everything
else and is the first row of the results table. It is also used inside the planner as one of the
candidate plans, so the planner can never do accidentally worse than it.

**Longer.** It is a useful floor precisely because nothing about it was tuned: any controller that
fails to beat it in wind has not learned anything about wind. Two notes for anyone re-implementing it.
The heuristic has a **leg-contact override**, once a leg touches it stops correcting its angle and
only damps the remaining descent, and omitting that costs about twenty points of landing rate,
because the controller keeps steering on the ground and tips over. The version used here is checked
action-for-action against the environment's own implementation over several thousand states, including
states with a leg touching. And its *fuel* figure is the lowest of any controller in the table, which
is an artefact and not a virtue: fuel is measured per successful landing, so a controller that
only manages the easy episodes looks cheap. Fuel is comparable only between controllers with
comparable landing rates.

### Computational overhead

**Short.** Planning dominates the cost. Measured by `--ablations` over every solve of a full ablation
run: **mean 2.90 ms per environment-step, 99th percentile 16.75 ms, maximum 35.18 ms**, against a
control period of 20 ms. The mean and the 99th percentile fit; the worst case does not. The residual
network's forward pass is negligible beside it.

**Longer.** Two caveats belong with those numbers and neither is in the reported figure.

First, they are **amortised across 16 environments solved together on shared arrays**. At this
problem size a solve is dominated by per-call overhead instead of arithmetic, so batching is worth
roughly a factor of four per environment. A single vehicle planning alone would not see 2.90 ms. The
honest statement is that the *throughput* fits the control period, not that a single-vehicle
real-time implementation has been demonstrated.

Second, the maximum exceeds the period. Over a long run some solves take 35 ms, so a real
implementation would need either a deadline with a fallback to the shifted previous plan, or the
cadence-2 schedule, which the ablation shows is free. That the fallback already exists in the code, it is what the cadence setting flies, is the useful part of the answer.

The number worth recording for anyone repeating this is that the planner, not the network, is the
entire cost of the residual method: about 25 times a standalone agent's cost per step. Batching every
parallel environment's search onto one set of arrays, at bit-identical output, is what made these
experiments affordable at all.

### Discounting in the planner and in PPO

**Short.** No, and they are not the same kind of quantity. The planner's objective is an
undiscounted sum over a fixed 48-step look-ahead plus a terminal cost. PPO uses γ = 0.99 with
generalised advantage estimation at λ = 0.95. Both are stated explicitly.

**Longer.** Discounting inside the planner would be a modelling error and not a tuning choice:
the look-ahead is short and bounded, the terminal cost already accounts for everything past it, and
discounting would make the planner prefer to defer a necessary braking burn. The PPO discount is
about the learning problem and has no relation to it.

### The 0.15 probability floor

**Short.** The floor exists because the residual network reads `log π_MPC`; without it a zero
probability becomes negative infinity. With the floor, every probability lies between 0.103 and
0.690, so the input is bounded. It is **not** a free parameter: agents retrained end to end with a
floor of 0.02 instead land 1.1 to 4.7 points less often at every wind (97.80 against 99.10 in
still air, 68.60 against 71.60 at wind 20, with the widest gap of 4.70 at wind 15).

**History of that number.** An earlier version of this row reported a 15-to-38-point gap. The
floor-0.02 agents had been *trained* against a 0.02 floor but, because of a defect in how a saved
agent's settings reached the planner it flies on, were *flown* against the 0.15 one, so the gap
confounded the floor's own effect with a training-to-flight mismatch. The defect is fixed, a saved
agent now carries its own floor, temperature and squash limit into flight, and the arm was re-flown
on the same protocol. The corrected gap is 1.1 to 4.7 points. This history is recorded and not
quietly overwritten because it is a class of error worth recognising, and because
the original number was written up before it was checked.

**Longer.** An earlier draft said the floor "makes no difference to the greedy policy". That was true
of the experiment it described, swapping the floor at evaluation time on an already-trained agent, and false of the parameter, and the difference matters. The floor shapes the distribution the network
learns to correct and the distribution training samples from, so changing it changes what is learnt.
Swapping it afterwards only reweights a policy that was fitted around the old value.

The mechanism is the one the earlier note gestured at, with the sign of its importance inverted. A
low floor lets an unlikely action be drawn often enough during training to end an episode, and the
resulting returns are noisier and occasionally catastrophic. A high floor is a variance-control
device for the training signal. It should be presented as a design parameter with a measured
justification, which is what the ablation now provides, and it is a warning about testing a
parameter only at evaluation time when it was also present during training.

### The tanh squash and the initialisation gain

**Short.** The network's last layer is initialised with gain 0.01/max(1, λ), which makes `λ·δ`
start near zero for every λ, so every cell begins as the planner and the comparison across λ is
fair. At initialisation the adjustment is around 0.015 against a limit of 3, so the tanh is nowhere
near saturating and only begins to bind later in training.

**Longer.** The gain rule compensates but not exactly: the largest initial `λ·δ` still roughly
doubles across the range of λ, so the cells do not start *identically*, only negligibly differently
by comparison with the eventual adjustments. A more interesting observation from the trained agents
is that the network **self-normalises against λ**: mean `|δ|` falls from about 1.28 at λ = 1 to about
0.34 at λ = 5, so the product `λ·|δ|` stays within roughly 1.3 to 1.7 across a fivefold change in λ.
That means λ is much less of a free parameter than the architecture suggests, and it predicts, correctly, that landing rate varies only weakly and non-monotonically across λ. It also means that
the spread across λ in the reported grid is comparable to the spread across training seeds, and
should not be over-read.

One thing anyone re-running this must know: **λ = 3 is identified as the strongest cell by comparing
the five cells on the `eval` episodes**, which are the same episodes its margins are then quoted
against, so +0.66 at wind 0 is a best-of-five. That is why a sixth episode set exists. λ = 3 was
re-flown on it with all 15 seeds and clears the bar at all five winds there too, worst margin +0.49
(`final_confirmation.csv`). The selection effect is real and the result survives it.

What does not survive is the wider claim the grid was making. At 5 seeds per cell, four of five cells
cleared the bar; at 15 seeds, **two do**. λ = 1 and λ = 5 both fall below it at wind 0 once the extra
runs are in. The lesson is the one a careful reader was pressing throughout: a 5-seed cell is a point
estimate with a standard error of two to four points on this problem, and four of five cells clearing
a bar by under a point was a statement about sampling, not about λ.

### Replanning cadence

**Short.** The planner re-solves at **every** step, and the text should say so throughout. Measured:
re-solving every second step is indistinguishable from every step at all five winds, and every sixth
step costs 12 points at wind 15 and 8.5 at wind 20. So per-step solving is the better option, and
halving the rate is available free if compute demands it.

**Longer.** The warm start is what makes per-step replanning affordable, the previous solution
shifted forward by one step is one of the candidates, so consecutive solves are not independent
searches, and it is also why cadence 2 costs nothing: flying the second action of a plan that was
about to be re-derived from that same plan is nearly the same action.

One thing to record, because it is the kind of error that is easy to miss. **The first version
of this ablation returned identical numbers for cadences 1, 2 and 6**, the same landing counts and
the same fuel to two decimals at all five winds. The setting was inert: the flight loop calls the
controller's `reset` once per step to clear state for environments that have just restarted, and
`reset` discarded the held plan unconditionally instead of only for the environments named in its
mask. Every cadence therefore re-solved every step. It was caught by the results being *too*
identical instead of by a test, the counter was made per-environment, and the arm was re-flown;
cadence 1 then reproduced the already-published planner row exactly, which is what shows the fix did
not disturb the reported controller. No headline number depended on it, because the reported planner
uses cadence 1 and so was never affected, but a table that reported three settings as identical was
one line of prose away from being a fabricated ablation.

### Why log space, and not additive renormalisation

**Short.** Both were implemented, trained to the same budget on the same seeds, and measured. The
add-and-normalise variant lands 98.40, 94.70, 85.00, 68.50, 47.20 against the log-space 99.10, 99.00,
97.40, 90.80, 71.60, **24 points worse at wind 20**, and close to the bare planner's own numbers. So
log-space combination is kept for a measured reason and not an argued one.

**Longer.** The structural argument is that in log space the adjustment is a shift of logits, bounded
by construction, and the result is always a valid distribution whose relation to the planner's is
smooth; adding in probability space needs clipping to stay positive, and the clip interacts badly
with the gradient.

The measurement supports the conclusion but not the strongest version of the story. An earlier draft
said the additive variant "failed every time, in some runs catastrophically, training collapsed to
near-zero landing rates." That is not what the controlled comparison shows. Trained properly, the
additive agents are stable and land 98.40 % in still air; what they fail to do is *improve on the
planner underneath them*, which is the entire claim. The interesting result is therefore more
specific than a collapse and worth stating precisely: the additive blend does not destroy the policy,
it neutralises the residual.

### Bounding the claims

**Short.** This is a 2-D, six-state, four-discrete-action game with a simplified physics engine and
no aerodynamics, mass depletion, actuator lag, sensor noise or three-dimensional attitude. Nothing
here transfers to a real vehicle. The contribution is a method comparison in a reproducible
benchmark, not a landing system.

**Longer.** Two specific gaps are worth naming for anyone tempted to extend the claims. The
action space is four discrete choices, so the "fuel" measured is engine-on time instead of a
throttle profile, and the real minimum-fuel problem is continuous. And the lander has no mass
depletion, so spending fuel does not make the vehicle lighter, which removes the coupling that
makes real powered descent hard.

### The binary leg-contact flags

**Short.** It does not need to differentiate them. The search is derivative-free, it samples
sequences and scores them, so discrete and binary quantities cost nothing. Contact enters the model
as the touchdown event that ends a rollout, and in the controller as an engine cut-off once both legs
are down.

**Longer.** The cut-off is not cosmetic. Without it the planner keeps firing on a landed vehicle, and
in wind that pushes it back off the pad; adding it was worth about 10 percentage points at wind 20.
The two contact flags are also the condition under which the wind estimator stops updating, since
contact forces are not a disturbance worth learning.

