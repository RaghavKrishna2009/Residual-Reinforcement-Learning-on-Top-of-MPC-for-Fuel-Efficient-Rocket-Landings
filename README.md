# Residual reinforcement learning on top of model predictive control

A learned policy is allowed to adjust a planner's decisions, but never to replace them. This
repository holds the code, the result files and the figure sources behind a study of that
arrangement in Gymnasium's `LunarLander-v3`.

Three controllers are built and compared on identical held-out episodes at five wind strengths:

| | what it is |
|---|---|
| **MPC** | a cross-entropy planner that looks ahead in a lander model identified from flight data, and picks the cheapest sequence of engine firings |
| **PPO** | a neural network trained by reinforcement learning, flying on its own |
| **Residual** | the planner proposes, a trained network nudges the proposal in log-probability space. This is the method under test |

## What it found

Nine residual configurations were trained across a grid of the correction scale λ. Five of the
nine match or beat every non-residual controller at all five wind strengths. The best, λ = 3,
clears that bar by at least 0.66 percentage points:

| landing rate (%) | wind 0 | wind 5 | wind 10 | wind 15 | wind 20 |
|---|---|---|---|---|---|
| built-in heuristic | 93.67 | 90.33 | 73.00 | 51.33 | 29.67 |
| MPC planner | 98.70 | 96.10 | 89.40 | 72.10 | 48.00 |
| best standalone PPO | 95.60 | 94.73 | 95.20 | 86.73 | 70.87 |
| **residual, λ = 3** | **99.36** | **98.80** | **96.87** | **90.87** | **74.22** |

Measured as speed change spent per landing actually delivered, every residual configuration is
14 % to 22 % cheaper than the planner; λ = 3 spends 45.52 m/s against the planner's 54.93 m/s.

The clearer result is consistency. In calm and moderate air the residual cells vary three to
seven times less across training seeds than the standalone agents do. A bad training run
degrades towards the planner instead of producing a bad controller.

On matched episodes, ones the planner landed and both learned controllers landed on at least
four fifths of their seeds, the picture is more modest. Only λ ≤ 2 flies an individual descent
more cheaply than the planner, by about 2 m/s, and no residual configuration flies one more
cheaply than a network flying alone. Most of the headline fuel figure is landing rate, not
economy. That distinction is the point of Table 5 in the paper and of `matched_fuel.csv` here.

## Running it

Python 3.10 or newer.

```
pip install "gymnasium[box2d]" numpy torch
python rocket_landing.py --check          # self-tests, about a minute
```

Everything is one file and one flag:

```
python rocket_landing.py --sysid            # refit the lander model from heuristic flights
python rocket_landing.py --train-ppo        # train the five standalone agents
python rocket_landing.py --train-residual   # train the residual grid
python rocket_landing.py --measure          # fly everything, write final_results.csv
python rocket_landing.py --spread           # per-seed landing rates
python rocket_landing.py --matched-fuel     # the three fuel views
python rocket_landing.py --train-ablations  # train the ablation agents
python rocket_landing.py --ablations        # the ablation sweep
python rocket_landing.py --refly-floor      # re-fly the probability-floor arm alone
python rocket_landing.py --final-check      # re-fly the best cells on reserved episodes
python rocket_landing.py --report           # print every table
python rocket_landing.py --watch 5          # watch five landings
```

`--workers` sets the process count and defaults to two below the core count. The full sequence
is days of CPU, dominated by `--train-residual`.

Planning is the entire cost of the method: a mean of 2.90 ms per environment-step against a
20 ms control period, 99th percentile 16.75 ms, worst case 35.18 ms. That figure is amortised
across sixteen environments planned together on shared arrays, which is roughly a factor of
four per environment and is what made the experiments affordable at all.

## What is in here

| file | what it holds |
|---|---|
| `rocket_landing.py` | everything: environment, model identification, wind estimator, planner, PPO, residual controller, experiments, CLI |
| `final_summary.csv` | landing rate and both Δv views, per controller per wind |
| `spread_summary.csv` | landing rate of every individual training run |
| `matched_fuel.csv` | the three fuel views, including matched episodes |
| `ablations.csv` | the planner and learning-component ablation sweep |
| `ablations_floor_reflown.csv` | the probability-floor arm, re-flown after the fix below |
| `figures-latex/` | TikZ and pgfplots sources for all three figures |
| `METHODS-NOTES.md` | the long-form record: why each constant has the value it has, and what is not settled |

`final_results.csv`, the per-episode table every summary is derived from, is about 46 MB and is
not committed. Regenerate it with `--measure`.

`final_summary.csv` is written without a header row. Its columns are: controller, wind strength,
episodes flown, episodes landed, landing rate (%), Δv per landing achieved (m/s), Δv per episode
(m/s). The other four carry their own headers.

## Episode sets

Episode seeds come from a fixed starting number, which is what makes the sets reproducible and
guarantees no episode used during development appears in the held-out set. `--check` asserts
the second. One trap: asking for a different *number* of episodes yields a different *set*, not
a prefix of the larger one, so two measurements at 150 and 300 episodes per wind are independent
samples. Fix the episode count before measuring and do not compare across counts.

## A defect that was found and fixed

The probability-floor ablation originally reported 83.90, 81.00, 73.50, 54.10 and 33.80, which
would have made the floor look load-bearing. It was not a result. The floor-0.02 agents had been
*trained* against a 0.02 floor but *flown* against the 0.15 one, because a saved agent's settings
were not reaching the planner it flew on. The planner now inherits the agent's floor, and the
re-flown arm gives 97.80, 97.80, 96.30, 86.10 and 68.60, between 1.1 and 4.7 points below the
reported controller. The floor matters, but far less than the broken row suggested.

The old numbers are recorded here and not quietly replaced, because the failure mode, a
configuration value silently not propagating from a saved artefact to the thing that consumes it, is worth knowing about if you reuse this code.

## What this does not show

The environment is a two-dimensional rigid-body game with simplified physics, no aerodynamics,
no mass depletion and no actuator lag. What is measured as fuel is engine-on time converted to
an impulse, whereas a real minimum-propellant descent is a continuous problem, and with no mass
depletion the vehicle never becomes lighter as it burns. None of these results transfer to a
physical vehicle.

What might transfer is the arrangement: a learned component whose failure mode is to fall back
on an inspectable base controller. That is cheap to arrange wherever a dependable baseline
already exists, and it is what the consistency numbers above are actually measuring.

Two further caveats belong with the grid. The four cross-entropy iterations add no measurable
improvement, which makes the planner closer to random shooting with prior knowledge than to a
true CEM search. And a 48-step horizon is not optimal, 72 is better at every wind, but 72 was
not affordable at this scale, so whether the residual cells improve on a better planner is
untested.
