# chefshat-RL-genAi-augmentation
Chef’s Hat (Variant 6) experiment: random baseline vs generative AI augmentation. Trains a lightweight action-prior model P(action|obs) from gameplay data and evaluates a masked sampling agent. Includes mean/std performance scores, loss curve, plots, and saved outputs.

# Overview

This repository contains my Task 2 implementation for the Chef’s Hat environment (Variant 6: Generative AI Augmentation).
I compare a Random baseline against a Generative-Augmented agent that uses a lightweight neural network trained to approximate:

p(action | observation)

The learned model biases action selection while respecting the environment’s valid-action mask.

Student ID: 16077270

Variant: 6 (Generative AI Augmentation)

# Environment

chefshatgym==3.0.0.1

gym==0.26.2

NumPy, PyTorch, Matplotlib

# Methods
**1) Baseline (Random Agent)**

The baseline uses the environment’s random agent (AgentRandon).
Performance is measured using Game_Performance_Score and summarized with mean and standard deviation across multiple games.

**2) Generative AI Augmentation (Variant 6)**

Data collection: Run multiple games with four random players. Log (observation, action) pairs and keep samples from the best-performing player per game (highest Game_Performance_Score).

Train generative prior: Train a small neural network with cross-entropy loss to predict the action index from the observation vector.

Generative-Augmented agent: During play, the network outputs action logits, applies a valid-action mask, and samples an action from the masked distribution. An epsilon fallback to random actions prevents degenerate behaviour.

# How to Run (Colab Recommended)

Open the notebook/script in Google Colab.

Install dependencies:

gym==0.26.2

chefshatgym==3.0.0.1

Run the pipeline top-to-bottom:

sanity check

baseline evaluation

data collection

prior training

generative evaluation

plot + save outputs

Outputs will be saved in task2_outputs/.

# Key Parameters

N_EVAL_GAMES = 6

N_TRAIN_GAMES = 8

MAX_SAMPLES_PER_GAME = 120

PRIOR_EPOCHS = 2

PRIOR_BATCH = 256

PRIOR_LR = 1e-3

EPSILON_FALLBACK = 0.30

TEMPERATURE = 1.0

# Results

Evaluation is based on Game_Performance_Score (mean ± std):

Baseline (Random): 0.3242 ± 0.3824

Gen-Aug (Variant 6): 0.6570 ± 0.3064

Delta (Gen-Aug − Baseline): +0.3327

This indicates the generative augmentation improved average performance over the baseline under the same evaluation protocol.

# Notes / Limitations

Chef’s Hat is stochastic and multi-agent, so variance is expected.

The generative prior is a lightweight supervised model trained from gameplay logs (not full RL training).

The valid-action mask is assumed to be available as a 200-length segment in the observation (as used in the script).
