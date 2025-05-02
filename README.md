# Identifying Dead-End States in Sepsis Patients Using Reinforcement Learning
## Project Overview
This project leverages offline reinforcement learning (RL) to identify critical states—known as "dead-end states"—in sepsis patients, where mortality becomes unavoidable despite any further treatments. Recognizing these states enables clinicians to proactively avoid treatments that may lead to negative outcomes, ultimately improving patient management in intensive care units (ICUs).

*This work was conducted to fulfill the CS 9670 course requirement at the University of Western Ontario.*

## Background
Sepsis is a severe, life-threatening medical condition caused by an extreme bodily response to infection, leading to rapid organ dysfunction and potential death if untreated. Reinforcement learning offers promising methods for analyzing historical patient data, particularly in identifying treatments to avoid rather than suggesting uncertain optimal interventions.

## Goals and Objectives
- Identify "dead-end" patient states in sepsis care using RL techniques.

- Improve upon existing methods by employing offline-specific RL algorithms tailored to limited data and safety-critical environments.

- Address data imbalance issues by augmenting the existing dataset with synthetic non-survivor trajectories.

## Methodology
- Data Source: Utilized patient records from the publicly available MIMIC-III ICU database, including features like vital signs, medications, medical interventions, and demographics.

- State Definition: Each state combines multiple patient-specific features (e.g., age, vital signs), updated in discrete 4-hour intervals.

- Actions: Defined by 25 combinations of treatment intensities (5 levels of IV fluids × 5 levels of vasopressors).

- Rewards: Structured as +1 (patient recovery), -1 (patient death), and 0 (no immediate outcome).

## Reinforcement Learning Approach
- Original Method: Implemented baseline models using Deep Q-Networks (DQN) based on the initial Dead-End Discovery (DeD) framework, to predict risks of patient death (D-network) and potential for recovery (R-network).

## Offline RL Improvements:

- Conservative Q-Learning (CQL): Addressed DQN’s over-estimation by explicitly penalizing unrealistic high-value predictions.

- Implicit Q-Learning (IQL): Regularized predictions towards observed clinical behaviors, thus avoiding unsafe extrapolations.

## Addressing Data Imbalance
- Recognized a severe imbalance between survivor and non-survivor patient trajectories.

- Generated synthetic non-survivor data using a nearest-neighbor interpolation technique to balance the dataset and enhance model sensitivity to rare, critical outcomes.

## Repository Contents
### RL Training Scripts:

- train_cql.py: Implements training for the Conservative Q-Learning algorithm.

- train_iql.py: Implements training for the Implicit Q-Learning algorithm.

- Supporting scripts (cql.py, cql_experiment.py, iql_experiment.py) containing implementations of the algorithms and training experiments.

### Analysis and Evaluation:

- Calculate_rates.py: Calculates the true positive and false positive rates for identifying dead-end states.

### Documentation:

- Detailed project report (RL_Paper.pdf) and presentation (CS 9670 Presentation.pdf) outlining methodology, background, and results.

## Results and Findings
- Replicated original Dead-End Discovery (DeD) outcomes, validating baseline results.

- Demonstrated significant performance improvements using offline-specific RL methods (CQL and IQL), achieving higher sensitivity in detecting critical states.

- Identified treatment combinations consistently linked to poor outcomes, thus providing actionable insights for clinical decision-making.

## Next Steps
- Explore advanced synthetic data generation methods (e.g., GANs, diffusion models) for further improving data quality and representativeness.

- Experiment with additional reward structures and RL configurations to better model patient progression.

## Technologies and Tools
- Programming Languages: Python

- Machine Learning Libraries: PyTorch, NumPy, Pandas

- Dataset: MIMIC-III Clinical Database

- External Resources: Modified and adapted methodologies from Microsoft's Medical Dead-End GitHub repository.
