# PhyRC 2027 — Robot-Assisted Dressing in Simulation

Research and development repository for learning robot-assisted dressing policies in the **PhyRC 2027 Phase 1 challenge**. This project focuses on teleoperated demonstration collection, dataset preparation, and learning-based policy development for simulated manipulation tasks involving clothing.

The goal is to develop and evaluate policies that can perform dressing-related manipulation through a structured, reproducible workflow.
<img width="1600" height="1000" alt="image" src="https://github.com/user-attachments/assets/ce2382d3-e5d6-4678-96f3-ced797905672" />


## Overview

Robot-assisted dressing is a challenging manipulation problem involving deformable objects, uncertain contact, and coordinated robot motion. This repository provides a workspace for experimenting with data-driven approaches to these challenges in simulation.

Current development focuses on:

- **Teleoperation:** Collecting demonstrations of dressing-related robot behaviors.
- **Dataset preparation:** Loading, inspecting, and preparing demonstration data for training.
- **Behavior cloning:** Learning policies from expert demonstrations.
- **Policy integration:** Connecting trained policies to the challenge inference interface.
- **Simulation evaluation:** Testing policies and iteratively improving their behavior.

## Repository Structure

```text
PhyRC_2027_Challenge/
├── policy.py                 # Policy inference entry point
├── zero_policy.py            # Baseline policy
├── data_loading.py           # Dataset loading utilities
├── participant/              # Participant policy and training code
├── examples/                 # Example policy implementations
├── run.sh                    # Project execution script
└── src/
    └── PhyRC_Sim/            # Simulation environment
```

*Note: This is a high-level overview. The exact contents and responsibilities of individual files may evolve as development progresses.*

## Development Pipeline

```text
Teleoperation
     ↓
Demonstration Collection
     ↓
Dataset Inspection & Preparation
     ↓
Behavior Cloning
     ↓
Policy Integration
     ↓
Simulation Evaluation
     ↓
Iterative Improvement
```

Each stage is intended to support a systematic approach to developing and evaluating learned manipulation policies.

## Current Progress

- Set up the local simulation development environment.
- Established a baseline policy and explored the challenge inference workflow.
- Set up teleoperation for collecting robot demonstrations.
- Began recording dressing-related behaviors for dataset development.
- Started exploring the data-loading and behavior-cloning pipeline.

**Current focus:** Turning collected demonstrations into a trained policy that can be evaluated in the challenge simulation.

## Getting Started

### Prerequisites

- A Linux environment compatible with the challenge simulator.
- The simulator and dependencies specified by the official PhyRC challenge documentation.
- Python and the required machine-learning dependencies.
- Any additional hardware or GPU resources required by the chosen simulation configuration.

### Setup

1. Clone this repository.
2. Follow the official PhyRC challenge instructions to install and configure the simulation environment.
3. Install the project dependencies required by the selected workflow.
4. Configure the appropriate dataset paths and policy settings.
5. Run the baseline or example policy to verify the environment before testing learned policies.

Refer to the official challenge documentation for the exact installation, execution, and submission procedures.

## Experiments and Evaluation

The repository is being developed to support experiments in:

- Demonstration-driven manipulation learning.
- Behavior cloning for robot-assisted dressing.
- Policy inference within the challenge environment.
- Qualitative and quantitative evaluation of learned behaviors.

Training configurations, evaluation metrics, and experimental results will be documented as they become available. Results should be interpreted in the context of the tested simulation configuration and evaluation protocol.

## Future Work

- Organize and validate the collected demonstration dataset.
- Implement and validate the behavior-cloning training pipeline.
- Train an initial policy from recorded demonstrations.
- Compare learned behavior against the baseline.
- Evaluate policy performance across multiple episodes.
- Document failure cases and iterate on data collection and training.

## Acknowledgements

This project builds on the PhyRC challenge infrastructure and simulation environment. Please refer to the official challenge materials for attribution, licensing, and usage requirements.

## Disclaimer

This is an independent development repository for research and experimentation related to the PhyRC 2027 challenge. It is not an official challenge repository. Challenge rules, submission requirements, and permitted assets are governed by the official organizers' documentation.
