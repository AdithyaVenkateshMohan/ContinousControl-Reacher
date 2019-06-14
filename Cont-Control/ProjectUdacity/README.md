[//]: # (Image References)

[image1]: Reaherdemogif.gif "Trained Agent"

# Reacher - Continuous Control

### Introduction

This repo trains a DDPG agent to solve the Unity [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

[![Trained Agent][image1]](https://www.youtube.com/watch?v=KEl6X9LpoA0)

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Getting Started

1. Clone this repo.
2. Install the python dependencies by running `pip install -r requirements.txt`
3. Download the Unity environment matching your OS to the root folder of the repo and unzip the file.

    - **_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

### Instructions

Follow the instructions in  `Cont_Control_final.ipynb` to get started with training the agent!

The project excution and methodlogy are described in detail in <a href="Report.md">Report</a> which is attached in the repo.

to run the trained agents just run the script <a href = "Testing.py"> Testing.py </a>
```
python3 Testing.py
```