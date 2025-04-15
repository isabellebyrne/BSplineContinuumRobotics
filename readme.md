# BSplineContinuumRobotics

A Python library for modeling and controlling continuum robots using B-splines.

## What It Does

This code lets you:
- Model a continuum robot's shape using B-spline curves
- Calculate forward kinematics (robot shape from control points)
- Solve inverse kinematics (finding control points to reach a target)
- Avoid obstacles in the robot's path
- Visualize the robot in 3D

## Installation

```bash
git clone https://github.com/isabellebyrne/BSplineContinuumRobotics.git
cd BSplineContinuumRobotics
```

## Requirements

- Python 3.x
- NumPy
- SciPy
- Matplotlib

## Quick Start

```python
import numpy as np
from continuum_robot import ContinuumRobot

# Create a robot with control points
robot = ContinuumRobot([
    [0.0, 0.0, 0.0],    # Base
    [0.2, 0.0, 0.3],    
    [0.4, 0.0, 0.6],    
    [0.6, 0.0, 0.9],   
    [0.8, 0.0, 1.2],
    [1.0, 0.0, 1.5],    # Tip
], degree=3)

# Add an obstacle (optional)
robot.add_obstacle([0.3, 0.3, 0.4], 0.15)  # center, radius

# Get the current robot shape
backbone_curve, tip_position, _ = robot.forward_kinematics()

# Move the robot to a target
target = np.array([0.5, 0.5, 0.7])
robot.inverse_kinematics(target)

# Show the robot
robot.visualize(target_position=target)
```

## Features

- **B-spline Modeling**: Uses B-splines to create smooth robot shapes
- **Efficient Algorithms**: Fast calculation of B-spline curves
- **Obstacle Avoidance**: Detects and avoids collisions
- **3D Visualization**: See your robot, control points, and frames

## License

MIT License

## Citation

```
@software{byrne2025bspline,
  author = {Byrne, Isabelle},
  title = {BSplineContinuumRobotics},
  year = {2025},
  url = {https://github.com/isabellebyrne/BSplineContinuumRobotics}
}
```
