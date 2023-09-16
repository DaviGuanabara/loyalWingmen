# Environment Description

In the level2, the ideia is to use RPMs as actions (level 1 were a velocity vector inserted directly to the pybullet simulation).
I began developing the PID Tunner with RL, but I realised that i can be overwhelming by introducing it inside the NN.
One of the difficulties were to map the action_space to the actual RPMs, until i realised that the max RPM were already calculated
in the operational constraints. Currently, i just make a linear interpolation to map it.
Action RPMs goes from -1 to 1, and the Resulting RPM goes from 0 to MAX_RPM.

In meaning, 0 to 1 would suit better. But to facilitate the learning process, i chose keep with symetric action_space: -1 to 1.
Strangly, the PID controller provided by the GYM_PYBULLET_DRONES got a different RPM range. This is shown bellow.

# RPM Ranges: Controller vs. Neural Network

In controlling a quadcopter, we analyze two distinct methods:

1. **Traditional PID Controller**
2. **Neural Network (NN) based Agent**

## 1. Controller-based RPM Calculation

Using the traditional PID control logic, RPM is determined by:

\[ \text{RPM} = \text{PWM2RPM\_SCALE} \times \text{pwm} + \text{PWM2RPM\_CONST} \]

Given the constants:
- `self.PWM2RPM_SCALE = 0.2685`
- `self.PWM2RPM_CONST = 4070.3`
- `self.MIN_PWM = 20000`
- `self.MAX_PWM = 65535`

From this:

- **Minimum RPM (`RPM_min_controller`):** 9440.3
- **Maximum RPM (`RPM_max_controller`):** 21954.0985

## 2. Neural Network-based RPM Calculation
The authors of GYM_PYBULLET_DRONES used the following non-linear action-to-RPM:

- Action `-1` -> `RPM_min_nn` -> 0
- Action `0` -> `HOVER_RPM` -> 14468.429183500699
- Action `1` -> `MAX_RPM` -> 21702.64377525105

HOVER_RPM and MAX_RPM are calculated in the Operational Constraints Calculator, inside the quadcopter Factory.

```
return np.where(action <= 0, (action+1)*self.HOVER_RPM, self.HOVER_RPM + (self.MAX_RPM - self.HOVER_RPM)*action) # Non-linear mapping: -1 -> 0, 0 -> HOVER_RPM, 1 -> MAX_RPM`
```

I have done a linear mapping:

- Action `-1` -> `RPM_min_nn` -> 0
- Action `1` -> `MAX_RPM` -> 21702.64377525105


## Comparison

- **Controller RPM Range:** [9440.3, 21954.0985]
- **Neural Network RPM Range:** [0, 21702.64377525105]

## Observations

- The maximum RPMs for both methods are closely aligned.
- The NN approach starts from 0 RPM, signifying motors off, granting broader control.
- The NN's hover RPM value is within the range of the controller's RPM.

## Conclusions

Distinct design philosophies are evident:

- **Controller's RPM Range:** Prioritizes stability and reliability. The RPM range doesn't drop to 0, preserving a minimum thrust.
  
- **Neural Network's RPM Range:** Emphasizes flexibility, allowing extensive agent exploration during the training phase.


# Environment Observations