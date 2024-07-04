# Coordinate System

For any IMU based application it is important that the Coordinate system (definition and orientation) is clearly 
defined.
Otherwise, algorithms might use the wrong axis or defined calculations might not work at all.
Therefore, before using any algorithm in mobgap, make sure that the coordinate system of your data matches our 
expectations.

## Basic properties of the IMU

- The mobgap coordinate system follows the [right-hand rule](https://en.wikipedia.org/wiki/Right-hand_rule).
  Your coordinate system, probably already follows this rule, but it is worth double-checking, as some IMU manufactures
  (e.g. old Shimmer 2R devices) might have used a different convention.
- The coordinate systems of the gyr and acc are aligned. Again this is a common convention, but it is worth checking.
- The direction of rotation defined by the positive direction of the angular velocity follows the right-hand rule around
  the respective positive direction of the acceleration.
- We name and define all axis with the names `["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]` and the two norms
  as `["acc_norm", "gyr_norm"]` (when used).

## Alignment with the Body Frame

- All algorithms assume that the IMU is attached to the body at the lower back in a fixed position and orientation.
- The three axis of the sensor should be aligned with the major axis of the body IS (inferior to superior), PA (posterior
  to anterior), ML (medial to lateral).
- For all mobilise-d datasets, the IMU axis (x, y, z) map to these axis as follows:
    - x -> IS (inferior-superior)
    - y -> ML (medial-lateral)
    - z -> PA (posterior-anterior)

  If your axis are defined differently, you should rotate and then rename the axis to match the expected axis.
  Have a look at the sections 3 and 4 in this 
  [guide in gaitmap](https://gaitmap.readthedocs.io/en/latest/source/user_guide/prepare_data.html#converting-into-the-correct-units) 
  for more information on how to do this.
  
```{warning}
At the moment the algorithms expect the axis to be named like `acc_x`, `acc_y`, ... .
However, we will change this soon to make the expectation of the body alignment more explicit 
([#103](https://github.com/mobilise-d/mobgap/issues/103)).
After this change, all data is expected to have anatomical names like `acc_is`, `acc_pa`, ... .
```

```{figure} ./images/coordinate_system.png
:name: coordinate_system

Expected direction of body frame axis (adapted based on [Palmerini et al](https://www.nature.com/articles/s41597-023-01930-9/figures/4))
```

## Global coordinate system

In addition to the body frame, we also define a global coordinate system.
This is mainly used for the calculation of the global orientation (e.g. through the use of the Madgwick algorithm).

To avoid confusion and allow you to use/implement other sensor fusion algorithms, we follow the "standard" global 
coordinate system, with a z-axis pointing upwards.
The global x-axis is defined as the forward direction of the participant at sample 0 of the data 
(i.e. the initial direction of sensor z-axis/PA axis).
The global y-axis is defined as the left direction of the participant (to ensure a right-handed coordinate system) at 
sample 0 of the data (i.e. the **negative** initial direction of sensor x-axis/IS axis).

This means the initial sensor orientation in the global coordinate system is defined by the following rotation matrix:

```{code-block} python
import numpy as np

# Initial orientation of the sensor in the global coordinate system
R = np.array([[0, 1, 0],
              [1, 0, 0],
              [0, 0, -1]])
```

The quaternion values of this rotation matrix are `[0.5, 0.5, 0.5, -0.5]`.
This value as scipy rotation object is available `mobgap` package as `mobgap.consts.INITIAL_MOBILISED_ORIENTATION` and
should be used as the initial orientation for any sensor fusion algorithm.

```{warning}
Directly applying the results of a sensor fusion algorithm defined in this coordinate system to the body frame sensor
data, will redefine what axis points upwards, forwards, etc.!
You should always use the results of the rotation methods we provide to do this transformation! (See below)
```




