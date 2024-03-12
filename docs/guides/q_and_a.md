# Q&A

## Walking Bouts vs Gait Sequences

Fundamentally, we differentiate between walking bouts and gait sequences.
Gait sequences are "regions likely to contain gait".
They don't follow strict rules, it is basically "whatever the GS detection algorithm think its gait".
Walking bouts follow strict defined rules.
These rules were defined as part of a consensus process within Mobilise-D and cover things like the minimum number of 
strides, the allowed break between two strides and others.
Because these rules can only be applied once all parameters are calculated, the final WB can be significantly different
from the original GSs.

We further differentiate between Walking Bout and Level Walking Bout (LWB).
The latter is a WB that should contains not incline or decline walking.
The idea behind this is, that clinically comparable gait parameters can likely only be extracted from level walking.
However, because the detection of inclines and declines from just a single lower back IMU is challenging, we rarely use 
this definition and usually do all comparisons on the WB.

Historic Naming: In older standardized data, WBs were referred to as "Continuous Walking Period (CWP)" or "MacroWbs" 
and Level WBs were called "MicroWBs".
When loading old data, we try to map these to the new terminology whenever possible.

## Why per-second value output for CAD/SL/Gaitspeed?

While we only implement Physics based CAD and SL algorithms, which can provide step-level output, in the original 
algorithm comparison, we also used ML algorithm that estimated the respective values per window of data.
The only common ground was to interpolate step based values to per-second values.
This seems a little awkward, and also introduces some level of error, but allows to easily replace the physics based 
algorithms with ML based algorithms.

Further, the interpolation per-second also allows for a level of outlier correction.
For example, in the case of Cadence, we can correct for potential missed initial contacts/steps by smoothing the 
detected step times.