# Q&A

(q&a__wb_vs_gs)=
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

(q&a__sec_vals)=
## Why per-second value output for CAD/SL/Gaitspeed?

While we only implement Physics based CAD and SL algorithms, which can provide step-level output, in the original 
algorithm comparison, we also used ML algorithm that estimated the respective values per window of data.
The only common ground was to interpolate step based values to per-second values.
This seems a little awkward, and also introduces some level of error, but allows to easily replace the physics based 
algorithms with ML based algorithms.

Further, the interpolation per-second also allows for a level of outlier correction.
For example, in the case of Cadence, we can correct for potential missed initial contacts/steps by smoothing the 
detected step times.

(q&a__other_paras)=
## Why do all algorithms store the input on the object? Does this increase memory consumption?

All algorithm objects make the input data available via `self.data` and `self.ic_list` (and others) available after 
calling the action method.
This is a "convention" that leads to some nice side effects.
This way, the final object has all the information about inputs and outputs. 
You could write for example a plot func, that just takes the algo instance as input and have all the information 
available that you need.
I also thought about memory consumption in this context, and the reality is, that it won't matter in 99% of the cases.
Just storing the object on the instance will not result in an increase in memory consumption.
It only stores the reference to the object.
This means the only thing that "could" cause issues here, is that as long as the instance with the results attached 
exists, the original data will not be cleared from memory.
But, it basically never happens, that the original data is out of scope and an algorithm object is still in scope, 
blocking the data deletion.
I played around with that a bit (as we are doing the same in gaitmap) and it was never an issue in any of our usecases.

(q&a__gaitmap)=
## This looks similar to [Gaitmap](https://github.com/mad-lab-fau/gaitmap). What is the difference/relationship?

Gaitmap can be considered a "sister" project to mobgap.
It was created by the same project lead (Arne Küderle) and its structure was used as a blueprint for mobgap.

Gaitmap focuses on foot-worn IMUs, while mobgap focuses on lower back IMUs.
Mobgap also improves on some of the design decisions made in Gaitmap, and includes more functionality with regards to
data loading and algorithm validation.

Algorithms should be mostly inter-compatible between the two projects.
Mobgap even vendors (i.e. copied) some of the algorithms from Gaitmap.
However, we explicitly do not use gaitmap as a dependency to untangle the two projects as much as possible and simplify
future maintenance.
