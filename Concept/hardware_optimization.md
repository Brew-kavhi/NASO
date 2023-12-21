Hardware notes
===
We are primarkly on gpu, like nvidia. nowe the problem obviously is that on edge devices, the gpu are primarly arm gpus and not nvidia gpus. Nevertheless, this may give some guidance for other GPU's as well.
This file will be a collection of things i notices during my work with tensorflow on a nvidia system.

# Current setup
Tensorflow uses oneDNN for intel CPu's. It is a framework for tensorflow tht enables AI optimizations on intel CPUs. This would be used if we train or infere on intel CPU. Therefor ithis is not interersting at all. As x86 chips are not used in mobiloe area.

Furthermore, we use nvidias-cuda-toolkit. It provides some optimization for nvidia gpus. We have several python packages installed that should provide features like sparse computations. Howver i am not sure if they are used on how they are used. 

## Problems
With the current setup, we get several errors regarding factories. It says attempting to load them , but could not register them. So i dont know what to do with that information.
