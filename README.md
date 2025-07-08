# Tensor Network Message Passing (TNMP)

## Overview

Tensor Network Message Passing (TNMP) is an algorithm designed for computing local observables in interacting systems. While inheriting the linear complexity of traditional message passing algorithms, TNMP achieves significantly higher accuracy through the incorporation of tensor networks, with a trade-off of introducing a larger constant factor. 

TNMP uses message passing to iteratively approximate the global environment while employing tensor networks to exactly contract local message updates. For implementation details, please refer to [the original paper](https://doi.org/10.1103/PhysRevLett.132.117401) and the detailed tutorial in `python/tnmp.ipynb`.

## Implementations

This repository provides two implementations, each targeting different applications:

### 1. Spin Glass Systems (Python Implementation)
- **Purpose**: Computing local observables (e.g., magnetization)
- **Location**: Available in the `python` folder
- **Features**: Direct implementation focusing on spin glass systems

### 2. General Tensor Networks (Julia Implementation)
- **Purpose**: Approximate contraction of general tensor networks with locally concentrated open-legs
- **Package**: Available as [GenericMessagePassing.jl](https://github.com/ArrogantGao/GenericMessagePassing.jl)
- **Examples**: Located in the `julia` folder
- **Additional Resource**: For a pictorial description of the algorithm in action, see [this specific example of K-SAT problems](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.110.034126)