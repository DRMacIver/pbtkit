# PBT-Kit

PBT-Kit is a highly modular Hypothesis-style property-based testing library, with the primary focus being on helping to understand how all the different parts fit together, and to make it easy to experiment with new implementation strategies in property-based testing.

It originally started out as an attempt to port over the new underpinnings of Hypothesis (which moved to a single uniform representation to a small number of primitives).
[Minithesis](https://github.com/DRMacIver/minithesis) was a minimalist implementation of the core idea of the original [Hypothesis](https://github.com/HypothesisWorks/hypothesis) implementation.[^1]

This increased its size enough that I started thinking about how to modularise it, and that part sortof took on a life of its own.

The theory behind PBT-Kit is that there is a very small core which is capable of implementing full Hypothesis PBT with shrinking, and everything else is an add on module to that. It can be used as a normal Python project, or it can be compiled down to a single file with any subset of the features you wish to include.

This project is extremely a work in progress and right now exists more for my understanding than yours, as I am using it as a research platform for developing next generation implementations to use in [Hegel](https://hegel.dev/).

Long-term, I intend for it to be a comprehensive toolkit for anyone to use to understand and experiment with implementations in this space, which is much easier to work with than Hypothesis.

[^1]: If you see parts that are still named "minithesis" or similar, that's why.
