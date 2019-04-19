# DeepLearningForMDPs
Some deep learning-based algorithms have been introduced and theoretically studied in [DeepLearningForMDPs_theoreticalPart](https://arxiv.org/abs/1812.04300) to solve Markovian Decisions Processes (MDPs). These latter have been tested and compared on many numerical applications, and the results are available on [DeepLearningForMDPs_applicationsPart](https://arxiv.org/abs/1812.05916).

Some codes used for the tests presented in [DeepLearningForMDPs_applicationsPart](https://arxiv.org/abs/1812.05916) are available in this repertory: 
* slpde_HybridNow.py is the code, written in Python and TensorFlow, for the ClassifHybrid algorithm used in the Semi-Linear PDE example.
* sgm_ClassifHybrid.py is the code, written in Python and TensorFlow, for the ClassifHybrid algorithm used in the Smart Grid Management example.
* sgm_Qknn.jl is the code, written in Julia, for the Qknn algorithm used in the Smart Grid Management example.


Decisions.mp4 is a video of the Qknn estimated optimal decisions to take w.r.t. time for the Smart Grid Management example. The terminal time was set to N=200.
