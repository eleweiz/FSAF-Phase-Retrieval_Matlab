# FSAF-Phase-Retrieval_Matlab
Implementation of the Frequency Subspace Amplitude Flow algorithm for Phase Retrieval
Include 1D, 2D coherent diffraction parttern, and natural image reconstrution from intensity measurements. 

Copyright © 2018,  National University of Singapore, Zhun Wei.
Matlab Implementation of the Frequency Subspace Amplitude Flow algorithm proposed in the paper ``Frequency Subspace Amplitude Flow for Phase Retrieval'' by Z. Wei, W. Chen, and X. Chen (2018). The code below is adapted from implementation of the Wirtinger Flow algorithm designed and implemented by E. J. Candes, X. D. Li, and M. Soltanolkotabi

The code use Frequency subspace to reduce the sample complexity at initialization stage; The code also use conjugate gradient and optimal stepsize to accelerate the convergence rate, where both of them can also be diretly used to Wirtinger Flow. The conjugate strategy is proposed in ``Conjugate gradient method for phase retrieval based on the Wirtinger derivative'' by Z. Wei, W. Chen, C.-W. QIU, and X. Chen (2017);
