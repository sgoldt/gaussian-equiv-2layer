# Code for "The Gaussian equivalence of generative models for learning with two-layer  neural networks"

Here we provide the code used to run all the experiments of our recent paper on
"The Gaussian Equivalence of generative models for learning in two-layer neural
networks" [[1, arXiv]](https://arxiv.org/abs/2006.14709). There are several parts to this package: (for step-by-step
explanations, see below)

| File                          | Description                                                                                                                                                    |
|-------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ```covariance_generator.py``` | Estimates the covariances of a generative neural network,<br>implemented using the [pyTorch](http://pytorch.org/) library,                                     |
| ```dcgan.py```                | An implementation of a deep convolutional GAN of Radford et<br>al. [1], provided by  [pyTorch examples](https://github.com/pytorch/examples/tree/master/dcgan) |
| ```deepgen_ode.cpp```         | An integrator for the dynamical equations derived in the paper,<br>with its ```Makefile```                                                                     |
| ```deepgen_online.py```       | Trains two-layer neural networks when inputs are drawn from a generator                                                                                        |
| ```generators.py```           | Provides fully-connected, deep generative neural networks                                                                                                      |
| ```libscmpp.h```              | C++ utility functions                                                                                                                                          |
| ```models```                  | random and pre-trained weights used for the experiments with the DCGAN,<br>as well as the corresponding covariance matrices                                    |
| ```twolayer.py```             | Python utility functions                                                                                                                                       |
| ```realnvp.py```              | pyTorch implementation of real NVP model by [Fangzhou Mu](https://github.com/fmu2)                                                                             |
| ```data_utils.py```           | Utility functions for real NVP model by [Fangzhou Mu](https://github.com/fmu2)                                                                                 |

## External packages included in this repository

We were fortunate to be able to use the implementation of the DCGAN from the
pyTorch example repository, provided together with pre-trained weights by
[Chandan Singh](https://github.com/csinva). We are also grateful to [Fangzhou
Mu](https://github.com/fmu2) for his pyTorch port of the original real NVP
implementation. We include both these packages in this repository to make
reproducing the paper's experiments as easy as possible, but you should check
out the other work of Chandan and Fangzhou, too !


# Compilation of the C++ code

To compile locally, simply type
```
make deepgen_ode.exe
``` 
This assumes that you have installed the [Armadillo
library](http://arma.sourceforge.net) on your machine.


# Step-by-step instructions to reproduce the experiments of Sec. 3

There are two parts to reproducing the experiments of Section 3: training a
neural network on inputs drawn from a particular generator, and integrating the
dynamical equations that predict the evolution of the test error and of the
order parameters.

## Training a neural network

To train a two-layer model on input drawn from a generative model
```dcgan_rand```, type
```
./deepgen_online.py -M 2 -K 2 --lr 0.2 --scenario dcgan_rand
```
The following scenarios are possible:
- ```rand```, which is the random one-layer
generator corresponding to Theorem 1,
- ```dcgan_rand```, which is the DCGAN with random weights
- ```dcgan_cifar10``` which is the DCGAN trained on CIFAR10, and
- ```nvp_cifar10``` which is the real NVP model trained on CIFAR10.

This command will train an actual network with K hidden nodes, while the teacher
has M hidden nodes, and the learning rate is 0.2. For a full overview over the
parameters, run
```
./deepgen_online.py --help.
```
A run of deepgen-online.py will generate various output files, with all have the
same file name root, in this case: deepgen_online_dcgan_rand_D100_N3072_hmm_erf_M2_K2_lr0.2_i1_s0.

## Integrating the dynamical equations

The second step is integrating the ODEs. To this end, you will have to first
compile the ODE integrator, which is written in C++ and uses the [Armadillo
library](http://arma.sourceforge.net). Once this programme is available, you can
simply run
```
./deepgen_ode.exe -N 3072 -M 2 -K 2 --lr 0.2 --prefix deepgen_online_dcgan_rand_D100_N3072_hmm_erf_M2_K2_lr0.2_i1_s0
```
Make sure that the parameters M, K, and lr match the values you provided for the
simulation. This will generate an additional file, in this case called
deepgen_online_dcgan_rand_D100_N3072_hmm_erf_M2_K2_lr0.2_i1_s0_ode.dat, that
contains the output of the ODE integrator. The columns of the output file are
explained in the comment lines in the header of the output file.


# References

[1] S. Goldt, G. Reeves, M. Mézard, F. Krzakala, L. Zdeborová [[arXiv:2006.14709]](https://arxiv.org/abs/2006.14709)
