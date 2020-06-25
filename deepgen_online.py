#!/usr/bin/env python3
#
# Training two-layer networks on inputs coming from various deep generators.
#
# Date: May 2020
#
# Author: Sebastian Goldt <goldt.sebastian@gmail.com>

import argparse
import math

import numpy as np  # for storing tensors in CSV format

import torch
import torch.distributions as distributions
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from dcgan import Generator
from generators import RandomGenerator, Sign
from mlp.twolayer import TwoLayer, identity, erfscaled

import realnvp, data_utils
from data_utils import Hyperparameters


NUM_TESTSAMPLES = 10000


class HalfMSELoss(nn.MSELoss):
    def __init__(self, size_average=None, reduce=None, reduction="mean"):
        super().__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        return 0.5 * F.mse_loss(input, target, reduction=self.reduction)


def eval_student(time, student, test_xs, test_ys, nus, T, tildeT, A, criterion):
    N = test_xs.shape[1]

    student.eval()
    with torch.no_grad():
        # compute the generalisation error w.r.t. the noiseless teacher
        preds = student(test_xs)
        eg = criterion(preds, test_ys)

        w = student.fc1.weight.data
        v = student.fc2.weight.data
        lambdas = w.mm(test_xs.T) / math.sqrt(N)
        Q_num = lambdas.mm(lambdas.T) / NUM_TESTSAMPLES
        R_num = lambdas.mm(nus.T) / NUM_TESTSAMPLES

        eg_analytical = get_eg_analytical(Q_num, R_num, T, A, v)

        msg = "%g, %g, %g, nan, " % (time, eg, eg_analytical)

        # upper triangular elements for symmetric matrices
        indices_K = Q_num.triu().nonzero().T
        indices_M = T.triu().nonzero().T
        Q_num_vals = Q_num[indices_K[0], indices_K[1]].cpu().numpy()
        msg += ", ".join(map(str, Q_num_vals)) + ", "
        msg += ", ".join(map(str, R_num.flatten().cpu().numpy())) + ", "
        T_vals = T[indices_M[0], indices_M[1]].cpu().numpy()
        msg += ", ".join(map(str, T_vals)) + ", "
        tildeT_vals = tildeT[indices_M[0], indices_M[1]].cpu().numpy()
        msg += ", ".join(map(str, tildeT_vals)) + ", "
        msg += ", ".join(map(str, A.flatten().cpu().numpy())) + ", "
        msg += ", ".join(map(str, v.flatten().cpu().numpy())) + ", "

        return msg[:-2]


def get_eg_analytical(Q, R, T, A, v):
    """
    Computes the analytical expression for the generalisation error of erf teacher
    and student with the given order parameters.

    Parameters:
    -----------
    Q: student-student overlap
    R: teacher-student overlap
    T: teacher-teacher overlap
    A: teacher second layer weights
    v: student second layer weights
    """
    eg_analytical = 0
    # student-student overlaps
    sqrtQ = torch.sqrt(1 + Q.diag())
    norm = torch.ger(sqrtQ, sqrtQ)
    eg_analytical += torch.sum((v.t() @ v) * torch.asin(Q / norm))
    # teacher-teacher overlaps
    sqrtT = torch.sqrt(1 + T.diag())
    norm = torch.ger(sqrtT, sqrtT)
    eg_analytical += torch.sum((A.t() @ A) * torch.asin(T / norm))
    # student-teacher overlaps
    norm = torch.ger(sqrtQ, sqrtT)
    eg_analytical -= 2.0 * torch.sum((v.t() @ A) * torch.asin(R / norm))
    return eg_analytical / math.pi


def get_samples(scenario, hmm, P, D, N, generator, teacher, mean_x, device):
    """
    Generates a set of test samples.

    Parameters:
    -----------

    scenario : string describing the scenario, e.g. dcgan_rand, nvp_cifar10, ...
    hmm : if True, the teacher is acting on the latent variables.
    P : number of samples
    D : latent dimension
    N : input dimension
    generator : generative model that transforms latent variables to inputs
    teacher : teacher networks
    mean_x : the mean of the generator's output
    """
    with torch.no_grad():
        cs = torch.randn(P, D).to(device)
        latent = cs
        if scenario.startswith("dcgan"):
            latent = latent.unsqueeze(-1).unsqueeze(-1)
        elif scenario.startswith("nvp"):
            latent = latent.reshape(-1, 3, 32, 32)
        xs = generator(latent).reshape(-1, N)
        xs -= mean_x
        teacher_inputs = cs if hmm else xs
        ys = teacher(teacher_inputs)

        return cs, xs, ys


def write_density(fname, density):
    """
    Stores the given order parameter density in a file of name fname in the Armadillo
    text format.

    Parameters:
    -----------
    density: (K, M, N)
    """
    K, M, N = density.shape
    output = open(fname, "w")
    output.write("ARMA_CUB_TXT_FN008\n")
    output.write("%d %d %d\n" % (K, M, N))
    for i in range(N):
        for k in range(K):
            for m in range(M):
                output.write("  %+.6e" % density[k, m, i])
        output.write("\n")

    output.close()


def log(msg, logfile):
    """
    Print log message to  stdout and the given logfile.
    """
    print(msg)
    logfile.write(msg + "\n")


def main():
    # define the command line arguments
    g_help = "teacher + student activation function: 'erf' or 'relu'"
    M_help = "number of teacher hidden nodes"
    K_help = "number of student hidden nodes"
    device_help = "which device to run on: 'cuda' or 'cpu'"
    scenario_help = "Some pre-configured scenarios: rand, dcgan_rand, dcgan_cifar10, nvp_imnet32."
    steps_help = "training steps as multiples of N"
    seed_help = "random number generator seed."
    hmm_help = "have teacher act on latent representation."
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--f", default="tanh", help=g_help)
    parser.add_argument("-g", "--g", default="erf", help=g_help)
    parser.add_argument("-L", "--depth", type=int, default=4, help="generator depth")
    parser.add_argument("-D", "--D", type=int, default=100, help="latent dimension")
    parser.add_argument("-N", "--N", type=int, default=1000, help="input dimension")
    parser.add_argument("-M", "--M", type=int, default=2, help=M_help)
    parser.add_argument("-K", "--K", type=int, default=2, help=K_help)
    parser.add_argument("--scenario", help=scenario_help, default="rand")
    parser.add_argument("--device", "-d", help=device_help)
    parser.add_argument("--lr", type=float, default=0.2, help="learning rate")
    parser.add_argument("--bs", type=int, default=1, help="mini-batch size")
    parser.add_argument("--steps", type=int, default=10000, help=steps_help)
    parser.add_argument("-q", "--quiet", help="be quiet", action="store_true")
    parser.add_argument("-s", "--seed", type=int, default=0, help=seed_help)
    parser.add_argument("--hmm", action="store_true", help=hmm_help)
    parser.add_argument("--store", action="store_true", help="store initial conditions")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    (L, D, N, M, K, lr) = (args.depth, args.D, args.N, args.M, args.K, args.lr)
    scenario = args.scenario
    args.hmm = True

    # Find the right generator for the given scenario
    loadweightsfrom = None
    scenario_desc = None
    num_gen_params = 0
    if scenario == "rand":
        Ds = [args.D] * L + [N]
        f = Sign
        generator = RandomGenerator(Ds, f, batchnorm=False)
        scenario_desc = "rand_sign_L%d" % L
        generator.eval()
        generator.to(device)
    elif args.scenario in ["dcgan_rand", "dcgan_cifar10"]:
        D = 100
        N = 3072

        generator = Generator(ngpu=1)
        # load weights
        loadweightsfrom = "models/%s_weights.pth" % args.scenario
        generator.load_state_dict(torch.load(loadweightsfrom, map_location=device))
        scenario_desc = scenario
        generator.eval()
        generator.to(device)
    elif args.scenario == "nvp_cifar10":
        D = 3072
        N = 3072
        scenario_desc = args.scenario

        flow = torch.load("models/nvp_cifar10.model", map_location=device)
        num_gen_params = sum(p.numel() for p in flow.parameters())
        generator = flow.g
    else:
        raise ValueError("Did not recognise the scenario here, will exit now.")

    if num_gen_params == 0:
        num_gen_params = sum(p.numel() for p in generator.parameters())

    # output file + welcome message
    hmm_desc = "hmm_" if args.hmm else ""
    log_fname = "deepgen_online_%s_D%d_N%d_%s%s_M%d_K%d_lr%g_i1_s%d.dat" % (
        scenario_desc,
        D,
        N,
        hmm_desc,
        args.g,
        M,
        K,
        lr,
        args.seed,
    )
    logfile = open(log_fname, "w", buffering=1)
    welcome = "# Two-layer nets on inputs from a generator (scenario %s)\n" % scenario
    welcome += "# M=%d, K=%d, lr=%g, batch size=%d, seed=%d\n" % (
        M,
        K,
        lr,
        args.bs,
        args.seed,
    )
    welcome += "# Generator has %d parameters\n" % num_gen_params
    if loadweightsfrom is not None:
        welcome += "# generator weights from %s\n" % (loadweightsfrom)
    welcome += "# Using device:" + str(device)
    log(welcome, logfile)

    # networks and loss
    g = erfscaled if args.g == "erf" else F.relu
    gs = (g, identity)
    student = TwoLayer(gs, N, args.K, 1, normalise1=True, std0=1)
    student.to(device)

    teacher_input_dim = D if args.hmm else N
    teacher = TwoLayer(gs, teacher_input_dim, args.M, 1, normalise1=True, std0=1)
    nn.init.constant_(teacher.fc2.weight, 1)
    teacher.freeze()
    teacher.to(device)
    B = teacher.fc1.weight.data
    A = teacher.fc2.weight.data

    # collect the parameters that are going to be optimised by SGD
    params = []
    params += [{"params": student.fc1.parameters()}]
    # If we train the last layer, ensure its learning rate scales correctly
    params += [{"params": student.fc2.parameters(), "lr": lr / N}]
    optimizer = optim.SGD(params, lr=lr)
    criterion = HalfMSELoss()

    print("# Generator, Teacher and Student: ")
    for net in [generator, teacher, student]:
        msg = "# " + str(net).replace("\n", "\n# ")
        log(msg, logfile)

    # when to print?
    end = torch.log10(torch.tensor([1.0 * args.steps])).item()
    times_to_print = list(torch.logspace(-1, end, steps=200))

    # Obtain the right covariance matrices
    Phi = None
    Omega = None
    mean_x = torch.zeros(N, device=device)
    if scenario == "rand":
        b = math.sqrt(2 / np.pi)
        c = 1
        b2 = pow(b, 2)
        for l in range(generator.num_layers):
            F = generator.generator[l * 2].weight.data
            if l == 0:
                Omega = b2 * F @ F.T
                Phi = b * F
            else:
                I = torch.eye(F.shape[1]).to(device)
                Omega = b2 * F @ ((c - b2) * I + Omega) @ F.T
                Phi = b * F @ Phi
        Omega[np.diag_indices(N)] = c

        torch.save(Omega, "models/rand_omega.pt")
        torch.save(Phi, "models/rand_phi.pt")
    elif scenario in ["fc_inverse", "dcgan_rand", "dcgan_cifar10", "nvp_cifar10"]:
        Omega = torch.load("models/%s_omega.pt" % scenario, map_location=device)
        Phi = torch.load("models/%s_phi.pt" % scenario, map_location=device)
        mean_x = torch.load("models/%s_mean_x.pt" % scenario, map_location=device)

    # generate the test set
    test_cs, test_xs, test_ys = get_samples(
        args.scenario,
        args.hmm,
        NUM_TESTSAMPLES,
        D,
        N,
        generator,
        teacher,
        mean_x,
        device,
    )

    teacher_inputs = test_cs if args.hmm else test_xs
    nus = B.mm(teacher_inputs.T) / math.sqrt(teacher_inputs.shape[1])

    msg = "# test xs: mean=%g, std=%g; test ys: std=%g" % (
        torch.mean(test_xs),
        torch.std(test_xs),
        torch.std(test_ys),
    )
    log(msg, logfile)

    T = 1.0 / B.shape[1] * B @ B.T
    rotation = Phi.T @ Phi
    tildeT = 1 / N * B @ rotation @ B.T
    if args.store:
        with torch.no_grad():
            # compute the exact densities of r and q
            exq = torch.zeros((K, K, N), device=device)
            exr = torch.zeros((K, M, N), device=device)
            extildet = torch.zeros((M, M, N), device=device)
            sqrtN = math.sqrt(N)
            w = student.fc1.weight.data
            v = student.fc2.weight.data

            rhos, psis = torch.symeig(Omega, eigenvectors=True)
            rhos.to(device)
            psis.to(device)
            #  make sure to normalise, orient evectors according to the note
            psis = sqrtN * psis.T

            GammaB = 1.0 / sqrtN * B @ Phi.T @ psis.T
            GammaW = 1.0 / sqrtN * w @ psis.T

            for k in range(K):
                for l in range(K):
                    exq[k, l] = GammaW[k, :] * GammaW[l, :]
                for n in range(M):
                    exr[k, n] = GammaW[k, :] * GammaB[n, :]
            for n in range(M):
                for m in range(M):
                    extildet[n, m] = GammaB[n, :] * GammaB[m, :]

            root_name = log_fname[:-4]
            np.savetxt(root_name + "_T.dat", T.cpu().numpy(), delimiter=",")
            np.savetxt(root_name + "_rhos.dat", rhos.cpu().numpy(), delimiter=",")
            np.savetxt(root_name + "_T.dat", T.cpu().numpy(), delimiter=",")
            np.savetxt(root_name + "_A.dat", A[0].cpu().numpy(), delimiter=",")
            np.savetxt(root_name + "_v0.dat", v[0].cpu().numpy(), delimiter=",")

            write_density(root_name + "_q0.dat", exq)
            write_density(root_name + "_r0.dat", exr)
            write_density(root_name + "_tildet.dat", extildet)

    time = 0
    dt = 1 / N

    msg = eval_student(time, student, test_xs, test_ys, nus, T, tildeT, A, criterion)
    log(msg, logfile)
    while len(times_to_print) > 0:
        # get the inputs
        cs, inputs, targets = get_samples(
            args.scenario, args.hmm, args.bs, D, N, generator, teacher, mean_x, device
        )

        for i in range(args.bs):
            student.train()
            preds = student(inputs[i])
            loss = criterion(preds, targets[i])

            # TRAINING
            student.zero_grad()
            loss.backward()
            optimizer.step()

            time += dt

            if time >= times_to_print[0].item() or time == 0:
                msg = eval_student(
                    time, student, test_xs, test_ys, nus, T, tildeT, A, criterion
                )
                log(msg, logfile)
                times_to_print.pop(0)

    print("Bye-bye")


if __name__ == "__main__":
    main()
