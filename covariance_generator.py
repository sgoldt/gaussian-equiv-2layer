#!/usr/bin/env python3
#
# Robust estimation of the mean and covariance of an arbitray generator.
#
# Author: Sebastian Goldt <goldt.sebastian@gmail.com>
#
# Date: May 2020

import argparse

import numpy as np

import torch
from tqdm import tqdm

from dcgan import Generator
from generators import RandomGenerator, Sign


def log(msg, logfile):
    """
    Print log message to  stdout and the given logfile.
    """
    print(msg)
    logfile.write(msg + "\n")


def main():
    parser = argparse.ArgumentParser()
    device_help = "which device to run on: 'cuda:x' or 'cpu'"
    scenario_help = "rand: four-layer random generator. dcgan: dcgan with random weights. cifar10: pre-trained dcgan."
    checkpoint_help = "checkpoint every ... steps"
    seed_help = "random number generator seed."
    parser.add_argument("--scenario", help=scenario_help, default="dcgan")
    parser.add_argument("--device", "-d", help=device_help)
    parser.add_argument("--bs", type=int, default=4096, help="batch size.")
    parser.add_argument("--steps", type=int, default=1e9, help="number of steps")
    parser.add_argument("--checkpoint", type=int, default=1000, help=checkpoint_help)
    parser.add_argument("-q", "--quiet", help="be quiet", action="store_true")
    parser.add_argument("-s", "--seed", type=int, default=0, help=seed_help)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Will use chunks of data of size (batch_size, N) or (batch_size, D) etc.
    batch_size = args.bs

    if args.scenario in ["dcgan", "cifar10"]:
        D = 100
        N = 3072

        generator = Generator(ngpu=1)
        generator.eval()
        generator.to(device)
        # load weights
        if args.scenario == "cifar10":
            loadedweightsfrom = "models/dcgan_cifar10.pth"
        else:
            loadedweightsfrom = "models/dcgan_rand.pth"
        generator.load_state_dict(torch.load(loadedweightsfrom, map_location=device))
    elif args.scenario == "rand":
        # Find the right generator for the given scenario
        D = 100
        N = 3072
        L = 4
        Ds = [D] * L + [N]
        f = Sign
        generator = RandomGenerator(Ds, f, batchnorm=False)
        generator.to(device)
        generator.eval()
    elif args.scenario == "spiked":
        # Find the right generator for the given scenario
        D = 100
        N = 3072
        L = 1
        Ds = [D] * L + [N]
        f = Sign
        generator = RandomGenerator(Ds, f, batchnorm=False)
        loadedweightsfrom = "models/rand_spiked_L1.pth"
        generator.load_state_dict(torch.load(loadedweightsfrom, map_location=device))
        generator.to(device)
        generator.eval()
    else:
        raise ValueError("Invalid scenario given.")

    max_P = args.steps * args.bs
    log_fname = "covariance_%s_P%g_s%d.dat" % (args.scenario, max_P, args.seed)
    logfile = open(log_fname, "w", buffering=1)
    welcome = "# Computing the covariance for %s\n" % args.scenario
    if loadedweightsfrom is not None:
        welcome += "# with weights from %s\n" % loadedweightsfrom
    welcome += "# batch size=%d, seed=%d\n" % (batch_size, args.seed)
    welcome += "# Using device: %s\n" % str(device)
    welcome += "# samples, diff E c, diff E x, diff Omega, diff Phi"
    log(welcome, logfile)

    # Hold the Monte Carlo estimators computed here
    variables = ["mean_c", "mean_x", "omega", "phi"]
    mc = {
        "mean_c": torch.zeros(D).to(device),  # estimate of mean of c
        "mean_x": torch.zeros(N).to(device),  # estimate of mean of x
        "omega": torch.zeros(N, N).to(device),  # input-input covariance
        "phi": torch.zeros(N, D).to(device),  # input-latent covariance
    }
    M2_omega = torch.zeros(N, N).to(device)  # running estimate of residuals
    M2_phi = torch.zeros(N, D).to(device)  # running estimate of residuals

    exact = {"mean_c": None, "mean_x": None, "omega": None, "phi": None}
    if args.scenario == "rand":
        exact["mean_c"] = torch.zeros(D).to(device)
        exact["mean_x"] = torch.zeros(N).to(device)

        b = np.sqrt(2 / np.pi)
        c = 1
        b2 = pow(b, 2)
        Phi = None
        Omega = None
        for l in range(generator.num_layers):
            F = generator.generator[l * 2].weight.data
            if l == 0:
                Omega = b2 * F @ F.T
                Phi = b * F
            else:
                Omega = (
                    b2 * F @ ((c - b2) * torch.eye(F.shape[1]).to(device) + Omega) @ F.T
                )
                Phi = b * F @ Phi
        Omega[np.diag_indices(N)] = c
        exact["omega"] = Omega
        exact["phi"] = Phi

    # store the values of the covariance matrices at the last checkpoint
    mc_last = dict()
    for name in variables:
        mc_last[name] = torch.zeros(mc[name].shape).to(device)

    step = -1
    with torch.no_grad():
        while step < args.steps:
            for _ in tqdm(range(args.checkpoint)):
                # slighly unsual place for step increment; is to preserve the usual notation
                # when computing the current estimate of the covariance outside this loop
                step += 1

                # Generate a new batch of data
                cs = torch.randn(batch_size, D).to(device)
                # add dimensions for the convolutions
                if args.scenario in ["rand", "spiked"]:
                    latent = cs
                else:
                    latent = cs.unsqueeze(-1).unsqueeze(-1)
                # pass through the generator
                xs = generator(latent).reshape(-1, N)

                # Update the estimators.
                ########################
                mc_mean_x_old = mc["mean_x"]
                # Start with the means
                dmean_c = torch.mean(cs, axis=0) - mc["mean_c"]
                mc["mean_c"] += dmean_c / (step + 1)
                dmean_x = torch.mean(xs, axis=0) - mc["mean_x"]
                mc["mean_x"] += dmean_x / (step + 1)
                # now the residuals
                M2_omega += (xs - mc_mean_x_old).T @ (xs - mc["mean_x"]) / batch_size
                M2_phi += (xs - mc_mean_x_old).T @ (cs - mc["mean_c"]) / batch_size

            mc["omega"] = M2_omega / (step + 1)
            mc["phi"] = M2_phi / (step + 1)

            # Build status message
            status = "%g" % (step * args.bs)
            for name in variables:
                diff = torch.sqrt(torch.mean((mc[name] - mc_last[name]) ** 2))
                status += ", %g" % diff

            # if exact expression is available, also compute the error of the current estimate
            for name in ["omega", "phi"]:
                if exact[name] is None:
                    status += ", nan"
                else:
                    diff = torch.sum((mc[name] - exact[name]) ** 2) / torch.sum(
                        exact[name] ** 2
                    )
                    status += ", %g" % diff

            log(status, logfile)

            # Write the estimates to files
            for name in variables:
                fname = log_fname[:-4] + ("_%s_%g.pt" % (name, step * batch_size))
                torch.save(mc[name], fname)

            for name in variables:
                mc_last[name] = mc[name].clone().detach()


if __name__ == "__main__":
    main()
