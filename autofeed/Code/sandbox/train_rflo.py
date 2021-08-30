import torch
from time import time
import numpy as np
import numpy.random as npr
from matplotlib import pyplot as plt
from plot import plot
from model_numpy import Network, Task, Plant, Algorithm
from model_torch import Trial
from run import run_torch
import pickle

# nlin
def phi(x):
    return torch.tanh(x) if torch.is_tensor(x) else np.tanh(x)

# derivative of nlin
def dphi(x):
    return 1 / (torch.cosh(10 * torch.tanh(x / 10)) ** 2) if torch.is_tensor(x) \
        else 1 / (np.cosh(10 * np.tanh(x / 10)) ** 2)

# load teacher from PATH
fname = 'feed.pt'
teacher = torch.load('C:\\Users\\jkl9\\PycharmProjects\\autofeed\\Data\\' + fname)

# instantiate student
net, task, plant, algo = Network('ThCtx', N=200), Task('Reaching', rand_tar=False, num_tar=4, auto=True), \
                             Plant('TwoLink'), Algorithm('rflo', lr=1e-1, B=1, Nepochs=4000, online=True)
add_sensoryprediction, add_sensorynoise, add_sensorydelay = False, False, False

# seed
npr.seed(4)

# figure preferences
doplot = True
num_to_plot = 1
plot_every = algo.Nepochs * 0.2

# frequently used vars
dt, NT, B, N, S, R = net.dt, int(task.T / net.dt), algo.B, net.N, net.S, net.R
t = dt * np.arange(NT)

# initial weight tensors
net.ws = net.g_in * np.random.standard_normal([S, N]).astype(np.float32) / np.sqrt(S)
for idx in range(task.num_tar):
    net.J.append(net.g_rec * np.random.standard_normal([N, N]).astype(np.float32) / np.sqrt(N))
net.wr = net.g_out * np.random.standard_normal([N, R]).astype(np.float32) / np.sqrt(N)
net.b = np.zeros([1, N]).astype(np.float32)
net.L = np.random.standard_normal([R, N]).astype(np.float32) / np.sqrt(R)

# OU noise parameters
ou_param_1 = np.exp(-dt / plant.noise_corr_time)
ou_param_2 = np.sqrt(1 - ou_param_1 ** 2)

mses, conds = [], []
for ei in range(algo.Nepochs):

    # create new trial
    trial = Trial()

    # initial and target positions
    trial.add_xinits(plant, B)
    cond = int(np.floor(npr.rand()*4))
    trial.add_xtargs(task, cond, B)
    x, x_predict, x_sense = trial.xinits, trial.xinits, trial.xinits
    # context cue
    trial.add_context(task, B)
    # go cue
    trial.add_go(task, net.dt, B)

    # simulate teacher learned using bptt
    msestar, ustar, xstar = run_torch(teacher, trial, doplot=False)    # simulate one trial
    ustar = ustar.detach().numpy()

    # convert to numpy array for training rflo
    for attr in list(trial.__dict__.keys()):
        if torch.is_tensor(getattr(trial, attr)): setattr(trial, attr, getattr(trial, attr).numpy())

    # random initialization of hidden state
    z = net.sig * npr.randn(B, N).astype(np.float32)   # hidden state (potential)
    h = phi(z)   # hidden state (rate)

    # initial arm pos and velocity in angular coordinates
    w = np.tile(plant.w_init, (B, 1))
    v = np.zeros((B, R))

    # initial noise
    noise = npr.randn(B, R) * plant.noise_scale

    # delay in sensory feedback
    delay = npr.rand(1, B, 1) * (plant.delay_feed[1] - plant.delay_feed[0]) + plant.delay_feed[0]
    delay = np.round(delay / dt).astype(int)
    delay = delay.repeat(R, axis=2)

    # std dev of noise in sensory
    noise_sig = npr.rand(B, 1) * (plant.noise_feed[1] - plant.noise_feed[0]) + plant.noise_feed[0]

    # save tensors for plotting
    sa = np.zeros((NT, B, S))  # save the inputs for each time bin for plotting
    ha = np.zeros((NT, B, N))  # save the hidden states for each time bin for plotting
    ua = np.zeros((NT, B, R))  # angular acceleration of joints
    na = npr.randn(NT, B, R) * plant.noise_scale  # multiplicative noise
    xa = np.zeros((NT, B, R))  # angular position of joints
    xpa = np.zeros((NT, B, R))  # angular position of joints (predicted)
    xsa = np.zeros((NT, B, R))  # angular position of joints (sensed)

    # errors
    err = np.zeros((NT, B, R))     # error in angular acceleration

    # eligibility traces p, q
    p = dphi(z).T * ha[0]
    q = dphi(z).T * sa[0]

    # store weight changes for offline learning
    dwr = np.zeros_like(net.wr)
    dJ = np.zeros_like(net.J)
    dws = np.zeros_like(net.ws)

    for ti in range(NT):
        # network update
        s = np.concatenate((trial.xtargs, x_sense * trial.c[:, :1], trial.c, trial.g[ti], x_predict * add_sensoryprediction), axis=-1)  # input
        z = np.matmul(s, net.ws) + np.matmul(h, net.J[cond]) + net.b
        h = h + dt * (-h + phi(z))  # dynamics
        u = np.matmul(h, net.wr)  # output

        # physics
        if add_sensoryprediction:
            x_predict = plant.forward_predict(u, v, w, dt)  # predict state from u alone
        noise = ou_param_1 * noise + ou_param_2 * na[ti]  # noise is OU process
        v, w, x = plant.forward(u, v, w, noise, dt)  # actual state

        # save values for plotting
        ha[ti], ua[ti], na[ti], xa[ti], xpa[ti], xsa[ti] = h, u, noise, x, x_predict, x_sense

        # add sensory delay
        if add_sensorydelay:
            x_sense = torch.squeeze(torch.gather(xa.clone(), 0, torch.maximum(ti - delay, torch.zeros_like(delay))))
        else:
            x_sense = x

        # add sensory noise
        if add_sensorynoise:
            x_sense = x_sense + (noise_sig * npr.randn(B, R)).astype(np.float32)

        # error
        err[ti] = ustar[ti] - u

        # update eligibility trace
        p = dt * dphi(z).T * h + (1 - dt) * p
        q = dt * dphi(z).T * s + (1 - dt) * q

        # online weight update
        if algo.online:
            net.wr += (algo.lr / NT) * h.T * err[ti]
            net.J[cond] += ((algo.lr / NT) * np.matmul(err[ti], net.L).T.repeat(N, axis=1) * p).T
            net.ws += ((algo.lr / NT) * np.matmul(err[ti], net.L).T.repeat(S, axis=1) * q).T
        else:
            dwr += (algo.lr / NT) * h.T * err[ti]
            dJ += ((algo.lr / NT) * np.matmul(err[ti], net.L).T.repeat(N, axis=1) * p).T
            dws += ((algo.lr / NT) * np.matmul(err[ti], net.L).T.repeat(S, axis=1) * q).T

    # offline update
    if not algo.online:
        net.wr += dwr
        net.J[cond] += dJ
        net.ws += dws

    # print loss
    mse = (err ** 2).mean().item() / 2
    print('\r' + str(ei + 1) + '/' + str(algo.Nepochs) + '\t Err:' + str(mse), end='')

    # save mse list and cond list
    mses.append(mse)
    conds.append(cond)

    if ei % 4000 == 0:
        plt.figure()
        plt.subplot(221)
        plt.plot(mses)
        plt.subplot(222)
        plt.plot(ua.squeeze())
        plt.plot(ustar.squeeze())
        plt.plot(err.squeeze())
        plt.subplot(223)
        plt.plot(xa[:, 0, 0], xa[:, 0, 1])
        plt.xlim(-1, 1)
        plt.ylim(-.2, 2.2)
        plt.subplot(224)
        plt.plot(xa.squeeze())
        #plot(1, ei, dt, t, task.T, ha, ua, na, noise, xa, trial.xinits, trial.xtargs, trial.starts, trial.stops,
        #     trial.c, mses, 0, 0, init=True, with_noise=True)
        plt.show()

    del p, q, x, x_predict, x_sense, dwr, dJ, dws, noise, delay, noise_sig, s, z, h, u, ustar, err, \
        sa, ha, ua, na, xa, xpa, xsa

##
np.save('C:\\Users\\jkl9\\PycharmProjects\\autofeed\\Data\\automulti_rflo.npy',
        {'net': net, 'task': task, 'plant': plant, 'algo': algo, 'mses': mses, 'add_sensoryprediction': add_sensoryprediction,
            'add_sensorynoise': add_sensorynoise, 'add_sensorydelay': add_sensorydelay}, allow_pickle=True)
