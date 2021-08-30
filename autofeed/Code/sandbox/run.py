import math
import torch
import numpy as np
import numpy.random as npr
from matplotlib import pyplot as plt
from plot import plot

# nlin
def phi(x):
    return torch.tanh(x) if torch.is_tensor(x) else np.tanh(x)

# derivative of nlin
def dphi(x):
    return 1 / (torch.cosh(10 * torch.tanh(x / 10)) ** 2) if torch.is_tensor(x) \
        else 1 / (np.cosh(10 * np.tanh(x / 10)) ** 2)

def run_torch(data, trial, doplot=False):
    net, task, plant, algo = data['net'], data['task'], data['plant'], data['algo']
    add_sensoryprediction, add_sensorynoise, add_sensorydelay = data['add_sensoryprediction'], \
                                                                data['add_sensorynoise'], data['add_sensorydelay']

    # frequently used vars
    dt, NT, N, S, R = net.dt, int(task.T / net.dt), net.N, net.S, net.R
    t = dt * np.arange(NT)

    # OU noise parameters
    ou_param_1 = np.exp(-dt / plant.noise_corr_time)
    ou_param_2 = np.sqrt(1 - ou_param_1 ** 2)

    mses = []

    # initial and target positions
    xinits = trial.xinits[0] #torch.as_tensor(np.array(plant.x_init, dtype=np.float32))
    # ang = npr.rand() * 2 * np.pi if task.rand_tar else np.pi / 2
    xtargs = trial.xtargs[0] #torch.as_tensor((np.cos(ang), np.sin(ang) + 1), dtype=torch.float32)
    x = xinits
    x_predict = xinits
    x_sense = xinits

    # context cue
    #c = trial.c[0]
    if task.feed and not task.auto:
        c = torch.as_tensor(np.array(task.c_feed, dtype=np.float32))
    elif task.auto and not task.feed:
        c = torch.as_tensor(np.array(task.c_auto, dtype=np.float32))
    else:  # randomly choose between auto and feed:
        c = torch.as_tensor(np.array(task.c_feed, dtype=np.float32)) if np.random.rand() > .5 \
            else torch.as_tensor(np.array(task.c_auto, dtype=np.float32))

    # go cue
    g = trial.g[:, :, 0] #torch.as_tensor(np.concatenate((np.ones((start, 1)), np.zeros((NT - start, 1))), dtype=np.float32))

    # random initialization of hidden state
    h = torch.as_tensor(net.sig * npr.randn(1, N).astype(np.float32))

    # initial arm pos and velocity in angular coordinates
    w = torch.as_tensor(plant.w_init, dtype=torch.float32)
    v = torch.zeros(R)

    # initial noise
    noise = torch.as_tensor(npr.randn(1, R).astype(np.float32)) * plant.noise_scale

    # delay in sensory feedback
    delay = npr.rand(1, 1, 1) * (plant.delay_feed[1] - plant.delay_feed[0]) + plant.delay_feed[0]
    delay = np.round(delay / dt).astype(int)
    delay = torch.tensor(delay).repeat(1, 1, R).long()

    # std dev of noise in sensory
    noise_sig = npr.rand(1, 1) * (plant.noise_feed[1] - plant.noise_feed[0]) + plant.noise_feed[0]

    # save tensors for plotting
    ha = torch.zeros(NT, 1, N)  # save the hidden states for each time bin for plotting
    ua = torch.zeros(NT, 1, R)  # angular acceleration of joints
    na = torch.as_tensor(npr.randn(NT, 1, R).astype(np.float32)) * plant.noise_scale  # multiplicative noise
    xa = torch.zeros(NT, 1, R)  # angular position of joints
    xpa = torch.zeros(NT, 1, R)  # angular position of joints (predicted)
    xsa = torch.zeros(NT, 1, R)  # angular position of joints (sensed)

    for ti in range(NT):
        # network update
        s = torch.cat((xtargs, x_sense * c[:1], c, g[ti], x_predict * add_sensoryprediction), dim=-1)  # input
        h = h + dt * (-h + torch.tanh(s.unsqueeze(0).mm(net.ws) + h.mm(net.J) + net.b))  # dynamics
        u = h.mm(net.wr).flatten()  # output

        # physics
        if add_sensoryprediction:
            x_predict = plant.forward_predict(u, v, w, dt)  # predict state from u alone
        noise = ou_param_1 * noise + ou_param_2 * na[ti]  # noise is OU process
        v, w, x = plant.forward(u, v, w, noise, dt)  # actual state
        v, w, x = v.flatten(), w.flatten(), x.flatten()

        # save values for plotting
        ha[ti], ua[ti], na[ti], xa[ti], xpa[ti], xsa[ti] = h, u, noise, x, x_predict, x_sense

        # add sensory delay
        if add_sensorydelay:
            x_sense = torch.squeeze(torch.gather(xa.clone(), 0, torch.maximum(ti - delay, torch.zeros_like(delay))))
        else:
            x_sense = x

        # add sensory noise
        if add_sensorynoise:
            x_sense = x_sense + torch.as_tensor((noise_sig * npr.randn(1, R)).astype(np.float32))

    # loss is sum of mse and regularization terms
    start, stop = torch.as_tensor(trial.starts[0]).unsqueeze(0), torch.as_tensor(trial.stops[0]).unsqueeze(0)
    mse, u_reg, du_reg = task.loss(dt, xa, ua, trial.xinits, trial.xtargs, trial.starts, trial.stops,
                                   algo.lambda_u, algo.lambda_du)

    # plot
    if doplot:
        plot(1, 1, dt, t, task.T, ha, ua, na, noise, xa, xinits, xtargs, start, stop, c, mse, with_noise=True)

    return mse, ua, xa


def run_numpy(data, trial):

    net, task, plant, algo = data['net'], data['task'], data['plant'], data['algo']
    add_sensoryprediction, add_sensorynoise, add_sensorydelay = data['add_sensoryprediction'], \
                                                                data['add_sensorynoise'], data['add_sensorydelay']

    # convert to numpy array for training rflo
    for attr in list(trial.__dict__.keys()):
        if torch.is_tensor(getattr(trial, attr)): setattr(trial, attr, getattr(trial, attr).numpy())

    x, x_predict, x_sense = trial.xinits, trial.xinits, trial.xinits

    # frequently used vars
    dt, NT, B, N, S, R = net.dt, int(task.T / net.dt), algo.B, net.N, net.S, net.R
    t = dt * np.arange(NT)

    # OU noise parameters
    ou_param_1 = np.exp(-dt / plant.noise_corr_time)
    ou_param_2 = np.sqrt(1 - ou_param_1 ** 2)

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

    for ti in range(NT):
        # network update
        s = np.concatenate((trial.xtargs, x_sense * trial.c[:, :1], trial.c, trial.g[ti], x_predict * add_sensoryprediction), axis=-1)  # input
        z = np.matmul(s, net.ws) + np.matmul(h, net.J) + net.b
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

    return ua, xa
