import math
import torch
import numpy as np
import numpy.random as npr
from matplotlib import pyplot as plt
from plot import plotdata
from model_numpy import Network, Task, Plant, Algorithm
from model_torch import Trial
from run import run_numpy

# load data from PATH
fname = 'auto_rflo.npy'
data = np.load('C:\\Users\\jkl9\\PycharmProjects\\autofeed\\Data\\' + fname, allow_pickle=True)
net, task, plant, algo = data.item()['net'], data.item()['task'], data.item()['plant'], data.item()['algo']
mses = data.item()['mses']
add_sensoryprediction, add_sensorynoise, add_sensorydelay = data.item()['add_sensoryprediction'], \
                                                            data.item()['add_sensorynoise'], data.item()['add_sensorydelay']

# seed
plant.noise_scale = 0.01
npr.seed(4)

Nepochs, trials, uas, xas = 10, [], [], []
for ei in range(Nepochs):

    # create new trial
    trial = Trial()

    # initial and target positions
    trial.add_xinits(plant, algo.B)
    trial.add_xtargs(task, algo.B)
    x, x_predict, x_sense = trial.xinits, trial.xinits, trial.xinits
    # context cue
    trial.add_context(task, algo.B)
    # go cue
    trial.add_go(task, net.dt, algo.B)

    # simulate teacher learned using bptt
    trials.append(trial)
    ua, xa = run_numpy(data.item(), trial)    # simulate one trial
    uas.append(ua)
    xas.append(xa)

plotdata(mses, trials, uas, xas, net.dt, B=4)