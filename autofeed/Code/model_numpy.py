import numpy as np
import math
import numpy.random as npr


class Network:
    def __init__(self, name='Ctx', N=200, M=0):
        self.name = name
        # network parameters
        self.N = N  # RNN units
        self.dt = .1  # time bin (in units of tau)
        self.g_in = 1.0  # initial input weight scale
        self.g_rec = 1.4  # initial recurrent weight scale
        self.g_out = 0.01  # initial output weight scale
        self.S_state, self.S_targ, self.S_context, self.S_go, self.S_pred = 2, 2, 2, 1, 2
        self.S = self.S_state + self.S_targ + self.S_context + self.S_go + self.S_pred  # input, used here
        self.R = 2  # readout
        self.sig = 0.01  # initial activity scale
        self.ws = []  # input, used here
        self.J = []  # readout
        self.wr = []  # time bin (in units of tau)
        self.b = []  # initial activity scale
        if self.name == 'ThCtx':
            self.M = M
            self.Jct = []

    # nlin
    def f(self, x):
        return np.tanh(x)

    # derivative of nlin
    def df(self, x):
        return 1 / (np.cosh(10*np.tanh(x/10)) ** 2)

    # add input channels
    def add_inputs(self, s):
        self.S += s

    # add input channels
    def add_units(self, n):
        self.N += n

    # add input channels
    def add_outputs(self, r):
        self.R += r


class Task:
    def __init__(self, name='Reaching', rand_tar=True, num_tar=None, auto=False, feed=False, both=False):
        self.name = name
        # task parameters
        self.T = 35  # duration (in units of tau)
        self.g_lims = (5, 20)
        self.rand_tar = rand_tar
        self.num_tar = num_tar
        self.move_time = 10
        self.c_auto, self.c_feed = [0, 1.0], [1.0, 0]
        self.auto, self.feed, self.both = auto, feed, both
        self.switch_after = .5  # fraction of epochs before multitasking

    def loss(self, dt, xt, ut, xinits, xtargs, starts, stops, lambda_u=0, lambda_du=0):
        if self.name == 'Reaching':
            mse = np.stack([((xt[:start] - xinit) ** 2).mean() + ((xt[stop:] - xtarg) ** 2).mean()
                               for xt, xinit, xtarg, start, stop in zip(xt.transpose(0, 1), xinits, xtargs, starts, stops)
                               ]).mean() / 2
            return mse


class Plant:
    def __init__(self, name='TwoLink'):
        self.name = name
        # physics parameters
        self.noise_scale = 0.01  # network output multiplicative noise scale (0.2)
        self.noise_corr_time = 4  # noise correlation time (units of tau)
        self.drag_coeff = 1.0
        self.noise_feed = (0, 0.2)    # sensory noise (range of SD)
        self.delay_feed = (3, 5)     # sensory delay (units of tau)
        self.w_init = [math.pi / 6, 2 * math.pi / 3]    # initial angles of links
        self.x_init = [0, 1.0]  # initial position of endpoint

    # plant dynamics (actual dynamics with noise)
    def forward(self, u, v, w, noise, dt):
        x = None
        # physics
        if self.name == 'TwoLink':
            accel = u + np.linalg.norm(u, axis=-1, keepdims=True) * noise - self.drag_coeff * v
            v_new = v + accel * dt
            w = w + v * dt + 0.5 * accel * dt ** 2
            v = v_new
            # hand location
            ang1, ang2 = w[:, 0], w.sum(axis=-1)
            x = np.stack((np.cos(ang1) + np.cos(ang2), np.sin(ang1) + np.sin(ang2)), axis=-1)
        # return
        return v, w, x

    # predicted plant dynamics (predicted dynamics unaware of noise)
    def forward_predict(self, u, v, w, dt):
        x_predict = None
        # physics
        if self.name == 'TwoLink':
            accel = u - self.drag_coeff * v
            v_new = v + accel * dt
            w = w + v * dt + 0.5 * accel * dt ** 2
            v = v_new
            # hand location
            ang1, ang2 = w[:, 0], w.sum(dim=-1)
            x_predict = np.stack((np.cos(ang1) + np.cos(ang2), np.sin(ang1) + np.sin(ang2)), dim=-1)
        # return
        return x_predict


class Algorithm:
    def __init__(self, name='Adam', Nepochs=1000, B=40, lr=1e-4, online=False):
        self.name = name
        # learning parameters
        self.Nepochs = Nepochs
        self.Nstart_anneal = 30000
        self.B = B  # batches per epoch
        self.lr = lr  # learning rate
        self.annealed_lr = 1e-6
        self.lambda_u = 5e-2
        self.lambda_du = 5e-1
        self.online = online


class Trial:
    def __init__(self):
        # learning parameters
        self.xinits = []    # initial states
        self.xtargs = []    # target states
        self.c = []     # context cue
        self.starts = []
        self.stops = []
        self.g = []     # go cue

    def add_xinits(self, plant, B=1):
        self.xinits = np.tile(np.array(plant.x_init, dtype=np.float32), (B, 1))

    def add_xtargs(self, task, B=1):
        ang = npr.rand(B, 1) * 2 * np.pi if task.rand_tar else np.ones((B, 1)) * np.pi / 2
        self.xtargs = np.concatenate((np.cos(ang), np.sin(ang) + 1), axis=-1, dtype=np.float32)

    def add_context(self, task, B=1):
        if task.feed and not task.auto:
            self.c = np.tile(np.array(task.c_feed, dtype=np.float32), (B, 1))
        elif task.auto and not task.feed:
            self.c = np.tile(np.array(task.c_auto, dtype=np.float32), (B, 1))
        else:  # randomly interleave auto and feed:
            c = np.cat((np.tile(np.array(task.c_feed, dtype=np.float32), (int(B / 2), 1)),
                           np.tile(np.array(task.c_auto, dtype=np.float32), (int(B / 2), 1))))
            self.c = c[np.randperm(B)]

    def add_go(self, task, dt, B=1):
        NT = int(task.T / dt)
        self.starts = np.round((npr.rand(B) * (task.g_lims[1] - task.g_lims[0]) + task.g_lims[0]) / dt).astype(int)
        self.stops = self.starts + int(task.move_time / dt)
        self.g = np.stack(
            [np.concatenate((np.ones((self.starts[bi], 1)), np.zeros((NT - self.starts[bi], 1))), dtype=np.float32)
             for bi in range(B)], axis=1)
