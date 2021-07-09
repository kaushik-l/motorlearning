import math
import torch
from time import time
import numpy as np
import numpy.random as npr
from matplotlib import pyplot as plt


def plot(B, epoch, fig=None, init=True, with_noise=False, final=False):
    if init:
        plot.fig = plt.figure(figsize=(12, 7)) if fig is None else fig
        plot.a = [plt.subplot(2, 1 + B, ai + 1) for ai in range(2 * (1 + B))]

    # loss plot
    a = plot.a[0]
    a.clear()
    plt.sca(a)
    plt.plot(mses, 'k')
    plt.plot(u_regs, '--', color='tab:red')
    plt.plot(du_regs, '--', color='tab:olive')
    plt.yscale('log')
    plt.legend(['MSE', '$\lambda_1|u|^2$', '$\lambda_2|du/dt|^2$'])
    plt.ylabel('Loss')
    plt.xlabel('Trial')

    # trajectories plot
    a = plot.a[B + 1]
    a.clear()
    plt.sca(a)
    for b in range(B):
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][b]
        plt.plot(xta[:, b, 0].detach(), xta[:, b, 1].detach(), color=color)
        plt.scatter(xinits[b, 0], xinits[b, 1], c='k')
        plt.scatter(xtargs[b, 0], xtargs[b, 1], color=color)
    plt.xlim((-1.2, 1.2))
    plt.ylim((-0.2, 2.2))
    plt.title('Trajectory')
    plt.ylabel('x_2 (m)')
    plt.xlabel('x_1 (m)')

    for b in range(B):
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][b]

        # network output plot
        a = plot.a[b + 1]
        a.clear()
        plt.sca(a)
        tmp = ua[:, b].detach()
        if with_noise:
            noise = tmp + torch.linalg.norm(tmp, dim=-1, keepdim=True) * na[:, b].detach()
        max_u = (noise if with_noise else tmp).abs().max()
        plt.plot([starts[b] * dt, starts[b] * dt], [-1.05 * max_u, 1.05 * max_u], 'k', linewidth=0.5)
        plt.plot([stops[b] * dt, stops[b] * dt], [-1.05 * max_u, 1.05 * max_u], 'k', linewidth=0.5)
        if with_noise:
            plt.plot(t, noise[:, 0], c=color, linewidth=0.5)
            plt.plot(t, noise[:, 1], c=color, linewidth=0.5, alpha=0.25)
        plt.plot(t, tmp[:, 0], c=color)
        plt.plot(t, tmp[:, 1], c=color, alpha=0.5)
        #         plt.ylim((-0.04, 0.04))
        plt.title('Trial type: autonomous' if np.array_equal([0, 1], c[b]) else 'Trial type: with feedback')
        if b == 0: plt.ylabel('Angular Acc (rad/s^2)')

        # hand position plot
        a = plot.a[B + 2 + b]
        a.clear()
        plt.sca(a)
        plt.plot([starts[b] * dt, starts[b] * dt], [-1, 2], 'k', linewidth=0.5)
        plt.plot([stops[b] * dt, stops[b] * dt], [-1, 2], 'k', linewidth=0.5)
        plt.plot([0, starts[b] * dt], [xinits[b, 0], xinits[b, 0]], 'k', linewidth=2)
        plt.plot([0, starts[b] * dt], [xinits[b, 1], xinits[b, 1]], 'k', linewidth=2)
        plt.plot([stops[b] * dt, T], [xtargs[b, 0], xtargs[b, 0]], 'k', linewidth=2)
        plt.plot([stops[b] * dt, T], [xtargs[b, 1], xtargs[b, 1]], 'k', linewidth=2)
        plt.plot(t, xta[:, b, 0].detach(), c=color)
        plt.plot(t, xta[:, b, 1].detach(), c=color, alpha=0.25)
        plt.ylim((-1.3, 2.3))
        plt.xlabel('Time')
        if b == 0: plt.ylabel('Arm Pos (m)')

    #     # heatmap of units
    #     a = plot.a[5]
    #     a.clear()
    #     plt.sca(a)
    #     plt.imshow(ha[:, 0].detach().T, vmin=-1, vmax=1)
    #     plt.ylabel('Unit')
    #     plt.xlabel('Time')

    plt.suptitle(f'epoch: {epoch}')
    plt.tight_layout()
    plot.fig.canvas.draw()
    plt.show()

    if final:
        plt.figure()
        plot.a = [plt.subplot(2, 2, ai + 1) for ai in range(4)]
        #plt.scatter(targangles, mses_by_targ, s=0.1, c='tab:gray')
        a = plot.a[0]
        a.clear()
        plt.sca(a)
        plt.plot(mses, 'k')
        plt.yscale('log')
        plt.ylabel('Loss', fontsize=14)
        plt.xlabel('Trial')
        a = plot.a[1]
        a.clear()
        plt.sca(a)
        plt.plot(delta_ws, color='k')
        plt.plot(delta1_ws, '--', color='tab:red')
        plt.plot(delta2_ws, '--', color='tab:green')
        plt.yscale('log')
        plt.legend(['All units', 'Input units', 'Output units'])
        plt.ylabel('$|\Delta w_s|$', fontsize=14)
        plt.xlabel('Trial')
        a = plot.a[2]
        a.clear()
        plt.sca(a)
        plt.plot(delta_J, color='k')
        plt.plot(delta1_J, '--', color='tab:red')
        plt.plot(delta2_J, '--', color='tab:green')
        plt.yscale('log')
        plt.legend(['All units', 'Input units', 'Output units'])
        plt.ylabel('$|\Delta J|$', fontsize=14)
        plt.xlabel('Trial')
        a = plot.a[3]
        a.clear()
        plt.sca(a)
        plt.plot(delta_wr, color='k')
        plt.plot(delta1_wr, '--', color='tab:red')
        plt.plot(delta2_wr, '--', color='tab:green')
        plt.yscale('log')
        plt.legend(['All units', 'Input units', 'Output units'])
        plt.ylabel('$|\Delta w_r|$', fontsize=14)
        plt.xlabel('Trial')
        plt.show()


##
doplot, auto, feed, both, upperthenlower, lowerthenupper = True, False, True, False, False, False  # plot?, autonomous?, w/ feedback?, both?
num_to_plot = 5

# network parameters
N = 200  # RNN units
g_in = 1.0  # initial input weight scale
g_rec = 1.4  # initial recurrent weight scale
g_out = 0.01  # initial output weight scale
S = 7  # input, used here
R = 2  # readout
dt = .1     # time bin (in units of tau)
sig = 0.01   # initial activity scale

# task parameters
T = 35      # duration (in units of tau)
g_lims = (5, 20)
rand_tar = True
move_time = 10

# physics parameters
noise_scale = 0.1  # network output multiplicative noise scale
noise_corr_time = 5  # noise correlation time (units of tau)
drag_coeff = 1.0

# learning parameters
Nepochs = 500
plot_every = 500
Nstart_anneal = 40000
B = 40  # batches per epoch
lr = 5e-5   # learning rate
annealed_lr = 1e-6
lambda_u = 5e-2
lambda_du = 5e-1

seed = 4


## seed
npr.seed(seed)

# useful values
NT = int(T / dt)
t = dt * np.arange(NT)

# OU noise parameters
ou_param_1 = np.exp(-dt / noise_corr_time)
ou_param_2 = np.sqrt(1 - ou_param_1**2) * noise_scale

# initial weights
ws0 = g_in * np.random.standard_normal([S, N]).astype(np.float32) / np.sqrt(S)
J0 = g_rec * np.random.standard_normal([N, N]).astype(np.float32) / np.sqrt(N)
wr0 = g_out * np.random.standard_normal([N, R]).astype(np.float32) / np.sqrt(N)
b0 = np.zeros([1, N]).astype(np.float32)

# initial weight tensors
ws = torch.tensor(ws0, requires_grad=True)
J = torch.tensor(J0, requires_grad=True)
wr = torch.tensor(wr0, requires_grad=True)
b = torch.tensor(b0, requires_grad=True)

# mask for zero weights
mask_ws = torch.cat([torch.ones(S, int(N/2)).T, torch.zeros(S, int(N/2)).T]).T
mask_wr = torch.cat([torch.zeros(R, int(N/2)).T, torch.ones(R, int(N/2)).T])
#ws.data = ws.data * mask_ws
#wr.data = wr.data * mask_wr

# optimizer
opt = torch.optim.Adam([J, wr, b, ws], lr=lr)

##
fig = plt.figure(figsize=(12, 7))


##
losses, mses, u_regs, du_regs = [], [], [], []
delta_ws, delta1_ws, delta2_ws, delta_J, delta1_J, delta2_J, delta_wr, delta1_wr, delta2_wr = [], [], [], [], [], [], [], [], []
prevt = time()
lr_ = lr
for ei in range(Nepochs):

    # initial and target positions
    xinits = torch.as_tensor(np.tile(np.array([0, 1.0], dtype=np.float32), (B, 1)))
    ang = npr.rand(B, 1) * 2 * np.pi if rand_tar else np.ones((B, 1)) * np.pi / 2
    xtargs = torch.as_tensor(np.concatenate((np.cos(ang), np.sin(ang) + 1), axis=-1, dtype=np.float32))
    xt = xinits

    # context cue
    if both and ei > int(Nepochs/2):
        auto, feed = True, True

    if feed and not auto:
        c = torch.as_tensor(np.tile(np.array([1.0, 0], dtype=np.float32), (B, 1)))
    elif auto and not feed:
        c = torch.as_tensor(np.tile(np.array([0, 1.0], dtype=np.float32), (B, 1)))
    else:  # randomly interleave auto and feed:
        c = torch.cat((torch.as_tensor(np.tile(np.array([1.0, 0], dtype=np.float32), (int(B/2), 1))),
                      torch.as_tensor(np.tile(np.array([0, 1.0], dtype=np.float32), (int(B/2), 1)))))
        c = c[torch.randperm(B)]

    # go cue
    starts = np.round((npr.rand(B) * (g_lims[1] - g_lims[0]) + g_lims[0]) / dt).astype(int)
    stops = starts + int(move_time / dt)
    g = torch.as_tensor(np.stack(
        [np.concatenate((np.ones((starts[bi], 1)), np.zeros((NT - starts[bi], 1))), dtype=np.float32) for bi in
         range(B)], axis=1))

    # random initialization of hidden state
    h = torch.as_tensor(sig * npr.randn(B, N).astype(np.float32))

    # initial arm pos and velocity in angular coordinates
    w = torch.as_tensor(np.tile([math.pi / 6, 2 * math.pi / 3], (B, 1)), dtype=torch.float32)
    v = torch.zeros(B, R)

    # initial noise
    noise = torch.as_tensor(npr.randn(B, R).astype(np.float32)) * noise_scale

    # save tensors for plotting
    ha = torch.zeros(NT, B, N)  # save the hidden states for each time bin for plotting
    ua = torch.zeros(NT, B, R)  # angular acceleration of joints
    na = torch.as_tensor(npr.randn(NT, B, R).astype(np.float32)) * ou_param_2  # multiplicative noise
    xta = torch.zeros(NT, B, R)  # angular position of joints

    for ti in range(NT):
        # network update
        s = torch.cat((xtargs, xt * c[:, :1], c, g[ti]), dim=-1)  # input
        h = h + dt * (-h + torch.tanh(s.mm(ws) + h.mm(J) + b))  # dynamics
        u = h.mm(wr)  # output

        # physics
        noise = ou_param_1 * noise + na[ti]  # noise is OU process
        accel = u + torch.linalg.norm(u, dim=-1, keepdim=True) * noise - drag_coeff * v
        v_new = v + accel * dt
        w = w + v * dt + 0.5 * accel * dt ** 2
        v = v_new

        # hand location
        ang1, ang2 = w[:, 0], w.sum(dim=-1)
        xt = torch.stack((torch.cos(ang1) + torch.cos(ang2), torch.sin(ang1) + torch.sin(ang2)), dim=-1)

        # save values for plotting
        ha[ti] = h
        ua[ti] = u
        na[ti] = noise
        xta[ti] = xt

    # loss is sum of mse and regularization terms
    mse = torch.stack([((xt[:start] - xinit) ** 2).mean() + ((xt[stop:] - xtarg) ** 2).mean()
                       for xt, xinit, xtarg, start, stop in zip(xta.transpose(0, 1), xinits, xtargs, starts, stops)
                       ]).mean() / 2
    u_reg = lambda_u * (ua ** 2).sum(dim=-1).mean()
    du_reg = lambda_du * ((torch.diff(ua, dim=0) / dt) ** 2).sum(dim=-1).mean()
    loss = mse + u_reg + du_reg

    # save loss, mse, and regularization terms
    losses.append(loss.item())
    mses.append(mse.item())
    u_regs.append(u_reg.item())
    du_regs.append(du_reg.item())

    # print loss
    print('\r' + str(ei + 1) + '/' + str(Nepochs) + '\t Err:' + str(loss.item()), end='')

    # update learning rate if needed
    if ei >= Nstart_anneal:
        lr_ *= np.exp(np.log(annealed_lr / lr) / (Nepochs - Nstart_anneal))
        opt.param_groups[0]['lr'] = lr_

    # save weights
    ws_old, J_old, wr_old = ws.detach().clone(), J.detach().clone(), wr.detach().clone()

    # do BPTT
    loss.backward()
    opt.step()
    opt.zero_grad()

    # mask network weights
    #ws.data = ws.data * mask_ws
    #wr.data = wr.data * mask_wr

    # freeze elements of J and b
    if ei <= int(Nepochs / 2):
        if lowerthenupper:
            J.data[:, :int(N/2)] = J_old.data[:, :int(N/2)]
        elif upperthenlower:
            J.data[:, int(N/2):] = J_old.data[:, int(N/2):]
    elif ei > int(Nepochs / 2):
        if lowerthenupper:
            J.data[:, int(N/2):] = J_old.data[:, int(N/2):]
        elif upperthenlower:
            J.data[:, :int(N/2)] = J_old.data[:, :int(N/2)]

    # compute delta weights
    delta_ws.append(torch.norm(ws.data - ws_old.data).item())
    delta1_ws.append(torch.norm(ws.data[:, :int(N/2)] - ws_old.data[:, :int(N/2)]).item())
    delta2_ws.append(torch.norm(ws.data[:, int(N/2):] - ws_old.data[:, int(N/2):]).item())
    delta_J.append(torch.norm(J.data - J_old.data).item())
    delta1_J.append(torch.norm(J.data[:, :int(N/2)] - J_old.data[:, :int(N/2)]).item())
    delta2_J.append(torch.norm(J.data[:, int(N/2):] - J_old.data[:, int(N/2):]).item())
    delta_wr.append(torch.norm(wr.data - wr_old.data).item())
    delta1_wr.append(torch.norm(wr.data[:int(N/2), :] - wr_old.data[:int(N/2), :]).item())
    delta2_wr.append(torch.norm(wr.data[int(N/2):, :] - wr_old.data[int(N/2):, :]).item())

    # plot
    if doplot and ei == 0:
        plot(num_to_plot, ei, fig=fig, with_noise=True)
    elif doplot and ei % plot_every == 0:
        plot(num_to_plot, ei, init=False, with_noise=True)

# final plot
if doplot:
    plot(num_to_plot, Nepochs, init=False, with_noise=True, final=True)

# plot training time
print("train time: ", time() - prevt)


##
torch.save({'ws': ws, 'J': J, 'wr': wr, 'b': b, 'mses': mses,
            'delta_ws': delta_ws, 'delta_J': delta_J, 'delta_wr': delta_wr,
            'delta1_ws': delta1_ws, 'delta1_J': delta1_J, 'delta1_wr': delta1_wr,
            'delta2_ws': delta2_ws, 'delta2_J': delta2_J, 'delta2_wr': delta2_wr},
           'C:\\Users\\jkl9\\PycharmProjects\\autofeed\\Data\\autoboth.pt')

##
torch.as_tensor(mses[-500:]).mean(), torch.as_tensor(u_regs[-500:]).mean(), torch.as_tensor(du_regs[-500:]).mean()