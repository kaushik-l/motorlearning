import math
import torch
import time
import numpy as np
import matplotlib.pyplot as plt


# plot
def plot(g, xtarg, c, u, xt, ha, losslist):
    if ei == 0:
        plt.figure(1)
        plt.subplot(341)
        plt.plot(g.numpy()[:, :, 0], 'k')
        plt.ylim((-0.1, 2.1))
        plt.ylabel('Go Signal')
        plt.subplot(342)
        plt.plot(xtarg.numpy()[:, :, 0])
        plt.ylim((-2.1, 2.1))
        plt.ylabel('Target Pos (m)')
        plt.subplot(343)
        plt.plot(c.numpy()[:, 0, 0], 'g')
        plt.plot(c.numpy()[:, 1, 0], 'r')
        plt.ylim((-0.1, 2.1))
        plt.ylabel('Context Signal')
        plt.subplot(345)
        plt.plot(u.detach().numpy()[:, :, 0])
        plt.ylim((-0.2, 0.2))
        plt.ylabel('Angular Acc (rad/s^2)')
        plt.subplot(346)
        plt.plot(xt.detach().numpy()[:, :, 0])
        plt.ylim((-2.1, 2.1))
        plt.ylabel('Arm Pos (m)')
        plt.subplot(347)
        plt.plot(xt.detach().numpy()[:, 0, 0], xt.detach().numpy()[:, 1, 0])
        plt.scatter(xinit.detach().numpy()[0, 0, 0], xinit.detach().numpy()[0, 1, 0], c='k')
        plt.scatter(xtarg.detach().numpy()[0, 0, 0], xtarg.detach().numpy()[0, 1, 0], c='r')
        plt.xlim((-2.1, 2.1))
        plt.ylim((-2.1, 2.1))
        plt.title('Trajectory')
        plt.ylabel('x_2 (m)')
        plt.subplot(348)
        h = ha.detach().numpy()[:, :, 0]
        plt.imshow(h[:, np.argsort(np.argmax(h, 0))].T, vmin=-1, vmax=1)
        plt.ylabel('Unit')
        plt.draw()
    elif ei+1 == Nepochs or ei % Nepochsplot == 0:
        plt.subplot(344)
        plt.loglog(losslist1, 'c')
        plt.loglog(losslist2, 'm')
        plt.loglog(losslist3, 'y')
        plt.loglog(losslist, 'k')
        plt.yscale('log')
        plt.legend(['Performance', 'Torque', 'Jerk', 'Total'])
        plt.ylabel('Loss')
        plt.xlabel('Trial')
        plt.subplot(349)
        plt.plot(u.detach().numpy()[:, :, 0])
        plt.ylim((-0.015, 0.015))
        plt.ylabel('Angular Acc (rad/s^2)')
        plt.xlabel('Time')
        plt.subplot(3, 4, 10)
        plt.plot(xt.detach().numpy()[:, :, 0])
        plt.ylim((-2.1, 2.1))
        plt.ylabel('Arm Pos (m)')
        plt.xlabel('Time')
        plt.subplot(3, 4, 11)
        plt.plot(xt.detach().numpy()[:, 0, 0], xt.detach().numpy()[:, 1, 0], alpha=0.5, lw=0.5)
        plt.scatter(xinit.detach().numpy()[0, 0, 0], xinit.detach().numpy()[0, 1, 0], c='k')
        plt.scatter(xtarg.detach().numpy()[0, 0, 0], xtarg.detach().numpy()[0, 1, 0], c='r')
        plt.xlim((-1.1, 1.1))
        plt.ylim((-0.1, 2.1))
        plt.title('Trajectory')
        plt.ylabel('x_2 (m)')
        plt.xlabel('x_1 (m)')
        plt.subplot(3, 4, 12)
        h = ha.detach().numpy()[:, :, 0]
        plt.imshow(h[:, np.argsort(np.argmax(h, 0))].T, vmin=-1, vmax=1)
        plt.ylabel('Unit')
        plt.xlabel('Time')
        plt.tight_layout()
        plt.show()
    elif Nepochs-ei < 20:
        plt.subplot(3, 4, 10)
        plt.plot(xt.detach().numpy()[:, :, 0], alpha=0.5, lw=0.5)
        plt.subplot(3, 4, 11)
        plt.plot(xt.detach().numpy()[:, 0, 0], xt.detach().numpy()[:, 1, 0], alpha=0.5, lw=0.5)


N = 200  # RNN units
S = 7  # input, used here
R = 2  # readout
B = 1  # batches per epoch
Nepochs = 20000
doplot, auto, feed = True, False, True  # plot?, autonomous?, feedback?
Nepochsplot = 200  # plot output every certain number of epochs
dt = .1     # time bin
T = 30      # duration (200 dt)
NT = int(T / dt)
tau = .5    # time constant of units (5 dt)
lr = .001   # learning rate
sig = 0.01   # noise factor
t = dt * np.arange(NT)

# initial parameters
hinit = 0.01 * torch.randn(N, 1)
ws0 = np.random.standard_normal([N, S]).astype(np.float32) / np.sqrt(S)
J0 = 1.4 * np.random.standard_normal([N, N]).astype(np.float32) / np.sqrt(N)
wr0 = 1e-2 * np.random.standard_normal([R, N]).astype(np.float32) / np.sqrt(N) # normal scaling slows learning (???)
b0 = np.zeros([N, 1]).astype(np.float32)
ws = torch.tensor(ws0, requires_grad=True)
J = torch.tensor(J0, requires_grad=True)
wr = torch.tensor(wr0, requires_grad=True)
b = torch.tensor(b0, requires_grad=True)

# optimizer
lambda1 = lambda epoch: 1/(1 + 3*np.log10(epoch+1))     # log decay
opt = torch.optim.Adam([J, wr, b, ws], lr=lr)
scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda1)

if feed:
    losslist1, losslist2, losslist3, losslist = [], [], [], []
    prevt = time.time()
    for ei in range(Nepochs):
        print(ei, "\r", end='')
        # initial position
        xinit = torch.tensor(np.concatenate((np.zeros([NT, 1, B]), np.ones([NT, 1, B])), axis=1), requires_grad=False)
        # target position
        xtarg = torch.tensor(np.concatenate((np.zeros([NT, 1, B]), 2*np.ones([NT, 1, B])), axis=1), requires_grad=False)
        # current position
        xt = torch.zeros([NT, 2, B], requires_grad=False)
        xt[0, :, :] = xinit[0, :, :]
        # context cue
        c = torch.tensor(np.concatenate((np.zeros([NT, 1, B]), np.ones([NT, 1, B])), axis=1), requires_grad=False)
        # go cue
        g = torch.tensor(np.concatenate((np.zeros([int(NT / 4), 1, B]), np.ones([int(3 * NT / 4), 1, B])), axis=0),
                         requires_grad=False)
        s = torch.cat([xtarg[0, :, :], xt[0, :, :], c[0, :, :], g[0, :, :]], dim=0).float()
        h = hinit + sig * torch.randn(N, B)                         # random initialization of hidden state
        sa = torch.zeros(NT, S, B)                                  # save the inputs for each time bin for plotting
        ha = torch.zeros(NT, N, B)                                  # save the hidden states for each time bin for plotting
        u = torch.zeros(NT, R, B, requires_grad=False)              # angular acceleration of joints
        v = torch.zeros(NT, R, B, requires_grad=False)              # angular velocity of joints
        w = torch.zeros(NT, R, B, requires_grad=False)              # angular position of joints
        w[0, 0, :], w[0, 1, :] = math.pi/6, 2*math.pi/3

        for ti in range(1, NT):
            sa[ti, :, :] = s
            h = h + dt/tau * (-h + torch.tanh(ws.mm(s) + J.mm(h) + b))
            ha[ti, :, :] = h
            u[ti, :, :] = wr.mm(h) + 0.2 * wr.mm(h) * torch.randn(1, R, B)
            v[ti, :, :] = v[ti - 1, :, :] + u[ti - 1, :, :] * dt
            w[ti, :, :] = w[ti - 1, :, :] + v[ti - 1, :, :] * dt + .5 * u[ti - 1, :, :] * dt * dt
            # the following two lines make it extremely slow (likely due to clone)
            xt[ti, 0, :] = torch.cos(w[ti, 0, :].clone()) + torch.cos(w[ti, 0, :].clone() + w[ti, 1, :].clone())
            xt[ti, 1, :] = torch.sin(w[ti, 0, :].clone()) + torch.sin(w[ti, 0, :].clone() + w[ti, 1, :].clone())

            s = torch.cat([xtarg[ti, :, :], xt[ti, :, :], c[ti, :, :], g[ti, :, :]], dim=0).float()

        loss1 = torch.sum(torch.pow(torch.cat([xt[:int(NT / 4), :, :], xt[int(1 * NT / 4):, :, :]]) -
                                    torch.cat([xinit[:int(NT / 4), :, :], xtarg[int(1 * NT / 4):, :, :]]), 2)) / (T * B)
        loss2 = 1e3 * torch.sum(torch.pow(u, 2)) / (T * B)
        loss3 = 1e3 * torch.sum(torch.pow(torch.diff(u, dim=0) / dt, 2)) / (T * B)
        loss = loss1 + loss2 + loss3

        losslist1.append(loss1.item()), losslist2.append(loss2.item()), losslist3.append(loss3.item()), losslist.append(loss.item())
        print('\r' + str(ei + 1) + '/' + str(Nepochs) + '\t Err:' + str(loss.item()), end='')

        # do BPTT
        loss.backward()
        opt.step()
        opt.zero_grad()
        scheduler.step()

        if doplot:
            plot(g, xtarg, c, u, xt, ha, losslist)

elif auto:
    losslist1, losslist2, losslist3, losslist = [], [], [], []
    prevt = time.time()
    for ei in range(Nepochs):
        print(ei, "\r", end='')
        # initial position
        xinit = torch.tensor(np.concatenate((np.zeros([NT, 1, B]), np.ones([NT, 1, B])), axis=1), requires_grad=False)
        # target position
        xtarg = torch.tensor(np.concatenate((np.zeros([NT, 1, B]), 2*np.ones([NT, 1, B])), axis=1), requires_grad=False)
        # current position
        xt = torch.zeros([NT, 2, B], requires_grad=False)
        xt[0, :, :] = xinit[0, :, :]
        # context cue
        c = torch.tensor(np.concatenate((np.ones([NT, 1, B]), np.zeros([NT, 1, B])), axis=1), requires_grad=False)
        # go cue
        g = torch.tensor(np.concatenate((np.zeros([int(NT / 4), 1, B]), np.ones([int(3 * NT / 4), 1, B])), axis=0),
                         requires_grad=False)
        s = torch.cat([xtarg[0, :, :], xt[0, :, :], c[0, :, :], g[0, :, :]], dim=0).float()
        h = hinit + sig * torch.randn(N, B)                         # random initialization of hidden state
        sa = torch.zeros(NT, S, B)                                  # save the inputs for each time bin for plotting
        ha = torch.zeros(NT, N, B)                                  # save the hidden states for each time bin for plotting
        u = torch.zeros(NT, R, B, requires_grad=False)              # angular acceleration of joints
        v = torch.zeros(NT, R, B, requires_grad=False)              # angular velocity of joints
        w = torch.zeros(NT, R, B, requires_grad=False)              # angular position of joints
        w[0, 0, :], w[0, 1, :] = math.pi/6, 2*math.pi/3

        for ti in range(1, NT):
            sa[ti, :, :] = s
            h = h + dt/tau * (-h + torch.tanh(ws.mm(s) + J.mm(h) + b))
            ha[ti, :, :] = h
            u[ti, :, :] = wr.mm(h) + 0.2 * wr.mm(h) * torch.randn(1, R, B)
            v[ti, :, :] = v[ti - 1, :, :] + u[ti - 1, :, :] * dt
            w[ti, :, :] = w[ti - 1, :, :] + v[ti - 1, :, :] * dt + .5 * u[ti - 1, :, :] * dt * dt
            # the following two lines make it extremely slow (likely due to clone)
            xt[ti, 0, :] = torch.cos(w[ti, 0, :].clone()) + torch.cos(w[ti, 0, :].clone() + w[ti, 1, :].clone())
            xt[ti, 1, :] = torch.sin(w[ti, 0, :].clone()) + torch.sin(w[ti, 0, :].clone() + w[ti, 1, :].clone())

        loss1 = torch.sum(torch.pow(torch.cat([xt[:int(NT / 4), :, :], xt[int(1 * NT / 4):, :, :]]) -
                                    torch.cat([xinit[:int(NT / 4), :, :], xtarg[int(1 * NT / 4):, :, :]]), 2)) / (T * B)
        loss2 = 1e3 * torch.sum(torch.pow(u, 2)) / (T * B)
        loss3 = 1e3 * torch.sum(torch.pow(torch.diff(u, dim=0) / dt, 2)) / (T * B)
        loss = loss1 + loss2 + loss3

        losslist1.append(loss1.item()), losslist2.append(loss2.item()), losslist3.append(loss3.item()), losslist.append(loss.item())
        print('\r' + str(ei + 1) + '/' + str(Nepochs) + '\t Err:' + str(loss.item()), end='')

        # do BPTT
        loss.backward()
        opt.step()
        opt.zero_grad()
        scheduler.step()

        if doplot:
            plot(g, xtarg, c, u, xt, ha, losslist)

print("train time: ", time.time() - prevt)