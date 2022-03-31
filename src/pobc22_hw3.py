#!/usr/bin/env python3raise

import brian2
from brian2 import NeuronGroup, SpikeMonitor, StateMonitor
from brian2 import mV, pA, ms, second, Hz, Gohm
import brian2.numpy_ as np
import matplotlib.pyplot as plt
import os.path
from os.path import join
import pickle as pkl


# generate data for fit

datafile = 'data_lif.pkl'


if not os.path.exists(datafile):

    # settings

    dt = .1 * ms
    brian2.defaultclock.dt = dt


    # lif neuron parameters

    u_rest = -65 * mV
    u_reset = -80 * mV
    u_th = -50 * mV
    tau_m = 10 * ms
    R_m = 0.03 * Gohm
    Delta_abs = 0 * ms

    I_mean = 500 * pA
    I_sigma = 250 * pA
    tau_I = 5 * ms

    sigma_u = 1 * mV

    t_sim = 60 * second


    # setup circuit

    net = brian2.Network()

    # note: for efficiency, we are simulating both the base neuron
    # and the passive model neuron as a single model here

    lif_eqs = '''
    dI/dt = -(I - I_mean) / tau_I + I_sigma * sqrt(2/tau_I) * xi_1 : ampere
    du/dt = ( -(u - u_rest) + R_m * I) / tau_m + sigma_u * sqrt(2/tau_m) * xi_2 : volt (unless refractory)
    du_pas/dt = ( -(u_pas - u_rest) + R_m * I) / tau_m : volt (unless refractory)
    '''

    thres = 'u >= u_th'  # LIF neuron crosses threshold
    reset = 'u = u_reset\nu_pas = u_reset'  # reset LIF, force passive neuron to reset

    neuron = NeuronGroup(1, lif_eqs, threshold=thres, reset=reset, refractory=Delta_abs, method='euler')
    neuron.u = u_rest
    neuron.u_pas = u_rest

    state_mon = StateMonitor(neuron, ['I', 'u', 'u_pas'], record=True)
    spike_mon = SpikeMonitor(neuron)

    net.add(neuron, state_mon, spike_mon)


    # simulate

    net.run(t_sim)


    # unpack data

    spikes = list(spike_mon.t)
    rate = len(spikes) / t_sim * second

    print('firing frequency: {0:.1f} Hz'.format(rate/Hz))

    t_ = state_mon.t[:]
    I_ = state_mon.I[0]
    u_ = state_mon.u[0]
    u0_ = state_mon.u_pas[0]

    # plot first 200 ms to visualize behavior

    m = (t_ < 200 * ms)
    ind = np.random.choice(len(u_), size=500, replace=False)

    plt.figure(figsize=(6, 3))

    plt.plot(t_[m]/ms,  u_[m]/mV, lw=2, label='with noise')
    plt.plot(t_[m]/ms,  u0_[m]/mV, lw=2, label='without noise')
    plt.autoscale(axis='x', tight=True)
    plt.locator_params(axis='x', nbins=3)
    plt.yticks([u_reset/mV, u_rest/mV, u_th/mV], [r'$u_\mathrm{reset}$', r'$u_\mathrm{rest}$', r'$\vartheta$'])
    plt.xlabel(r'$t$ / ms')
    plt.ylabel(r'$u(t)$ / mV')
    plt.grid(axis='y')
    plt.legend(loc='best')

    plt.tight_layout()

    data = [dt, t_, I_, u_, u0_, spikes, u_rest, u_reset, u_th, tau_m, R_m, Delta_abs, sigma_u]

    with open(datafile, 'wb') as f:
        pkl.dump(data, f)

else:
    with open(datafile, 'rb') as f:
        data = pkl.load(f)

    dt, t_, I_, u_, u0_, spikes, u_rest, u_reset, u_th, tau_m, R_m, Delta_abs, sigma_u = data

# ----------------------------------------------------------------------
# fit exponential

#
# define bins
#
# note that functions like np.histogram use bin edges, but for
# plotting we want to plot the values over the centers of the bins

bin_centers = ...

#
# calculate histograms using these bins
#

u0_hist = ...
u0_at_spike_hist = ...

#
# calculate p(spike|u) and rho (for each bin)
#

p_spike = ...
rho = ...

#
# fit an exponential of the form
#
#   rho(V) = 1/dt * exp(c1*V + c0)
#

c1, c0 = ...
rho_fit = ...


# plots

plt.figure(figsize=(8, 6))

plt.subplot(2, 2, 1)
plt.plot(bin_centers, np.log10(u0_hist), lw=2, label='all')
plt.plot(bin_centers, np.log10(u0_at_spike_hist), lw=2, label='at spike')
plt.autoscale(axis='x', tight=True)
plt.xlabel(r'$u_0$ / mV')
plt.ylabel(r'$\log_{10}$ count')
plt.legend(loc='best')

plt.subplot(2, 2, 2)
plt.scatter(bin_centers, p_spike)
plt.autoscale(axis='x', tight=True)
plt.xlabel(r'$u_0$ / mV')
plt.ylabel(r'$p(\mathrm{spike} | u)$')

plt.subplot(2, 2, 3)
plt.scatter(bin_centers, rho)
plt.autoscale(axis='x', tight=True)
plt.autoscale(False)
plt.plot(bin_centers, rho_fit, c='k', label='fit')
plt.autoscale(axis='x', tight=True)
plt.xlabel(r'$u_0$ / mV')
plt.ylabel(r'$\rho(u)$ / ms$^{-1}$')
plt.legend(loc='best')

plt.subplot(2, 2, 4)
plt.scatter(...)  # TODO, scatter plot of data in log space where we perform the linear fit
plt.plot(...)  # plot linear fit
plt.autoscale(axis='x', tight=True)
plt.xlabel(r'$u_0$ / mV')
plt.ylabel(r'$\log (-\log(1 - p(\mathrm{spike} | u)$')
plt.legend(loc='best')

plt.tight_layout()

# ----------------------------------------------------------------------
# evaluate fit

...

t_rho = ...
lif_rho = ...
srm_rho = ...

# plot, e.g. like this:

plt.figure(figsize=(6, 3.5))

plt.fill_between(t_rho/ms, np.zeros_like(lif_rho*ms), lif_rho*ms, label=r'lif')
plt.autoscale(axis='x', tight=True)
plt.autoscale(enable=False)
plt.plot(t_rho/ms, srm_rho*ms, c='k', lw=1, ls='--', label=r'srm')
plt.locator_params(axis='x', nbins=5)
plt.locator_params(axis='y', nbins=2)
plt.xlabel(r'$t$ / ms')
plt.ylabel(r'$\rho$ / ms$^{-1}$')
plt.legend(loc='best')

plt.tight_layout()


plt.show() # avoid having multiple plt.show()s in your code
