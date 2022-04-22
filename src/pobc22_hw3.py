#!/usr/bin/env python3raise

import brian2
from brian2 import NeuronGroup, SpikeMonitor, StateMonitor, exp
from brian2 import mV, pA, ms, second, Hz, Gohm, volt
import brian2.numpy_ as np
import matplotlib.pyplot as plt
import os.path
from os.path import join
import pickle as pkl


# generate data for fit

datafile_3a = 'data_lif_3a.pkl'


def __task3a_prepare_data(datafile):
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

        print('firing frequency: {0:.1f} Hz'.format(rate / Hz))

        t_ = state_mon.t[:]
        I_ = state_mon.I[0]
        u_ = state_mon.u[0]
        u0_ = state_mon.u_pas[0]

        # plot first 200 ms to visualize behavior

        m = (t_ < 200 * ms)
        ind = np.random.choice(len(u_), size=500, replace=False)

        plt.figure(figsize=(6, 3))

        plt.plot(t_[m] / ms, u_[m] / mV, lw=2, label='with noise')
        plt.plot(t_[m] / ms, u0_[m] / mV, lw=2, label='without noise')
        plt.autoscale(axis='x', tight=True)
        plt.locator_params(axis='x', nbins=3)
        plt.yticks([u_reset / mV, u_rest / mV, u_th / mV], [r'$u_\mathrm{reset}$', r'$u_\mathrm{rest}$', r'$\vartheta$'])
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

    return dt, spikes, t_, u0_

def __task3a_stochastic_srm_model(dt, spikes, t_, u0_):
    bins = np.arange(u0_.min(), u0_.max(), 1 * mV)
    [u0_hist, edges] = np.histogram(u0_, bins)
    bin_centers = (edges + (edges[1] - edges[0]) / 2)[:-1] * mV * 1000  # one bin less than edges

    u0_spikes = u0_[np.isin(t_, spikes)]  # u0_t of spike train S(t)
    [u0_at_spike_hist, _] = np.histogram(u0_spikes, bins)

    p_spike = u0_at_spike_hist / u0_hist  # p(spike|u0)
    rho = - np.log(1 - p_spike) / dt  # rho_fit(u0): reformulation of rho for stochastic Poisson process
    b = np.log(dt * rho)
    inf_mask = b != -np.inf
    b = b[inf_mask]
    A = np.array([bin_centers / mV, np.ones(bin_centers.shape[0])]).T
    A = A[inf_mask]
    x_ = np.linalg.lstsq(A, b, rcond=None)

    c1 = x_[0][0]
    c0 = x_[0][1]
    rho_fit = 1 / dt * np.exp(c1 * bin_centers / mV + c0)

    return u0_hist, bin_centers, u0_at_spike_hist, p_spike, rho, b, inf_mask, c1, c0, rho_fit

def __task3a_figures(u0_hist, bin_centers, u0_at_spike_hist, p_spike, rho, b, inf_mask, c1, c0, rho_fit):
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
    plt.scatter(bin_centers, rho / 1000)
    plt.autoscale(axis='x', tight=True)
    plt.autoscale(False)
    plt.plot(bin_centers, rho_fit / 1000, c='k', label='fit')
    plt.autoscale(axis='x', tight=True)
    plt.xlabel(r'$u_0$ / mV')
    plt.ylabel(r'$\rho(u)$ / ms$^{-1}$')
    # plt.legend(loc='best')

    plt.subplot(2, 2, 4)
    plt.scatter(bin_centers[inf_mask], b)
    plt.plot(bin_centers[inf_mask], bin_centers[inf_mask] * c1 / mV + c0)  # plot linear fit
    plt.autoscale(axis='x', tight=True)
    plt.xlabel(r'$u_0$ / mV')
    plt.ylabel(r'$\log (-\log(1 - p(\mathrm{spike} | u)$')
    plt.legend(loc='best')

    plt.tight_layout()

dt, spikes, t_, u0_ = __task3a_prepare_data(datafile_3a)
u0_hist, bin_centers, u0_at_spike_hist, p_spike, rho, b, inf_mask, c1, c0, rho_fit = __task3a_stochastic_srm_model(dt, spikes, t_, u0_)
# __task3a_figures(u0_hist, bin_centers, u0_at_spike_hist, p_spike, rho, b, inf_mask, c1, c0, rho_fit)

# -------------------------------------------------------------------

# task 3b

datafile_3b_base = 'data_lif_3b_base.pkl'
datafile_3b_stochastic = 'data_lif_3b_stochastic.pkl'

def __task3b_prepare_base_neurons(datafile):
    num_neurons = 1000

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

        t_sim = 200 * ms

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

        neuron = NeuronGroup(num_neurons, lif_eqs, threshold=thres, reset=reset, refractory=Delta_abs, method='euler')
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

        print('firing frequency: {0:.1f} Hz'.format(rate / Hz))

        t_ = state_mon.t[:]
        I_ = state_mon.I[0]
        u_ = state_mon.u[0]
        u0_ = state_mon.u_pas[0]

        # plot first 200 ms to visualize behavior

        m = (t_ < 200 * ms)
        ind = np.random.choice(len(u_), size=500, replace=False)

        plt.figure(figsize=(6, 3))

        plt.plot(t_[m] / ms, u_[m] / mV, lw=2, label='with noise')
        plt.plot(t_[m] / ms, u0_[m] / mV, lw=2, label='without noise')
        plt.autoscale(axis='x', tight=True)
        plt.locator_params(axis='x', nbins=3)
        plt.yticks([u_reset / mV, u_rest / mV, u_th / mV], [r'$u_\mathrm{reset}$', r'$u_\mathrm{rest}$', r'$\vartheta$'])
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

    return dt, spikes, t_, u0_, num_neurons

def __task3b_prepare_stochastic_neurons(datafile, c0, c1):
    num_neurons = 1000

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

        t_sim = 200 * ms

        # setup circuit

        net = brian2.Network()

        # note: for efficiency, we are simulating both the base neuron
        # and the passive model neuron as a single model here

        lif_eqs = '''
    dI/dt = -(I - I_mean) / tau_I : ampere
    du/dt = ( -(u - u_rest) + R_m * I) / tau_m : volt (unless refractory)
    du_pas/dt = ( -(u_pas - u_rest) + R_m * I) / tau_m : volt (unless refractory)
    '''

        thres = 'rand() >= 1 - exp(-exp((c1 * u) / mV + c0))'  # LIF neuron crosses threshold
        reset = 'u = u_reset\nu_pas = u_reset'  # reset LIF, force passive neuron to reset

        neuron = NeuronGroup(num_neurons, lif_eqs, threshold=thres, reset=reset, refractory=Delta_abs, method='euler')
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

        print('firing frequency: {0:.1f} Hz'.format(rate / Hz))

        t_ = state_mon.t[:]
        I_ = state_mon.I[0]
        u_ = state_mon.u[0]
        u0_ = state_mon.u_pas[0]

        # plot first 200 ms to visualize behavior

        m = (t_ < 200 * ms)
        ind = np.random.choice(len(u_), size=500, replace=False)

        plt.figure(figsize=(6, 3))

        plt.plot(t_[m] / ms, u_[m] / mV, lw=2, label='with noise')
        plt.plot(t_[m] / ms, u0_[m] / mV, lw=2, label='without noise')
        plt.autoscale(axis='x', tight=True)
        plt.locator_params(axis='x', nbins=3)
        plt.yticks([u_reset / mV, u_rest / mV, u_th / mV], [r'$u_\mathrm{reset}$', r'$u_\mathrm{rest}$', r'$\vartheta$'])
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

    return dt, spikes, t_, u0_, num_neurons

def __task3b_stochastic_srm_model(dt, spikes, t_, u0_, num_neurons):

    # 1. Estimate rho_base
    spike_indices = np.digitize(spikes, np.arange(0 * ms, 200 * ms, dt))
    rho_base = np.zeros(spike_indices.shape[0])

    for i in np.arange(1, 2000):
        rho_base[i] = np.count_nonzero(spike_indices == i) / num_neurons

    # 2. Set up same number of stochastic neurons

    pass


dt, spikes, t_, u0_, num_neurons = __task3b_prepare_base_neurons(datafile_3b_base)
dt, spikes, t_, u0_, num_neurons = __task3b_prepare_stochastic_neurons(datafile_3b_stochastic, c0, c1)
u0_hist, bin_centers, u0_at_spike_hist, p_spike, rho, b, inf_mask, c1, c0, rho_fit = __task3b_stochastic_srm_model(dt, spikes, t_, u0_, num_neurons)

t_rho = ...
lif_rho = ...
srm_rho = ...

# plot, e.g. like this:

# plt.figure(figsize=(6, 3.5))

# plt.fill_between(t_rho/ms, np.zeros_like(lif_rho*ms), lif_rho*ms, label=r'lif')
# plt.autoscale(axis='x', tight=True)
# plt.autoscale(enable=False)
# plt.plot(t_rho/ms, srm_rho*ms, c='k', lw=1, ls='--', label=r'srm')
# plt.locator_params(axis='x', nbins=5)
# plt.locator_params(axis='y', nbins=2)
# plt.xlabel(r'$t$ / ms')
# plt.ylabel(r'$\rho$ / ms$^{-1}$')
# plt.legend(loc='best')

plt.tight_layout()


plt.show() # avoid having multiple plt.show()s in your code
