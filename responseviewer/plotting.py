import os
import matplotlib.pyplot as plt
import numpy as np


class Plotting():

    states_avail = ['x [m]', 'y [m]', 'z [m]', 'Phi [deg]', 'Theta [deg]', 'Psi [deg]',
                    'u [m/s]', 'v [m/s]', 'w [m/s]', 'p [deg]', 'q [deg]', 'r [deg]']
    commands_avail = ['Xi [deg]', 'Eta [deg]', 'Zeta [deg]', 'Thrust [N]', 'Stabilizer [deg]', 'Flaps [deg]']
    loadfactors_avail = ['Nx [-]', 'Ny [-]', 'Nz [-]']
    other_avail = ['q_dyn [Pa]', 'alpha [deg]', 'beta [deg]', 'p1 [m]', 'F1 [N]']
    all_quantities = states_avail + commands_avail + loadfactors_avail + other_avail

    def __init__(self, fig):
        plt.rcParams.update({'font.size': 16,
                             'svg.fonttype': 'none'})
        self.fig = fig
        im = plt.imread(os.path.dirname(__file__) + '/../graphics/LK_logo2.png')
        newax = fig.add_axes([0.04, 0.02, 0.10, 0.08])
        newax.imshow(im, interpolation='hanning')
        newax.axis('off')

        self.responses = None

    def plot_nothing(self):
        self.fig.clf()

    def add_responses(self, responses):
        self.responses = responses

    def timehistories(self, subcases, quantities,):
        self.fig.clf()
        # Create subplots for each state to plot, sharing the x-axis (time).
        ax = self.fig.subplots(len(quantities), sharex=True)
        # Make sure that ax is always iterable, even if there is only one state to plot.
        if len(quantities) == 1:
            ax = [ax]
        # Plotting the time histories for each state and subcase.
        for a, quantity in zip(ax, quantities):
            for subcase in subcases:
                time = self.responses[subcase]['t'][()]
                if quantity in self.states_avail:
                    # States are stored in 'X' in the same order as in states_avail.
                    idx = self.states_avail.index(quantity)
                    data = self.responses[subcase]['X'][:, idx]
                elif quantity in self.commands_avail:
                    # Commands are stored in the last 6 rows of 'X' in the same order as in commands_avail.
                    idx = self.commands_avail.index(quantity)
                    commands = self.responses[subcase]['X'][:, -6:]
                    data = commands[:, idx]
                elif quantity in self.loadfactors_avail:
                    # Load factors are stored in 'Nxyz' in the same order as in loadfactors_avail.
                    idx = self.loadfactors_avail.index(quantity)
                    data = self.responses[subcase]['Nxyz'][:, idx]
                else:
                    # For the remaining quantities, we need to check if they are available in the response.
                    # Most quantities have units in their name, so we split by space to obtain the base name for
                    # lookup in the response.
                    q = quantity.split()[0]
                    if quantity in self.other_avail and q in self.responses[subcase]:
                        data = self.responses[subcase][q][()]
                    else:
                        # In case the quantity is not found, create some dummy data.
                        subcase = 'Not found'
                        data = np.zeros_like(time)
                if '[deg]' in quantity:
                    data *= 180.0 / np.pi
                # Plot the time history for the current subcase and quantity.
                a.plot(time, data, label=subcase)
            # Format current axis.
            a.ticklabel_format(style='sci', axis='y', scilimits=(-2, 2))
            a.grid(True)
            a.set_ylabel(quantity)
            # Push left bound to the right to make space for the ylabel.
            left, bottom, width, height = a.get_position().bounds
            a.set_position([left + 0.05, bottom, width - 0.05, height])
            a.get_yaxis().set_label_coords(x=-0.18, y=0.5)
            # Show legend per plot
            a.legend(loc='upper right')
        ax[-1].set_xlabel('Time [s]')
