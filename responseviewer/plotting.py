import os
import matplotlib.pyplot as plt


class Plotting():

    states_avail = ['X [m]', 'Y [m]', 'Z [m]', 'Phi [rad]', 'Theta [rad]', 'Psi [rad]',
                    'u [m/s]', 'v [m/s]', 'w [m/s]', 'p [rad]', 'q [rad]', 'r [rad]']

    def __init__(self, fig):
        plt.rcParams.update({'font.size': 16,
                             'svg.fonttype': 'none'})
        self.fig = fig
        im = plt.imread(os.path.dirname(__file__) + '/graphics/LK_logo2.png')
        newax = fig.add_axes([0.04, 0.02, 0.10, 0.08])
        newax.imshow(im, interpolation='hanning')
        newax.axis('off')

        self.responses = None

    def plot_nothing(self):
        self.fig.clf()

    def add_responses(self, responses):
        self.responses = responses

    def timehistories(self, subcases, states,):
        self.fig.clf()
        # Create subplots for each state to plot, sharing the x-axis (time).
        ax = self.fig.subplots(len(states), sharex=True)
        # Make sure that ax is always iterable, even if there is only one state to plot.
        if len(states) == 1:
            ax = [ax]
        # Plotting the time histories for each state and subcase.
        for a, state in zip(ax, states):
            for subcase in subcases:
                subcase = str(subcase)
                a.plot(self.responses[subcase]['t'][()], self.responses[subcase]['X'][:, state], label=subcase)
                # a.ticklabel_format(style='sci', axis='x', scilimits=(-2, 2))
                a.ticklabel_format(style='sci', axis='y', scilimits=(-1, 1))
                a.grid(True)
                a.set_ylabel(self.states_avail[state])
                ya = a.get_yaxis()
                ya.set_label_coords(x=-0.1, y=0.5)
        # Make plots look nice.
        ax[0].legend(loc='upper right')
        ax[-1].set_xlabel('Time [s]')
