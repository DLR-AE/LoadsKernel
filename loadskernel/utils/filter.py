from scipy import signal


class Butter:
    # Application of Butterworth filter to a time signal
    # fs     =  sample rate, Hz
    # cutoff =  desired cutoff frequency of the filter, Hz
    # Usage:
    # butter = filter.Butter(fs=40.0, cutoff=2.0)
    # data_filtered = butter.butter_lowpass_filter(data)

    def __init__(self, cutoff, fs, order=5):
        self.order = order
        self.fs = fs  # sample rate, Hz
        self.cutoff = cutoff  # desired cutoff frequency of the filter, Hz

    def butter_lowpass(self):
        nyq = 0.5 * self.fs
        normal_cutoff = self.cutoff / nyq
        b, a = signal.butter(self.order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(self, data):
        b, a = self.butter_lowpass()
        y = signal.filtfilt(b, a, data)
        return y
