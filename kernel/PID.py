
class PID_ideal:
    def __init__(self, Kp=0.2, Ki=0.0, Kd=0.0, t=0.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.sample_time = 0.00
        self.current_time = t
        self.last_time = self.current_time

        self.clear()

    def clear(self):
        """Clears PID computations and coefficients"""
        self.SetPoint = 0.0
        self.PTerm = 0.0
        self.ITerm = 0.0
        self.DTerm = 0.0
        self.last_error = 0.0

        # Windup Guard
        self.int_error = 0.0
        self.windup_guard = 20.0

        self.output = 0.0
        
    def setSetPoint(self, SetPoint):
        self.SetPoint = SetPoint
        
    def update(self, t, feedback_value):
        """Calculates PID value for given reference feedback
        .. math::
            u(t) = K_p e(t) + K_i \int_{0}^{t} e(t)dt + K_d {de}/{dt}
        """
        error = self.SetPoint - feedback_value

        self.current_time = t
        delta_time = self.current_time - self.last_time
        delta_error = error - self.last_error

        if (delta_time >= self.sample_time):
            self.PTerm = self.Kp * error
            self.ITerm += error * delta_time

            if (self.ITerm < -self.windup_guard):
                self.ITerm = -self.windup_guard
            elif (self.ITerm > self.windup_guard):
                self.ITerm = self.windup_guard

            self.DTerm = 0.0
            if delta_time > 0:
                self.DTerm = delta_error / delta_time

            # Remember last time and last error for next calculation
            self.last_time = self.current_time
            self.last_error = error

            self.output = self.PTerm + (self.Ki * self.ITerm) + (self.Kd * self.DTerm)

class PID_standart:
    def __init__(self, Kp=0.2, Ti=0.0, Td=0.0, t=0.0):
        self.Kp = Kp
        self.Ti = Ti
        self.Td = Td
        self.sample_time = 0.00
        self.current_time = t
        self.last_time = self.current_time

        self.clear()

    def clear(self):
        """Clears PID computations and coefficients"""
        self.SetPoint = 0.0

        self.PTerm = 0.0
        self.ITerm = 0.0
        self.DTerm = 0.0
        self.last_error = 0.0

        # Windup Guard
        self.int_error = 0.0
        self.windup_guard = 20.0

        self.output = 0.0
    
    def setSetPoint(self, SetPoint):
        self.SetPoint = SetPoint
        
    def update(self, t, feedback_value):
        """Calculates PID value for given reference feedback
        .. math::
            u(t) = K_p * [ e(t) + 1/T_i \int_{0}^{t} e(t)dt + T_d {de}/{dt} ]
        """
        error = self.SetPoint - feedback_value

        self.current_time = t
        delta_time = self.current_time - self.last_time
        delta_error = error - self.last_error

        if (delta_time >= self.sample_time):
            self.PTerm = error
            self.ITerm += error * delta_time

            if (self.ITerm < -self.windup_guard):
                self.ITerm = -self.windup_guard
                #print 'windup'
            elif (self.ITerm > self.windup_guard):
                self.ITerm = self.windup_guard
                #print 'windup'

            self.DTerm = 0.0
            if delta_time > 0:
                self.DTerm = delta_error / delta_time

            # Remember last time and last error for next calculation
            self.last_time = self.current_time
            self.last_error = error

            self.output = self.Kp * (self.PTerm + (1.0/self.Ti * self.ITerm) + (self.Td * self.DTerm))
