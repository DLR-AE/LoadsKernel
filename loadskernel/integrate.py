import numpy as np
import logging

class RungeKutta4():
    # Klassiches Runge-Kutta Verfahren fuer ein Anfangswertprobelm 1. Ordnung
    # Implementiert wie beschieben in: https://de.wikipedia.org/wiki/Klassisches_Runge-Kutta-Verfahren 
    def __init__(self, odefun):
        self.odefun = odefun # Aufrufbare Funktion einer Differenziagleichung vom Typ dy = odefun(t, y)
        self.success = True

    def successful(self):
        # Check status of success flag
        return self.success

    def set_integrator(self, stepwidth):
        self.stepwidth = float(stepwidth)
        return self

    def set_initial_value(self, y0, t0):
        self.y = y0
        self.t = t0

    def integrate(self, t_end):
        self.success = False # uncheck success flag
        # Integration mit fester Schrittweite bis t_end erreicht ist
        while self.t < t_end :
            self.RungeKuttaStep()

        # check success flag
        if np.isnan(self.y).any():
            logging.warning('Encountered NaN during integration at t={}.'.format(self.t))
        else:
            self.success = True 

    def RungeKuttaStep(self):
        # Ausgangswerte bei y(t0) holen
        h = self.stepwidth
        t0 = self.t
        y0 = self.y

        # Berechnung der Koeffizienten
        k1 = self.odefun(t0, y0)
        k2 = self.odefun(t0 + h/2.0, y0 + h/2.0 * k1)
        k3 = self.odefun(t0 + h/2.0, y0 + h/2.0 * k2)
        k4 = self.odefun(t0 + h, y0 + h*k3)

        # Berechnung der Naeherungsloesung fuer y(t1)
        t1 = t0 + h
        y1 = y0 + h/6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4)

        # Daten speichern fuer den naechsten Schritt
        self.t = t1
        self.y = y1.copy()

class ExplicitEuler(RungeKutta4):
    # Explizites Euler Verfahren fuer ein Anfangswertprobelm 1. Ordnung
    # https://de.wikipedia.org/wiki/Explizites_Euler-Verfahren
    def integrate(self, t_end):
        self.success = False # uncheck success flag
        # Integration mit fester Schrittweite bis t_end erreicht ist
        while self.t < t_end :
            self.EulerStep()

        # check success flag
        if np.isnan(self.y).any():
            logging.warning('Encountered NaN during integration at t={}.'.format(self.t))
        else:
            self.success = True 

    def EulerStep(self):
        # Ausgangswerte bei y(t0) holen
        h = self.stepwidth
        t0 = self.t
        y0 = self.y

        # Berechnung der Koeffizienten
        k1 = self.odefun(t0, y0)
        
        # Berechnung der Naeherungsloesung fuer y(t1)
        t1 = t0 + h
        y1 = y0 + h*k1

        # Daten speichern fuer den naechsten Schritt
        self.t = t1
        self.y = y1.copy()
    