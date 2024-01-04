import logging
import numpy as np

"""
Achtung - Verwirrungsgefahr!
Nomenklature der Integrationsverfahren: dy = f(t,y)
Nomenklature in den Modellgleichungen: Y = f(t,X)
"""


class RungeKutta4():

    # Klassiches Runge-Kutta Verfahren fuer ein Anfangswertprobelm 1. Ordnung
    # Implementiert wie beschieben in: https://de.wikipedia.org/wiki/Klassisches_Runge-Kutta-Verfahren
    def __init__(self, odefun):
        # Aufrufbare Funktion einer Differenziagleichung vom Typ dy = odefun(t, y)
        self.odefun = odefun
        self.success = True
        self.stepwidth = 1.0e-4
        self.t = 0.0
        self.y = None
        self.dy = []
        self.output_dict = None
        self.i = 1

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
        self.success = False  # uncheck success flag
        # Integration mit fester Schrittweite bis t_end erreicht ist
        while self.t < t_end:
            self.time_step()

        # check success flag
        if np.isnan(self.y).any():
            logging.warning(
                'Encountered NaN during integration at t={}.'.format(self.t))
        else:
            self.success = True

    def time_step(self):
        # Ausgangswerte bei y(t0) holen
        h = self.stepwidth
        t0 = self.t
        y0 = self.y

        # Berechnung der Koeffizienten
        k1 = self.odefun(t0, y0)
        k2 = self.odefun(t0 + h / 2.0, y0 + h / 2.0 * k1)
        k3 = self.odefun(t0 + h / 2.0, y0 + h / 2.0 * k2)
        k4 = self.odefun(t0 + h, y0 + h * k3)

        # Berechnung der Naeherungsloesung fuer y(t1)
        t1 = t0 + h
        y1 = y0 + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # Daten speichern fuer den naechsten Schritt
        self.t = t1
        self.y = y1.copy()


class ExplicitEuler(RungeKutta4):
    # Explizites Euler Verfahren fuer ein Anfangswertprobelm 1. Ordnung
    # https://de.wikipedia.org/wiki/Explizites_Euler-Verfahren

    def time_step(self):
        # Ausgangswerte bei y(t0) holen
        h = self.stepwidth
        t0 = self.t
        y0 = self.y

        # Berechnung der Koeffizienten
        k1 = self.odefun(t0, y0)

        # Berechnung der Naeherungsloesung fuer y(t1)
        t1 = t0 + h
        y1 = y0 + h * k1

        # Daten speichern fuer den naechsten Schritt
        self.t = t1
        self.y = y1.copy()


class AdamsBashforth(RungeKutta4):
    # Explizites Adams-Bashforth Verfahren fuer ein Anfangswertprobelm 1. Ordnung
    # https://en.wikipedia.org/wiki/Linear_multistep_method

    def time_step(self):
        # Ausgangswerte bei y(t0) holen
        h = self.stepwidth
        t0 = self.t
        y0 = self.y

        # Berechnung der Koeffizienten
        out = self.odefun(t0, y0)
        # Handhabung des Outputs, falls es sich um ein Dictionary handelt
        if isinstance(out, dict):
            self.output_dict = out
            self.dy.append(out['dy'])
        else:
            self.dy.append(out)

        # Berechnung der Naeherungsloesung fuer y(t1)
        # Je nach dem wie viele zurückliegende Schritte verfügbar sind, wird die Ordnung des Verfahrens erhöht.
        if self.i == 1:
            # Dies ist die Euler Methode
            t1 = t0 + h
            y1 = y0 + h * self.dy[-1]
        elif self.i == 2:
            t1 = t0 + h
            y1 = y0 + h * (3.0 / 2.0 * self.dy[-1] - 1.0 / 2.0 * self.dy[-2])
        elif self.i == 3:
            t1 = t0 + h
            y1 = y0 + h * (23.0 / 12.0 * self.dy[-1] - 16.0 / 12.0 * self.dy[-2] + 5.0 / 12.0 * self.dy[-3])
        elif self.i == 4:
            t1 = t0 + h
            y1 = y0 + h * (55.0 / 24.0 * self.dy[-1] - 59.0 / 24.0 * self.dy[-2] + 37.0 / 24.0 * self.dy[-3]
                           - 9.0 / 24.0 * self.dy[-4])
        elif self.i >= 5:
            t1 = t0 + h
            y1 = y0 + h * (1901.0 / 720.0 * self.dy[-1] - 2774.0 / 720.0 * self.dy[-2] + 2616.0 / 720.0 * self.dy[-3]
                           - 1274.0 / 720.0 * self.dy[-4] + 251.0 / 720.0 * self.dy[-5])

        # Daten speichern fuer den naechsten Schritt
        self.t = t1
        self.y = y1.copy()
        # Überflüssige zurückliegende Schritte löschen
        self.dy = self.dy[-4:]
        # Zeitschrittzähler erhöhen
        self.i += 1
