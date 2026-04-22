import numpy as np

class Sim:
    def __init__(self, n=120):
        self.n = n
        self.E = np.zeros((n, n))
        self.phi = 0

    def step(self, t):
        cx = self.n//2
        cy = self.n//3

        mw = np.sin(2*np.pi*2*t)
        ir = np.sin(2*np.pi*5*t + self.phi)

        self.E[cx, cy] += mw + 0.3*ir

        # simple diffusion (안 터지는 모델)
        self.E = 0.99*self.E + 0.01*np.roll(self.E, 1, axis=0)

        # feedback
        signal = np.mean(np.abs(self.E[50:70, 80:100]))
        self.phi += 0.05*(0.5 - signal)

        return self.E, signal
