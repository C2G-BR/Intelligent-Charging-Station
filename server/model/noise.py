from __future__ import annotations

import numpy as np

from typing import Union

class OrnsteinUhlenbeckActionNoise():
    """Ornstein Uhlenbeck Action Noise
    """

    def __init__(self, mean:np.array[float], sigma:float=0.2, theta:float=0.15,
        dt:float=0.01, initial_noise:Union[None, float]=None) -> None:
        """Constructor

        Args:
            mean (np.array[float]):  Mean value of the noise.
            sigma (float, optional): Scale of the noise in form of standard
                deviation. Defaults to 0.2.
            theta (float, optional): Rate of mean reversion. Defaults to 0.15.
            dt (float, optional): Timestep for the noise. Defaults to 0.01.
            initial_noise (Union[None, float], optional): Initial value for the
                noise. Defaults to None.
        """
        self._theta = theta
        self._mu = mean
        self._sigma = sigma
        self._dt = dt
        self.initial_noise = initial_noise
        self.noise_prev = None
        self.reset()

    def reset(self) -> None:
        """Reset noise to initial values
        """
        self.noise_prev = self.initial_noise if self.initial_noise is not None \
            else np.zeros_like(self._mu)

    def __call__(self) -> float:
        """Get Noise

        Returns:
            float: Current noise.
        """
        noise = self.noise_prev + self._theta * (self._mu - self.noise_prev) * \
            self._dt + self._sigma * np.sqrt(self._dt) * \
            np.random.normal(size=self._mu.shape)
        self.noise_prev = noise
        return noise