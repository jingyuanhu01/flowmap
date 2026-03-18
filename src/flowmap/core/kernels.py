import numpy as np


class BaseKernel:
    """
    Scalar kernel interface for vectorized spline evaluation.

    The spline handles all tensor contractions.
    The kernel only provides scalar radial factors.
    """

    def phi(self, r: np.ndarray) -> np.ndarray:
        """Kernel value φ(r)."""
        raise NotImplementedError

    def jacobian_factor(self, s: np.ndarray, eps: float) -> np.ndarray:
        """
        Factor for gradient:

            ∇φ = (x - c) * factor(s)

        Parameters
        ----------
        s : (M, m)
            Squared distances

        Returns
        -------
        factor : (M, m)
        """
        raise NotImplementedError

    def hessian_factors(self, s: np.ndarray, eps: float):
        """
        Factors for Hessian:

            Hφ = factor_dd * (x-c)(x-c)^T + factor_I * I

        Returns
        -------
        factor_dd : (M, m)
        factor_I  : (M, m)
        """
        raise NotImplementedError


class PolyharmonicKernel(BaseKernel):
    """
    φ(r) = r^4 log r
    """

    def phi(self, r):
        out = np.zeros_like(r)
        mask = r > 0
        out[mask] = r[mask]**4 * np.log(r[mask])
        return out

    def jacobian_factor(self, s, eps):
        factor = np.zeros_like(s)

        mask = s > eps
        if np.any(mask):
            factor[mask] = s[mask] * (2*np.log(s[mask]) + 1)

        return factor

    def hessian_factors(self, s, eps):
        factor_dd = np.zeros_like(s)
        factor_I = np.zeros_like(s)

        mask = s > eps
        if np.any(mask):
            log_s = np.log(s[mask])

            factor_dd[mask] = 2*(2*log_s + 3)
            factor_I[mask]  = s[mask]*(2*log_s + 1)

        return factor_dd, factor_I


class ThinPlateKernel(BaseKernel):
    """
    Dimension-aware TPS kernel.
    """

    def __init__(self, d):
        self.d = d

    def phi(self, r):
        out = np.zeros_like(r)
        mask = r > 0

        if self.d == 2:
            out[mask] = r[mask]**2 * np.log(r[mask])
        elif self.d == 3:
            out[mask] = r[mask]
        else:
            out[mask] = r[mask]**(2 - self.d)

        return out

    def jacobian_factor(self, s, eps):
        factor = np.zeros_like(s)
        mask = s > eps

        if self.d == 2:
            if np.any(mask):
                r = np.sqrt(s[mask])
                factor[mask] = (2*np.log(r) + 1)
        else:
            if np.any(mask):
                factor[mask] = 1.0 / (np.sqrt(s[mask]) + eps)

        return factor

    def hessian_factors(self, s, eps):
        factor_dd = np.zeros_like(s)
        factor_I = np.zeros_like(s)

        mask = s > eps

        if self.d == 2:
            if np.any(mask):
                r = np.sqrt(s[mask])
                log_r = np.log(r)

                factor_dd[mask] = 2.0 / s[mask]
                factor_I[mask]  = (2*log_r + 1)
        else:
            if np.any(mask):
                factor_dd[mask] = 0.0
                factor_I[mask]  = 1.0 / (np.sqrt(s[mask]) + eps)

        return factor_dd, factor_I

