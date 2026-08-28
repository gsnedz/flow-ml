# This file adds flow by nudging.  It didn't really work for our training, so we didn't really use it.

from random import uniform
import numpy as np

def compute_v2_truth(phi, psi2=0.0):
    """Compute v2 = <cos(2(phi-Psi2))>."""
    phi = np.asarray(phi)
    return np.mean(np.cos(2 * (phi - psi2)))


def nudge_v2(phi, target_v2=0.5, tol=1e-12, max_iter=50):
    """
    Adjust phi by the smallest L2 change needed to reach a target v2.
    Parameters
    ----------
    phi : array_like
        Particle azimuths.
    target_v2 : float
        Desired v2.
    psi2 : float
        Event-plane angle.
    """
    phi = np.asarray(phi, dtype=float).copy()
    M = len(phi)
    psi2 = uniform.rnduin(0, 2 * np.pi)
    if M == 0:
        return phi
    for _ in range(max_iter):
        current_v2 = compute_v2_truth(phi, psi2 = psi2)
        dv = target_v2 - current_v2
        if abs(dv) < tol:
            break
        # dv2/dphi_i
        g = -(2.0 / M) * np.sin(2 * (phi - psi2))
        g2 = np.dot(g, g)
        if g2 < 1e-20:
            raise RuntimeError("Gradient vanished.")
        # Minimum-norm correction
        dphi = (dv / g2) * g
        phi += dphi
        # keep angles in [0,2π)
        phi %= 2*np.pi
    return 

def sample_phi_exact(n, v):
    """
    Draw n samples from

        f(phi) = (1 + 2*v*cos(2phi))/(2*pi)

    using inversion of the exact CDF with Newton iterations.
    """
    out = np.empty(n)

    for i in range(n):

        # uniform random number
        u = np.random.random()

        # initial guess: uniform distribution
        phi = 2*np.pi*u - np.pi

        # Newton iterations
        for _ in range(4):
            F = (phi + np.pi + v*np.sin(2*phi))/(2*np.pi) - u
            dF = (1 + 2*v*np.cos(2*phi))/(2*np.pi)
            phi -= F/dF

        out[i] = phi

    return out

def sample_flow_phis(phis):
    """Samples phis from a flow distribution."""
    target_v2 = .05
    new_phis = []
    for event_i in range(len(phis)):
        event = phis[event_i]
        new_phis.append(sample_phi_exact(len(event), target_v2))
    return new_phis
        
