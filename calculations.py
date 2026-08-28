# some useful math

import numpy as np

# -------------------------
# Correlation functions
# -------------------------

def cut(data, endpoints):
    """Returns only data within a specific range.
    
    data: (N) array
    endpoints: (2) array of minimum and maximum values desired."""
    array_data = np.array(data)
    return (endpoints[0] <= array_data) & (array_data <= endpoints[1])

def Qmoment(a, n=2):   
    """Calculates Q-vector for a given array of particle phis."""
    return np.sum(np.exp(1j * n * a))  

def correlation_4(phi, pt, n, momentum_cut=0):  
    """Calculates <4> for a given event."""
    phi = phi[pt > momentum_cut]  
    M = len(phi)

    if M <= 3:
        return 1234
    else:
        Qn = Qmoment(phi, n) 
        Q2n = Qmoment(phi, 2*n)   

        denom = M*(M-1)*(M-2)*(M-3)

        first = (
            np.abs(Qn)**4
            + np.abs(Q2n)**2
            - 2*np.real(Q2n * np.conjugate(Qn) * np.conjugate(Qn))
        )

        second = 2*(M-2)*np.abs(Qn)**2 - M*(M-3)

        return (first - 2*second) / denom  


def correlation_2(phi, pt, n, momentum_cut=0):  
    """Calculates <2> for a given event."""
    M = len(phi)

    if M <= 1:
        return 1234
    else:
        return (np.abs(Qmoment(phi, n))**2 - M) / ((M-1)*M)

def get_truth_correlations(phi, pt):
    """Calculates <2>, <4> for a set of events.
    
    phi: (num_events) list of all phis in each event
    pt: list of pts
    
    Returns (num_events) <2> list, and <4> list"""
    cor2, cor4 = [], []

    for i in range(len(phi)):
        cor2.append(correlation_2(phi[i], pt[i], 2))
        cor4.append(correlation_4(phi[i], pt[i], 2))

    return cor2, cor4
