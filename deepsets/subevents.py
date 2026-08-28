import numpy as np
# I had slurm files calling this from different folders, so I was changing between "from calculations" and "from deepsets.calculations"
from calculations import *

def mask_phi(phi, rapidity):
    """Returns phi split into 4 subevents."""
    maska = (rapidity >= -2.4) & (rapidity < -1.2)
    maskb = (rapidity >= -1.4 )& (rapidity < 0)
    maskc = (rapidity >= 0) & (rapidity < 1.2 ) 
    maskd = (rapidity >= 1.2) & (rapidity <= 2.4 ) 
    phi_a = phi[maska] 
    phi_b = phi[maskb]
    phi_c = phi[maskc]
    phi_d = phi[maskd]
    return phi_a, phi_b, phi_c, phi_d

def get_multiplicities(phis):
    """Input phi which has previously been split into four subevents, get multiplicities of each subevent."""
    M_a = len(phis[0])
    M_b = len(phis[1])
    M_c = len(phis[2])
    M_d = len(phis[3])
    return np.array((M_a, M_b, M_c, M_d))

def get_subevents(phi, rapidity, pt, momentum_cut):
    """Get phis and multiplicities for 4 subevents in one easy-to-use function.  Incredible!"""
    phi = phi[pt>momentum_cut]
    rapidity = rapidity[pt>momentum_cut]
    phis = mask_phi(phi, rapidity)
    mults = get_multiplicities(phis)
    return phis, mults


def sub_cors(phi, rapidity, weight, pt, n = 2, momentum_cut = 0):
    """Supposedly this function calculates the subevent correlators but we didn't train on those so I never used it."""
    #a, b, c, d based on rapity ranges
    phis, mults = get_subevents(phi, rapidity, pt, momentum_cut)
    if (mults < 1).any():
        #return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, weight
        return 1234, 1234, 1234, 1234, 1234, 1234, 1234, weight
        
    else:
        Qna = Qmoment(phis[0], n)
        Qnb = Qmoment(phis[1], n)
        Qncc = np.conjugate(Qmoment(phis[2], n) )
        Qndc = np.conjugate(Qmoment(phis[3], n) )
        #<4>, euqation 44a in paper 2
        cor4_4e = (Qna* Qnb *Qncc*Qndc 
                /mults[0]/mults[1]/mults[2]/mults[3])
        #these are need to define Cn{4} in second line eq 44 in paper 2. these take the place of <2>
        #note their is not <2> here
        cor_ac = Qna*Qncc/mults[0]/mults[2]
        cor_bd = Qnb*Qndc/mults[1]/mults[3]
        cor_ad = Qna*Qndc/mults[0]/mults[3]
        cor_bc = Qnb*Qncc/mults[1]/mults[2]
    
        #defining stuff for 2 subevents
        phi_aa = np.concatenate((phis[0], phis[1])) #subevent a
        phi_bb = np.concatenate((phis[2], phis[3])) #subevent b
        # cor2_2e = Qmoment(phi_aa, n) *np.conjugate(Qmoment(phi_bb, n))/len(phi_aa)/len( phi_bb) #eq 19 paper 2
        cor4_2e = ((Qmoment(phi_aa, n)**2- Qmoment(phi_aa, 2*n))*np.conjugate(Qmoment(phi_bb, n)**2-Qmoment(phi_bb, 2*n)) #eq 20 paper 2
                       /(len( phi_aa)*(len(phi_aa)-1)*len( phi_bb)*(len( phi_bb)-1)) )
        return cor4_4e, cor_ac, cor_bd, cor_ad, cor_bc, weight

def get_subevent_qvecs(phi, rapidity, pt, n=2, momentum_cut=0):
    """ Calculates Q-vectors for 4 subevents.
    Args:
        phi: (N) array of phi for one event
        rapidity: (N) array
        pt: (N) array
        n(int): which flow coefficient you want.  We're trying to get at v_2, so we have n=2.
        momentum_cut(float): momentum cut
        
    Returns: components of Q-vecs for 4 subevents."""
    
    num_subevents = 4
    phis, mults = get_subevents(phi, rapidity, pt, momentum_cut)

    if (mults <= 1).any():
        return [1234 for _ in range(8)]
    else:
        qvecs = [Qmoment(phis[i], n) for i in range(num_subevents)]
        return qvecs[0].real, qvecs[0].imag, qvecs[1].real, qvecs[1].imag, qvecs[2].real, qvecs[2].imag, qvecs[3].real, qvecs[3].imag

    
    
