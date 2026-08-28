from calculations import *


def get_differential_truth(phi, pt, poi_cut = (3, 5)):
    """Returns the p_n, q_n, q_2n, Q_n, and Q_2n components necessary for differential flow."""
    reference_cut = (.3, 3)
    n = 2

    poi, poi_pt, m_p = cut_particles(phi, pt, poi_cut)
    if len(poi) == 0:
        return [1234 for i in range(10)]
    rfp, _, m_q = cut_particles(poi, poi_pt, reference_cut)
    q_particles, _, m = cut_particles(phi, pt, reference_cut)
    
    poi_label = (poi_cut[0] <= phi) & (phi <= poi_cut[1])
    ref_label = (reference_cut[0] <= phi) & (phi <= reference_cut[1])

    p_vec = Qmoment(poi, n)
    q_vec = Qmoment(rfp, n)
    q2_vec = Qmoment(rfp, 2 * n)
    big_q = Qmoment(q_particles, n)
    big_q2 = Qmoment(q_particles, 2 * n)
    """
    old_q = Qmoment(phi, n)
    old_q2 = Qmoment(phi, 2 * n)

    return big_q.real, big_q.imag, big_q2.real, big_q2.imag, old_q.real, old_q.imag, old_q2.real, old_q2.imag
    """
    #return poi_label, ref_label, p_vec.real, p_vec.imag, q_vec.real, q_vec.imag, q2_vec.real, q2_vec.imag, big_q.real, big_q.imag, big_q2.real, big_q2.imag
    return poi_label, ref_label, p_vec.real, p_vec.imag, q2_vec.real, q2_vec.imag
    #return poi_label, ref_label, p_vec.real, p_vec.imag

def cut_particles(phi, pt, endpoints):
    """Returns phi, pt, and multiplicity for a specific range of """
    new_phi = np.array(phi, dtype=np.float64)[cut(pt, endpoints)]
    mult = len(new_phi)
    new_pt = np.array(pt, dtype=np.float64)[cut(pt, endpoints)]
    return new_phi, new_pt, mult


def get_differential_mults(phi, pt, poi_cut = (3, 5)):
    """Returns M, m_p, and m_q multiplicities for differential calculations."""
    reference_cut = (.3, 3)
    poi, poi_pt, m_p = cut_particles(phi, pt, poi_cut)
    _, _, m_q = cut_particles(poi, poi_pt, reference_cut)
    q_particles, _, m = cut_particles(phi, pt, reference_cut)
    return m, m_p, m_q

    
"""
def Full_0sub(phi,pt, n, POI_start=1 , POI_end = 2):
    #returns dcor4, dcor4w, dcor2, dcor2w
    #reff = 3

    #phi = phi[pt>momentum_cut]
    #def dcor_4(phi, pt, n, POI_cut): #equ 32, with, weights by eq. 25
    phi = np.array(phi); pt = np.array(pt)
    mask_POI = (pt>=POI_start) & (pt<= POI_end)
    mask_Both = (pt>=POI_start) & (pt<= POI_end) & (pt<reff_cut)
    POI = phi[mask_POI]
    Ref = phi[pt<reff_cut]
    Both = phi[mask_Both]
    mp = len(POI); M=len(Ref); mq = len(Both)
    pn = Qmoment(POI, n)
    Qn = Qmoment(Ref, n); Qnc = np.conjugate(Qn)
    qn = Qmoment(Both,  n); q2n = Qmoment(Both, 2*n)
    if ( (mp==0 or M==0 )and mq==0):
        return -1234, -1234, -1234, -1234
    d2cor = (pn*Qnc-mq)/(mp*M-mq)
    d2corw = (mp*M-mq)
    if (M<3 or ((M==0 or mp==0) and mq==0)):
        return -1234, -1234, d2cor, d2corw
    d4cor =( (pn*Qn*Qnc*Qnc -q2n*Qnc*Qnc
            -pn*Qn*np.conjugate(Qmoment(Ref, 2*n)) -2*M*pn*Qnc
            -2*mq*np.abs(Qn)**2+7*qn*Qnc
            -Qn*np.conjugate(qn)+q2n*np.conjugate(Qmoment(Ref, 2*n))
            +2 *pn*Qnc+2*mq*M-6*mq
            )/( (mp*M-3*mq)*(M-1)*(M-2))       )
    d4corw = (mp*M-3*mq)*(M-1)*(M-2)
    return d4cor, d4corw, d2cor, d2corw
"""