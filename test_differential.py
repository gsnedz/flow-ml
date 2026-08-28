from testing import *
from test_subevents import *
from matplotlib import colors

from collections import defaultdict

def plot_differential_corrs(name, data, pt_bins, to_weight=True, bin_size=20):
    """Main function, plots output from NN calculating differential flow.
    
    Args:
        name(str): name of model
        data(str): name of dataset to be loaded.  There should be one dataset for each pt bin, saved under the name "data_i" for each bin i.
        pt_bins(list): list of pt bin edges for POI ranges.  Very important that this matches what the dataset was originally saved as."""
    num_pt_bins = len(pt_bins) - 1

    fig, axs = plt.subplots(1, 2)

    plot_x = []
    
    truth_y = []
    truth_err = []
    out_y = []
    out_err = []

    vecfig, vecaxs = plt.subplots(3, 2)
    outs = 10
    
    for bin_i in range(num_pt_bins):
        #j = bin_i % 3
        #i = bin_i // 3
        data_name = f"{data}_{bin_i}"
        dataset = load_data(name, data_name, ins = 7, outs=outs)

        output, truth, event_weights, mults, pts = dataset[:-1]
            
        total_mults = mults[:, 0]
        diff_mults = mults[:, 1:]

        p_out = np.array([output[output_i::outs] for output_i in range(outs)])
        p_truth = np.array([truth[output_i::outs] for output_i in range(outs)])
        #perr = p_out - p_truth

        """
        h = vecaxs[i, j].hist2d(perr[0], perr[1], bins=25, cmap = "Purples", norm=colors.LogNorm())
        fig.colorbar(h[3], ax=vecaxs[i, j])
        vecaxs[i, j].set_xlabel("$p_{2x}$ error")
        vecaxs[i, j].set_ylabel("$p_{2y}$ error")
        vecaxs[i, j].set_title(f"POI range {pt_bins[bin_i]}-{pt_bins[bin_i + 1]}")
        #graph_tilted_qvecs(vecs, vecfig, vecaxs)
    vecfig.tight_layout()
    vecfig.savefig(f"images/differential/{name}.png", dpi=500)
        """
        p_out, q_out, big_q_out = process_array_to_differential_vecs(output)
        p_truth, q_truth, big_q_truth = process_array_to_differential_vecs(truth)
        
        p_err = p_out - p_truth
        q_err = q_out - q_truth
        big_q_err = big_q_out - big_q_truth

        num_bins = 25
        
        h = vecaxs[0, 0].hist2d(p_err[0], p_err[1], bins=num_bins, cmap = "Purples", norm=colors.LogNorm())
        vecfig.colorbar(h[3], ax=vecaxs[0, 0])
        h = vecaxs[1, 0].hist2d(q_err[0, 0], q_err[0, 1], bins=num_bins, cmap = "Purples", norm=colors.LogNorm())
        vecfig.colorbar(h[3], ax=vecaxs[1, 0])
        h = vecaxs[1, 1].hist2d(q_err[1, 0], q_err[1, 1], bins=num_bins, cmap = "Purples", norm=colors.LogNorm())
        vecfig.colorbar(h[3], ax=vecaxs[1, 1])
        h = vecaxs[2, 0].hist2d(big_q_err[0, 0], big_q_err[0, 1], bins=num_bins, cmap = "Purples", norm=colors.LogNorm())
        vecfig.colorbar(h[3], ax=vecaxs[2, 0])
        h = vecaxs[2, 1].hist2d(big_q_err[1, 0], big_q_err[1, 1], bins=num_bins, cmap = "Purples", norm=colors.LogNorm())
        vecfig.colorbar(h[3], ax=vecaxs[2, 1])

        vecaxs[0, 0].set_xlabel("$p_{2x}$ error")
        vecaxs[0, 0].set_ylabel("$p_{2y}$ error")
        vecaxs[1, 0].set_xlabel("$q_{2x}$ error")
        vecaxs[1, 0].set_ylabel("$q_{2y}$ error")
        vecaxs[1, 1].set_xlabel("$q_{4x}$ error")
        vecaxs[1, 1].set_ylabel("$q_{4y}$ error")
        vecaxs[2, 0].set_xlabel("$Q_{2x}$ error")
        vecaxs[2, 0].set_ylabel("$Q_{2y}$ error")
        vecaxs[2, 1].set_xlabel("$Q_{4x}$ error")
        vecaxs[2, 1].set_ylabel("$Q_{4y}$ error")
        vecfig.suptitle(f"The Errors of Bin {bin_i}")
        vecfig.tight_layout()
        vecfig.savefig(f"images/differential/errors_{name}_{data}_{bin_i}.png", dpi=500)

        corr2_out, diffcorr2_out, diffcorr4_out = process_diffvecs_to_corrs(p_out, q_out, big_q_out, diff_mults)
        corr2_truth, diffcorr2_truth, diffcorr4_truth = process_diffvecs_to_corrs(p_truth, q_truth, big_q_truth, diff_mults)
        weights = process_weights(event_weights, total_mults)
    
        mult_range = (100, 200)
        corr2_out = filter_array(corr2_out, total_mults, mult_range)
        diffcorr2_out = filter_array(diffcorr2_out, total_mults, mult_range)
        diffcorr4_out = filter_array(diffcorr4_out, total_mults, mult_range)
        
        corr2_truth = filter_array(corr2_truth, total_mults, mult_range)
        diffcorr2_truth = filter_array(diffcorr2_truth, total_mults, mult_range)
        diffcorr4_truth = filter_array(diffcorr4_truth, total_mults, mult_range)
        
        weights = filter_array(weights, total_mults, mult_range)
        diff_mults = filter_array(diff_mults, total_mults, mult_range)
        total_mults = filter_array(total_mults, total_mults, mult_range)
        
        cumul_out, error_out = calculate_differential_cumul_and_error(corr2_out, diffcorr2_out, diffcorr4_out, weights, to_weight)
        cumul_truth, error_truth = calculate_differential_cumul_and_error(corr2_truth, diffcorr2_truth, diffcorr4_truth, weights, to_weight)

        plot_x.append(np.average(pt_bins[bin_i : bin_i + 2]))
        truth_y.append(cumul_truth)
        truth_err.append(error_truth)
        out_y.append(cumul_out)
        out_err.append(error_out)

    plot_x, truth_y, truth_err, out_y, out_err = np.array(plot_x), np.array(truth_y), np.array(truth_err), np.array(out_y), np.array(out_err)

    axs[0].errorbar(plot_x, out_y[:, 0], yerr=out_err[:, 0], marker=".", ls="", label="output")
    axs[0].errorbar(plot_x + .02, truth_y[:, 0], yerr=truth_err[:, 0], marker=".", ls="", label="truth")

    axs[0].legend()
    fig.suptitle("250k testing")
    axs[0].set_title("2-particle cumulant")
    axs[0].set_xlabel("$p_T$ of POIs")
    axs[0].set_ylabel("$d_2\{2\}$")

    axs[1].errorbar(plot_x, out_y[:, 1], yerr=out_err[:, 1], marker=".", ls="")
    axs[1].errorbar(plot_x + .02, truth_y[:, 1], yerr=truth_err[:, 1], marker=".", ls="")
    axs[1].set_title("4-particle cumulant")
    axs[1].set_xlabel("$p_T$ of POIs")
    axs[1].set_ylabel("$d_2\{4\}$")
    fig.tight_layout()
    fig.savefig(f"images/differential/cumul_{name}_{data}.png", dpi=500)
    

def process_array_to_differential_vecs(outputs):
    outs = 10
    comps = np.array([outputs[i::outs] for i in range(outs)])

    p = comps[:2]
    q = np.stack((comps[2:4], comps[4:6]))
    big_q = np.stack((comps[6:8], comps[8:10]))

    return p, q, big_q

def process_diffvecs_to_corrs(p, q, big_q, mults):
    """Calculates correlators from NN output
    
    Args:
        p: (2, n) array.
        q: (2, 2, n) array.  q[0] is q2 and q[1] is q4.
        big_q: (2, 2, n) array.  Q[0] is Q2 and Q[1] is Q4.
        mults: (3, n) array. mults[0] is M, mults[1] is m_p, mults[2] m_q.
        
    Returns: <2>, <2'>, <4'>"""
    
    corr2 = (mag2(big_q[0]) - mults[:, 0]) / (mults[:, 0] * (mults[:, 0] - 1))
    diff_corr2 = (to_complex(p) * to_complex(conj(big_q[:, 0])) - mults[:, 2]) / (mults[:, 1] * mults[:, 0] - mults[:, 2])
    diff_corr4 = (to_complex(p) * to_complex(big_q[0]) * to_complex(conj(big_q[0])) * to_complex(conj(big_q[0])) \
        - to_complex(q[1]) * to_complex(conj(big_q[0])) * to_complex(conj(big_q[0])) \
        - to_complex(p) * to_complex(big_q[0]) * to_complex(conj(big_q[1])) \
        - 2 * mults[:, 0] * to_complex(p) * to_complex(conj(big_q[0])) - 2 * mults[:, 2] * mag2(big_q[0]) \
        + 7 * to_complex(q[0]) * to_complex(conj(big_q[0])) - to_complex(big_q[0]) * to_complex(conj(q[0])) \
        + to_complex(q[1]) * to_complex(conj(big_q[1])) + 2 * to_complex(p) * to_complex(conj(big_q[0])) \
        + 2 * mults[:, 2] * mults[:, 0] - 6 * mults[:, 2]) \
        / ((mults[:, 1] * mults[:, 0] - 3 * mults[:, 2]) * (mults[:, 0] - 1) * (mults[:, 0] - 2))

    return corr2, diff_corr2, diff_corr4

def calculate_differential_cumul_and_error(corr2, diffcorr2, diffcorr4, weights, to_weight):
    """"""
    orders = (2, 4)
    cumuls = []
    errors = []

    for i in range(len(orders)):
        order = orders[i]
        cumuls.append(calculate_differential_cumul(corr2, diffcorr2, diffcorr4, order, weights, to_weight))
        errors.append(calculate_differential_error(corr2, diffcorr2, diffcorr4, order, weights, to_weight))

    return np.array(cumuls), np.array(errors)

def calculate_differential_error(corr2, diffcorr2, diffcorr4, order, weights, to_weight):
    """Calculates error by subsampling."""
    num_splits = min(len(corr2), 20)
    splits = np.array_split(np.arange(len(corr2)), num_splits)
    cumuls = []
    for split in splits:
        corr2_split = corr2[split]
        diffcorr2_split = diffcorr2[split]
        diffcorr4_split = diffcorr4[split]
        weight_split = weights[split]
        cumuls.append(calculate_differential_cumul(corr2_split, diffcorr2_split, diffcorr4_split, order, weight_split, to_weight))
    return np.std(cumuls) / np.sqrt(num_splits)

def calculate_differential_cumul(corr2, diffcorr2, diffcorr4, order, weights, to_weight):
    """Calculates differential cumulants."""
    if not to_weight:
        weights = np.ones(weights.shape)

    if order == 2:
        cumul = np.average(diffcorr2, weights = weights)
    elif order == 4:
        cumul = np.average(diffcorr4, weights = weights) - 2 * np.average(diffcorr2, weights = weights) * np.average(corr2, weights = weights)
    return cumul

def filter_array(data, labels, label_range):
    """Returns only the parts of the data of a specific range.  For example, if data is phi, labels is pt, and label_range is (.3, .5), this function returns only the phi in the pt range of (.3, .5)."""
    return data[(label_range[0] < labels) & (labels <= label_range[1])]

def graph_tilted_qvecs(vecs_out, fig, axs):
    """This ccreates graphs for a net we trained which compared the old Q-vectors (all phi) with the differential Q-vectors (only reference particles)."""
    num_bins = 25
    h = axs[0, 0].hist2d(vecs_out[0, 0], vecs_out[2, 0], bins=num_bins, cmap = "Purples", norm=colors.LogNorm())
    axs[0, 0].set_xlabel("old $Q_{2x}$")
    axs[0, 0].set_ylabel("diff $Q_{2x}$")
    #axs[0, 0].set_xlim(l, r)
    #axs[0, 0].set_ylim(l, r)
    fig.colorbar(h[3], ax=axs[0, 0])
        
    h = axs[0, 1].hist2d(vecs_out[0, 1], vecs_out[2, 1], bins=num_bins, cmap = "Purples", norm=colors.LogNorm())
    axs[0, 1].set_xlabel("old $Q_{2y}$")
    axs[0, 1].set_ylabel("diff $Q_{2y}$")
    #axs[0, 1].set_xlim(l, r)
    #axs[0, 1].set_ylim(l, r)
    fig.colorbar(h[3], ax=axs[0, 1])
        
    h = axs[1, 0].hist2d(vecs_out[1, 0], vecs_out[3, 0], bins=num_bins, cmap = "Purples", norm=colors.LogNorm())
    axs[1, 0].set_xlabel("old $Q_{4x}$")
    axs[1, 0].set_ylabel("diff $Q_{4x}$")
    #axs[1, 0].set_xlim(l, r)
    #axs[1, 0].set_ylim(l, r)
    fig.colorbar(h[3], ax=axs[1, 0])
        
    h = axs[1, 1].hist2d(vecs_out[1, 1], vecs_out[3, 1], bins=num_bins, cmap = "Purples", norm=colors.LogNorm())
    axs[1, 1].set_xlabel("old $Q_{4y}$")
    axs[1, 1].set_ylabel("diff $Q_{4y}$")
    #axs[1, 1].set_xlim(l, r)
    #axs[1, 1].set_ylim(l, r)
    fig.colorbar(h[3], ax=axs[1, 1])
        

    fig.tight_layout()
    fig.savefig("images/differential/qvecs_truth_hist.png", dpi=500)