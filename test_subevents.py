from testing import *
from collections import defaultdict

def plot_subevent_corrs(name, datas, labels="", to_weights=True, bin_size=20, outs=8):
    """Main function to plot output from subevent net."""
    # ==============================
    # Load data
    # ==============================
    if isinstance(datas, str):
        datas = (datas,)
        labels = (labels,)
    if isinstance(to_weights, bool):
        to_weights = (to_weights,)
    if labels == "":
        labels = ["" for _ in range(len(datas))]
    if len(datas) > 1 and len(to_weights) == 1:
        to_weights = [to_weights[0] for i in range(len(datas))]

    collected_data = [load_data(name, data) for data in datas]

    fig, axs = plt.subplots(1)
    
    
    for data_i in range(len(datas)):
        output, truth, event_weights, mults = collected_data[data_i]
        to_weight = to_weights[data_i]
        
        qvecs_out = process_array_to_subevents_qvecs(output)
        qvecs_truth = process_array_to_subevents_qvecs(truth)

        corr2_out, corr4_out = process_subevents_to_corrs(qvecs_out, mults)
        corr2_truth, corr4_truth = process_subevents_to_corrs(qvecs_truth, mults)
        weights = process_weights(event_weights, mults)

        mult_bins = np.arange(50, 211, 20)

        corr2_out_binned = bin_subevents_data(corr2_out, mults, mult_bins)
        corr4_out_binned = bin_subevents_data(corr4_out, mults, mult_bins)

        corr2_truth_binned = bin_subevents_data(corr2_truth, mults, mult_bins)
        corr4_truth_binned = bin_subevents_data(corr4_truth, mults, mult_bins)

        weight_binned = bin_subevents_data(weights, mults, mult_bins)

        cumul_out, error_out = calculate_subevent_cumul_and_error(corr2_out_binned, corr4_out_binned, weight_binned, to_weight)
        cumul_truth, error_truth = calculate_subevent_cumul_and_error(corr2_truth_binned, corr4_truth_binned, weight_binned, to_weight)
    
        plot_mults = mult_bins[:-1]
        axs.errorbar(plot_mults, cumul_out, yerr=error_out, marker=".", ls="", label=f"{labels[data_i]} output")
        axs.errorbar(plot_mults + 2, cumul_truth, yerr=error_truth, marker=".", ls="", label=f"{labels[data_i]} truth")
        """
        newfig, newax = plt.subplots(1)
        newax.hist(mults)
        newfig.savefig(f"images/subevents/mults.png", dpi = 500
        """
        
        #plot_4corr_by_mult(mult_bins, corr4_out_binned, corr4_truth_binned)

    axs.set_xlabel("Multiplicity")
    axs.set_ylabel(r"$c_2{4}$")
    axs.set_title("Subevent cumulants, 200k val")
    axs.legend()
    fig.tight_layout()
    data = "_data_"
    for i in range(len(datas)):
        weight = ""
        if to_weights[i]:
            weight = "weighted"
        data += f"{datas[i]}_{weight}_"
    fig.savefig(f"images/subevents/{name}{data}subevents.png", dpi = 500)
    
    

    # plot_4corr_by_mult(mult_bins, corr4_out_binned, corr4_truth_binned)

def get_total_mults(mults):
    """Takes the multiplicities of each subevent (n, 4)-size array and returns to total multiplicity of each event."""
    return np.sum(mults, axis=1)

def bin_subevents_data(data, mults, mult_bins):
    """Subevent 2-particle correlators are stored in dictionaries, so this will bin either dictionaries or arrays."""
    total_mults = get_total_mults(mults)
    if isinstance(data, dict):
        binned_data = defaultdict(list)
        keys = data.keys()
        for key in keys:
            binned_data[key] = bin_array(data[key], total_mults, mult_bins)
        return binned_data
    else:
        return bin_array(data, total_mults, mult_bins)

def calculate_subevent_corr2(qvecs, mults, i, j):
    """Calculates subevent <2>."""
    return (to_complex(qvecs[i]) * to_complex(conj(qvecs[j]))) / (mults[:, i] * mults[:, j])

def process_weights(weights, mults):
    """Multiplies the four-particle combinatoric weight."""
    if mults.ndim != 1:
        total_mults = get_total_mults(mults)
    else:
        total_mults = mults
    return weights * total_mults * (total_mults - 1) * (total_mults - 2) * (total_mults - 3)

def process_subevents_to_corrs(qvecs, mults):
    """Calculates correlators from subevents."""
    # Events with insufficient multiplicity were already selected out
    corr2 = defaultdict(list)
    corr2["ac"] = calculate_subevent_corr2(qvecs, mults, 0, 2)
    corr2["bd"] = calculate_subevent_corr2(qvecs, mults, 1, 3)
    corr2["ad"] = calculate_subevent_corr2(qvecs, mults, 0, 3)
    corr2["bc"] = calculate_subevent_corr2(qvecs, mults, 1, 2)

    corr4 = to_complex(qvecs[0]) * to_complex(qvecs[1]) * to_complex(conj(qvecs[2])) * to_complex(conj(qvecs[3])) \
            / (mults[:, 0] * mults[:, 1] * mults[:, 2] * mults[:, 3])
    
    return corr2, corr4

def process_array_to_qvecs(output, outs=8):
    """Processes a one-dimensional array from the net into an (outs / 2, 2, n) size array."""
    q_comps = np.array([output[i::outs] for i in range(outs)], dtype = np.float64)

    qvecs = [q_comps[i:i+2] for i in range(0, outs, 2)]
    return np.array(qvecs, dtype = np.float64)

def get_dict_bin(dictionary, bin_i):
    """Returns bin_i from each item in a dictionary."""
    new_dict = defaultdict(list)
    for key in dictionary.keys():
        new_dict[key] = dictionary[key][bin_i]
    return new_dict

def calculate_subevent_cumul_and_error(corr2_binned, corr4_binned, weight_binned, to_weight):
    """Calculates the cumulants and error (by subsampling) for subevents."""
    cumuls = []
    errors = []
    for bin_i in range(len(corr4_binned)):
        corr4_bin = corr4_binned[bin_i]
        corr2_bin = get_dict_bin(corr2_binned, bin_i)

        weight_bin = weight_binned[bin_i]
        cumuls.append(calculate_subevent_cumul(corr2_bin, corr4_bin, weight_bin, to_weight))
        errors.append(calculate_subevent_error(corr2_bin, corr4_bin, weight_bin, to_weight))
    return np.array(cumuls), np.array(errors)
        

def calculate_subevent_cumul(corr2, corr4, weight, to_weight):
    """Calculate correlators for subevents."""
    if not to_weight:
        weight = np.ones(weight.shape)
    return np.average(corr4, weights=weight) - np.average(corr2["ac"], weights=weight) * np.average(corr2["bd"], weights=weight) \
            - np.average(corr2["ad"], weights=weight) * np.average(corr2["bc"], weights=weight)

def split_dict(dictionary, split):
    """Returns a selection from each item in a dictionary."""
    new_dict = defaultdict(list)
    for key in dictionary.keys():
        new_dict[key] = dictionary[key][split]
    return new_dict

def calculate_subevent_error(corr2, corr4, weight, to_weight):
    """Calculate error for subevents by subsampling."""
    num_splits = 20
    splits = np.array_split(np.arange(len(corr4)), num_splits)
    cumuls = []
    #print("Hi! ===========================================================")
    
    for split in splits:
        corr4_split = corr4[split]
        corr2_split = split_dict(corr2, split)
        weight_split = weight[split]
        #print("corr4", corr4_split)
        #print("corr2", corr2_split["ac"], corr2_split["bd"], corr2_split["ad"], corr2_split["bc"])
        cumuls.append(calculate_subevent_cumul(corr2_split, corr4_split,  weight_split, to_weight))
        #print("cumul", cumuls[-1])
        #print("========")
        
    return np.std(cumuls) / np.sqrt(num_splits)
        
