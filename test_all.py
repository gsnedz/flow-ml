from test_subevents import *
import time

def test_all(model, data, to_weight = True, bin_size = 20, chunks = True):
    """Main function to plot net which outputs Q2, Q4, and all subevent Q-vecs."""
    outs = 12

    if chunks:
        chunk_i = 0
        calc_time = 0
        while os.path.exists(f"deepsets/deepset_data/{data}_{chunk_i}/data.pt"):
            print(f"Dataset: deepsets/deepset_data/{data}_{chunk_i}/data.pt")
            dataset = f"{data}_{chunk_i}"
            output, truth, event_weights, mults, pts, runtime = load_data(model, dataset, outs=outs)
            calc_time += runtime
            chunk_i += 1
        print(chunk_i)
        print("total time:", calc_time)
    else:
        output, truth, event_weights, mults = load_data(model, data, outs=outs)
        
        #mults_total = mults[:, 0]
        mults_sube = mults[:, 1:]
        mults_total = np.sum(mults_sube, axis=1)
    
        qvecs_out = process_array_to_qvecs(output, outs)
        qvecs_truth = process_array_to_qvecs(truth, outs)
    
        # No subevents
        qvecs_out_reg = qvecs_out[:2]
        qvecs_truth_reg = qvecs_truth[:2]
    
        # 2 subevents
        qvecs_out_sube = qvecs_out[2:]
        qvecs_truth_sube = qvecs_truth[2:]
    
        # Get all the correlators
        corrs_out, corrs_truth, _, _ = process_qvecs_to_corrs(2, qvecs_out_reg, qvecs_truth_reg, event_weights, mults_total)
        corr2_out_sube, corr4_out_sube = process_subevents_to_corrs(qvecs_out_sube, mults_sube)
        corr2_truth_sube, corr4_truth_sube = process_subevents_to_corrs(qvecs_truth_sube, mults_sube)
    
        # Pythia event weight * Hijing weight
        weights = process_weights(event_weights, mults_sube)
        
        mult_bins = np.arange(0, 201, 20)
    
        corr_out_bins = bin_data(2, mult_bins, mults_total, corrs_out)
        corr_truth_bins = bin_data(2, mult_bins, mults_total, corrs_truth)
    
        corr2_out_bin_sube = bin_subevents_data(corr2_out_sube, mults_sube, mult_bins)
        corr4_out_bin_sube = bin_subevents_data(corr4_out_sube, mults_sube, mult_bins)
        corr2_truth_bin_sube = bin_subevents_data(corr2_truth_sube, mults_sube, mult_bins)
        corr4_truth_bin_sube = bin_subevents_data(corr4_truth_sube, mults_sube, mult_bins)
    
        weight_bins = bin_subevents_data(weights, mults_sube, mult_bins)
    
        
        cumul_out, err_out = calculate_cumul_and_error(corr_out_bins, weight_bins, to_weight)
        cumul_truth, err_truth = calculate_cumul_and_error(corr_truth_bins, weight_bins, to_weight)
    
        cumul_out_sube, err_out_sube = calculate_subevent_cumul_and_error(corr2_out_bin_sube, corr4_out_bin_sube, weight_bins, to_weight)
        cumul_truth_sube, err_truth_sube = calculate_subevent_cumul_and_error(corr2_truth_bin_sube, corr4_truth_bin_sube, weight_bins, to_weight)
    
        
                 
        fig, axs = plt.subplots(1, 2)
    
        plot_bins = mult_bins[:-1]
        start = 1
        
        axs[0].errorbar(plot_bins[start:], cumul_out[0, start:], yerr=err_out[0, start:], marker=".", ls="", label="no subevents, output")
        axs[0].errorbar(plot_bins[start:] + 2, cumul_truth[0, start:], yerr=err_truth[0, start:], marker=".", ls="", label="no subevents, truth")
        axs[0].set_xlabel("Multiplicity")
        axs[0].set_ylabel("$c_2\{2\}$")
        axs[0].set_title("2-particle")
    
        axs[1].errorbar(plot_bins[start:], cumul_out[1, start:], yerr=err_out[1, start:], marker=".", ls="", label="no subevents, output")
        axs[1].errorbar(plot_bins[start:] + 2, cumul_truth[1, start:], yerr=err_truth[1, start:], marker=".", ls="", label="no subevents, truth")
        axs[1].errorbar(plot_bins[start:] + 4, cumul_out_sube[start:], yerr=err_out_sube[start:], marker=".", ls="", label="4 subevents, output")
        axs[1].errorbar(plot_bins[start:] + 6, cumul_truth_sube[start:], yerr=err_truth_sube[start:], marker=".", ls="", label="4 subevents, truth")
        axs[1].set_xlabel("Multiplicity")
        axs[1].set_ylabel("$c_2\{4\}$")
        axs[1].set_title("4-particle")
        axs[1].legend()
    
        fig.suptitle("Cumulants, million minbias val")
        fig.tight_layout()
        fig.savefig(f"images/combined/cumul_{model}_{data}.png", dpi=500)
    
    