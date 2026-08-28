from testing import *

def cut_by_pt(data, pt, pt_cuts):
    cut_data = []
    for cut in pt_cuts:
        cut_data.append(data[pt > cuts])
    return cut_data

def get_data_by_mult(data, mults, mult_range):
    mult_min = mult_range[0]
    mult_max = mult_range[1]
    cut = data[mult_min <= mults && mults <= mult_max]
    return data

def plot_corr_by_pt(name, datas, labels="", to_weights=True, bin_size=20, outs=8):
    # ==============================
    # Load data
    # ==============================
    if isinstance(datas, str):
        datas = (datas,)
        labels = (labels,)
    if isinstance(to_weights, bool):
        to_weights = (to_weights, )
    if labels == "":
        labels = ["" for _ in range(len(datas))]
    if len(datas) > 1 and len(to_weights) == 1:
        to_weights = [to_weights[0] for i in range(len(datas))]

    collected_data = [load_data(name, data) for data in datas]

    fig, axs = plt.subplots(2, 2)
    
    for data_i in range(len(datas)):
        pt_cuts = (0, .5)
        mult_range (100, 200)
        
        output_cor, truth_cor, event_weights, mults, pt = collected_data[data_i]
        to_weight = to_weights[data_i]
        
        q_out_vecs, q_truth_vecs = process_output_to_qvecs(output_cor, truth_cor, 8)
    
        corr_out, corr_truth, mults_by_order, final_weights = process_qvecs_to_corrs(outs // 2, q_out_vecs, q_truth_vecs, event_weights, mults)

        corr_out