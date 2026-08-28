from dnn.dnn import *
from deepsets.calculations import *
from deepsets.subevents import *
import time
import matplotlib.colors as mcolors


def run_correlations(model, name, device):
    """
    Takes in the trained model and the root folder path for a dataset.  Runs the dataset through the model.  It's called run_correlations because I was originally calculating correlators with it.

    Args:
        model: NN loaded in
        name(str): name of the dataset to feed through the model.  The dataset you want should be stored at deepsets/deepset_data/{name}/data.pt.
        device: the device to do the PyTorch calculations on.

    Returns:
        output: A one-dimensional array containing all NN outputs.  If, for example, the net has three outputs, the output array will look like [output1, output2, output3, output1, output2, output3, output1, ...].
        truth: A one-dimensional truth array of the same format.
        weight: The weight of each event.
        mults: multiplicity of each event.  For regular correlators, this was just a single number of the multiplicity.  For differential or subevet calculations, this was an array of all the multiplicities necessary for calculations.  Check your dataset so you know what the values here will be.
        pts: The momenta of particles in each event.  This is necessary for differential calculations.
        runtime: the total time it took the net to do the calculations.
    """
    # Load data
    if isinstance(name, str):
        dataset = torch.load(f"deepsets/deepset_data/{name}/data.pt")
    else:
        dataset = name
    # loader = DataLoader(dataset[int(len(dataset) * .8):], batch_size=256, shuffle=False, collate_fn=collate_fn)
    # loader = DataLoader(dataset[-1:], batch_size=1, shuffle=False, collate_fn=collate_fn)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, drop_last=True)
    

    # Run model by batches
    outputs = []
    truths = []
    multiplicities = []
    weights = []
    pts = []
    net_start = time.time()
    with torch.no_grad():
        
        for item in loader:
            x, mask, y, weight, mult, pt = item
            
            x = x.to(device)
            mask = mask.to(device)
            y = y.to(device).view(-1)
            mult = mult.to(device)
            
            
            pred = model(x, mask)
            
            
            #outputs.append(comp_to_mag2(pred))
            outputs.append(pred)
            truths.append(y)
            multiplicities.append(mult)
            weights.append(weight)

            pt = item[5]
            for event in pt:
                pts.append(event)
    net_end = time.time()
    runtime = net_end - net_start
        
        #print("Net time:", net_end - net_start)
    """
    with torch.no_grad():
        
        events = dataset
        print(events)

        calc_time = 0

        # No subevents
        for event_i in range(len(dataset)):
            phi = np.asarray(x[event_i])[0][:, 0]
            eta = torch.tensor(x[event_i])[0][:, 4]
            pt = torch.tensor(x[event_i])[0][:, 3]
            calc_start = time.time()
            truth = Qmoment(phi, 2), Qmoment(phi, 4)
            subevents = get_subevent_qvecs(phi, eta, pt)
            calc_end = time.time()
            calc_time += calc_end - calc_start
        print("calc time", calc_time)
    """

    # All 2nd-order correlations.
    total_output = torch.cat(outputs)
    total_truth = torch.cat(truths)
    total_mults = torch.cat(multiplicities)
    total_weights = torch.cat(weights)

    #if pt == []:
    #return total_output.cpu(), total_truth.cpu(), total_weights.cpu(), total_mults.cpu()
    #else:
    return total_output.cpu(), total_truth.cpu(), total_weights.cpu(), total_mults.cpu(), pts, runtime

def calc_qmoment(phi, n):
    """Calculates the Q-moment of a bunch along axis 2.  I'm not sure why I made this function.  I probably had some weirdly-shaped array."""
    return torch.sum(torch.exp(1j * n * phi), 2)

def calc_subevent_qvec(phi, eta, pt):
    """Calculates the subevent Qvectors for many events."""
    truth = []
    for i in range(len(phi)):
        event_phi = phi[i]
        event_eta = eta[i]
        event_pt = pt[i]
        truth.append(get_subevent_qvecs(event_phi, event_eta, event_pt))
    return truth
        

#def calculate_corr_24(q_vecs, weights, mult, order, weight = True):
#    q2, q4, weights = q_vecs[0], q_vecs[1], weights.squeeze()
#    
#    if order == 2:
#        corr = (mag2(q2) - mult) / (mult * (mult - 1))
#        total_weight = weights * mult * (mult - 1)
#        
#    if order == 4:
#        corr = (mag2(q2)**2 + mag2(q4) - 2 * (to_complex(q4) * to_complex(conj(q2)) * to_complex(conj(q2))).real - 2 * (2 * (mult - #2) * mag2(q2) - mult * (mult - 3))) \
#        / (mult * (mult - 1) * (mult - 2) * (mult - 3))
#        total_weight = weights * mult * (mult - 1) * (mult - 2) * (mult - 3)
#
#    return corr, total_weight

def calculate_corr(q_vecs, weights, mult, order):
    """Calculates the correlators.  No differential flow or subevents or anything fancy.  I couldn't find a formula for 8th order correlators and I never got around to looking, so fix that if you're trying to use that for something.
    
    Args:
        q_vecs: (n, 2) array for n-many Q-vectors.
        weights: The weight of each Q-vector.
        mult: The multiplicity of each event.
        order: The order of the correlator you want (2, 4, 6, 8).
        
    Returns:
        corr: an array of the correlator for each event.
        total_weights: The event weight for each event (given weight * combinatoric weight)."""
    if order == 2 or order == 4:
        q2, q4, weights = q_vecs[0], q_vecs[1], weights.squeeze()
    else:
        q2, q4, q6, q8, weights = q_vecs[0], q_vecs[1], q_vecs[2], q_vecs[3], weights.squeeze()
    
    if order == 2:
        corr = (mag2(q2) - mult) / (mult * (mult - 1))
        total_weight = weights * mult * (mult - 1)
        
    if order == 4:
        corr = (mag2(q2)**2 + mag2(q4) - 2 * (to_complex(q4) * to_complex(conj(q2)) * to_complex(conj(q2))).real - 2 * (2 * (mult - 2) * mag2(q2) - mult * (mult - 3))) \
        / (mult * (mult - 1) * (mult - 2) * (mult - 3))
        total_weight = weights * mult * (mult - 1) * (mult - 2) * (mult - 3)
        
    if order == 6:
        corr = (mag2(q2)**3 + 9 * mag2(q4)**2 * mag2(q2) - 6 * (to_complex(q4) * to_complex(q2) * to_complex(conj(q2))**3).real \
                + 4 * ((to_complex(q6) * to_complex(conj(q2))**3).real - 3 * (to_complex(q6) * to_complex(conj(q4)) * to_complex(conj(q2))).real) \
                + 2 * (9 * (mult - 4) * (to_complex(q4) * to_complex(conj(q2))**2).real  + 2 * mag2(q6)) \
                - 9 * (mag2(q2)**2 + mag2(q4))) \
                / (mult * (mult - 1) * (mult - 2) * (mult - 3) * (mult - 4) * (mult - 5)) \
                + 18 * mag2(q2) / (mult * (mult - 1) * (mult - 3) * (mult - 4)) \
                - 6 / ((mult - 1) * (mult - 2) * (mult - 3))
        total_weight = weights * mult * (mult - 1) * (mult - 2) * (mult - 3) * (mult - 4) * (mult - 5)
        
    if order == 8:
        corr = np.zeros(mult.shape) #FIX
        total_weight = weights * mult * (mult - 1)

    return corr, total_weight

def mag2(q):
    """Square magnitude of a vector."""
    return q[0]**2 + q[1]**2

def conj(q):
    """Complex conjugate"""
    return q[0], -q[1]

def to_complex(q):
    """The Q-vectors are all stored as 2-component vectors, so here they get converted to complex numbers so you can do multiplication of complex numbers."""
    return q[0] + q[1] * 1j

def subsample(array, num_splits=20):
    """I don't think I used this function ever.  But it's here."""
    splits = np.array_split(array, n)
    means = [np.average(split) for split in splits]
    return np.std(means) / np.sqrt(n)

def load_data(model_name, data_name, ins = 5, outs=8):
    """Loads dataset, runs it through the model, and saves the output to arrays under the "modelname_dataname" folder.  NOTE: If you would like the output to be recalculated (like if you have a new dataset of the same name), you need to delete the already-saved arrays.  Otherwise those will simply be loaded in.
    
    Args:
        model_name(str): Name of the NN (minus .pth extension).
        data_name(str): Name of the dataset folder.
        ins(int): number of inputs on the NN
        outs(int): number of outputs on the NN.
        
    Returns:
        """
    folder = f"arrays/{model_name}_{data_name}"
    output_path = f"{folder}/output.npy"
    truth_path = f"{folder}/truth.npy"
    mult_path = f"{folder}/mults.npy"
    weight_path = f"{folder}/weights.npy"
    pt_path = f"{folder}/pts.pt"
    if not (os.path.exists(output_path)
            and os.path.exists(truth_path)
            and os.path.exists(mult_path)
            and os.path.exists(weight_path)
            and os.path.exists(pt_path)):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DNN(ins, outs).to(device)
        model.load_state_dict(torch.load("models/" + model_name + ".pth", weights_only=True, map_location=torch.device('cpu')))
        model.eval()
    
        output_cor, truth_cor, weights, mults, pts, runtime = run_correlations(model, data_name, device)
        #output_cor, truth_cor, weights, mults = run_correlations(model, data_name, device)
    
        output_cor, truth_cor, weights, mults = np.array(output_cor), np.array(truth_cor), np.squeeze(weights), np.array(mults)
        
        os.makedirs(folder, exist_ok=True)
        
        np.save(output_path, output_cor)
        np.save(truth_path, truth_cor)
        np.save(weight_path, weights)
        np.save(mult_path, mults)
        torch.save(pts, pt_path)
        
    else:
        output_cor = np.load(output_path)
        truth_cor = np.load(truth_path)
        weights = np.load(weight_path)
        mults = np.load(mult_path)
        pts = torch.load(pt_path)
        runtime = 0
    
    return output_cor, truth_cor, weights.squeeze(), mults, pts, runtime
    #return output_cor, truth_cor, weights.squeeze(), mults, pts
    #return output_cor, truth_cor, weights.squeeze(), mults

def process_output_to_qvecs(output_cor, truth_cor, outs):
    """The NN outputs a one-dimensional array.  This function reformats it into an (n, 2)-dimensional array so you can do vector math with it.  Ignore the fact that the parameters are called cor."""
    q_out = np.array([output_cor[i::outs] for i in range(outs)], dtype = np.float64)
    q_truth = np.array([truth_cor[i::outs] for i in range(outs)], dtype = np.float64)
    
    q_out_vecs = []
    q_truth_vecs = []
    for i in range(0, outs, 2):
        q_out_vecs.append(q_out[i:i+2])
        q_truth_vecs.append(q_truth[i:i+2])
    
    q_out_vecs = np.array(q_out_vecs, dtype=np.float64)
    q_truth_vecs = np.array(q_truth_vecs, dtype=np.float64)

    return q_out_vecs, q_truth_vecs

def process_qvecs_to_corrs(num_qvecs, q_out_vecs, q_truth_vecs, event_weights, mults):
    """So now that you processed the output into q-vectors using the process_output_to_qvecs, this function calculates the correlators from them."""
    corr_out = []
    corr_truth = []
    plot_mults = []
    final_weights = []
        
    for i in range(num_qvecs):
        order = 2 * i + 2
        condition = (mults >= order)
        output, weight = calculate_corr(q_out_vecs[:, :, condition], event_weights[condition], mults[condition], order)
        corr_out.append(np.array(output, dtype=np.float64))
            
        truth, _ = calculate_corr(q_truth_vecs[:, :, condition], event_weights[condition], mults[condition], order)
        corr_truth.append(np.array(truth, dtype=np.float64))
            
        plot_mults.append(mults[condition])
        final_weights.append(np.array(weight, dtype=np.float64))
    return corr_out, corr_truth, plot_mults, final_weights


def bin_data(num_data, mult_bins, mults, data):
    """Bins multiple piece of data by multiplicity.  For n types of data, data should be an array or list of length n.  Length of each type of data may differ."""
    num_bins = len(mult_bins) - 1
    data_bins = [[] for _ in range(num_data)]
    # format data
    #print(mults)
    if len(mults) != num_data:
        mults = [mults for _ in range(num_data)]
        
    for i in range(num_data):
        for j in range(num_bins):
            to_bin = (mult_bins[j] < mults[i]) & (mults[i] <= mult_bins[j + 1])
                    
            data_bins[i].append(data[i][to_bin])

    return np.array(data_bins, dtype=object)

def bin_array(data, labels, bins):
    """Bins a one-dimensional array.  Simpler than bin_data, which bins multiple arrays."""
    num_bins = len(bins) - 1
    data_binned = []
    for i in range(num_bins):
        to_bin = (bins[i] < labels) & (labels <= bins[i + 1])
        data_binned.append(data[to_bin])
    return np.array(data_binned, dtype=object)

def calculate_cumul_and_error(corr_bins, weight_bins, to_weight):
    """Calculates cumulants and errors.  For n-many Q-vectors and m-many bins, returns an array of size (n, m) for both cumulants and error."""
    num_vecs = len(corr_bins)
    num_bins = len(corr_bins[0])
    
    cumul = np.zeros((num_vecs, num_bins))
    cumul_err = np.zeros((num_vecs, num_bins))
    corr_bins = np.array(corr_bins, dtype=object)
    weight_bins = np.array(corr_bins, dtype=object)
    
    for i in range(num_vecs):
        order = 2 * i + 2
        for j in range(num_bins):
            cumul[i, j] = calculate_cumul(corr_bins[:, j], weight_bins[:, j], order, to_weight)
            cumul_err[i, j] = calculate_error(corr_bins[:, j], weight_bins[:, j], order, to_weight)
            #if len(weight_bins) == len(corr_bins):
            #    print(corr_bins[:, j])
            #    cumul[i, j] = calculate_cumul(corr_bins[:, j], weight_bins[:, j], order, to_weight)
            #    cumul_err[i, j] = calculate_error(corr_bins[:, j], weight_bins[:, j], order, to_weight)
            #else:
            #    cumul[i, j] = calculate_cumul(corr_bins[j], weight_bins[j], order, to_weight)
            #    cumul_err[i, j] = calculate_error(corr_bins[j], weight_bins[j], order, to_weight)

    return cumul, cumul_err

# Takes a model, runs some data through it and prints the output and truth
# values so that the accuracy of the model may be verified
def plot_corr(name, datas, labels="", to_weights=True, bin_size=20, outs=8):
    """This is the main function.  It runs the data through the net, calculates and plots the cumulants.
    
    Args:
        name: name of the model (minus .pth extension)
        datas: str or list of strings.  Names of datasets to be plotted.
        labels: str or list of strings.  This is if you want certain labels on the legend of the plots for each dataset.
        to_weights: bool or list of bools, whether or not each dataset should be weighted when calculating the cumulants.
        bin_size(int): size of multiplicity bins.
        outs(int): number of outputs on the NN."""
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
    
    # 0 - q2x
    # 1 - q2y
    # 2 - q4x
    # 3 - q4y
    # 4 - q6x
    # 5 - q6y
    # 6 - q8x
    # 7 - q8y
    
    fig, axs = plt.subplots(2, 2)
    
    for data_i in range(len(datas)):
        output_cor, truth_cor, event_weights, mults = collected_data[data_i][:4]
        to_weight = to_weights[data_i]
        
        q_out_vecs, q_truth_vecs = process_output_to_qvecs(output_cor, truth_cor, 8)
    
        corr_out, corr_truth, plot_mults, final_weights = process_qvecs_to_corrs(outs // 2, q_out_vecs, q_truth_vecs, event_weights, mults)
    
        """
        fig, axs = plt.subplots(1)
        fig.suptitle("Pythia event weights, 200k val")
        h = axs.hist2d(mults, event_weights, bins=50, range=((0, np.max(mults)), (np.min(event_weights), np.max(event_weights))), cmap="Purples", norm=mcolors.LogNorm())
        axs.set_xlabel("Multiplicity")
        axs.set_ylabel("Pythia event weights")
        axs.set_title(r"$\langle 2 \rangle$ weights")
        fig.colorbar(h[3], ax=axs)
        
        h = axs[1].hist2d(plot_mults[1], final_weights[1], bins=100, cmap="Purples")
        axs[1].set_xlabel("Multiplicity")
        axs[1].set_ylabel("Event weight \n (Pythia weight * Hijing combinatoric weight)")
        axs[1].set_title(r"$\langle 4 \rangle$ weights")
        fig.colorbar(h[3], ax=axs[1])
        
        
        fig.tight_layout()
        fig.savefig(f"weight_v_mult_{data_i}.png", dpi=500)
        """
    
        # 0 - q2
        # 1 - q4
        # 2 - q6
        # 3 - q8

        mult_bins = np.arange(0, 201, bin_size)
    
        corr_out_bins = bin_data(outs // 2, mult_bins, plot_mults, corr_out)
        corr_truth_bins = bin_data(outs // 2, mult_bins, plot_mults, corr_truth)
        weight_bins = bin_data(outs // 2, mult_bins, plot_mults, final_weights)

        # if data_i == 0:
            # plot_2corr_by_mult(mult_bins
        start=2
        cumul_out, cumul_err_out = calculate_cumul_and_error(corr_out_bins, weight_bins, to_weight)
        cumul_truth, cumul_err_truth = calculate_cumul_and_error(corr_truth_bins, weight_bins, to_weight)
        
        axs[0, 0].errorbar(mult_bins[start:-1], cumul_out[0, start:], yerr=cumul_err_out[0, start:], marker=".",
                           label=f"{labels[data_i]} output", ls="")
        axs[0, 0].errorbar(mult_bins[start:-1] + 2, cumul_truth[0, start:], yerr=cumul_err_truth[0, start:], marker=".", 
                           label=f"{labels[data_i]} truth", ls="")
        
        axs[1, 0].hist(mults, bins = 25, histtype="step")

        
        axs[0, 1].errorbar(mult_bins[start:-1], cumul_out[1, start:], yerr=cumul_err_out[1, start:], marker=".", ls="")
        axs[0, 1].errorbar(mult_bins[start:-1] + 2, cumul_truth[1, start:], yerr=cumul_err_truth[1, start:], marker=".", ls="")
        
        axs[1, 1].errorbar(mult_bins[start:-1], cumul_out[2, start:], yerr=cumul_err_out[2, start:], marker=".", ls="")
        axs[1, 1].errorbar(mult_bins[start:-1] + 2, cumul_truth[2, start:], yerr=cumul_err_truth[2, start:], marker=".", ls="")

        #plot_4corr_by_mult(mult_bins, corr_out_bins, corr_truth_bins)
        #print(cumul_out)
        #print(cumul_truth)
        #print(cumul_out - cumul_truth)

    fig.suptitle("Computing Cumulants by multiplicity")

    axs[0, 0].set_title("$c_2\{2\}$ output")
    axs[0, 0].set_xlabel("multiplicity")
    axs[0, 0].set_ylabel("$c_2\{2\}$")
    lgd = fig.legend(bbox_to_anchor=(1.02, 1))

    axs[1, 0].set_title("Multiplicity distribution")
    axs[1, 0].set_xlabel("Multiplicity")
    axs[1, 0].set_ylabel("N")

    axs[0, 1].set_title("$c_2\{4\}$ (cut low mult)")
    axs[0, 1].set_xlabel("multiplicity")
    axs[0, 1].set_ylabel("$c_2\{4\}$")

    axs[1, 1].set_title("$c_2\{6\}$ (cut low mult)")
    axs[1, 1].set_xlabel("multiplicity")
    axs[1, 1].set_ylabel("$c_2\{6\}$")
    
    fig.tight_layout()
    data = "_data_"
    for i in range(len(datas)):
        weight = ""
        if to_weights[i]:
            weight = "weighted"
        data += f"{datas[i]}_{weight}_"
    fig.savefig(f"images/{name}{data}part246.png", bbox_extra_artists=(lgd,), bbox_inches="tight", dpi = 500)

    
    
    # Plot corr by bin
    # dims = (4, 5)
    # fig, axs = plt.subplots(dims[0], dims[1], figsize=(20, 10))
    # 
    # fig.suptitle("2-particle correlators by multiplicity bin")
    # 
    # bin_i = 0
    # colors = list(mcolors.TABLEAU_COLORS.items())
    # 
    # for i in range(0, dims[0], 2):
    #     for j in range(dims[1]):
    #         if bin_i < len(mult_bins) - 1:
    #             
    #             axs[i, j].set_title(f"Multiplicity {mult_bins[bin_i]}-{mult_bins[bin_i+1]}, Output")
    #             
    #             axs[i, j].hist(corr_out_bins[0, bin_i], bins = 100, color=colors[bin_i][i // 2])
    #             axs[i, j].set_xlabel(r"$\langle 2 \rangle$")
    #             axs[i, j].set_ylabel("N")
    # 
    #             axs[i+1, j].hist(corr_truth_bins[0, bin_i], bins = 100, color=colors[bin_i][i // 2])
    #             axs[i+1, j].set_title("Truth")
    #             axs[i+1, j].set_xlabel(r"$\langle 2 \rangle$")
    #             axs[i+1, j].set_ylabel("N")
    # 
    #             bin_i += 1
    # 
    # fig.tight_layout()
    # fig.savefig(f"images/{name}_{data}_by_mult.png", dpi = 500)
    # 
    # fig, axs = plt.subplots(2, 2)
    # fig.suptitle("Computing 2, 4, 6, 8-Cumulants by multiplicity, 200k val")



def plot_2corr_by_mult(mult_bins, corr_out_bins, corr_truth_bins):
    """Generates a bunch of histograms of the 2-particle correlators in each multiplicity bin, so the distribution which is averaged to calculate the cumulant can be seen."""
    fig, axs = plt.subplots(2, 5, figsize=(10, 4))
    fig.suptitle("2-particle correlator by multiplicity bin. 200k flow")
    num_bins = len(mult_bins) - 1
    for bin_i in range(num_bins):
        ax = axs[bin_i // 5, bin_i % 5]
        _, out_bins, _ = ax.hist(corr_out_bins[0][bin_i], histtype="step", log=True, label="Output", bins=30)
        ax.hist(corr_truth_bins[0][bin_i], histtype="step", log=True, label="Truth", bins=out_bins)
        ax.set_ylabel("N")
        ax.set_xlabel(r"$\langle 2 \rangle$")
        if bin_i == 0:
            ax.legend()
        ax.set_title(f"Multiplicity {mult_bins[bin_i]}-{mult_bins[bin_i+1]}")

    
    fig.tight_layout()
    fig.savefig("images/2corr_mult_bins.png", dpi=500)

def plot_4corr_by_mult(mult_bins, corr_out_bins, corr_truth_bins):
    """Generates a bunch of histograms of the 4-particle correlators in each multiplicity bin, so the distribution which is averaged to calculate the cumulant can be seen."""
    fig, axs = plt.subplots(2, 5, figsize=(10, 4))
    fig.suptitle("4-particle correlator by multiplicity bin. 200k minbias val")
    num_bins = len(mult_bins) - 1
    for bin_i in range(num_bins):
        ax = axs[bin_i // 5, bin_i % 5]
        _, out_bins, _ = ax.hist(corr_out_bins[bin_i], histtype="step", log=True, label="Output", bins=30)
        ax.hist(corr_truth_bins[bin_i], histtype="step", log=True, label="Truth", bins=out_bins)
        ax.set_ylabel("N")
        ax.set_xlabel(r"$\langle 4 \rangle$")
        if bin_i == 0:
            ax.legend()
        ax.set_title(f"Multiplicity {mult_bins[bin_i]}-{mult_bins[bin_i+1]}")

    fig.tight_layout()
    fig.savefig("images/4corr_mult_bins.png", dpi=500)

def calculate_error(corrs, weight, order, to_weight=False):
    """Calculates error by subsampling.
    
    Args:
        corrs: """
    num_splits = 20
    num_data = len(corrs)
    
    corr_splits = [[] for _ in range(num_data)]
    weight_splits = [[] for _ in range(num_data)]

    #if len(weight) != num_data:
     #   weight = [weight for i in range(num_data)]

    for i in range(num_data):
        split_indices = np.array_split(np.arange(corrs[i].size), num_splits)
        
        for split_i in split_indices:
            corr_splits[i].append(np.array(corrs[i], dtype=np.float64)[split_i])
            weight_splits[i].append(np.array(weight[i], dtype=np.float64)[split_i])
    
    cumuls = []
    for i in range(num_splits):
        corr_split = [corr_splits[vec][i] for vec in range(num_data)]
        weight_split = [weight_splits[vec][i] for vec in range(num_data)]
        cumuls.append(calculate_cumul(corr_split, weight_split, order, to_weight))
    return np.std(cumuls) / np.sqrt(num_splits)

def calculate_cumul(corrs, weight, order, to_weight):
    """Calculates error by subsampling."""
    print(len(corrs))
    if len(corrs) == 2:
        corr2, corr4 = np.array(corrs[0], dtype=np.float64), np.array(corrs[1], dtype=np.float64)
    elif len(corrs) == 4:
        corr2, corr4, corr6, corr8 = np.array(corrs[0], dtype=np.float64), np.array(corrs[1], dtype=np.float64), \
            np.array(corrs[2], dtype=np.float64), np.array(corrs[3], dtype=np.float64)

    if not to_weight:
        weight = [np.ones(weight[i].shape) for i in range(len(corrs))]

    #if len(weight) != len(corrs]):
     #   weight = [weight for i in range(order // 2)]
    
    if order == 2:
        return np.average(corr2, weights = weight[0])
    elif order == 4:
        return np.average(corr4, weights = weight[1]) - 2 * np.average(corr2, weights = weight[0]) ** 2
    elif order == 6:
        return np.average(corr6, weights = weight[2]) - 9 * np.average(corr2, weights = weight[0]) * np.average(corr4, weights = weight[1]) \
        + 12 * np.average(corr2, weights = weight[0])**2
    elif order == 8:
        return np.average(corr2, weights = weight[0])


    
