import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from calculations import *
from subevents import *

import time
import numpy as np
from matplotlib import pyplot as plt
import uproot as ur
import os
from add_flow import *
from differential import *


def get_alphas(num):
    """Returns the unit circle split into num + 1 pieces.
    
    num(int): number of additional rotations to make."""
    
    return [i * 2 * np.pi / (num + 1) for i in range(num + 1)]



# -------------------------
# Data loading
# -------------------------

def get_data(code, start, end):
    """Extracts data from a file, indicated by code.
    
    Args:
        code(str): indicates which file should be opened.  Options are \"minbias\", \"hijing\", \"8160\", \"isomerge\", and \"pbpb\".
        start(int): event index to begin extraction.
        end(int): event index to end extraction.
        
    Returns: (num events, num particles) list of phi, pt, eta, and (num events) list of weights."""

    file = get_file(code)

    # For some reason the keys in the files are different.  Don't ask me, I'm just the messenger.
    if code in ("minbias", "8160", "isomerge"):
        tree = file["jet_tree;1"]
    
        vtrackphis = tree['vtrackphi'].array(entry_start=start, entry_stop=end)
        weights = tree['weight'].array(entry_start=start, entry_stop=end)
        pts = tree['vtrackpt'].array(entry_start=start, entry_stop=end)
        rapidity = tree['vtracketa'].array(entry_start=start, entry_stop=end)

    elif code in ("pbpb", "hijing"):
        tree = file["tree;1"]
    
        vtrackphis = tree['phi'].array(entry_start=start, entry_stop=end)
        weights = tree['weight'].array(entry_start=start, entry_stop=end)
        pts = tree['pt'].array(entry_start=start, entry_stop=end)
        rapidity = tree['eta'].array(entry_start=start, entry_stop=end)

    phi = [np.array(x) for x in vtrackphis]
    pt = [np.array(x) for x in pts]
    eta = [np.array(x) for x in rapidity]

    weights = weights / np.sum(weights)

    return phi, pt, eta, weights




# -------------------------
# Deep Sets sample builder
# -------------------------

def fake_pt(phi, pt):
    if len(phi) >= 60:
        for i in range(60):
            pt[-i] = 8
    return pt

def get_inputs(phi, eta, pt, ins):
    """This functions gets the most common inputs that I trained on.  (phi, pt, eta) for 3 inputs and (phi, cos(phi), sin(phi), pt, eta) for 5 inputs."""
    if ins == 3:
        x = np.stack([phi, pt, eta], axis=1)
    elif ins == 5:
        x = np.stack([
            phi,
            np.cos(phi),
            np.sin(phi),
            pt,
            eta,
            ], axis=1)
    return torch.tensor(x, dtype=torch.float32)

def get_differential_inputs(phi, eta, pt, poi_label, ref_label):
    x = np.stack([
        phi,
        np.cos(phi),
        np.sin(phi),
        pt,
        eta,
        poi_label,
        ref_label
    ], axis=1)
    return torch.tensor(x, dtype=torch.float32)


def get_truth(phi, outs):
    """This function is for the truth values when I was training on Q-vectors.  Every additional 2 inputs gets you another Q-vector output to train on."""
    n = 2
    truth = []
    while n <= outs:
        q = Qmoment(phi, n)
        truth.append(q.real)
        truth.append(q.imag)
        n += 2
    return torch.tensor(truth, dtype=torch.float32)
    

def build_deepset(ins, outs, phi, pt, eta, weight, true_cor2, true_cor4, subevents, calc_time, differential = False, poi_cut = (3, 5)):
    """Does all the calculation and assembles the tuple to be added to the dataset.
    
    Args:
        ins(int): number of inputs on the model
        outs(int): number of outputs on the model
        phi: array of all particle phis in an event
        pt: array of all pts in an event
        eta: array of all eta in an event
        weight: event weight
        true_cor2: <2> truth value.  I didn't actually use this, but it was in the code when I got it and I never bothered to change it.
        true_cor4: <4> truth value.  See above.
        subevents: Can be True, False, or 'Both'.
        calc_time: This is for doing time calculations.  It gets continually added to as the dataset is created.
        Differential: whether you are calculating differential cumulants or not.
        poi_cut: pt bin for labelling POIs.
        """
    
    x = get_inputs(phi, eta, pt, ins)
    # Q2, Q4, Q6, Q8, no subevents
    if subevents == "both":
        calc_start = time.time()
        y = torch.cat((torch.tensor(get_truth(phi, 4)), torch.tensor(get_subevent_qvecs(phi, eta, pt))))
        calc_end = time.time()
        calc_time += calc_end - calc_start
        
        mult_total = torch.unsqueeze(torch.tensor(x.shape[0], dtype=torch.float32), 0)
        _, mult_sube = get_subevents(phi, eta, pt, momentum_cut=0)
        mult_sube = torch.tensor(mult_sube)
        multiplicity = torch.cat((mult_total, mult_sube))
    elif not subevents:
        if differential:
            calc_start = time.time()
            y = get_differential_truth(phi, pt, poi_cut)
            calc_end = time.time()
            calc_time += calc_end - calc_start
            poi_label, ref_label, y = y[0], y[1], torch.tensor(y[2:])

            mult_total = torch.unsqueeze(torch.tensor(x.shape[0], dtype=torch.float32), 0)
            mult_diff = torch.tensor(get_differential_mults(phi, pt, poi_cut))
            multiplicity = torch.cat((mult_total, mult_diff))
            if 1234 not in y:
                x = get_differential_inputs(phi, eta, pt, poi_label, ref_label)
        else:
            calc_start = time.time()
            y = get_truth(phi, outs)
            calc_end = time.time()
            calc_time += calc_end - calc_start

            multiplicity = torch.tensor(x.shape[0], dtype=torch.float32)
    else:
        calc_start = time.time()
        y = torch.tensor(get_subevent_qvecs(phi, eta, pt), dtype=torch.float32)
        calc_end = time.time()
        calc_time += calc_end - calc_start
        
        _, multiplicity = get_subevents(phi, eta, pt, momentum_cut=0)
        multiplicity = torch.tensor(multiplicity)
            
            
        
    weight = torch.tensor([weight], dtype=torch.float32)

    momentum = torch.tensor(pt)

    return x, y, weight, multiplicity, momentum, calc_time


# -------------------------
# Dataset saving
# -------------------------

def save_dataset(samples, path):
    """Save a dataset at the given path with the name 'data.pt'."""
    os.makedirs(path, exist_ok=True)

    file_path = os.path.join(path, "data.pt")
    torch.save(samples, file_path)

    print(f"Saved dataset to {file_path}")


# -------------------------
# Main pipeline
# -------------------------

def assemble_dataset(ins, outs, phi, pt, eta, weight, cor2, cor4, alphas, subevents, differential, poi_bins):
    """Puts together dataset according to the given parameters.
    
    Args:
        ins: number of inputs on the NN.
        outs: number of outputs on the NN.
        data_i: index of event when parsing through data.
        counter: Index of how many events (including rotations) have been added to the dataset.
        num_events: number of events (including rotations) to include in the dataset.
        phi: phis of all events.  Lengths should be longer than the number of events needed, because low-multiplicity events are discarded when necessary.
        pt: pts of all events.  See phi.
        eta: rapidities of all events.  See phi.
        weight: weights of all evnets.  See phi.
        cor2: Truth value of cor2.  This isn't actually used, but it was in the code when I got it and I never got around to changing it.
        cor4: Truth value of cor4.  See cor2.
        alphas: list of rotations.  [0,] means the the dataset will be inputted raw.  [0, np.pi], for example, means the raw data will be inputted, and the data will be inputted a second time but with all phi rotated by pi.
        subevents: Can be False (no subevents), True(4 subevents), or 'both' (12 outputs: Q2, Q4 (no subevents, and Qa, b, c, d (4 subevents).
        differential: True (10 outputs: p2, q2, q4, Q2, Q4) for differential d2 calculations, or False (Q2, 4, 6, 8) for regular c2 calculations.
        poi_cut: if differential is True, this specificies the pt bin for POIs.  If None, a copy of each event will be put into each pt bin.
        
    Returns:
        dataset: the created dataset
        counter: the updated counter for number of events in the dataset
        data_i: the updated data index for parsing the data file."""
    dataset = []

    calc_time = 0

    counter = 0
    for i in range(len(phi)):
        if cor2[i] != 1234:

            if differential:
                num_bins = len(poi_bins) - 1
    
                for bin_i in range(num_bins):
                    poi = poi_bins[bin_i:bin_i + 2]
                    for alpha in alphas:
                        x, y, w, m, momentum, calc_time = build_deepset(ins, outs,
                            phi[i] + alpha,
                            pt[i],
                            eta[i],
                            weight[i],
                            cor2[i],
                            cor4[i],
                            subevents,
                            calc_time,
                            differential,
                            poi
                        )
                        
                        if 1234 not in y:
                            dataset.append([x, y, w, m, momentum])

    return dataset

def get_file(code):
    """Returns the root file given by the code.  Code options are for the files I used, which are 'minbias', 'pbpb', '8160', 'hijing', and 'isomerge'."""
    data_folder = "deepsets/Data"
    if code == "minbias":
        file = ur.open(f"{data_folder}/1merged_2pt (1).root")
    elif code == "pbpb":
        file = ur.open(f"{data_folder}/1pbpb_50k.root")
    elif code == "8160":
        file = ur.open(f"{data_folder}/com_8160_minpt_0_merged_tracks (2).root")
    elif code == "hijing":
        file = ur.open(f"{data_folder}/1p_pbHijing_1M.root")
    elif code == "isomerge":
        file = ur.open(f"{data_folder}/pp_isomerge_all.root")
    return file
        

def create_dataset(path, codes="minbias", begins = 0, ends=50000, ins = 5, outs = 8, alphas=[0,], add_flows=False, subevents=False, differential=False, poi_bins=[.3, .5, 1, 3, 5, 8], chunk_size=-1):
    """Main function to create and save dataset(s).
    
    path(str): folder name to save the dataset to.
    codes(str or strings): identifies the file(s) to draw from.  For this, begins, ends, and add_flows, if multiple are listed then multiple data files will be drawn from to create the dataset.  This is useful if for training, ie. have a dataset with both pPb and PbPb data mixed in it.
    begins(int or ints): index in the file at which to begin.
    ends(int or ints): index in the file at which to end.
    ins(int): number of inputs on the NN.
    outs(int): number of outputs on the NN.
    alphas: list of rotations.  [0,] means the the dataset will be inputted raw.  [0, np.pi], for example, means the raw data will be inputted, and the data will be inputted a second time but with all phi rotated by pi.
    subevents: Can be False (no subevents), True(4 subevents), or 'both' (12 outputs: Q2, Q4 (no subevents, and Qa, b, c, d (4 subevents).
    differential: True (10 outputs: p2, q2, q4, Q2, Q4) for differential d2 calculations, or False (Q2, 4, 6, 8) for regular c2 calculations.
    poi_bins(list): if differential is True, this specificies the pt bins for POIs.
    """
    if isinstance(ends, int):
        begins = (begins,)
        ends = (ends,)
        codes = (codes,)
        add_flows = (add_flows,)

    if chunk_size == -1: # All one dataset, no chunk
        dataset = []
        for i in range(len(codes)):
            code = codes[i]
            end = ends[i]
            begin = begins[i]
            add_flow = add_flows[i]
        
            phi, pt, eta, weight = get_data(code, begin, end)
            
            if add_flow:
                phi = get_flow(phi, code, begin, end)
    
            cor2, cor4 = get_truth_correlations(phi, pt)
    
            dataset += assemble_dataset(ins, outs, phi, pt, eta, weight, cor2, cor4, alphas, subevents, differential, poi_bins)
            #dataset +=

        save_dataset(dataset, "deepsets/deepset_data/" + path)

    else:
        for i in range(len(codes)):
            code = codes[i]
            end = ends[i]
            begin = begins[i]
            add_flow = add_flows[i]

            total_counter = 0
            data_i = begin
            chunk_i = 0
            num_total_events = (end - begin) * len(alphas)

            while total_counter < num_total_events:
                phi, pt, eta, weight = get_data(code, data_i, data_i + chunk_size // len(alphas))
                print(len(phi))
                if add_flow:
                    phi = get_flow(phi, code, begin, end)
                    
                cor2, cor4 = get_truth_correlations(phi, pt)

                dataset = create_dataset(ins, outs, phi, pt, eta, weight, cor2, cor4, alphas, subevents, differential, poi_bins)
                #print(len(dataset))
                save_dataset(dataset, f"deepsets/deepset_data/{path}_{chunk_i}")
        
                chunk_i += 1
                total_counter += chunk_size
                data_i += chunk_size // len(alphas)
                print(total_counter, data_i)

            print("total data size:", total_counter)

def get_flow(phi, code, begin, end):
    """Samples phi according to a distribution with flow."""
    flow_path = f"deepsets/flow_arrays/{code}_{begin}_to_{end}"
    array_path = flow_path + "/flowphi.npy"

    # Only generate once and saves so that it doesn't do the calculation over and over again.
    if not os.path.exists(array_path):
        phi = sample_flow_phis(phi)
        os.makedirs(flow_path, exist_ok=True)
        torch.save(phi, array_path)
    else:
        phi = torch.load(array_path)
    return phi
