#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import pyfastx
import csv
import argparse
import deeplift

from deeplift.conversion import kerasapi_conversion as kc

import numpy as np
import pandas as pd
import multiprocessing as mp
import logomaker as lm

def plotPromoters():
    
    ########################
    #command line arguments#
    ########################
    
    parser = argparse.ArgumentParser()

    #PARAMETERS
    parser.add_argument("--sequences",help="Full path to a fasta-file containing the promoter sequences.",type=str)
    parser.add_argument("--outdir",help="Full path to the output directory.")
    parser.add_argument("--N",help="How many references are used for averaging single signal sequence contributions.",type=int,default=10)
    parser.add_argument("--model",help="Full path to the trained keras model.",type=str,default=None)
    parser.add_argument("--background",help="Full path to a fasta-file containing the background sequences.",type=str)
    parser.add_argument("--target_layer",help="Target layer index for deeplift (default=-3).",type=int,default=-3)
    parser.add_argument("--ylim",help="Limits for y-axis.",type=float,nargs=2,default=None)
    parser.add_argument("--labels",help="Full path to a file containing labels used as figure titles. If not given, use fasta IDs.",type=str,default=None)
    parser.add_argument("--logoType",help="Logo image file extension (default=pdf).",type=str,default='pdf',choices=['png','pdf'])
    
    args = parser.parse_args()
    
    #reading in the promoter sequences
    ids = []
    signal = []
    signal_seq = []
    for seq in pyfastx.Fasta(args.sequences):
        ids.append(seq.name)
        signal_seq.append(str(seq.seq).upper())
    #and one-hot encoding
    for i in range(0,len(signal_seq)): signal.append(vectorizeSequence(signal_seq[i]))
    signal = np.array(signal)
    
    #reading in the background sequences
    bg = []
    for seq in pyfastx.Fasta(args.background): bg.append(str(seq.seq).upper())
    #and one-hot encoding
    for i in range(0,len(bg)): bg[i] = vectorizeSequence(bg[i])
    bg = np.array(bg)
    
    #reading in labels if given
    if args.labels!=None:
        labels = []
        f = open(args.labels,'rt')
        for row in f: labels.append(row)
        f.close()
    else: labels = ids
    
    #initialize the deeplift model
    deeplift_model = kc.convert_model_from_saved_files(args.model,nonlinear_mxts_mode=deeplift.layers.NonlinearMxtsMode.DeepLIFT_GenomicsDefault)
    find_scores_layer_idx = 0 #computes importance scores for inpur layer input
    deeplift_contribs_func = deeplift_model.get_target_contribs_func(find_scores_layer_idx=find_scores_layer_idx,target_layer_idx=args.target_layer)
    
    #and then score each sequence against args.N different background sequences
    scores = np.zeros(shape=(args.N,signal.shape[0],signal.shape[1]))
    
    for i in range(0,args.N):
        scores[i,:,:] = np.sum(deeplift_contribs_func(task_idx=1,input_data_list=[signal],input_references_list=[bg[:signal.shape[0],:,:]],batch_size=10,progress_update=None),axis=2)
        bg = np.roll(bg,1,axis=0)
        
    scores = np.mean(scores,axis=0)
    
    #now the contributions have been calculated, next plotting the sequence logos weighted by the contributions
    for ind in range(0,len(signal_seq)):
        #first plotting the sequence
        seq = signal_seq[ind]
        fig, ax = plt.subplots()
        matrix_df = lm.saliency_to_matrix(seq,scores[ind,:])#pd.DataFrame(scores[i,:])
        logo = lm.Logo(df=matrix_df,color_scheme='classic')
        logo.ax.set_xlabel('position')
        logo.ax.set_ylabel('contribution')
        title = labels[ind]
        logo.ax.set_title(title)
        if args.ylim!=None: logo.ax.set_ylim(args.ylim)
        plt.tight_layout()
        plt.savefig(args.outdir+ids[ind]+'.'+args.logoType,dpi=150,bbox_inches='tight',pad_inches=0) 
        plt.close(fig)
        plt.clf()
        plt.cla()
        
        #and then saving the importance scores to a file
        np.savetxt(args.outdir+ids[ind]+'.txt',scores[ind,:])

    #end

def vectorizeSequence(seq):
    # the order of the letters is not arbitrary.
    # Flip the matrix up-down and left-right for reverse compliment
    ltrdict = {'A':[1,0,0,0],'C':[0,1,0,0],'G':[0,0,1,0],'T':[0,0,0,1]}
    return np.array([ltrdict[x] for x in seq])

plotPromoters()
