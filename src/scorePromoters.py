#!/usr/bin/env python

import argparse
import keras
import tensorflow
import csv
import pyfastx

import numpy as np
import multiprocessing as mp

from keras.models import load_model

def scorePromoters():

    ########################
    #command line arguments#
    ########################

    parser = argparse.ArgumentParser()

    #PARAMETERS
    parser.add_argument("--outfile",help="Full path to the output directory.",type=str)
    parser.add_argument("--model",help="Full path to the .h5 file containing the model architecture.",type=str)
    parser.add_argument("--sequences",help="Full paths to a fasta-file containing the promoter sequences.",type=str)
    parser.add_argument("--nproc",help="Number of parallel processes used when scoring sequences (default=1).",type=int,default=1)
    args = parser.parse_args()
    
    #first read in all sequences from the input fasta-file
    seqs = [] #all promoter sequences
    labels = [] #IDs of promoter sequences as read from the Fasta file
    for seq in pyfastx.Fasta(args.sequences):
        seqs.append(str(seq.seq).upper())
        labels.append(seq.name)
        
    #score the sequences with the promoter model, use multiprocessing to parallelize
    pool = mp.Pool(args.nproc)
    ilist = []
    N = len(seqs)
    batch = int(N/args.nproc)
    for i in range(0,N,batch): ilist.append([i,i+batch])
    ilist[-1][1] = N

    res = [pool.apply_async(calcScore_batch,args=(seqs[i[0]:i[1]],args)) for i in ilist]
    res = [p.get() for p in res]
    pool.close()
    pool.join()
    pool.terminate()
    
    predicted_probabilities = list(res[0])
    if len(res)>1:
        for i in range(1,len(res)): predicted_probabilities += list(res[i])
    
    with open(args.outfile,'wt') as csvfile:
        w = csv.writer(csvfile,delimiter='\t')
        for i in range(0,len(labels)): w.writerow([labels[i],predicted_probabilities[i]])
    
#end

def calcScore_batch(seqs,args):
    #use the model to calculate predicted promoter score for sequences in seqs
    
    #first we need to one-hot encode each sequence so they can be scored by the model
    seqs_onehot = []
    for seq in seqs: seqs_onehot.append(vectorizeSequence(seq))
    
    #load the model from file
    loaded_model = load_model(args.model)
    
    #score the one-hot encoded sequences
    P = loaded_model.predict(np.stack(np.array(seqs_onehot)))[:,1]
    
    return P

def vectorizeSequence(seq):
    # the order of the letters is not arbitrary.
    # Flip the matrix up-down and left-right for reverse compliment
    ltrdict = {'A':[1,0,0,0],'C':[0,1,0,0],'G':[0,0,1,0],'T':[0,0,0,1]}
    return np.array([ltrdict[x] for x in seq])

scorePromoters()
