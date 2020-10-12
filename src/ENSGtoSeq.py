#!/usr/bin/env python

import argparse
import csv
import gzip

from os import system

def ENSGtoSeq():
    
    ########################
    #command line arguments#
    ########################

    parser = argparse.ArgumentParser()

    #PARAMETERS
    parser.add_argument("--ids",help="Full path to the csv-file with ENSGG-ids. ENSG id first entry of each row.")
    parser.add_argument("--outdir",help="Full path to the output directory.",type=str)
    parser.add_argument("--ensembl",help="Full path to the gtf-file containing the coordinates for each Ensembl gene.",type=str)
    parser.add_argument("--genome",help="Full path to the human genome as fasta-file.",type=str)
    parser.add_argument("--prefix",help="Prefix to output file names (default=promoters).",type=str,default="promoters")
    args = parser.parse_args()
    
    #first read in the desired Ensembl IDs
    if args.ids[-3:]=='.gz': f = gzip.open(args.ids,'rt')
    else: f = open(args.ids,'rt')
    
    with f as csvfile:
        r = csv.reader(csvfile,delimiter=',')
        genelist = []
        for row in r:
            if row[0].count('ID')>0: continue
            genelist.append(row)

    #find the corresponding Ensembl gene coordinates:
    if args.ensembl[-3:]=='.gz': f = gzip.open(args.ensembl,'rt')
    else: f = open(args.ensembl,'rt')
    
    with f as csvfile:
        r = csv.reader(csvfile,delimiter='\t')
        genes = []
        for row in r:
            if row[0][0]=='#': continue
            feature_type = row[2]
            if feature_type!='gene': continue
            chrom = 'chr'+row[0]
            start = int(row[3])
            end = int(row[4])
            strand = row[6]
            attributes = row[8]
            for gene in genelist:
                if attributes.count(gene[0])>0:
                    genes.append([chrom,start,end,gene[0]+'_'+gene[1],1,strand])
                    break
    
    #save promoter coordinates as a bed-file
    with open(args.outdir+args.prefix+'.bed','wt') as csvfile:
        w = csv.writer(csvfile,delimiter='\t')
        for row in genes: w.writerow([row[0],row[1]-100,row[1]+20]+row[3:])
    
    #fetch the promoter sequences using bedtools getfasta
    system("bedtools getfasta -fi "+args.genome+" -bed "+args.outdir+args.prefix+'.bed'+" -fo "+args.outdir+args.prefix+'.fasta'+" -name -s")
    
#end

ENSGtoSeq()
