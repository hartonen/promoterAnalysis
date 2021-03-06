# 1. Installation

All scripts are pure Python, so no compiling is needed. Easiest way is to clone the repository to your own computer. In a desired location, type:

`git clone https://github.com/hartonen/promoterAnalysis.git`

The scripts in this repository need specific Python packages to function properly. The easiest way to make sure everything works is to (create a virtual environment) [https://docs.python.org/3/library/venv.html#module-venv] containing the tested versions of each package and then run the scripts in this environment. This is done by first creating a new virtual environment:

`python3 -m venv /path/to/new/virtual/environment`

Then one needs to install all the required packages. These are listed in `data/requirements.txt`. So activate the new virtual environment:

`source /path/to/new/virtual/environment/bin/activate`

and install the packages with pip:

`pip install -r data/requirements.txt`

# 2. Usage examples

Help message for each of the scripts can be invoked by calling the script with flag '-h'. For example, assuming installation path is `/path/to/repo/`:

`/path/to/repo/src/ENSGtoSeq.py -h`

Given a file with Ensembl gene ids on each line, one can fetch the corresponding promoter sequences from reference genome with `ENSGtoSeq.py`, for example:

`ENSGtoSeq.py --ids sc_shared_hits.csv --outdir sc_shared_hits/ --ensembl /ssd-raidz/ssd/thartone/wrk/genomes/hg19/Homo_sapiens.GRCh37.87.gtf.gz --genome /ssd-raidz/ssd/thartone/wrk/genomes/hg19/hg19.fa --prefix sc_shared_hits`

Given a fasta-file of promoter sequences and a Keras-model, the promoters are scored with `scorePromoters.py`. For example:

`scorePromoters.py --outfile sc_shared_hits/sc_shared_hits_pred_probs.txt --model ../TERT_promoters/model-199-0.87.h5 --sequences sc_shared_hits/sc_shared_hits.fasta --nproc 1`

A convenient way to include the predicted promoter probabilities in the title of the Deeplift sequence-logos is to use a file that defines the titles for each promoter:

`sed 's/\t/:/g' sc_shared_hits_pred_probs.txt > sc_shared_hits_labels.txt`

Finally, `plotPromoters.py` uses Deeplift and the Keras model to visualize importance of each promoter position on the prediction verdict. For example:

`plotPromoters.py --sequences sc_shared_hits/sc_shared_hits.fasta --outdir sc_shared_hits/ --N 100 --model ../TERT_promoters/model-199-0.87.h5 --background sc_background/sc_background_noN.fasta --labels sc_shared_hits/sc_shared_hits_labels.txt --logoType png`