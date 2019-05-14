# HALK
Evaluation code for the paper "Investigating Extensions to Random Walk Based Graph Embedding", IEEE Services/ICCC 2019.

## Setup
The easiest way is to setup a conda environment with the provided environment.yml:
`conda env create -f environment.yml`

The evaluation is contained in the jupyter notebook [eval.ipynb](eval.ipynb). This notebook downloads and extracts the datasets first via shell commands and contains the evaluation pipeline afterwards. Simply run it cell by cell.

For HARP, the correct binary (sfdp_linux, sfdp_osx, sfdp_windows.exe) according to the operating system has to be specified upon training the embeddings. Preset is Linux.

Due to its size, the Youtube dataset requires up to 100GB of RAM during training the embeddings. Might be eased by storing the walks to disk and loading only a part of it, instead of keeping all in memory. The other datasets should not pose a problem for common desktop RAM sizes.

## Acknowledgement
The implementation is based on:
- https://github.com/GTmac/HARP (HARP reference implementation by the authors)
- https://github.com/aditya-grover/node2vec (node2vec reference implementation by the authors)
- https://github.com/phanein/deepwalk (deepwalk reference implementation by the authors)
- https://github.com/benedekrozemberczki/walklets (walklets)
- https://github.com/adocherty/node2vec_linkprediction (link prediction task)
