from Bio import SeqIO
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import support_functions as sf
import plotter
import matplotlib.pyplot as plt
from datetime import datetime 

dir_data=os.getcwd()
os.chdir(dir_data+"\\data_set")
plot_dir= dir_data+"\\plots\\"

fasta_sequences_neg = SeqIO.parse(open("negative.fa"), 'fasta')
fasta_sequences_pos = SeqIO.parse(open("positive.fa"), 'fasta')
order=0; no_of_splits = 4; no_of_points_in_roc= 30

neg_seq, pos_seq, list_of_sequences, list_of_ids= sf.read_fasta(fasta_sequences_neg, fasta_sequences_pos)
training_sequences, training_ids, testing_sequences, testing_ids = sf.to_split_data(list_of_sequences, list_of_ids, no_of_splits)
negative_likelihoods_all, positive_likelihoods_all= [], []
tpr_fpr_all, prec_rec_all= [],[]
for c in range (0, no_of_splits):
    training_sequences_per_fold = training_sequences[c]
    training_ids_per_fold = training_ids[c]
    testing_sequences_per_fold = testing_sequences[c]
    testing_ids_per_fold = testing_ids[c]
    
    trained_neg_matrices, trained_averaged_neg_matrix, trained_pos_matrices, trained_averaged_pos_matrix, labels = sf.to_train_model(training_sequences_per_fold, training_ids_per_fold, order)
    negative_likelihoods, positive_likelihoods = sf.to_test_model(testing_sequences_per_fold, order, trained_averaged_neg_matrix, trained_averaged_pos_matrix, labels)
    negative_likelihoods_all.append(negative_likelihoods); positive_likelihoods_all.append(positive_likelihoods)
    
    store_tpr_fpr, store_prec_rec= plotter.get_metrics(positive_likelihoods, negative_likelihoods, no_of_points_in_roc, testing_ids[c])
    tpr_fpr_all.append(store_tpr_fpr); prec_rec_all.append(store_prec_rec)

os.chdir(plot_dir)
plotter.plot_roc_all(tpr_fpr_all, order, no_of_splits)
plotter.plot_prec_rec_all(prec_rec_all, order, no_of_splits)
