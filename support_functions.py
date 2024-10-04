from Bio import SeqIO
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import create_matrix as cm

def read_fasta(fasta_sequences_neg, fasta_sequences_pos):
    neg_seq =[]
    pos_seq = []
    list_of_sequences = []
    list_of_ids = []
    for fasta in fasta_sequences_neg:
        name, sequence = fasta.id, str(fasta.seq)
        # print(name, sequence)
        neg_seq.append(sequence)
        list_of_sequences.append(sequence)
        list_of_ids.append('neg')
    for fasta in fasta_sequences_pos:
        name, sequence = fasta.id, str(fasta.seq)
        # print(name, sequence)
        
        pos_seq.append(sequence)
        list_of_sequences.append(sequence)
        list_of_ids.append('pos')
    return neg_seq, pos_seq, list_of_sequences, list_of_ids

def loglikelihood_given_matrix(sequence, order, matrix, matrix_rows):
    
    '''Retunrs the likelihood score for the given sequence and an averaged matrix
        
        Parameters
        ----------
        sequence: str
            The input sequence to calculate the likelihood score for
        order: int
            The order of the Markov Model
        matrix: numpy matrix
            The averaged negative or positive matrix used to calculate the likelihood of teh sequence belonging to either of the given matrices
        matrix rows: list
            The list of matrix row labels for given order
             
        Returns
        -------
        score_likelihood: float
            The likelihood of the sequence being derived from matrix'''
            
    sequence_prefix= "$"*(order-1)
    sequence= sequence_prefix+sequence
    nucleotides= ["A", "T", "G", "C"]
    score_loglikelihood=0
    if order==0:
        for char_index in range(order,len(sequence)):
            index_col= nucleotides.index(sequence[char_index])
            # print(index_col)
            score_loglikelihood+= matrix[index_col]
    else:  
        for char_index in range(order,len(sequence)):
            index_col= nucleotides.index(sequence[char_index])
            index_row= matrix_rows.index(sequence[char_index-order: char_index])
            score_loglikelihood+= matrix[index_row, index_col]
    return score_loglikelihood

def to_split_data(sequences_for_split, ids_for_split, no_of_splits):
    
    '''Returns the k-fold splits for a given k
        
        Parameters
        ----------  
        sequences_for_split: list
            The list of sequences to split into k folds
        ids_for_split: list
            The tag for each sequence, ie., whether it is a promoter ot not
        no_of_splits: int
            The input defined number of times cross-validation needs to occur
        
        Returns
        -------
            xtrain_all_folds: list
                The list of the training sequences for all folds
            ytrain_all_folds: list
                The training tags for all sequences for all folds
            xtest_all_folds: list
                The list of the testing sequences for all folds
            ytest_all_folds: list
                The testing tags for all sequences for all folds'''
                
    #defining the number of splits we want to make
    sgkf = StratifiedKFold(n_splits = no_of_splits, shuffle=True, random_state=42)
    xtrain_all_folds = []
    ytrain_all_folds = []
    xtest_all_folds = []
    ytest_all_folds = []
    fold_no = 0
    #generating the splits 
    for train_index, test_index in sgkf.split(sequences_for_split, ids_for_split):

        print(f'Train and test indices have been obtained for fold {fold_no}')

        xtrain, ytrain = np.take(sequences_for_split, train_index, axis = 0), np.take(ids_for_split, train_index)
        xtest, ytest = np.take(sequences_for_split, test_index, axis = 0), np.take(ids_for_split, test_index)
        # print(xtrain, ytrain, xtest, ytest)
        xtrain_all_folds.append(list(xtrain))
        ytrain_all_folds.append(list(ytrain))
        xtest_all_folds.append(list(xtest))
        ytest_all_folds.append(list(ytest))
        
        fold_no += 1
    return xtrain_all_folds, ytrain_all_folds, xtest_all_folds, ytest_all_folds

def to_train_model(sequences_for_training, ids_for_training, order):
    
    '''Returns the averaged trained positive and negative matrix
        
        Parameters
        ----------
        sequences_for_training: list
            The list of sequences for training per fold
        ids_for_training: list
            The tags for all seqeucnes for training per fold
        order: int
            The order of the Markov Model
        
        Returns 
        --------
        neg_matrices: numpy matrix
            All matrices computed for each sequence that does not show promoter activity
        average_neg_matrix: numpy matrix
            The average of all the compuated neg_matrices
        pos_matrices: numpy matrix
            All matrices computed for each sequence that does show promoter activity
        average_pos_matrix: numpy matrix
            The average of all the computed pos_matrices
        matrices_rows: list
            The list of row labels for given order of Markov Model'''
            
    neg_matrices = []; pos_matrices = []
    average_neg_matrix = []; average_pos_matrix = []
    finalDf_neg = []; finalDf_pos = []
    for it in range(len(sequences_for_training)):
        if ids_for_training[it]=="pos":
            finalDf_pos.append(sequences_for_training[it])
        else:
            finalDf_neg.append(sequences_for_training[it])
    print(len(finalDf_pos), len(finalDf_neg))
    for negative_indices in range(0, len(finalDf_neg)):
        matrices_neg, matrices_rows = cm.create_matrix_given_seq_and_order(finalDf_neg[negative_indices], order)
        neg_matrices.append(matrices_neg)
    for positive_indices in range(0, len(finalDf_pos)):
        matrices_pos, matrices_rows_pos = cm.create_matrix_given_seq_and_order(finalDf_pos[positive_indices], order)
        pos_matrices.append(matrices_pos)
    
    average_neg_matrix = np.average(neg_matrices, axis = 0)
    average_pos_matrix = np.average(pos_matrices, axis = 0)
    if order==0:
        matrices_rows= ["NULL"]
    return neg_matrices, average_neg_matrix, pos_matrices, average_pos_matrix, matrices_rows

def to_test_model(sequences_for_testing, order, trained_neg_matrix, trained_pos_matrix, labels_for_order):
    
    '''Returns the likelihoods of each test sequence belonging to either negative or positive matrix
        
        Parameters
        ----------
        sequences_for_testing: list
            The list of testing sequences per fold
        order: int
            The order of the Markov Model
        trained_neg_matrix: numpy matrix
            The averaged negative matrix from the Markov Model
        trained_pos_matrix: numpy matrix
            The averaged positive matrix from th Markov Model
        labels_for_order: list
            List of row labels for given order of model
        
        Returns
        -------
        testing_likelihoods_neg: list
            The list of all likelihoods of each sequence not showing promoter activity
        testing_likelihoods_pos: list
            The list of all likelihoods of each sequence showing promoter activity'''
            
    testing_likelihoods_neg = []
    testing_likelihoods_pos = []
    sequence_dataframe = pd.DataFrame(sequences_for_testing, columns= ['Sequences']) 
    for indices in range(0, len(sequence_dataframe['Sequences'])):
        likelihoods_neg = loglikelihood_given_matrix(sequence_dataframe['Sequences'][indices], order, trained_neg_matrix, labels_for_order)
        testing_likelihoods_neg.append(likelihoods_neg)
    for indices_1 in range(0, len(sequence_dataframe['Sequences'])):
        likelihoods_pos = loglikelihood_given_matrix(sequence_dataframe['Sequences'][indices_1], order, trained_pos_matrix, labels_for_order)
        testing_likelihoods_pos.append(likelihoods_pos)
    return testing_likelihoods_neg, testing_likelihoods_pos

def train_test_given_order_kfold(training_sequences, testing_sequences, testing_ids, c):
    training_sequences_per_fold = training_sequences[c]
    training_ids_per_fold = testing_ids[c]
    # print(type(training_sequences_per_fold), type(training_ids_per_fold))
    testing_sequences_per_fold = testing_sequences[c]
    testing_ids_per_fold = testing_ids[c]
    # print(type(testing_sequences_per_fold), type(testing_ids_per_fold))
    order=3
    trained_neg_matrices, trained_averaged_neg_matrix, trained_pos_matrices, trained_averaged_pos_matrix, labels = to_train_model(training_sequences_per_fold, training_ids_per_fold, order)
    negative_likelihoods, positive_likelihoods = to_test_model(testing_sequences_per_fold, order, trained_averaged_neg_matrix, trained_averaged_pos_matrix, labels)

# def to_split_data (sequences_for_split, ids_for_split, no_of_splits):
#     pos_sequences = []
#     neg_sequences = []
#     for id in range(0, len(sequences_for_split)):
#         if ids_for_split[id] == 'neg':
#             neg_sequences.append(sequences_for_split[id])
#         else:
#             pos_sequences.append(sequences_for_split[id])
    
#     size_of_split_neg, size_of_split_pos= int(np.floor(len(neg_sequences)/no_of_splits)), int(np.floor(len(pos_sequences)/no_of_splits))
#     list_of_ranges_pos=[list(np.arange(i, i+size_of_split_pos)) for i in range(no_of_splits)] #no shuffling
#     list_of_ranges_neg=[list(np.arange(i, i+size_of_split_neg)) for i in range(no_of_splits)] #no shuffling
#     training_for_all_folds = []; training_ids_for_all_folds = []; testing_for_all_folds = []; testing_ids_for_all_folds = [] #initialize the arrays
    
#     for i in range(no_of_splits):
        
#         training_for_fold, testing_for_fold= [], []
#         training_ids_for_fold = []
#         testing_ids_for_fold = []
#         testing_range_pos, testing_range_neg= list_of_ranges_pos[i], list_of_ranges_neg[i]
#         training_range_pos= [ind for ind in range(len(pos_sequences)) if ind not in testing_range_pos]
#         training_range_neg= [ind for ind in range(len(neg_sequences)) if ind not in testing_range_neg]
#         for r in training_range_neg:
#             training_for_fold.append(neg_sequences[r])
#             training_ids_for_fold.append('neg')
#         for x in testing_range_neg:
#             testing_for_fold.append(neg_sequences[x])
#             testing_ids_for_fold.append('neg')
#         for t in training_range_pos:
#             training_for_fold.append(pos_sequences[t])
#             training_ids_for_fold.append('pos')
#         for y in testing_range_pos:
#             testing_for_fold.append(pos_sequences[y])
#             testing_ids_for_fold.append('pos')
#         training_for_all_folds.append(training_for_fold)
#         training_ids_for_all_folds.append(training_ids_for_fold)
#         testing_for_all_folds.append(testing_for_fold)
#         testing_ids_for_all_folds.append(testing_ids_for_fold)
        
#         print("Done for Fold number "+str(i))
    
#     return training_for_all_folds, training_ids_for_all_folds, testing_for_all_folds, testing_ids_for_all_folds