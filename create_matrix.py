from Bio import SeqIO
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
import pandas as pd

def create_matrix_labels_given_order( order):
    
    '''Returns the row labels for the matrix for given order
        
        Parameters
        ----------
        order: int
            The order that the user inputs
        
        Returns
        -------
        new_output: list
            The list of row labels for given order'''
    
    nucleotides= ["$", "A", "T", "G", "C"]
    counter=0
    if order==0:
        return ["NULL"]
    if order==1:
        return nucleotides
    else:
        past_output= create_matrix_labels_given_order(order-1)
        new_output= []
        for entries in past_output:
            for nucleotide in nucleotides:
                new_output.append(entries + nucleotide)
        return new_output
    
def create_matrix_given_order_zero(sequence):
    
    '''Returns the likelihood values for matrix for order given zero
    
        Parameters
        ----------
        sequence: str
            The input sequence for which we are to calculate the occurence matrix

        Returns
        -------
        matrix_for_order_zero: list
            The probabilities for order zero matrix'''
            
    counts_array, nucleotides= np.ones(shape=(4))*0.000001, ["A", "T", "G", "C"]
    total_count = 0
    for index in range(0, len(sequence)):
        get_index= nucleotides.index(sequence[index])
        counts_array[get_index]+=1
        total_count+=1
    matrix_order_zero= [i/total_count for i in counts_array]
    matrix_rows= ["NULL"]
    return matrix_order_zero, matrix_rows

def create_matrix_given_seq_and_order(sequence, order):
    
    '''Returns the occurence matrix as well as the labels for the rows for given sequence and order
        
        Parameters
        ----------      
        sequence: str
            The input sequence given to calculate matrix of given order
        order: int
            The order of the Markov model
        
        Returns
        -------   
        matrix: numpy matrix
            The occurence matrix for the order given
        matrix_rows: list
            The labels for the rows for given matrix'''
    
    sequence_prefix= "$"*(order-1)
    sequence= sequence_prefix+sequence
    # print(sequence)
    if order ==0:
        return create_matrix_given_order_zero(sequence)
    
    nucleotides= ["$", "A", "T", "G", "C"]
    matrix_rows= create_matrix_labels_given_order(order)
        
    matrix= np.zeros(shape= (len(matrix_rows), len(nucleotides)))
    matrix= 0.0001* np.ones(shape= (len(matrix_rows), len(nucleotides))) + matrix

    for char_index in range(order,len(sequence)):
        index_col= nucleotides.index(sequence[char_index])
        index_row= matrix_rows.index(sequence[char_index-order: char_index])
        matrix[index_row, index_col]+=1
    
    # denom= np.sum(matrix, axis=0)
    matrix= np.array(matrix)
    matrix= matrix/ matrix.sum(axis=0)
            
    return matrix, matrix_rows

