Intermediate Goal: Markov Model


The goal of the project is to build a Markov Model classifier for determining if the given DNA sequence shows promoter activity or not. 
The code takes the order of the markov model, the number of folds for cross-validation and the sequences for training and testing as inputs. The user can also set the number of thresholds they want to use to plot to the roc-curve. 
The final output of the code is the roc curves for each cross-validation fold, the precision-recall curve for each cross-validation fold and the average area under the roc-curves for all cross validation folds. 

create_matrix.py

Creates the matrix for the markov model. The inputs take the order of the markov model, the sequence, for which to create the matrix. It then generates the row labels for the given order and then computes the log-likelihoods to fill in the matrix. 

