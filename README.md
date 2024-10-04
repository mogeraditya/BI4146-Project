# Intermediate Goal: Markov Model

## Overview
The goal of the project is to build a Markov Model classifier for determining if the given DNA sequence shows promoter activity or not. 
The code takes the order of the markov model, the number of folds for cross-validation and the sequences for training and testing as inputs. The user can also set the number of thresholds they want to use to plot to the roc-curve. 
The final output of the code is the roc curves for each cross-validation fold, the precision-recall curve for each cross-validation fold and the average area under the roc-curves for all cross validation folds. 

### create_matrix.py

Creates the matrix for the markov model. The inputs take the order of the markov model, the sequence, for which to create the matrix. It then generates the row labels for the given order and then computes the log-likelihoods to fill in the matrix. 

### support_functions.py

It takes the order and splits the data into k-many cross-validation folds (the number of splits and order can be inputted by user in the code). Then it trains the data and calculates the averaged likelihood matrix for negative data and positive data (i.e., sequences that do not show promoter activity and sequences that show promoter activity). Once the averaged matrices are obtained for a particular fold, the model then tests it on the split that was left out for the training and calculates the likelihood of the test sequence either showing promoter activity or not (by pulling values from the averaged matrix and simply adding them). 

### plotter.py

Predicts labels for each test sequence using the scores obtained from the test function in support_functions.py. Then it calculates the number of true positives, false positives, true negatives and false negatives and finally plots roc-curves, precision-recall curves and the average area under roc-curve for all folds. The number of thresholds to calculate the roc can be inputted by the user. 

### run_functions.py

It runs all the above functions. The values for number of splits, order of markov model and the number of thresholds to calculate roc can be inputted by user in this function. 
