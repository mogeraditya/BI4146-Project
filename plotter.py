import numpy as np
import matplotlib.pyplot as plt
def given_threshold_get_tpr_fpr(score_array, threshold, testing_ids):
    prediction_array= ["a" for i in score_array]
    for prediction_index in range(len(prediction_array)):
        if score_array[prediction_index]<threshold:
            prediction_array[prediction_index]= "NULL"
        else:
            if score_array[prediction_index]==0:
                prediction_array[prediction_index]= "NULL"
            if score_array[prediction_index]>0:
                prediction_array[prediction_index]= "pos"
            if score_array[prediction_index]<0:
                prediction_array[prediction_index]= "neg"   
            
    # positive_counts= prediction_array.count("pos")
    # negative_counts= prediction_array.count("neg")
    # print(prediction_array)
    tp=0; fp=0; tn=0; fn =0
    for id in range(len(testing_ids)):
        if prediction_array[id]!= "NULL":
            if testing_ids[id]=="pos":
                if prediction_array[id]=="pos":
                    tp+=1
                if prediction_array[id]=="neg":
                    fn+=1
            if testing_ids[id]=="neg":
                if prediction_array[id]=="pos":
                    fp+=1
                if prediction_array[id]=="neg":
                    tn+=1
    try:
        tpr= tp/(tp+fn); fpr= fp/(fp+tn)
    except ZeroDivisionError:
        tpr, fpr= "NULL", "NULL"
        
    try:
        precision= tp/(tp+fp); recall= tp/(tp+fn)
    except ZeroDivisionError:
        precision, recall= "NULL", "NULL"
        
    return tpr, fpr, precision, recall
    
def get_metrics(positive_likelihoods, negative_likelihoods, resolution, testing_ids):
    score_array= np.array(positive_likelihoods)-np.array(negative_likelihoods)
    score_array= list(score_array)
    res= (max(score_array)-min(score_array))/resolution
    range_of_thresholds= np.arange(min(score_array),max(score_array)+res, res)
    store_tpr_fpr= []; store_prec_rec=[]
    for threshold in range_of_thresholds:
        tpr, fpr, precision, recall = given_threshold_get_tpr_fpr(score_array, threshold, testing_ids)
        if tpr!="NULL":
            store_tpr_fpr.append([tpr,fpr])
        if recall!="NULL":
            store_prec_rec.append([precision,recall])
    store_tpr_fpr= np.array(store_tpr_fpr)
    store_prec_rec= np.array(store_prec_rec)
    return store_tpr_fpr, store_prec_rec
def plot_roc_curve(store_tpr_fpr, order, c):
    # print(store_tpr_fpr)
    plt.scatter(store_tpr_fpr[:,1], store_tpr_fpr[:,0], label= "fold number" +str(c))
    plt.xlim(0,1); plt.ylim(0,1)
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("roc for order "+str(order))

def plot_roc_all(store_tpr_fpr_all, order, no_of_splits):
    for c in range (0, no_of_splits):
        plot_roc_curve(store_tpr_fpr_all[c], order, c)    
    x=[0,1]; y=[0,1]
    plt.plot(x,y,linestyle= "dashed")
    plt.legend()
    plt.savefig("order_"+str(order)+"TPR_FPR_plot_w_"+str(no_of_splits)+"_splits.png")
    
def plot_prec_rec_curve(store_prec_rec, order, c):
    # print(store_tpr_fpr)
    plt.scatter(store_prec_rec[:,1], store_prec_rec[:,0], label= "fold number" +str(c))
    plt.xlim(0,1); plt.ylim(0,1)
    plt.xlabel("RECALL"); plt.ylabel("PRECISION")
    plt.title("precision_recall curve for order "+str(order))

def plot_prec_rec_all(store_prec_rec_all, order, no_of_splits):
    for c in range (0, no_of_splits):
        plot_prec_rec_curve(store_prec_rec_all[c], order, c)    
    x=[0,1]; y=[0,1]
    plt.plot(x,y,linestyle= "dashed")
    plt.legend()
    plt.savefig("order_"+str(order)+"PREC_REC_plot_w_"+str(no_of_splits)+"_splits.png")    