# %%
import tensorflow as tf
import numpy as np
totClasses=8
alpha=10
beta=10

# %%
def MSEAndSCCLoss(y_true, y_pred):  
    scc=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    sccloss=scc(y_true[:,:1],y_pred[:,:totClasses])
    
    MSE=tf.keras.losses.MeanSquaredError()
    mseloss1=MSE(y_true[:,1:3],y_pred[:,totClasses:totClasses+2])
    mseloss2=MSE(y_true[:,3:],y_pred[:,totClasses+2:])
    mseloss=alpha*mseloss1+mseloss2
    totloss=beta*mseloss+sccloss
    return totloss

