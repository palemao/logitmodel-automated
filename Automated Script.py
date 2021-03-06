# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 11:29:41 2020

@author: palemao
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.utils import resample
from sklearn.metrics import roc_curve

'''download data'''

data2 = pd.read_excel(r'C:\Users\directory\data.xlsx') #change to correct directory 
data2 = data2.dropna().reset_index(drop=True)
'''
#Checking Correlation between independent variables & Multi-collinearity with VIF test
'''
def heatmap(x_variables):
    corr = x_variables.corr()
    sns.heatmap(corr, cmap='RdBu')
'''
VIF - TEST for MUlticollinearity if VIF stat > 5 then collinear
'''
def vif_test(x_variables):
    X_constant = sm.tools.add_constant(x_variables)
    VIF = pd.Series([variance_inflation_factor(X_constant.values, i) for i in range(X_constant.shape[1])], index=X_constant.columns)
    return (VIF)

def dummyvariables(dataset, column, attribute):  #creates dummy variables from categorical column
    intermediate = []
    for i in dataset[column]:
        if i == attribute:
            i = 1
            intermediate.append(i)
        else:
            i = 0
            intermediate.append(i)
  
    name = attribute +'flag'
    dataset[name] = intermediate

##################################################################################################################


'''
Define X_Variables models
'''
x_1 = ['CHAS', 'RM']
x_2 = [ 'AGE', 'DIS']
x_3 = ['DIS', 'RM', 'TAX', 'INDUS']

x_list = [x_1, x_2, x_3]

model_name_list = ['x_1', 'x_2', 'x_3']

y = 'highcrimerate'

yall_variable = data2[y]

train_test = [train_test_split(data2[i], yall_variable, test_size=0.2, random_state=42) for i in x_list]

x_data_list = [i[0] for i in train_test]
y_variable =  [i[2] for i in train_test]

x_data_list_test = [i[1] for i in train_test]
y_variable_test =  [i[3] for i in train_test]

#No major Multi - Collineraity found in this model so keep all variables
heatmap(x_data_list[0])
vif_results = [vif_test(x_data_list[i]) for i in range(len(x_data_list))]

'''Run Machine Learning Model with Cross - Validation'''
model_list_std = [LogisticRegressionCV(cv=5, max_iter=1000, solver='newton-cg', penalty='l2').fit(x_data_list[i], y_variable[i]) for i in range(len(x_data_list))]

#training data
predictedvalues_list_std = [pd.DataFrame(model_list_std[i].predict_proba(x_data_list[i]), columns = ['No', 'Yes']) for i in range(len(x_data_list))]

predictedvalues_table = [pd.merge(y_variable[i], predictedvalues_list_std[i], how='inner',
         left_index=True, right_index=True).drop(columns=['No']) for i in range(len(predictedvalues_list_std))]
    
#testing data 
predictedvalues_list_test = [pd.DataFrame(model_list_std[i].predict_proba(x_data_list_test[i]), columns = ['No', 'Yes']) for i in range(len(x_data_list_test))]

predictedvalues_table_test = [pd.merge(y_variable_test[i], predictedvalues_list_test[i], how='inner',
         left_index=True, right_index=True).drop(columns=['No']) for i in range(len(predictedvalues_list_test))]
    
######################################################################################################################

'''
Below I'm Using OVERSAMPLING to balance the data. Use only if data is very unbalanced
'''
tester = [pd.merge(x_data_list[i], y_variable[i], how='inner',  left_index=True, right_index=True) for i in range(len(train_test))]

df_majority = [i[i[y]==0] for i in tester]
df_minority = [i[i[y]==1] for i in tester]

over_sampled_y = [resample(df_minority[i],
                              replace=True,     # sample with replacement
                              n_samples=len(df_majority[i]),    # to match majority class
                              random_state=123) for i in range(len(df_minority))]

inputed_datasets = [pd.concat([df_majority[i], over_sampled_y[i]], ignore_index=True) for i in range(len(over_sampled_y))]

'''
# Define variables for inputed dataset
'''
y_inputed = [i[y] for i in inputed_datasets]

x_inputed = [i.drop(columns=[y]) for i in inputed_datasets] 

model_list_std_inputed = [LogisticRegressionCV(cv=5, max_iter=1000, solver='newton-cg', penalty='l2').fit(x_inputed[i], y_inputed[i]) for i in range(len(x_inputed))]

predictedvalues_list_test_ovr = [pd.DataFrame(model_list_std_inputed[i].predict_proba(x_data_list_test[i]), columns=['No', 'Yes']) for i in range(len(x_data_list_test))]

#ROC/AUC

class ROC():
    
    def __init__(self, test_y, predictedvalues):

            self.test_y = test_y
            self.predictedvalues = predictedvalues

    def optimal_threshold_fpr_tpr(self):

            fpr, tpr, thresholds2 = roc_curve(self.test_y, self.predictedvalues, pos_label=1)
            specificity = [1-i for i in fpr]
            sensitivity = tpr
            test = [specificity[i] + sensitivity[i] for i in range(len(specificity))]
            test = pd.Series(test)
            test = test.sort_values(ascending=False)
            test = test.reset_index()
            test = test.rename(columns={"index": "old_index"})

            optimal_threshold = thresholds2[test.old_index[0]]
            return  specificity[test.old_index[0]], sensitivity[test.old_index[0]],  optimal_threshold
        
    def fpr_tpr_thresholds(self):
            
        fpr, tpr, thresholds2 = roc_curve(self.test_y, self.predictedvalues, pos_label=1)
        return fpr, tpr, thresholds2

    def fpr_tnr_df(self):
           
        fpr, tpr, thresholds2 = roc_curve(self.test_y, self.predictedvalues, pos_label=1)
        tnr = [1-i for i in fpr]

        f = {'thresholds': thresholds2, 'tpr': tpr , 'tnr' : tnr}
        df2 = pd.DataFrame(data=f)

        return df2

roc1_train = [ROC(y_variable[i] ,predictedvalues_list_std[i]['Yes']) for i in range(len(y_variable))]
roc1_test = [ROC(y_variable_test[i] ,predictedvalues_list_test[i]['Yes']) for i in range(len(y_variable))]
roc1_test_ovr = [ROC(y_variable_test[i] , predictedvalues_list_test_ovr[i]['Yes']) for i in range(len(y_variable))]

'''
PLotting ROC Training & Testing
'''
def plot_rocs(roc_output, title='unknown'):
    list_of_roc_points = [roc_output[i].fpr_tpr_thresholds() for i in range(len(roc_output))]
    fpr_points = []
    tpr_points = []

    for i in list_of_roc_points:
        fpr_points.append(i[0])
        tpr_points.append(i[1])

    fig1, ax1 = plt.subplots()

    [ax1.plot(fpr_points[i], tpr_points[i]) for i in range(len(fpr_points))]

    ax1.legend(model_name_list, loc='lower right')
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title(title)

plot_rocs(roc1_train, title='Train')

plot_rocs(roc1_test, title='Test')

plot_rocs(roc1_test_ovr, title='Inputedmodel_Testdata')

'''Comparing optimal threshold'''

def optthreshold_andCVscore(roc, x_train_data_list):
    list_of_opt_std = [roc[i].optimal_threshold_fpr_tpr() for i in range(len(roc))]
    threshold_opt_std = []
    tpr_opt_std = []
    tnr_opt_std = []

    for i in range(len(list_of_opt_std)):
        threshold_opt_std.append(list_of_opt_std[i][2]), tnr_opt_std.append(list_of_opt_std[i][0]), tpr_opt_std.append(list_of_opt_std[i][1])
    summed_tprandtnr_std = [tpr_opt_std[i] + tnr_opt_std[i] for i in range(len(tpr_opt_std))]
    cvscores = [model_list_std[i].score(x_data_list[i], y_variable[i]) for i in range(len(x_train_data_list))]
    d_std = {'Model' : model_name_list, 'threshold_opt': threshold_opt_std, 'tpr_opt': tpr_opt_std, 
             'tnr_opt' : tnr_opt_std, 'Cross-Validation Accuracy': cvscores, 'summed_tprandtnr' : summed_tprandtnr_std}

    return pd.DataFrame(data=d_std)


optimal_train = optthreshold_andCVscore(roc1_train, x_data_list)
optimal_test = optthreshold_andCVscore(roc1_test, x_data_list)  
optimal_test_ovr = optthreshold_andCVscore(roc1_test_ovr, x_data_list)  
    
'''
Choosing any threshold you want
'''
list_of_tpr_tnr_train = [roc1_train[i].fpr_tnr_df() for i in range(len(roc1_train))]
list_of_tpr_tnr_test = [roc1_test[i].fpr_tnr_df() for i in range(len(roc1_test))]
list_of_tpr_tnr_test_ovr = [roc1_test_ovr[i].fpr_tnr_df() for i in range(len(roc1_test))]

'''
Coverting parameters to odds ratio and summarising
'''

def odds_ratios(model_list):
    coef_list  = [model_list[i].coef_ for i in range(len(model_list))]
    coef_list =  [i.reshape(-1,) for i in coef_list]
    odds_ratios = [np.exp(model_list[i].coef_) for i in range(len(model_list))]
    odds_ratios =  [i.reshape(-1,) for i in odds_ratios]
    d1 = [{'Variables' : x_list[i], 'odds ratio': odds_ratios[i], 'coef_list': coef_list[i]} for i in range(len(odds_ratios))]
    df1 = [pd.DataFrame(data=d1[i]) for i in range(len(d1))]
    return df1

odds_ratio_table = odds_ratios(model_list_std)
odds_ratio_table_ovr = odds_ratios(model_list_std_inputed) 


