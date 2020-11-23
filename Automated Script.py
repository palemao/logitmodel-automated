# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 11:29:41 2020

@author: palemao
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
#from sklearn.linear_model import LogisticRegression
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
Below I'm Using OVERSAMPLING to balnace the data. Use only if data is very unbalanced
'''
df_majority = data2[data2[y]==0]
df_minority = data2[data2[y]==1]

over_sampled_y = resample(df_minority,
                                 replace=True,     # sample with replacement
                                 n_samples=len(df_majority),    # to match majority class
                                 random_state=123)

inputed_dataset = pd.concat([df_majority, over_sampled_y], ignore_index=True)

'''
# Define variables for inputed dataset
'''
y_inputed = inputed_dataset[y]

x_inputed = [inputed_dataset[i] for i in x_list]

model_list_std_inputed = [LogisticRegressionCV(cv=5, max_iter=1000, solver='newton-cg', penalty='l2').fit(x_inputed[i], y_inputed) for i in range(len(x_inputed))]

predictedvalues_list_std_inputed = [pd.DataFrame(model_list_std_inputed[i].predict_proba(x_data_list[i]), columns=['No', 'Yes']) for i in range(len(x_data_list))]

 
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

#Training data ROC plot
plot_rocs(roc1_train, title='Train')

#Testing ROC  plot
plot_rocs(roc1_test, title='Test')

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
    
'''
Choosing any threshold you want
'''
list_of_tpr_tnr_train = [roc1_train[i].fpr_tnr_df() for i in range(len(roc1_train))]
list_of_tpr_tnr_test = [roc1_test[i].fpr_tnr_df() for i in range(len(roc1_test))]


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


    

    

 

 

 

'''

 

 

'''

#Log-Liklihood of models & Log-liklihood Test

 

'''

 

 

 

 

def loglike(model_from_LogisticRegressionCV, x_inputs, y_actual):
    predicted_prob = pd.DataFrame(model_from_LogisticRegressionCV.predict_proba(x_inputs), columns =['no', 'yes'])
    predicted_prob['actual'] = y_actual
    all_1s = predicted_prob[predicted_prob['actual']==1]
    all_1s = all_1s.drop(columns=['no', 'actual'])
    all_0s = predicted_prob[predicted_prob['actual']==0]
    all_0s = all_0s.drop(columns=['yes', 'actual'])

 

    correct_prob = pd.concat([all_0s, all_1s], ignore_index=True)

    correct_prob = correct_prob['no'].fillna(correct_prob['yes'])

    correct_prob = correct_prob.rename('relevant_prob')

 

    log_like = np.log(correct_prob)

    model_log_like= np.sum(log_like)

    return(model_log_like)

 

 

#loglike_model = loglike(model, x1, y)

 

#loglike_model1 = loglike(model1, x2, y)

 

#print('Restricted Model:' , loglike_model)

#print('UnRestricted Model:' , loglike_model1)

 

 

 

 

 

def loglikelyhoodtest(log_likehood_restricted, log_likehood_unrestricted, dof, p=None):

 

p = 0.95   

 #dof = len(log_likehood_unrestricted)  

 test_stat = -2*(log_likehood_restricted - log_likehood_unrestricted)

crit_value = chi2.ppf(p, dof)

confirmedp = chi2.cdf(crit_value, dof)

print("confirmed", confirmedp, "p-level")

 if test_stat < 0:

  modulus_v =  test_stat*-1

else:

  modulus_v = test_stat

print("test stat is", modulus_v)

if modulus_v >= crit_value:

    return("Unresticted model is statistically different to restricted model")

else:

    return("Unresticted model is NOT statistically different to restricted model")

 

 

 

#likelihoodtest = loglikelyhoodtest(loglike_model, loglike_model1, dof=3)

 

#print(loglikelyhoodtest(loglike_model, loglike_model1, dof=3))

 

 

 

 

 

from scipy.stats import norm

 

def logit_pvalue(model, x):

    """ Calculate z-scores for scikit-learn LogisticRegression.

    parameters:

        model: fitted sklearn.linear_model.LogisticRegression with intercept and large C

        x:     matrix on which the model was fit

    This function uses asymtptics for maximum likelihood estimates.

    """

    p = model.predict_proba(x)

    n = len(p)

    m = len(model.coef_[0]) + 1

    coefs = np.concatenate([model.intercept_, model.coef_[0]])

    x_full = np.matrix(np.insert(np.array(x), 0, 1, axis = 1))

    ans = np.zeros((m, m))

    for i in range(n):

        ans = ans + np.dot(np.transpose(x_full[i, :]), x_full[i, :]) * p[i,1] * p[i, 0]

    vcov = np.linalg.inv(np.matrix(ans))

    se = np.sqrt(np.diag(vcov))

    t =  coefs/se 

    p = (1 - norm.cdf(abs(t))) * 2

    return p

'''