# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 13:48:44 2020

@author: Main
"""


#CHECK READ-ME FILE FOR EXPLAINATIONS





#Runs Logistic Regression Model (step2 readme)
model = LogisticRegression(max_iter=1000, solver='newton-cg', penalty='l2').fit(x_variables, y_treatment_dummy) 

# ROC Class 
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
    

#matching
def matching(table1, table2):
    if len(table1) > len(table2):
        treatment = table2
        control = table1
    else:
        treatment = table1
        control = table2
    absdiff =  [[abs(control['No'] - i)] for i in treatment['No']]
    chosen_index =[]
    for i in range(len(absdiff)):
       t = absdiff[i][0].drop(chosen_index)[absdiff[i][0].drop(chosen_index) == min(absdiff[i][0].drop(chosen_index))].index[0]
       chosen_index.append(t)
   
    treatment = treatment.reset_index()
    treatment_probs_with_control_index = control.iloc[chosen_index].reset_index()
    collated = treatment.merge(treatment_probs_with_control_index, right_index=True, left_index=True, left_on=treatment.index, right_on=control.index)
    collated1 = collated.drop(columns=['index_x', 'index_y'])
    collated1 = collated1.rename(columns={"identifeiableid_y": "identifeiableidControl", "identifeiableid_x": "identifeiableidTreatment", "No_x":"Prob_treament", "No_y":"Prob_control"})
    collated1['abs_diff'] = abs(collated1['Prob_treament'] -  collated1['Prob_control'])

    return collated1

#takes as input dfs of treatment patients, control patients, and all unmatched patients
def cohen_d(treatment_variable_table, matched_variable_table, unmateched_variable_table, desired_variables):
        nx = len(unmateched_variable_table)
        ny = len(treatment_variable_table)
        dof = nx + ny - 2
        unmatched_cohensd = [abs((mean(treatment_variable_table[i]) - mean(unmateched_variable_table[i])) / sqrt(((nx-1)*std(treatment_variable_table[i], ddof=1) ** 2 + (ny-1)*std(unmateched_variable_table[i], ddof=1) ** 2) / dof)) for i in desired_variables]
        matched_cohensd = [abs((mean(treatment_variable_table[i]) - mean(matched_variable_table[i])) / sqrt((std(treatment_variable_table[i], ddof=1) ** 2 + std(matched_variable_table[i], ddof=1) ** 2) / 2.0)) for i in desired_variables]
        d = { 'Variables': desired_variables,'matched_cohensd': matched_cohensd, 'unmatched_cohensd': unmatched_cohensd}
        df = pd.DataFrame(data=d)
        return df

#visualise cohen's d output to compare SMDs
