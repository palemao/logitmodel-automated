# Evaluation-of-an-Intervention-using-Control-Group
Evaluation of an Intervention using Control Group: 

treatment_group = people that used the intervention

untreated_group = people that did not use the intervention

control_group = people that did not use the intervention that are very similar to the treatment group

## Identify Control Group

Step 1: Identify a large range of variables of all people (treated and untreated) in our dataset (i.e. demographics, past behaviour etc.)

Step 2: First oversampling treatment group (if necessary), input treatment (actual or oversmapled) data and untreated group data into a logistic regression with 1=treated_person
0=untreated_person

    a: Plot ROC plot using *Use fpr_tnr_df* which outputs the true positive rate and true negative rate at different probability thresholds (*fpr_tpr_thresholds* outputs different probabiity threshold)
    
    b: Choose model with the most "right angled" ROC curve (has highest tnr and tpr for different probability estimates). The key trade off to understand: By adding more variables we could reduce the probabilty overlap between untreated and treated patients too much so there aren't enough valid matches, but reducing the the number of variables we could be reducing the quality of matches

Step 3: Use matching() function in repository to find matched patients based on logistic regression probability estimates
       
       a: There are differnt matching algorithms that can be used, I chose an optimal alogrithm  
        
Step 4: Use cohen_d() function in respository to test how good matched patients are compared to the wider dataset of untreated patients. 
      
      a: It calculates the standardised mean difference (SMD) of each variable for treatment group and untreated group, and SMD of treatment and control groups
      
      b: Its useful to plot "control_group" cohen's d and "untreated_group" cohen's d using a barchart to see how the step 1 to 3 have found a control group similar to treatment
      
      C: as a rule of thumb SMD below 0.1 of the control_group are good matches
 
## Use control group and treatment group to compare the objectives of the intervention

(i.e. look at outcomes before and after intervention for both group, to determine if there was a real impact from the intervention)
