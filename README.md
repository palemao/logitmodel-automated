# Testing-Intervention-using-Control-Group
Evaluation of an Intervention using Control Group: 

treatment_group = people that used the intervention

untreated_group = people that did not use the intervention

control_group = people that did not use the intervention that are very similar to the treatment group

## Identify Control Group

Step 1: Identify a large range of variables of all people (treated and untreated) in our dataset (i.e. demographics, past behaviour etc.)

Step 2: First oversampling treatment group (if necessary), input treatment (actual or oversmapled) data and untreated group data into a logistic regression with 1=treated_person
0=untreated_person

   a:USING ROC CLASS Plot ROC CURVE plot using *Use fpr_tnr_df* which outputs the true positive rate and true negative rate at different probability thresholds (*fpr_tpr_thresholds* outputs different probabiity threshold)
    
   b: Choose model with the most "right angled" ROC curve (has highest tnr and tpr for different probability estimates). The key trade off to understand: By adding more variables we could reduce the probabilty overlap between untreated and treated patients too much so there aren't enough valid matches, but reducing the the number of variables we could be reducing the quality of matches

Step 3: Use *matching()* function in repository to find matched patients based on logistic regression probability estimates
       
   a: There are differnt matching algorithms that can be used, I chose an optimal alogrithm  
        
Step 4: Use cohen_d() function in respository to test how good matched patients are compared to the wider dataset of untreated patients. 
      
   a: It calculates the standardised mean difference (SMD) of each variable for treatment group and untreated group, and SMD of treatment and control groups
      
   b: Its useful to plot "control_group" cohen's d and "untreated_group" cohen's d using a barchart to see how the step 1 to 3 have found a control group similar to treatment
      
   C: as a rule of thumb SMD below 0.1 of the control_group are good matches

look at outcomes before and after intervention for both group, to determine if there was a real impact from the intervention)

## Automated/streamlined logistic regression comparison script (Automated Script.py)

Import example dataset (data.xlsx) from the respository and run script to see script outputs  

### User Instructions

1. Import, clean and have data set up properly (use *dummyvariables()* function to create dummyvariables for all domain values of categorical columns and drop the "base" categorical value.

a. Ensure your cleaned df is saved in a variable called data2
 
      2. Inputs
      
      x_1 = ['CHAS', 'RM'] #model1
      x_2 = [ 'AGE', 'DIS'] #model2
      x_3 = ['DIS', 'RM', 'TAX', 'INDUS'] #model3
      
      #add more models if you have the computing power!
      x_list = [x_1, x_2, x_3] # put model variables in a list
      model_name_list = ['x_1', 'x_2', 'x_3'] # put string labels for each model
      y = 'highcrimerate'  # specify y column string
      
      hit run! 

3. Main Outputs: 
  
  a. pythonvariable *vif_results* outputs vif results of all models
ROC plots of training and testing data

  b. Predicted probabilities of training and testing data; *predictedvalues_list_std* and *predictedvalues_list_test* 
  
  c. Training and testing data ROCs plotted (see console, or plot tab)
  
  d. Training and testing data optimal threshold to maximise classification accuracy 
    d2. Cross-Validation Scores 
  
  e. Odd ratios of trained model
    
  

4. Useful notes

a. heatmap function is used to visualise multi-collinearity. It can run for all models in a list, it must be used for each model individually, unlike most outputs.

b. I aim to include statiscal inference to improve this




