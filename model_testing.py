from sklearn.datasets import load_iris,load_breast_cancer
import pandas as pd
from sklearn.model_selection import train_test_split
import interpret
from interpret import show
from interpret.glassbox import ExplainableBoostingClassifier
import numpy as np
import einops
import glob
import time
import einops
from sklearn.metrics import accuracy_score,recall_score,roc_curve, classification_report,confusion_matrix,precision_score,roc_auc_score, auc, balanced_accuracy_score
import random
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from interpret.perf import ROC
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import resample
import kaleido
import plotly
import plotly.graph_objects as go



def kendall_tau(arr1, arr2):
    if len(arr1) != len(arr2):
        raise ValueError("Input arrays must have the same length.")    
    n = len(arr1)
    concordant_pairs = 0
    discordant_pairs = 0
    
    for i in range(n - 1):
        for j in range(i + 1, n):
            # Compare the order of elements in both arrays
            if (arr1[i] < arr1[j] and arr2[i] < arr2[j]) or (arr1[i] > arr1[j] and arr2[i] > arr2[j]):
                concordant_pairs += 1
            elif (arr1[i] < arr1[j] and arr2[i] > arr2[j]) or (arr1[i] > arr1[j] and arr2[i] < arr2[j]):
                discordant_pairs += 1
    
    tau = (concordant_pairs - discordant_pairs) / (0.5 * n * (n - 1))
    return tau


n_iterations = 100
#function to do the bootstraping
def bootstrap(a, b, calculate_statistic,name):
    # run bootstrap
    stats = list()
    for i in range(n_iterations):
        # prepare sample
        sample_a, sample_b = resample(a, b, stratify=a,  random_state=i)
        
        stat = calculate_statistic(sample_a, sample_b)
        stats.append(stat)
    average = np.average(stats)
    print(len(a),len(sample_a),str(calculate_statistic))
    metric = pd.DataFrame(stats)    
    metric.to_csv("result/{}.csv".format(name))
    # confidence intervals
    alpha = 0.95
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(stats, p))
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1.0, np.percentile(stats, p))
    print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))
    
    return [average,lower, upper]
    
    
    
#It is not possible to save the whole model and put it to the gitlab, so I saved the biomarkers output from the CNN part of the saved GL-ICNN, including the data and label of traing and testing sets.
label = pd.read_csv("/data/train_data_ADCN.csv")
data = pd.read_csv("/data/train_label_ADCN.csv")

X_train,y_train = data,label['diagnosis']

label = pd.read_csv("/data/test_data_ADCN.csv")
data = pd.read_csv("/data/test_label_ADCN.csv")


X_test,y_test = data,label['diagnosis']

train_size = len(y_train)
test_size = len(y_test)
acc_list = []
sen_list = []
spe_list = []

#model training
ebm = ExplainableBoostingClassifier(interactions = 0)
ebm.fit(X_train,y_train)

output_pro = ebm.predict_proba(X_test)
output_pro = output_pro[:,1]
output = ebm.predict(X_test)
print(list(output_pro))
print(list(y_test))

accuracy_list = []
t_list = []

accuracy_pre = 0
for i in range(0,100):
   t = 0.2 + (0.8-0.2)*i/100
   t_list.append(t)
   output = np.where(output_pro > t, 1, 0)
   accuracy = balanced_accuracy_score(y_test,output)
   accuracy_list.append(accuracy)
   accuracy_pre = accuracy
   output_pre = output 

t_choose = t_list[accuracy_list.index(max(accuracy_list))]
output = np.where(output_pro > t_choose, 1, 0)

#calculate metrics and CIs
ci_acc = bootstrap(y_test,output , accuracy_score,'V_acc_ADNI')
ci_ba = bootstrap(y_test,output , balanced_accuracy_score,'V_ba_ADNI')
ci_re = bootstrap(y_test,output , recall_score,'re_ADNI') 
ci_pre = bootstrap(y_test,output , precision_score,'pre_ADNI') 
ci_auc = bootstrap(y_test,output_pro , roc_auc_score,'V_auc_ADNI') 

matrix=confusion_matrix(y_test,output)
sensitivity = float(matrix[1][1])/np.sum(matrix[1])
specificity = float(matrix[0][0])/np.sum(matrix[0][0]+matrix[0][1])

print(ci_acc)
print(ci_ba)
print(ci_re)
print(ci_pre)
print(ci_auc)

auc=roc_auc_score( y_test,output_pro)
ba = balanced_accuracy_score(y_test,output)

print('auc score:',auc)
print('balanced accuracy:',ba)



###calculating feature importance
#calculate group-level feature importance
CI_importance = []
prediction_subject = ebm.eval_terms(X_train)
subject_contribution = abs(prediction_subject)

#calculate the CIs of group-level feature importance
for j in range(len(ebm.term_names_)):
    stats = []
    for i in range(n_iterations):
        # prepare sample
        sample_a, sample_b = resample(y_train, subject_contribution[:,j], stratify=y_train,  random_state=i)
        stat = np.average(sample_b,axis = 0)
        stats.append(stat)
    average = np.average(stats)
    alpha = 0.95
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(stats, p))
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1.0, np.percentile(stats, p))
    CI_importance.append((upper-lower)*0.5)
    
    print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))
    

subject_contributions = np.average(subject_contribution,axis = 0)
print('subject_contributions:',subject_contributions)

subjectimportances_index = np.argsort(-subject_contributions)
subjectimportances_index = subjectimportances_index.tolist()
print('subjectimportances:',subjectimportances_index)
print(list_original)



list_testset = []
k=0
for (term, importance, CI) in zip(names, subject_contributions, CI_importance):
    list_testset.append([term,importance,CI])
    k=k+1
    #print(f"Term {term} importance: {importance}")

list = pd.DataFrame(list_testset)
list= list.rename(columns={list.columns[0]:'Feature'})
list= list.rename(columns={list.columns[1]:'Importance'})
list= list.rename(columns={list.columns[2]:'Errorbar'})
list_original = list
group_list = list
list=list.sort_values('Importance',ascending=True,ignore_index=False)
list = list[-10:]
group_index = list.index

    
    
list_testimportance = np.array(list)
top_biomarkers = 3
significant_differ_tau= []
diagnosis_correct = []
tau_list = []
    
### Plot the group-level feature importance
f, ax = plt.subplots(figsize=(20, 10))
sns.set_color_codes("muted")
sns.barplot(x="Importance", y="Feature", data=list,label="EBM", color="blue", alpha=0.9, errorbar=('ci', 85))

#use sns to draw the plot
ax.legend(ncol=1, loc="lower right", frameon=True)
ax.set( ylabel="",
       xlabel="Mean Absolute Score")
sns.despine(left=True, bottom=True)
#plt.savefig('plot/ADCN_group.png', dpi=600)
 
#use plotly to draw the plot (preferred)    
#marker_color = 'red',or it can change to other colors like cornflowerblue
fig = go.Figure(go.Bar(x=list["Importance"], y=list["Feature"],error_x_array= list['Errorbar'],orientation='h', name = 'Feature Importances of EBM',marker_color = 'red'))
fig.update_layout(title=dict(
            text="Mean Absolute Feature Importance"),width=800,height=800,font=dict(size=15),template = 'plotly_white')
fig.write_image('plot/ADCN_group.png')



###Plot the individual-level feature importance
#draw the feature importance of one specific subject
subject_test_num = 0 
prediction_subject = ebm.eval_terms(X_test)
subject_contributions = prediction_subject[:,subject_test_num]
if output_pro[subject_test_num][1] >=0.5:
  prediction = 'AD'
if output_pro[subject_test_num][1] <0.5:
  prediction = 'CN'

if y_test[subject_test_num] >=0.5:
  label = 'AD'
if y_test[subject_test_num] <0.5:
  label = 'CN'

list_testset = [] 

for (term, importance, CI,group_importance) in zip(names, subject_contributions, CI_importance,group_list['Importance']):
    list_testset.append([term,importance,CI,group_importance])
    k=k+1

list = pd.DataFrame(list_testset)
list= list.rename(columns={list.columns[0]:'Feature'}) 
list= list.rename(columns={list.columns[1]:'Importance'}) 
list= list.rename(columns={list.columns[3]:'group_importance'}) 



list.index = group_list.index
print('group_index:',group_index)
list=list.sort_values('group_importance',ascending=True)
print('subject_index:',list.index)
list = list[-10:]

colors = []
for feature in list["Importance"]:
  print(feature)
  if feature <0:
    colors.append('blue')
    continue
  if feature >0:
    colors.append('orange')
    continue

title = 'Prediction:' + prediction+' ('+str(round(output_pro[subject_test_num][1],2))+ ')  '+'Reference label:'+label
fig = go.Figure(go.Bar(x=list["Importance"], y=list["Feature"],orientation='h', name = 'Feature Importances of EBM',marker_color = colors))
fig.update_layout(title=dict(
            text=title),width=800,height=800,font=dict(size=15),template = 'plotly_white')
fig.write_image('plot/individual_{}_ADCN.png'.format(subject_test_num))


#draw the feature importance of all subjects
#read subjectname of testing set
subjectlist_test = pd.read_csv("/data/test_data_ADCN.csv")
subjectlist_test = subjectlist_test['subjectname']
testdatalist = []

for i in range(len(subjectlist_test)):

   subject_contributions = prediction_subject[i,:]
   print('output_pro:',output_pro[i][1])
   
   
   list_testset = []

   for (term, importance, CI,group_importance) in zip(names, subject_contributions, CI_importance,group_list['Importance']):
    list_testset.append([term,importance,CI,group_importance])
    k=k+1
    

   list = pd.DataFrame(list_testset)
   list= list.rename(columns={list.columns[0]:'Feature'})
   list= list.rename(columns={list.columns[1]:'Importance'}) 
   list= list.rename(columns={list.columns[3]:'group_importance'}) 

   list.index = group_list.index
   list=list.sort_values('group_importance',ascending=True)
   list = list[-10:]
   
   
   if output_pro[i][1] >=0.5:
    prediction = 'AD'
   if output_pro[i][1] <0.5:
    prediction = 'CN'
   if y_test[i] >=0.5:
    label = 'AD'
   if y_test[i] <0.5:
    label = 'CN'
    
   colors = []
   for feature in list["Importance"]:
    if feature <0:
     colors.append('blue')
     continue
    if feature >0:
     colors.append('orange')
     continue
     
   title = 'Prediction:' + prediction+' ('+str(round(output_pro[i][1],2))+ ')  '+'Reference label:'+label
   fig = go.Figure(go.Bar(x=list["Importance"], y=list["Feature"],orientation='h', name = 'Feature Importances of EBM',marker_color = colors))
   fig.update_layout(title=dict(
            text=title),width=800,height=800,font=dict(size=15),template = 'plotly_white')
   fig.write_image('plot/{}_ADCN_{}.png'.format(subjectlist_test[i],label))






































    