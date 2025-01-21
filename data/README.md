[TOC]

# Introduction

Dataset1 (**D1**) : `MATHInstruct`.  [Download](https://huggingface.co/datasets/TIGER-Lab/MathInstruct). 

- Fill in the Blanks Questions(**FIBQ**) : CoT/aqua_rat
- Multiple Choice Questions(**MCQ**) : CoT/gsm_rft
- Programming Questions (**PQ**) : PoT/mathqa

Dataset2 (**D2**) : `GLUE` . [Download](https://gluebenchmark.com/tasks). 

- Single Sentence Classification (**SSC**) : SST-2
- Sentence Pair Classification (**SPC**) : QQP
- Natural Language Inference (**NLI**) : WNLI

# Process Code

## D1 Heterogeneous

```python
import json
with open('./MathInstruct.json') as file:
    datas = json.load(file)
# len(datas) 262039
datas_source={}
for i in datas:
    if i['source'] in datas_source:
        datas_source[i['source']].append([i['instruction'],i['output']])
    else:
        datas_source[i['source']]=[[i['instruction'],i['output']]]
# len(datas_source) 14

import re
trainset={'choice':[], 'blank_fill':[], 'program':[]}
testset={'choice':[], 'blank_fill':[], 'program':[]}
trainset_num,testset_num=5000,500
dataset_keys={'choice':'data/CoT/aqua_rat.json',
              'blank_fill':'data/CoT/gsm_rft.json',
              'program':'data/PoT/mathqa.json'}
#blank_fill
ANS_RE_blank_fill = re.compile(r"\nThe answer is (\-?[\$0-9\.\,]+)")
i_bf,j_bf=0,0
bf_data=datas_source[dataset_keys['blank_fill']]
while i_bf<trainset_num:
    if ANS_RE_blank_fill.search(bf_data[j_bf][1]):
        trainset['blank_fill'].append({'task':'blank_fill',
                        'qestion':bf_data[j_bf][0],
                        'answer':bf_data[j_bf][1]})
        i_bf+=1
    j_bf+=1
i_bf=0
while i_bf<testset_num:
    if ANS_RE_blank_fill.search(bf_data[j_bf][1]):
        testset['blank_fill'].append({'task':'blank_fill',
                        'qestion':bf_data[j_bf][0],
                        'answer':bf_data[j_bf][1]})
        i_bf+=1
    j_bf+=1
print(j_bf) #5500
#choice
ANS_RE_blank_fill = re.compile(r"\nThe answer is (\-?[\$A-Z\.\,]+)")
i_bf,j_bf=0,0
bf_data=datas_source[dataset_keys['choice']]
while i_bf<trainset_num:
    if ANS_RE_blank_fill.search(bf_data[j_bf][1]):
        trainset['choice'].append({'task':'choice',
                        'qestion':bf_data[j_bf][0],
                        'answer':bf_data[j_bf][1]})
        i_bf+=1
    j_bf+=1
i_bf=0
while i_bf<testset_num:
    if ANS_RE_blank_fill.search(bf_data[j_bf][1]):
        testset['choice'].append({'task':'choice',
                        'qestion':bf_data[j_bf][0],
                        'answer':bf_data[j_bf][1]})
        i_bf+=1
    j_bf+=1
print(j_bf) #7801
#program
i_bf,j_bf=0,0
bf_data=datas_source[dataset_keys['program']]
while i_bf<trainset_num:
    try:
        if 'scipy' not in bf_data[j_bf][1]:
            exec(bf_data[j_bf][1])
            trainset['program'].append({'task':'program',
                            'qestion':bf_data[j_bf][0],
                            'answer':bf_data[j_bf][1]})
            i_bf+=1
    except:
        0
    j_bf+=1
i_bf=0
while i_bf<testset_num:
    try:
        if 'scipy' not in bf_data[j_bf][1]:
            exec(bf_data[j_bf][1])
            testset['program'].append({'task':'program',
                            'qestion':bf_data[j_bf][0],
                            'answer':bf_data[j_bf][1]})
            i_bf+=1
    except:
        0
    j_bf+=1
print(j_bf) #5566
with open('./math_trainset_15k.json','w') as file:
    json.dump(trainset['ssc']+trainset['spc']+trainset['nli'],file,indent=4)
with open('./choice_trainset_5k.json','w') as file:
    json.dump(trainset['choice'],file,indent=4)
with open('./choice_testset_500.json','w') as file:
    json.dump(testset['choice'],file,indent=4)
with open('./choice_10.json','w') as file:
    json.dump(testset['choice'][:10],file,indent=4)
with open('./blank_fill_trainset_5k.json','w') as file:
    json.dump(trainset['blank_fill'],file,indent=4)
with open('./blank_fill_testset_500.json','w') as file:
    json.dump(testset['blank_fill'],file,indent=4)
with open('./blank_fill_10.json','w') as file:
    json.dump(testset['blank_fill'][:10],file,indent=4)
with open('./program_trainset_5k.json','w') as file:
    json.dump(trainset['program'],file,indent=4)
with open('./program_testset_500.json','w') as file:
    json.dump(testset['program'],file,indent=4)
with open('./program_10.json','w') as file:
    json.dump(testset['program'][:10],file,indent=4)
```

## D1 Homogeneous

```python
import json
with open('./MathInstruct.json') as file:
    datas = json.load(file)
# len(datas) 262039
datas_source={}
for i in datas:
    if i['source'] in datas_source:
        datas_source[i['source']].append([i['instruction'],i['output']])
    else:
        datas_source[i['source']]=[[i['instruction'],i['output']]]
import re
trainset={'choice1':[], 'choice2':[], 'choice3':[]}
testset={'choice1':[], 'choice2':[], 'choice3':[]}
ANS_RE_blank_fill = re.compile(r"\nThe answer is (\-?[\$A-Z\.\,]+)")
bf_data=datas_source['data/CoT/aqua_rat.json']
j_bf=0
for i in range(3):
    i_bf=0
    while i_bf<5000:
        if ANS_RE_blank_fill.search(bf_data[j_bf][1]):
            trainset['choice'+str(i+1)].append({'task':'choice',
                            'qestion':bf_data[j_bf][0],
                            'answer':bf_data[j_bf][1]})
            i_bf+=1
        j_bf+=1
    i_bf=0
    while i_bf<500:
        if ANS_RE_blank_fill.search(bf_data[j_bf][1]):
            testset['choice'+str(i+1)].append({'task':'choice',
                            'qestion':bf_data[j_bf][0],
                            'answer':bf_data[j_bf][1]})
            i_bf+=1
        j_bf+=1
print(j_bf)
with open('./choice1_trainset_5k.json','w') as file:
    json.dump(trainset['choice1'],file,indent=4)
with open('./choice2_trainset_5k.json','w') as file:
    json.dump(trainset['choice2'],file,indent=4)
with open('./choice3_trainset_5k.json','w') as file:
    json.dump(trainset['choice3'],file,indent=4)
with open('./choice1_testset_500.json','w') as file:
    json.dump(testset['choice1'],file,indent=4)
with open('./choice2_testset_500.json','w') as file:
    json.dump(testset['choice2'],file,indent=4)
with open('./choice3_testset_500.json','w') as file:
    json.dump(testset['choice3'],file,indent=4)
```

## D2 Heterogeneous

```python
import json
import pandas as pd
trainset_num,testset_num=5000,500
trainset={'ssc':[], 'spc':[], 'nli':[]}
testset={'ssc':[], 'spc':[], 'nli':[]}
ssc_train=pd.read_csv('./SST-2/train.tsv',nrows=trainset_num+1, sep='\t')
ssc_test=pd.read_csv('./SST-2/dev.tsv',nrows=testset_num, sep='\t')
spc_train=pd.read_csv('./QQP/train.tsv',nrows=trainset_num+1, sep='\t')
spc_test=pd.read_csv('./QQP/dev.tsv',nrows=testset_num, sep='\t')
nli_train=pd.read_csv('./QNLI/train.tsv',nrows=trainset_num+100, sep='\t')
nli_test=pd.read_csv('./QNLI/dev.tsv',nrows=testset_num, sep='\t')

for i in range(trainset_num):
    trainset['ssc'].append({'task':'ssc',
                            'qestion':'Analyze the sentiment of the following movie review, answer with Positive or Negative\nReview: '+ssc_train.loc[i].sentence,
                            'answer':'Positive' if ssc_train.loc[i].label==1 else 'Negative'})
    trainset['spc'].append({'task':'spc',
                            'qestion':'Determine whether the following two questions ask the same thing, answer with yes or no.\n'+'Question1: '+spc_train.loc[i].question1+'\nQuestion2: '+spc_train.loc[i].question2,
                            'answer':'Yes' if spc_train.loc[i].is_duplicate==1 else 'No'})
    trainset['nli'].append({'task':'nli',
                            'qestion':'Determine whether the given text contains the answer to a question, answering with yes or no.\n'+'Question: '+nli_train.loc[i].question+'\nSentence: '+nli_train.loc[i].sentence,
                            'answer':'Yes' if nli_train.loc[i].label=='entailment' else 'No'})
for i in range(testset_num):
    testset['ssc'].append({'task':'ssc',
                            'qestion':'Analyze the sentiment of the following movie review, answer with Positive or Negative\nReview: '+ssc_test.loc[i].sentence,
                            'answer':'Positive' if ssc_test.loc[i].label==1 else 'Negative'})
    testset['spc'].append({'task':'spc',
                            'qestion':'Determine whether the following two questions ask the same thing, answer with yes or no.\n'+'Question1: '+spc_test.loc[i].question1+'\nQuestion2: '+spc_test.loc[i].question2,
                            'answer':'Yes' if spc_test.loc[i].is_duplicate==1 else 'No'})
    testset['nli'].append({'task':'nli',
                            'qestion':'Determine whether the given text contains the answer to a question, answering with yes or no.\n'+'Question: '+nli_test.loc[i].question+'\nSentence: '+nli_test.loc[i].sentence,
                            'answer':'Yes' if nli_test.loc[i].label=='entailment' else 'No'})

with open('./ssc_trainset_5k.json','w') as file:
    json.dump(trainset['ssc'],file,indent=4)
with open('./ssc_testset_500.json','w') as file:
    json.dump(testset['ssc'],file,indent=4)
with open('./ssc_10.json','w') as file:
    json.dump(testset['ssc'][:10],file,indent=4)
with open('./spc_trainset_5k.json','w') as file:
    json.dump(trainset['spc'],file,indent=4)
with open('./spc_testset_500.json','w') as file:
    json.dump(testset['spc'],file,indent=4)
with open('./spc_10.json','w') as file:
    json.dump(testset['spc'][:10],file,indent=4)
with open('./nli_trainset_5k.json','w') as file:
    json.dump(trainset['nli'],file,indent=4)
with open('./nli_testset_500.json','w') as file:
    json.dump(testset['nli'],file,indent=4)
with open('./nli_10.json','w') as file:
    json.dump(testset['nli'][:10],file,indent=4)
with open('./glue_trainset_15k.json','w') as file:
    json.dump(trainset['ssc']+trainset['spc']+trainset['nli'],file,indent=4)
```

## D2 Homogeneous

```python
import json
import pandas as pd
ssc_data=pd.read_csv('./SST-2/train.tsv',nrows=16500, sep='\t')#5k*3  0.5k*3
trainset={'ssc1':[], 'ssc2':[], 'ssc3':[]}
testset={'ssc1':[], 'ssc2':[], 'ssc3':[]}
for i in range(3):
    idx=0
    for j in range(5000):
        trainset['ssc'+str(i+1)].append({'task':'ssc',
                            'qestion':'Analyze the sentiment of the following movie review, answer with Positive or Negative\nReview: '+ssc_data.loc[idx].sentence,
                            'answer':'Positive' if ssc_data.loc[idx].label==1 else 'Negative'})
        idx+=1
    for j in range(500):
        testset['ssc'+str(i+1)].append({'task':'ssc',
                            'qestion':'Analyze the sentiment of the following movie review, answer with Positive or Negative\nReview: '+ssc_data.loc[idx].sentence,
                            'answer':'Positive' if ssc_data.loc[idx].label==1 else 'Negative'})
        idx+=1
with open('./ssc1_trainset_5k.json','w') as file:
    json.dump(trainset['ssc1'],file,indent=4)
with open('./ssc2_trainset_5k.json','w') as file:
    json.dump(trainset['ssc2'],file,indent=4)
with open('./ssc3_trainset_5k.json','w') as file:
    json.dump(trainset['ssc3'],file,indent=4)
with open('./ssc1_testset_500.json','w') as file:
    json.dump(testset['ssc1'],file,indent=4)
with open('./ssc2_testset_500.json','w') as file:
    json.dump(testset['ssc2'],file,indent=4)
with open('./ssc3_testset_500.json','w') as file:
    json.dump(testset['ssc3'],file,indent=4)
```

