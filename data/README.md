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

```python
# Dataset1

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

```python
# Dataset1

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
with open('./glue_trainset_15k.json','w') as file:
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

