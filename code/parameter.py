server_port = 8080
client_num = 3
round = 4
strategy = 1

ports = [29500+i for i in range(client_num+1)]
# datasets = ['','blank_fill','choice','program','ssc','spc','nli']
datasets=['','program','program','program']

models=['','gemma2b','llama3b','smollm1_7b']
# models=['','blank_fill','choice','program']

layers=[1,18,28,24]
max_layer=max(layers[1:client_num+1])
epochs = [1 ,1 ,1 ,1 ]
batchs= [1, 2, 2, 2]
learn_rates=[0,2e-4,2e-4,2e-4]
lora_rs=[0, 16, 64, 128]
max_r=max(lora_rs[1:client_num+1])

models_path= ["../autodl-tmp/"+model for model in models]
train_path = ["./data/"+dataset+"_trainset_5k.json" for dataset in datasets]
test_path = ["./data/"+dataset+"_testset_500.json" for dataset in datasets]
# test_path = ["./data/"+dataset+"_10.json" for dataset in datasets]



finetune_sh_train=["sh ./code/finetune.sh "+str(p)+" "+train_path[p-ports[0]]+" "+models_path[p-ports[0]]+" {round} "+str(epochs[p-ports[0]])+" "+str(batchs[p-ports[0]])+" "+str(lora_rs[p-ports[0]])+" "+str(max_r)+" "+str(max_layer)+" "+str(strategy)+" "+str(learn_rates[p-ports[0]]) for p in ports]
finetune_sh_eval=["python ./code/eval.py --port "+str(p)+" --round {round} --strategy "+str(strategy) for p in ports]


def txtlog(log_str):
    with open('FL_log.txt','a') as f:
        f.write(log_str+"\n")

few_shot={
    'blank_fill':
    {'qestion':"Trent caught 180 tadpoles then let 75% of them go. How many did he keep?",
    'answer':"He released 180 x 75% = 135 tadpoles\nHe keeps 180 - 135 = 45 tadpoles\nThe answer is 45"},
    'choice':
    {'qestion':"5020−(502÷100.4)=?\nAnswer Choices: (A) 15 (B) 20 (C) 5015 (D) 25 (E) 35",
    'answer':"Let's think about the multi-choice question.\n=5020−(502/1004×10)\n=5020−5=5015\nThe answer is C"},
    'program':
    {'qestion':"1 / 0.08 is equal to ? Let's write a program.",
    'answer':"n0 = 1.0\nn1 = 0.08\n\nanswer = n0 / n1\nprint(answer)"},
    'ssc':
    {'qestion':"Analyze the sentiment of the following movie review, answer with Positive or Negative\nReview: entirely stale concept.",
    'answer':"Negative"},
    'spc':
    {'qestion':"Determine whether the following two questions ask the same thing, answer with yes or no.\nQuestion1 : How do I divide my time between physics and chemistry for NEET?\nQuestion2 : How do I divide my time between physics, chemistry and biology for NEET?",
    'answer':"No"},
    'nli':
    {'qestion':"Determine whether the given text contains the answer to a question, answering with yes or no.\nQuestion : Who wrote Argonautica?\nSentence : Another poet, Apollonius of Rhodes, attempted to revive the epic for the Hellenistic world with his Argonautica.",
    'answer':"Yes"}
}

PROMPT_TEMPLATE = [
    (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{qestion}\n\n### Response:"
    ),
    (
        "You are supposed to follow an instruction to generate proper response."
        "### Instruction:\n{qestion}\n\n### Response:"
    ),
    (
        "Please follow the instruction to give a response."
        "### Instruction:\n{qestion}\n\n### Response:"
    ),
    (
        "You are an expert, please listen to human instruction to generate the response.\n\n"
        "### Instruction:\n{qestion}\n\n### Response:"
    ),
    (
        "Let's follow the instruction to generate a response.\n\n"
        "### Instruction:\n{qestion}\n\n### Response:"
    ),
    (
        "The instruction is a description of the task. You need to follow that and respond.\n\n"
        "### Instruction:\n{qestion}\n\n### Response:"
    ),
    (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "Instruction:\n{qestion}\n\nResponse:"
    ),
    (
        "You are supposed to follow an instruction to generate proper response."
        "Instruction:\n{qestion}\n\nResponse:"
    ),
    (
        "Please follow the instruction to give a response."
        "Instruction:\n{qestion}\n\nResponse:"
    ),
    (
        "You are an expert, please listen to human instruction to generate the response.\n\n"
        "Instruction:\n{qestion}\n\nResponse:"
    ),
    (
        "Let's follow the instruction to generate a response.\n\n"
        "Instruction:\n{qestion}\n\nResponse:"
    ),
    (
        "The instruction is a description of the task. You need to follow that and respond.\n\n"
        "Instruction:\n{qestion}\n\nResponse:"
    ),
]

