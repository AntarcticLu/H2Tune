# H2Tune: Federated Foundation Model Fine-Tuning with Hybrid Heterogeneity

## Dataset

Please read `./data/README.md`.

## Parameter.py

Some important parameters can be modified in file `./code/parameter.py`.

```markdown
client_num : Number of clients
round  :  Number of federated learning rounds
datasets  :  Client dataset
models  :  Client LLM
layers  :  Number of LLM layers of the client
epochs  :  Number of LLM fine-tuning epochs of the client
batchs  :  Number of LLM fine-tuning batchs of the client
learn_rates  :  LLM fine-tuning learn rates of the client
lora_rs  :  Number of LLM fine-tuning lora ranks of the client
few_shot  :  Few-shots of LLM fine-tuning on the client
PROMPT_TEMPLATE  :  Prompt of LLM fine-tuning on the client
```

## Run

```shell
python ./code/serverclient_low.py
```

## Environment

```markdown
transformers  4.45.2
accelerate  1.0.0
deepspeed  0.15.1
peft  0.13.0
vllm  0.6.3
```



