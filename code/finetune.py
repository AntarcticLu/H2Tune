# This code is based on the revised code from fastchat based on tatsu-lab/stanford_alpaca.


from dataclasses import dataclass, field
import json
import numpy as np
import os
from typing import Optional, List
import torch
from torch.utils.data import Dataset
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import transformers
from transformers import Trainer, deepspeed
from transformers.trainer_pt_utils import LabelSmoother
from peft import LoraConfig, get_peft_model

import random
import copy


IGNORE_TOKEN_ID = LabelSmoother.ignore_index

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="../Meta-Llama-3-8B")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = False
    round: int = 0
    strategy: int = 0
    max_r: int = 16
    forzen_state: int = 0
    max_layer: int = 32


@dataclass
class LoraArguments:
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    q_lora: bool = False


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)



def _tokenize_fn(strings,
                 tokenizer):
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )
def preprocess(
    sources,
    targets,
    tokenizer,
):
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_TOKEN_ID
    return dict(input_ids=input_ids, labels=labels)


import parameter as para
class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    def __init__(self, 
                 raw_data, 
                 tokenizer, 
                 max_len: int):
        super(SupervisedDataset, self).__init__()
        prompt_input = para.PROMPT_TEMPLATE[1]
        sources=[prompt_input.format_map(example) for example in raw_data]
        targets = [f"{example['answer']}{tokenizer.eos_token}" for example in raw_data]
        self.sources = sources
        self.targets = targets
    def __len__(self):
        return len(self.sources)
    def __getitem__(self, i):
        return dict(input_ids=self.sources[i], labels=self.targets[i])
    
@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer
    def __call__(self, instances):
        sources = []
        targets = []
        for instance in instances:
            source = instance['input_ids']
            target = instance['labels']
            sources.append(source)
            targets.append(target)
        data_dict = preprocess(sources, targets, self.tokenizer)
        input_ids, labels = data_dict['input_ids'], data_dict['labels']
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_TOKEN_ID)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )



def make_supervised_data_module(
        tokenizer, 
        data_args,
        max_len,):
    train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = SupervisedDataset(train_json,
                                    tokenizer=tokenizer, max_len=max_len)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    if training_args.strategy==1:
        if training_args.round>0:
            training_args.forzen_state=1
        else:
            training_args.forzen_state=0
        import overload_peft   
        overload_peft.reset_fun(training_args.max_r) 
        import overload_transformers
        overload_transformers.reset_fun() 

    local_rank = training_args.local_rank

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
        use_cache=False,  
        trust_remote_code=True
    )
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )

    tokenizer.pad_token_id = tokenizer.eos_token_id

    def find_all_linear_names(model):
        cls = torch.nn.Linear
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        if 'lm_head' in lora_module_names:  # needed for 16-bit
            lora_module_names.remove('lm_head')
        return list(lora_module_names)

    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=find_all_linear_names(model),
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    # Load data
    data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args, max_len=training_args.model_max_length
    )

    if training_args.strategy==1:
        import math
        model.num_hidden_layers=model.config.num_hidden_layers
        model.allset=torch.nn.Parameter(torch.randn(training_args.max_layer,7,training_args.max_r,training_args.max_r)).to(model.device)
        model.allset.requires_grad = False
        model.layer_proj=torch.nn.Parameter(torch.randn(training_args.max_layer,model.config.num_hidden_layers)).to(model.device)
        torch.nn.init.kaiming_uniform_(model.allset, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(model.layer_proj, a=math.sqrt(5))
    if training_args.round>0:
        loadlp={}
        if training_args.strategy==1:
            localpara=torch.load('./savedir/temp_para_old_'+training_args.output_dir[-5:])
            loadlp.update({k:localpara[k] for k in localpara if 'lora_A' in k})
            loadlp.update({k:localpara[k] for k in localpara if 'lora_B' in k})
            loadlp.update({k:localpara[k] for k in localpara if 'lora_mT' in k})
            loadlp.update({'base_model.layer_mask':localpara['layer_proj']})
            loadlp.update({'base_model.allset':torch.load('./savedir/temp_para_new_'+training_args.output_dir[-5:])['allset'].to(model.allset.device)})
        elif training_args.strategy==0:
            loadlp=torch.load('./savedir/temp_para_new_'+training_args.output_dir[-5:])
        # model.load_state_dict(torch.load('./savedir/temp_para_new_'+training_args.output_dir[-5:]),strict=False)
        model.load_state_dict(loadlp,strict=False)

    # print(model.model.model.layers[0].self_attn.q_proj.lora_lT.default.weight,"aa")
    if training_args.strategy==1:
        import re
        layer_matrix=(model.allset.permute(1,2,3,0)@model.layer_proj).permute(3,0,1,2)
        lT_data={}
        for k,_ in model.named_parameters():
            if 'lora_lT' in k:
                idx=int(re.findall(r'\d+',k)[0])
                if 'q_proj' in k:
                    lT_data[k]=layer_matrix.data[idx][0]
                elif 'k_proj' in k:
                    lT_data[k]=layer_matrix.data[idx][1]
                elif 'v_proj' in k:
                    lT_data[k]=layer_matrix.data[idx][2]
                elif 'o_proj' in k:
                    lT_data[k]=layer_matrix.data[idx][3]
                elif 'gate_proj' in k:
                    lT_data[k]=layer_matrix.data[idx][4]
                elif 'up_proj' in k:
                    lT_data[k]=layer_matrix.data[idx][5]
                elif 'down_proj' in k:
                    lT_data[k]=layer_matrix.data[idx][6]
        model.load_state_dict(lT_data,strict=False)
    # print(model.model.model.layers[0].self_attn.q_proj.lora_lT.default.weight,"bb")

    # print(model.model.model.layers[0].self_attn.q_proj.lora_mT.default.weight)
    # Start trainner
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module
    )

    with torch.autocast("cuda"):
        trainer.train()

    # print(model.model.model.layers[0].self_attn.q_proj.lora_T.default.weight)
    trainer.save_state()

    # print([k for k,p in model.named_parameters() if p.requires_grad])

    if trainer.args.use_lora:
        state_dict = get_peft_state_maybe_zero_3(
            trainer.model.named_parameters(), lora_args.lora_bias
        )
    else:
        state_dict = trainer.model.state_dict()
    
    if training_args.strategy==1:
        layer_matrix=torch.randn(model.num_hidden_layers,7,training_args.max_r,training_args.max_r)
        for k in state_dict:
            if 'lora_lT' in k:
                idx=int(re.findall(r'\d+',k)[0])
                if 'q_proj' in k:
                    layer_matrix[idx][0]=lT_data[k]
                elif 'k_proj' in k:
                    layer_matrix[idx][1]=lT_data[k]
                elif 'v_proj' in k:
                    layer_matrix[idx][2]=lT_data[k]
                elif 'o_proj' in k:
                    layer_matrix[idx][3]=lT_data[k]
                elif 'gate_proj' in k:
                    layer_matrix[idx][4]=lT_data[k]
                elif 'up_proj' in k:
                    layer_matrix[idx][5]=lT_data[k]
                elif 'down_proj' in k:
                    layer_matrix[idx][6]=lT_data[k]
        state_dict['allset']=(layer_matrix.permute(1,2,3,0)@model.layer_proj.t().detach().cpu().float()).permute(3,0,1,2)
        state_dict['layer_proj']=model.layer_proj.detach().cpu()
        

    if trainer.args.should_save and trainer.args.local_rank == 0:
        torch.save(state_dict,"./savedir/temp_para_old_"+training_args.output_dir[-5:])
        trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    train()
    with open('./FL_log.txt','a') as f:
        f.write('train done!\n')
