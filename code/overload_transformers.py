import transformers
import torch
import torch.nn as nn
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
_is_peft_model=transformers.trainer._is_peft_model
MODEL_FOR_CAUSAL_LM_MAPPING_NAMES=transformers.trainer.MODEL_FOR_CAUSAL_LM_MAPPING_NAMES


def compute_loss(self, model, inputs, return_outputs=False, pred_value=None):
    """
    How the loss is computed by Trainer. By default, all models return the loss in the first element.

    Subclass and override for custom behavior.
    """
    if self.label_smoother is not None and "labels" in inputs:
        labels = inputs.pop("labels")
    else:   #this
        labels = None
    outputs = model(**inputs)
    # Save past state if it exists
    # TODO: this needs to be fixed and made cleaner later.
    if self.args.past_index >= 0:
        self._past = outputs[self.args.past_index]

    if labels is not None:
        unwrapped_model = self.accelerator.unwrap_model(model)
        if _is_peft_model(unwrapped_model):
            model_name = unwrapped_model.base_model.model._get_name()
        else:
            model_name = unwrapped_model._get_name()
        if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
            loss = self.label_smoother(outputs, labels, shift_labels=True)
        else:
            loss = self.label_smoother(outputs, labels)
    else:
        if isinstance(outputs, dict) and "loss" not in outputs:
            raise ValueError(
                "The model did not return a loss from the inputs, only the following keys: "
                f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
            )
        # We don't use .loss here since the model may return tuples instead of ModelOutput.

        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0] #True
        loss_norm=[]
        if self.args.forzen_state!=2:
            for k,para in model.named_parameters():
                if "lora_mT" in k:
                    loss_norm.append(torch.norm(para,p=2))
                    # mTsize=para.shape[0]
                if "lora_T" in k:
                    loss_norm.append(torch.norm(para,p=2))
                if "layer_proj" in k:
                    loss_norm.append(torch.norm(para,p=2))
            loss+=sum(loss_norm)/len(loss_norm)
        if self.args.forzen_state==1:
            new_loss=[]
            kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
            old_T=torch.load('../autodl-tmp/temp_para_old_'+str(self.args.output_dir[-5:]))
            old_T={ke:old_T[ke] for ke in old_T if 'lora_T' in ke}
            new_T={k:p for k,p in model.named_parameters() if 'lora_T' in k}
            for k in old_T:
                new_loss.append(kl_loss(torch.nn.functional.log_softmax(new_T['module.'+k].reshape(1,-1), dim=-1),
                                        torch.nn.functional.softmax(old_T[k].reshape(1,-1).to(new_T['module.'+k].device), dim=-1)))
            new_loss=sum(new_loss)/len(new_loss)
            loss+=new_loss

            return loss, outputs['logits'].detach()

        elif self.args.forzen_state==2:
            kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
            pred_logit=pred_value
            tc_logit=outputs['logits']
            
            new_loss=kl_loss(torch.nn.functional.log_softmax(tc_logit, dim=-1),
                             torch.nn.functional.softmax(pred_logit, dim=-1))
            loss+=new_loss
    return (loss, outputs) if return_outputs else loss

def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
    """
    Perform a training step on a batch of inputs.

    Subclass and override to inject custom behavior.

    Args:
        model (`nn.Module`):
            The model to train.
        inputs (`Dict[str, Union[torch.Tensor, Any]]`):
            The inputs and targets of the model.

            The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
            argument `labels`. Check your model's documentation for all accepted arguments.

    Return:
        `torch.Tensor`: The tensor with training loss on this batch.
    """
    model.train()
    if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
        self.optimizer.train()
    inputs = self._prepare_inputs(inputs)
    # print(model)
    if self.args.forzen_state==1:
        for k,p in model.named_parameters():
            if "lora_A" in k:
                p.requires_grad = False
            if "lora_B" in k:
                p.requires_grad = False
            if "lora_T" in k:
                p.requires_grad = True
            if "lora_mT" in k:
                p.requires_grad = True
            if "layer_proj" in k:
                p.requires_grad = True
        with self.compute_loss_context_manager():
            loss1,pred = self.compute_loss(model, inputs)
        kwargs = {}
        if self.args.n_gpu > 1:
            loss1 = loss1.mean()
        self.accelerator.backward(loss1, **kwargs)
        self.args.forzen_state=2
        for k,p in model.named_parameters():
            if "lora_A" in k:
                p.requires_grad = True
            if "lora_B" in k:
                p.requires_grad = True
            if "lora_T" in k:
                p.requires_grad = False
            if "lora_mT" in k:
                p.requires_grad = False
            if "layer_proj" in k:
                p.requires_grad = False
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, pred_value=pred)
        del inputs
        self.args.forzen_state=1
        kwargs = {}
        if self.args.n_gpu > 1:
            loss = loss.mean()
        self.accelerator.backward(loss, **kwargs)
    else:
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        del inputs
        kwargs = {}
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        self.accelerator.backward(loss, **kwargs)

    if self.args.forzen_state!=2:
        import re
        lT_data={}
        for k,p in model.named_parameters():
            if 'lora_T' in k:
                paranamelist=k.split('.')
                npn0='model'
                npn1='model'
                for paraname in paranamelist:
                    if paraname == 'lora_T':
                        npn0+=".lora_lT['default'].weight"
                        npn1+=".lora_pT['default']"
                        break
                    elif paraname.isdigit():
                        npn0+='['+paraname+']'
                        npn1+='['+paraname+']'
                    else:
                        npn0+='.'+paraname
                        npn1+='.'+paraname
                eval(npn0).data.reshape(-1)[eval(npn1)]=p.reshape(-1)
    return loss.detach() / self.args.gradient_accumulation_steps

def reset_fun():
    transformers.trainer.Trainer.compute_loss=compute_loss
    transformers.trainer.Trainer.training_step=training_step