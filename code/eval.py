import argparse
import json
import transformers
import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import parameter as para
from tqdm import tqdm
import re

class BatchDatasetLoader:
    def __init__(self, dataset: str, batch_size: int):
        self.inputs, self.outputs, self.tasks=[],[],[]
        for data in json.load(open(dataset, "r")):
            self.inputs.append(data['qestion'])
            self.outputs.append(data['answer'])
            self.tasks.append(data['task'])
        self.index = 0
        self.batch_size = batch_size
        self.length = len(self.inputs)
    def __len__(self):
        if self.batch_size == -1:
            return 1
        else:
            return self.length // self.batch_size
    def __getitem__(self, index):
        if self.batch_size == -1:
            if index >= self.__len__():
                raise StopIteration
            else:
                return self.inputs, self.outputs, self.tasks
        else:
            if self.length % self.batch_size == 0:
                if index >= self.__len__():
                    raise StopIteration
                else:
                    tmp_inputs, tmp_outputs, tmp_tasks = [], [], []
                    for i in range(index * self.batch_size, min((index + 1) * self.batch_size, self.length)):
                        tmp_inputs.append(self.inputs[i])
                        tmp_outputs.append(self.outputs[i])
                        tmp_tasks.append(self.tasks[i])
                    return tmp_inputs, tmp_outputs, tmp_tasks
            else:
                if index > self.__len__():
                    raise StopIteration
                else:
                    tmp_inputs, tmp_outputs, tmp_tasks = [], [], []
                    for i in range(index * self.batch_size, min((index + 1) * self.batch_size, self.length)):
                        tmp_inputs.append(self.inputs[i])
                        tmp_outputs.append(self.outputs[i])
                        tmp_tasks.append(self.tasks[i])
                    return tmp_inputs, tmp_outputs, tmp_tasks

def eval_model(args):
    datatset = para.datasets[args.port-para.ports[0]]
    data_path = para.test_path[args.port-para.ports[0]]
    lora_r=para.lora_rs[args.port-para.ports[0]]
    model_path =  para.models_path[args.port-para.ports[0]]
    few_shot = para.few_shot[datatset]
    if args.strategy==1:
        import overload_vllm 
        import overload_vllm2 
        overload_vllm.reset_fun() 
        overload_vllm2.reset_fun() 


    triggers = ['The answer is:', 'The answer is', 'the answer is']
    stop_tokens = ["USER:", "ASSISTANT:",  "### Instruction:", "Response:",
                   "\n\nProblem", "\nProblem", "Problem:", "<|eot_id|>", "####"]
    sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=512, stop=stop_tokens)

    llm = LLM(model=model_path, tensor_parallel_size=torch.cuda.device_count(), dtype=torch.bfloat16, trust_remote_code=True, enable_lora=True,max_lora_rank=lora_r)
    tokenizer = llm.get_tokenizer()

    questions, groundtruths, tasks = BatchDatasetLoader(data_path, -1)[0]
    if datatset=='program':
        questions = [q + " Let's write a program." for q in questions]
    tmp = "You are supposed to provide a solution to a given problem.\n\n"
    prefix = '\n' + 'Problem:\n{qestion}\nSolution:\n'
    tmp += '\n' + 'Problem:\n{qestion}\nSolution:\n{answer}\n'.format_map(few_shot)
    input_strs=[tmp+prefix.format(qestion=q) for q in questions]
    outputs = llm.generate(input_strs, sampling_params, lora_request=LoRARequest(str(args.port), 1, './savedir/llm_'+str(args.port)))
    outputs = [output.outputs[0].text for output in outputs]
    correct, wrong = 0, 0
    for output, groundtruth, task in zip(outputs, groundtruths, tasks):
        if task == 'blank_fill':
            pred = re.split('|'.join(triggers), output)[-1]
            pred = pred.strip('\n').rstrip('.').rstrip('/').strip(' ')
            pred = pred.replace(",", "")
            pred=re.findall(r'-?\d+/?\.?\d*', pred)
            if len(pred)>0:
                pred = pred[0]
                label = re.split('|'.join(triggers), groundtruth)[-1]
                label = label.strip('\n').rstrip('.').rstrip('/').strip(' ')
                label = label.replace(",", "")
                label = re.findall(r'-?\d+/?\.?\d*', label)[0]
                if label==pred:
                    correct+=1
                else:
                    wrong+=1
            else:
                wrong+=1
        elif task=='choice':
            pred = re.split('|'.join(triggers), output)[-1]
            pred = pred.strip('\n').rstrip('.').rstrip('/').strip(' ')
            pred=re.findall(r'\b(A|B|C|D|E)\b', pred.upper())
            if len(pred)>0:
                pred = pred[0]
                label = re.split('|'.join(triggers), groundtruth)[-1]
                label = label.strip('\n').rstrip('.').rstrip('/').strip(' ')
                label = re.findall(r'\b(A|B|C|D|E)\b', label.upper())[0]
                if label==pred:
                    correct+=1
                else:
                    wrong+=1
            else:
                wrong+=1
            
        elif task=='program':
            try:
                answer={}
                exec(output, answer)
                pred=answer['answer']

                answer={}
                exec(groundtruth, answer)
                label=answer['answer']
                if label==pred:
                    correct+=1
                else:
                    wrong+=1
            except:
                wrong+=1
        elif task in ['ssc','spc','nli']:
            if groundtruth.lower() in output.lower():
                correct+=1
            else:
                wrong+=1
            # print(output)          
    res_str="[round:"+str(args.round)+"] client: "+datatset+" result: "+str(correct)+"/"+str(correct+wrong)+"\n"
    with open('./FL_log.txt','a') as f:
        f.write(res_str)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=29500)
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--strategy", type=int, default=0)
    args = parser.parse_args()

    eval_model(args)
