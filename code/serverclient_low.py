import os
import time
import socket
import pickle
import torch
import threading
import parameter as para

txtlog=para.txtlog
ports=para.ports
datasets=para.datasets
client_num=para.client_num
round=para.round
server_port=para.server_port
strategy=para.strategy

finetune_sh_train=para.finetune_sh_train
finetune_sh_eval=para.finetune_sh_eval

t=time.localtime()
txtlog(str(t.tm_year)+'/'+str(t.tm_mon)+'/'+str(t.tm_mday))

def parameter_send(tensor_to_send,client_socket):
    tensor_bytes = pickle.dumps(tensor_to_send)
    parameter_size=str(len(tensor_bytes))
    parameter_size='0'*(20-len(parameter_size))+parameter_size
    client_socket.send(pickle.dumps(parameter_size))
    start_time=time.time()
    client_socket.send(tensor_bytes)
    end_time=time.time()
    return int(parameter_size),end_time-start_time
def parameter_recv(client_socket):
    tensor_size=int(pickle.loads(client_socket.recv(35)))
    tensor_bytes=b''
    red_byt=0
    start_time=time.time()
    while red_byt<tensor_size:
        tensorbyte=client_socket.recv(tensor_size-red_byt)
        tensor_bytes+=tensorbyte
        red_byt+=len(tensorbyte)
    end_time=time.time()
    tensor_received = pickle.loads(tensor_bytes)
    return tensor_received,tensor_size,end_time-start_time
def extract_para(port):
    param_dict=torch.load('./savedir/temp_para_old_'+str(port))
    if strategy==1:
        return {ke:param_dict[ke] for ke in param_dict if 'allset' in ke}
    elif strategy==0:
        return {ke:param_dict[ke] for ke in param_dict}

def parameter_non_sum(tensor_sum):
    para={}
    for i in tensor_sum[0]:
        para[i]=tensor_sum[0][i]
        for j in range(1,len(tensor_sum)):
            para[i]+=tensor_sum[j][i]
        para[i]=para[i]/len(tensor_sum)
    return para
def parameter_last_sum(tensor_sum):
    layer_lens=[len(i)//7 for i in tensor_sum]
    max_lens=max(layer_lens)
    lj=['base_model.model.model.layers.{layer}.self_attn.q_proj.lora_lT.default.weight',
        'base_model.model.model.layers.{layer}.self_attn.k_proj.lora_lT.default.weight',
        'base_model.model.model.layers.{layer}.self_attn.v_proj.lora_lT.default.weight',
        'base_model.model.model.layers.{layer}.self_attn.o_proj.lora_lT.default.weight',
        'base_model.model.model.layers.{layer}.mlp.gate_proj.lora_lT.default.weight',
        'base_model.model.model.layers.{layer}.mlp.up_proj.lora_lT.default.weight',
        'base_model.model.model.layers.{layer}.mlp.down_proj.lora_lT.default.weight']
    for i in range(max_lens):
        for lij in lj:
            avg_para=[]
            for ij,j in enumerate(layer_lens):
                if i-(max_lens-j)>=0:
                    avg_para.append(tensor_sum[ij][lij.format(layer=i-(max_lens-j))])
            for ij,j in enumerate(layer_lens):
                if i-(max_lens-j)>=0:
                    tensor_sum[ij][lij.format(layer=i-(max_lens-j))]=sum(avg_para)/len(avg_para)
def parameter_hetlora(tensor_sum):
    for i in tensor_sum[0]:
        if "lora_B" in i:
            max_lorar=max([tensor_sum[j][i].shape[1] for j in range(len(tensor_sum))])
            avg_para=sum([torch.nn.functional.pad(tensor_sum[j][i],(0,max_lorar-tensor_sum[j][i].shape[1])) for j in range(len(tensor_sum))])/len(tensor_sum)
            for j in range(len(tensor_sum)):
                tensor_sum[j][i]=avg_para@torch.pca_lowrank(avg_para.float(),tensor_sum[j][i].shape[1])[2].half()
        elif "lora_A" in i:
            max_lorar=max([tensor_sum[j][i].shape[0] for j in range(len(tensor_sum))])
            avg_para=sum([torch.nn.functional.pad(tensor_sum[j][i].t(), (0,max_lorar-tensor_sum[j][i].shape[0])) for j in range(len(tensor_sum))])/len(tensor_sum)
            for j in range(len(tensor_sum)):
                tensor_sum[j][i]=(avg_para@torch.pca_lowrank(avg_para.float(),tensor_sum[j][i].shape[0])[2].half()).t()



class ClientThread(threading.Thread):
    def __init__(self,client_socket,client_addr):
        threading.Thread.__init__(self)
        self.client_socket=client_socket
        self.client_addr=client_addr[0]+"/"+str(client_addr[1])
    def recv_param(self,r):
        tensor_received,tensor_size,spent_time= parameter_recv(self.client_socket)
        if r>=0:
            txtlog("[round:"+str(r)+"] server : receive; size: "+str(tensor_size)+"bytes, time: "+str(spent_time)+"s, client: "+self.client_addr)
        return tensor_received
    def send_param(self,tensor_to_send,r):
        parameter_size,spent_time= parameter_send(tensor_to_send,self.client_socket)
        if r>=0:
            txtlog("[round:"+str(r+1)+"] server : send; size: "+str(parameter_size)+"bytes, time: "+str(spent_time)+"s, client: "+self.client_addr)
class SCThread(threading.Thread):
    def __init__(self,port):
        threading.Thread.__init__(self)
        self.port=port
        if self.port==29500:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.bind(('localhost', server_port))
        else:
            self.client_socket={}
            for p in self.port:
                self.client_socket[p]=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.client_socket[p].connect(('localhost', server_port))
    def run(self):
        if self.port==29500:
            self.server_end()
        else:
            self.client_end()
    def server_end(self):
        self.server_socket.listen(client_num)
        clients={}
        while True:
            conn, addr = self.server_socket.accept()
            clients[addr]=conn
            if len(clients)>=client_num:
                client_thread=[ClientThread(clients[c],c) for c in clients]
                port_inf=[c.recv_param(-1) for c in client_thread]
                txtlog("**server client info**")
                for cln,cl in enumerate(client_thread):
                    txtlog("port: "+str(port_inf[cln])+" ;address: "+cl.client_addr+" ;dataset: "+datasets[port_inf[cln]-29500])
                txtlog("**FL start**")
                for r in range(round):
                    tensor_sum=[c.recv_param(r) for c in client_thread]
                    para=parameter_non_sum(tensor_sum)
                    for c in client_thread:
                        c.send_param(para,r)
                    # parameter_last_sum(tensor_sum)
                    # for ci,c in enumerate(client_thread):
                    #     c.send_param(tensor_sum[ci],r)
                for c in clients:
                    clients[c].close()
                break
        self.server_socket.close()
    def client_end(self):
        for p in self.port:
            _,_= parameter_send(p, self.client_socket[p])
        for r in range(round):
            for p in self.port:
                start_time=time.time()
                os.system(finetune_sh_train[p-ports[0]].format(round=r))
                end_time=time.time()
                txtlog("[round:"+str(r)+"] client :"+str(p)+" train_time: "+str(end_time-start_time)+"s")
                os.system(finetune_sh_eval[p-ports[0]].format(round=r))
                os.system("rm -rf ./savedir/llm_"+str(p)+"/*")
                tensor_to_send=extract_para(p)
                parameter_size,spent_time= parameter_send(tensor_to_send, self.client_socket[p])
                # txtlog("[round:"+str(r)+"] client : send; size: "+str(parameter_size)+"bytes, time: "+str(spent_time)+"s")
            for p in self.port:
                tensor_received,tensor_size,spent_time= parameter_recv(self.client_socket[p])
                torch.save(tensor_received, "./savedir/temp_para_new_"+str(p))
                # txtlog("[round:"+str(r+1)+"] client : receive; size: "+str(tensor_size)+"bytes, time: "+str(spent_time)+"s")
        for p in self.port:
            start_time=time.time()
            os.system(finetune_sh_train[p-ports[0]].format(round=r+1))
            end_time=time.time()
            txtlog("[round:"+str(r+1)+"] client :"+str(p)+" train_time: "+str(end_time-start_time)+"s")
            os.system(finetune_sh_eval[p-ports[0]].format(round=r+1))
            os.system("rm -rf ./savedir/llm_"+str(p)+"/*")
            self.client_socket[p].close

server_thread=SCThread(ports[0])
server_thread.start()
client_threads=SCThread(ports[1:])
client_threads.start()
    
