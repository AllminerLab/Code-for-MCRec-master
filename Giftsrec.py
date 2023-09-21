import torch
import numpy as np
from torch.utils.data import Dataset
import scipy.sparse as sp
from time import time
import pickle
import gc
import os
import pandas as pd
import openpyxl
import itertools
from collections import Counter
import random
from tqdm import *
from torch.utils.data import DataLoader
from sklearn import metrics
from torch.nn.init import xavier_normal_, xavier_uniform_

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(set_seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_seed(2022)
print('torch.cuda.is_available(): {}'.format(torch.cuda.is_available()))

config={}
if torch.cuda.is_available():
    config['device']='cuda:0'
else:
    config['device']='cpu'

config['dataset_name']=''
config['data_path']=''
config['user_path']=''
config['dict_path']=''
config['i2i_wdict']=''
config['seq_data']=''


config['val_step']=1
config['patience']=5
config['result_topk']=5
config['train_ratio']=0.8
config['val_ratio']=0.1
config['test_ratio']=0.1
config['epoch']=50

config['item_num']=None
config['user_attr_num']=10 #number of attribution of users
config['max_gifts_len']=20 #maximum of session sequence
config['max_gifts_size']=20 #maximum of length of session
config['item_pad']=0 #item padding of session
config['item_holder']=1 #item holder of session, i.e., channel pad in paper
config['start_itemID']=2
config['candidate_channels']=['buy', 'copy', 'sale' , 'equip_train', 'role_train', 'gen_play']
config['channel2id_dict']=dict()
config['channel_buy_index']=None
config['channel_pad']=0 #channel padding of session
config['start_channelID']=1

config['lr']=0.01
config['weight_decay']=10e-5
config['batch_size']=32
config['dropout']=0.3

# setting hyper-parameters
config['item_id_embbed_dim']=64
config['item_attr_embbed_dim']=32
config['item_embbed_dim']=64
config['num_embbed_dim']=1
config['user_id_embbed_dim']=32
config['user_attr_embbed_dim']=16
config['user_embbed_dim']=128
config['gru_latent_dim']=64
config['gru_num_layers']=2
config['n_layers']=2
config['lam'] = 0.5
config['user_gifts_embbed_dim']=8


# %% [markdown]
# # Dataset

# %%
class LGCNDataset(Dataset):

    def __init__(self, config):
        self.config=config
        self.user2id_dict=None
        self.item2id_dict=None
        self.item_num=None
        self.dict_generation()
        print('dict_generation has been finish...')
        self.channels=self.item2item_weight()
        print('item2item_weight has been finish...')

    def dict_generation(self):
        if os.path.exists(self.config['dict_path']):
            with open(self.config['dict_path'],'rb') as f:
                self.user2id_dict = pickle.load(f)
                self.item2id_dict = pickle.load(f)
                self.item_num=len(self.item2id_dict)+self.config['start_itemID']
            print('{} already exists...'.format(self.config['dict_path']))
        else:
            all_data=pd.read_csv(self.config['data_path'])

            user_set=list(set(all_data['userID'].values))
            self.user2id_dict=dict(zip(user_set,list(range(1,len(user_set)+1))))

            item_set=list(set(all_data['itemID'].values))
            self.item_num=len(item_set)+self.config['start_itemID']
            self.item2id_dict=dict(zip(item_set,list(range(self.config['start_itemID'],self.item_num))))
            
            with open(self.config['dict_path'],'wb') as f:
                pickle.dump(self.user2id_dict,f)
                pickle.dump(self.item2id_dict,f)
            print('{} is saved...'.format(self.config['dict_path']))

    
    def item2item_weight(self):
        if os.path.exists(self.config['i2i_wdict']):
            with open(self.config['i2i_wdict'],'rb') as f:
                channels=pickle.load(f)
            print('{} already exists...'.format(self.config['i2i_wdict']))
        else:
            all_data=pd.read_csv(self.config['data_path'])
            all_data=all_data.groupby(['userID'],as_index=False).apply(lambda x:x[x['giftsID']<x['giftsID'].max()]).reset_index(drop=True)
            cg_inter_dict=all_data.groupby('channelID').groups
            channels=cg_inter_dict.keys()
            with open(config['i2i_wdict'], 'wb') as f:
                pickle.dump(list(channels),f)
                for c in channels:
                    i2i_wdict=Counter()
                    c_data=all_data.iloc[cg_inter_dict[c]][['userID','itemID', 'giftsID']].reset_index(drop=True)
                    cug_data_dict=c_data.groupby(['userID','giftsID']).groups
                    for cug in cug_data_dict.keys():
                        cug_data=c_data.iloc[cug_data_dict[cug]]
                        t_counter=Counter(sorted(list(itertools.product(list(map(lambda a:self.item2id_dict[a], cug_data['itemID'])),repeat=2))))
                        i2i_wdict.update(t_counter)
                    pickle.dump(i2i_wdict,f)
            print('{} is saved...'.format(self.config['i2i_wdict']))
        return channels
    
    
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

        
    def getI2IGraphs(self):
        i2i_Graphs=[]
        with open(self.config['i2i_wdict'],'rb') as f:
            channels=pickle.load(f)
            for _ in channels:
                i2i_mat = sp.dok_matrix((self.item_num, self.item_num), dtype=np.float32)
                i2i_wdict=pickle.load(f)
                for k in i2i_wdict.keys():
                    if k[0]==k[1]:
                        i2i_mat[k[0],k[1]]=1
                    else:
                        i2i_mat[k[0],k[1]]=i2i_wdict[k]
                        i2i_mat[k[1],k[0]]=i2i_wdict[k]
                
                rowsum = np.array(i2i_mat.sum(axis=1))+10e-24
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_i2i = d_mat.dot(i2i_mat)
                norm_i2i = norm_i2i.dot(d_mat)
                norm_i2i = norm_i2i.tocsr()

                Graph = self._convert_sp_mat_to_sp_tensor(norm_i2i)
                Graph = Graph.coalesce()
                i2i_Graphs.append(Graph)
        return torch.stack(i2i_Graphs,dim=0).to(self.config['device'])

# %%
class GiftsDataset(Dataset):
    def __init__(self, config:dict, dataset:LGCNDataset, mode='train'):
        self.config=config
        self.dataset=dataset
        self.seq_data=None
        self.user_set=list(self.dataset.user2id_dict.keys())
        self.user_set.sort()
        if mode=='train':
            self.user_set=self.user_set[:int(len(self.user_set)*self.config['train_ratio'])]
        elif mode=='val':
            self.user_set=self.user_set[int(len(self.user_set)*self.config['train_ratio']):int(len(self.user_set)*(self.config['train_ratio']+self.config['val_ratio']))]
        elif mode=='test':
            self.user_set=self.user_set[-int(len(self.user_set)*self.config['test_ratio']):]
        else:
            raise ValueError("Wrong mode! Please select a mode in \{ train, test\}.")
        if os.path.exists(self.config['seq_data']):
            with open(self.config['seq_data'],'rb') as f:
                self.seq_data=pickle.load(f)
            print('{} already exists...'.format(self.config['seq_data']))
        else:
            all_data=pd.read_csv(self.config['data_path'])
            self.seq_data=all_data.groupby(['userID']).apply(
                    lambda x:x.groupby(['giftsID']).apply(
                    lambda x:x[['userID','itemID','itemNum','channelID']].values))
            with open(self.config['seq_data'],'wb') as f:
                pickle.dump(self.seq_data,f)
            print('{} is saved...'.format(self.config['seq_data']))
        
        self.user_attrs=pd.read_csv(self.config['user_path'])

        
    def __getitem__(self, index):
        gifts_seq=list(map(lambda x:list(map(lambda y:tuple([self.dataset.user2id_dict[y[0]], self.dataset.item2id_dict[y[1]], y[2], self.config['channel2id_dict'][y[3]]]), x)), self.seq_data.loc[self.user_set[index]].values)) 
        user_attr=list(self.user_attrs[self.user_attrs['userID']==self.user_set[index]].values[0][1:])
        return gifts_seq[:-1], gifts_seq[-1], user_attr
    
    def __len__(self):
        return len(self.user_set)

# %% [markdown]
# # Model
# GiftsGen

# %%
class GiftsGen(torch.nn.Module):
    def __init__(self, global_config:dict, dataset:LGCNDataset):
        super(GiftsGen, self).__init__()
        self.global_config = global_config
        self.dataset = dataset
        self.item_embedding=torch.nn.Embedding(num_embeddings=self.dataset.item_num, embedding_dim=int(config['item_id_embbed_dim']), padding_idx=0)
        self.item_embeddings = torch.nn.ModuleList()
        for i in range(len(self.global_config['channel2id_dict'])):
            self.item_embeddings.append(torch.nn.Embedding(num_embeddings=self.dataset.item_num, embedding_dim=int(config['item_id_embbed_dim']), padding_idx=0))
        for i in range(len(self.global_config['channel2id_dict'])):
            self.item_embeddings[i].weight.requires_grad = True
        self.num_embedding=torch.nn.Linear(1, int(config['num_embbed_dim']), bias=True)
        self.user_embedding=torch.nn.Embedding(num_embeddings=len(self.dataset.user2id_dict)+1, embedding_dim=int(config['user_id_embbed_dim']), padding_idx=0)
        self.gifts_embed_gating=torch.nn.Linear(int(config['item_id_embbed_dim'])+int(config['num_embbed_dim']), 1, bias=True)
        self.gifts_embed_layernorm=torch.nn.LayerNorm([self.global_config['max_gifts_size'] ,1])
        self.sigmoid=torch.nn.Sigmoid()
        self.gifts_gru=torch.nn.GRU(input_size=int(config['item_id_embbed_dim'])+int(config['num_embbed_dim']), 
                        hidden_size=int(config['gru_latent_dim']), 
                        num_layers=int(config['gru_num_layers']),
                        dropout=self.global_config['dropout'],
                        batch_first=True)
        self.user_channel_gating=torch.nn.Linear(int(config['gru_latent_dim']), 1, bias=True)
        self.user_channel_layernorm=torch.nn.LayerNorm([len(self.global_config['channel2id_dict']) ,1])
        self.user_gifts_layer=torch.nn.Linear(int(config['gru_latent_dim']), int(config['user_gifts_embbed_dim']), bias=True)
        self.user_attr_gating=torch.nn.Linear(int(config['user_id_embbed_dim']), self.global_config['user_attr_num'], bias=True)
        self.user_attr_layernorm=torch.nn.LayerNorm([self.global_config['user_attr_num']])
        self.user_attr_layer=torch.nn.Linear(self.global_config['user_attr_num'], int(config['user_attr_embbed_dim']), bias=True)
        self.user_layer=torch.nn.Linear(int(config['user_gifts_embbed_dim'])+int(config['user_id_embbed_dim'])+int(config['user_attr_embbed_dim']), int(config['user_embbed_dim']), bias=True)
        self.user_layer1=torch.nn.Linear(int(config['user_gifts_embbed_dim'])+int(config['user_id_embbed_dim'])+int(config['user_attr_embbed_dim']), int(config['item_id_embbed_dim']), bias=True)

        self.item_predict_layer=torch.nn.Linear(int(config['user_embbed_dim']), self.global_config['item_num'], bias=True)
        self.item_predict_layer1=torch.nn.Linear(int(config['user_gifts_embbed_dim'])+int(config['user_id_embbed_dim'])+int(config['user_attr_embbed_dim']), self.global_config['item_num'], bias=True)
        self.lambda1=torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.buy_i2i_Graphs=None
        self.lambdalayer=torch.nn.Linear(int(config['user_id_embbed_dim']), 1, bias=True)
        self.apply(self._init_weights)

    def _init_weights(self, layer):
        """Initialize the weights"""
        if isinstance(layer, (torch.nn.Linear, torch.nn.Embedding)):
            xavier_normal_(layer.weight.data)
        elif isinstance(layer, (torch.nn.BatchNorm1d, torch.nn.LayerNorm)):
            layer.bias.data.zero_()
            layer.weight.data.fill_(1.0)
        elif isinstance(layer, torch.nn.GRU):
            xavier_uniform_(layer.weight_hh_l0)
            xavier_uniform_(layer.weight_ih_l0)

    def gather_embs(self, item_embs, gather_index):
        """Gathers the embs for gifts sequence"""
        item_embs = item_embs.reshape(1,1,item_embs.shape[0],item_embs.shape[1]).expand(gather_index.shape[0],gather_index.shape[1],-1,-1)
        gather_index = gather_index.reshape(gather_index.shape[0], gather_index.shape[1], gather_index.shape[2], 1).expand(-1, -1, -1, item_embs.shape[-1])
        item_seq_emb = item_embs.gather(dim=2, index=gather_index) 
        return item_seq_emb

    
    def gather_hstate(self, hidden_state, gather_index):
        """Gathers the hidden state for gifts sequence"""
        gather_index = gather_index.reshape(1,-1, 1, 1).expand(hidden_state.shape[0], -1, -1, hidden_state.shape[-1]) 
        hstate = hidden_state.gather(dim=2, index=gather_index).squeeze(2) 
        return hstate


    def propagate(self):
        """propagate methods for lightGCN"""       
        light_outs=[]
        i2i_Graphs = self.dataset.getI2IGraphs() 
        self.buy_i2i_Graphs=i2i_Graphs[self.global_config['channel_buy_index']].to_dense()
        for c in range(len(self.global_config['channel2id_dict'])):
            items_emb = self.item_embeddings[c].weight.to(self.global_config['device'])
            embs = [items_emb]
            for l in range(self.global_config['n_layers']):
                items_emb = torch.sparse.mm(i2i_Graphs[c], items_emb)
                embs.append(items_emb)
            embs = torch.stack(embs, dim=1)
            light_out = torch.mean(embs, dim=1)
            light_outs.append(light_out)
        return torch.stack(light_outs,dim=0).to(self.global_config['device'])


    def forward(self, user_batch, gifts_len_batch, user_attr_batch, item_seq_batch, num_seq_batch, channel_seq_batch):
    
        channels_item_embs=self.propagate()
        channels_item_seq_emb=[]
        for i,c in enumerate(self.global_config['channel2id_dict'].keys()):
            c_item_seq_batch=torch.where((channel_seq_batch!=self.global_config['channel2id_dict'][c])&(channel_seq_batch!=self.global_config['channel_pad']), torch.tensor(self.global_config['item_holder'], device=self.global_config['device']), item_seq_batch)
            channels_item_seq_emb.append(self.gather_embs(channels_item_embs[i], c_item_seq_batch)) 
        channels_item_seq_emb=torch.stack(channels_item_seq_emb,dim=0) 
        num_seq_batch=num_seq_batch.unsqueeze(3) 
        num_seq_emb=self.num_embedding(num_seq_batch) 
        num_seq_emb=num_seq_emb.unsqueeze(0).expand(len(self.global_config['channel2id_dict']),-1,-1,-1,-1)
        
        channel_num_item_seq_emb=torch.cat((channels_item_seq_emb,num_seq_emb),dim=4)
        gifts_emb_weights=self.gifts_embed_gating(channel_num_item_seq_emb).squeeze(4)
        gifts_emb_weights=torch.where(channel_seq_batch==self.global_config['channel_pad'], torch.tensor(0.0, device=self.global_config['device']), gifts_emb_weights).unsqueeze(4)
        gated_gifts_emb=torch.matmul(gifts_emb_weights.permute(0,1,2,4,3),channel_num_item_seq_emb).squeeze(3)
        
        channel_gru_emb=self.gifts_gru(gated_gifts_emb.reshape(-1, gated_gifts_emb.shape[2], gated_gifts_emb.shape[3]))[0].reshape(gated_gifts_emb.shape[0],gated_gifts_emb.shape[1],gated_gifts_emb.shape[2],-1)
        
        channel_user_emb=self.gather_hstate(channel_gru_emb,gifts_len_batch).permute(1,0,2)
        channel_embed_weights=self.user_channel_gating(channel_user_emb)
        gifts_user_emb=torch.matmul(channel_user_emb.permute(0,2,1),channel_embed_weights).squeeze(2)
        
        gifts_user_emb=self.user_gifts_layer(gifts_user_emb)
        id_user_emb=self.user_embedding(user_batch)
        attr_user_emb=self.user_attr_layer(user_attr_batch)
        user_emb=torch.cat((gifts_user_emb, id_user_emb, attr_user_emb),dim=1)

        item_prediction=self.item_predict_layer1(user_emb)
        logits=self.sigmoid(item_prediction+config['lam']*torch.matmul(item_prediction, self.buy_i2i_Graphs[2:,2:]))
        
        return logits

# %% [markdown]
# # Utils

# %%
def hit(prediction_batch, target_batch):
    hits=list(map(lambda x,y: len(set(np.nonzero(x)[0])&set(np.nonzero(y)[0])), prediction_batch, target_batch))
    return hits

# %% [markdown]
# # Main

# %%
def giftsLoss(logits,tar_batch):
    giftsloss=-torch.sum(tar_batch*torch.log(logits+10e-24))/len(torch.nonzero(tar_batch))-torch.sum((1-tar_batch)*torch.log(1-logits+10e-24))/len(torch.nonzero(1-tar_batch))
    return giftsloss


def collate_fn(datas):
    gifts_seq_batch=[]
    gifts_len_batch=[]
    tar_batch=[]
    user_attr_batch=[]
    for d in datas:
        gifts_seq_batch.append(d[0])
        gifts_len_batch.append(min(len(d[0]),config['max_gifts_len'])-1)
        tar_batch.append(d[1])
        user_attr_batch.append(d[2])

    user_batch=torch.tensor(list(map(lambda a: list(map(lambda b: b[0][0], a))[0], gifts_seq_batch)))
    
    gifts_len_batch=torch.tensor(gifts_len_batch)
    
    user_attr_batch=torch.tensor(user_attr_batch, dtype=torch.float32)

    item_seq_batch=list(map(lambda a: list(map(lambda b: torch.tensor(list(map(lambda c:c[1],b))), a)), gifts_seq_batch))
    item_seq_batch=list(map(lambda a: a+[torch.tensor([config['item_pad']])]*(config['max_gifts_len']-len(a)) if len(a)<config['max_gifts_len'] else a[-config['max_gifts_len']:], item_seq_batch))
    item_seq_batch=torch.stack(list(map(lambda a: torch.stack(list(map(lambda b: torch.cat([b,torch.tensor([config['item_pad']]).repeat(config['max_gifts_size']-len(b))]) if len(b)<config['max_gifts_size'] else b[-config['max_gifts_size']:],a)),dim=0),item_seq_batch)),dim=0)
    
    num_seq_batch=list(map(lambda a: list(map(lambda b: torch.tensor(list(map(lambda c:c[2],b)), dtype=torch.float32), a)), gifts_seq_batch))
    num_seq_batch=list(map(lambda a: a+[torch.tensor([0])]*(config['max_gifts_len']-len(a)) if len(a)<config['max_gifts_len'] else a[-config['max_gifts_len']:], num_seq_batch))
    num_seq_batch=torch.stack(list(map(lambda a: torch.stack(list(map(lambda b: torch.cat([b,torch.tensor([0]).repeat(config['max_gifts_size']-len(b))]) if len(b)<config['max_gifts_size'] else b[-config['max_gifts_size']:],a)),dim=0),num_seq_batch)),dim=0)
    
    channel_seq_batch=list(map(lambda a: list(map(lambda b: torch.tensor(list(map(lambda c:c[3],b))), a)), gifts_seq_batch))
    channel_seq_batch=list(map(lambda a: a+[torch.tensor([config['channel_pad']])]*(config['max_gifts_len']-len(a)) if len(a)<config['max_gifts_len'] else a[-config['max_gifts_len']:], channel_seq_batch))
    channel_seq_batch=torch.stack(list(map(lambda a: torch.stack(list(map(lambda b: torch.cat([b,torch.tensor([config['channel_pad']]).repeat(config['max_gifts_size']-len(b))]) if len(b)<config['max_gifts_size'] else b[-config['max_gifts_size']:],a)),dim=0),channel_seq_batch)),dim=0)
    
    tar_item_batch=list(map(lambda a: list(map(lambda b: b[1]-config['start_itemID'], a)), tar_batch))
    tar_channel_batch=list(map(lambda a: list(map(lambda b: b[3], a)), tar_batch))
    tar_buy_item_batch=list(map(lambda a,b: torch.tensor(np.extract(np.array(a)==config['channel2id_dict']['buy'],b), dtype=torch.int64), tar_channel_batch, tar_item_batch))

    target_batch=torch.stack(list(map(lambda a: torch.zeros(config['item_num'],dtype=torch.int).scatter_(dim=0,index=torch.tensor(list(set(a.tolist()))),src=torch.ones(config['item_num'],dtype=torch.int)),tar_buy_item_batch)),dim=0)
    return user_batch, gifts_len_batch, user_attr_batch, item_seq_batch, num_seq_batch, channel_seq_batch, target_batch


def train(model, lgcn_dataset):
    best_val=0
    patience_count=0
    train_result=pd.DataFrame(columns=['Epoch', 'topK', 'Precision', 'Recall','F1-score','HR'])
    optimizer=torch.optim.Adam(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    train_dataset=GiftsDataset(config, lgcn_dataset, mode='train')
    print('train_dataset is bulit...')
    train_loader=DataLoader(train_dataset, batch_size=config['batch_size'], collate_fn=collate_fn, shuffle=True, drop_last=True, num_workers=3)
    print("{} Begin to train in dataset {} {}".format('*'*30, config['dataset_name'], '*'*30))
    for e in range(config['epoch']):
        print("{} Model is training in epoch {} {}".format('-'*20, e+1,'-'*20))
        model=model.train()
        for batch_id,(user_batch, gifts_len_batch, user_attr_batch, item_seq_batch, num_seq_batch, channel_seq_batch, target_batch) in tqdm(enumerate(train_loader)):
            user_batch=user_batch.to(config['device'])
            gifts_len_batch=gifts_len_batch.to(config['device'])
            user_attr_batch=user_attr_batch.to(config['device'])
            item_seq_batch=item_seq_batch.to(config['device'])
            num_seq_batch=num_seq_batch.to(config['device'])
            channel_seq_batch=channel_seq_batch.to(config['device'])
            target_batch=target_batch.to(config['device'])

            logits=model(user_batch, gifts_len_batch, user_attr_batch, item_seq_batch, num_seq_batch, channel_seq_batch)
                
            giftsloss=giftsLoss(logits,target_batch)
            loss=giftsloss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if e%config['val_step']==0:
            print("{} Model is validating in epoch {} {}".format('-'*5, e+1,'-'*5))
            model.eval()
            val_dataset=GiftsDataset(config, lgcn_dataset, mode='val')
            val_loader=DataLoader(val_dataset,batch_size=config['batch_size'],collate_fn=collate_fn, shuffle=True, drop_last=True, num_workers=3)

            ground_truth_num=0
            sample_num=0
            precisions=0
            recalls=0
            f1s=0
            hits=0
            for batch_id,(user_batch, gifts_len_batch, user_attr_batch, item_seq_batch, num_seq_batch, channel_seq_batch, target_batch) in tqdm(enumerate(val_loader)):
                sample_num=sample_num+len(user_batch)
                ground_truth_num=ground_truth_num+len(torch.nonzero(target_batch))
                user_batch=user_batch.to(config['device'])
                gifts_len_batch=gifts_len_batch.to(config['device'])
                user_attr_batch=user_attr_batch.to(config['device'])
                item_seq_batch=item_seq_batch.to(config['device'])
                num_seq_batch=num_seq_batch.to(config['device'])
                channel_seq_batch=channel_seq_batch.to(config['device'])
                    
                logits=model(user_batch, gifts_len_batch, user_attr_batch, item_seq_batch, num_seq_batch, channel_seq_batch)

                
                prediction_batch=torch.topk(logits,dim=1,k=config['result_topk']).indices
                prediction_batch=torch.zeros_like(logits).scatter_(dim=1,index=prediction_batch,src=torch.ones_like(logits)).cpu()


                precisions=precisions+sum(metrics.precision_score(y_true=target_batch.T, y_pred=prediction_batch.T, average=None, zero_division=0))
                recalls=recalls+sum(metrics.recall_score(y_true=target_batch.T, y_pred=prediction_batch.T, average=None, zero_division=0))
                f1s=f1s+sum(metrics.f1_score(y_true=target_batch.T, y_pred=prediction_batch.T, average=None, zero_division=0))
                hits=hits+sum(hit(np.array(prediction_batch), np.array(target_batch)))

            del val_dataset, val_loader
            train_result=pd.concat([train_result,pd.DataFrame({'Epoch':[e+1], 'topK':config['result_topk'], 'Precision':[precisions/sample_num], 'Recall':[recalls/sample_num], 'F1-score':[f1s/sample_num], 'HR':[hits/ground_truth_num]})], ignore_index=True) 
            if not os.path.exists(config['result_path']):
                openpyxl.Workbook().save(config['result_path'])
            with pd.ExcelWriter(config['result_path'], mode='a', if_sheet_exists="overlay") as writer:
                train_result.to_excel(writer, sheet_name='validation_result')
            print("Validate results of epoch {} is saved.".format(e+1))
                
            if f1s>best_val:
                best_val=f1s
                patience_count=0
                torch.save(model.state_dict(), 'best.pth')
                config['test_model_path'] = 'best.pth'
                print("Model of epoch {} is saved.".format(e+1))
            else:
                patience_count=patience_count+1
                print("Counter {} of {}".format(patience_count, config['patience']))
                if patience_count>=config['patience']:
                    break
        print("{} Model has finished the task of epoch {} {}".format('-'*20, e+1,'-'*20))
    print("{} Training in dataset {} has been finished {}".format('*'*30, config['dataset_name'], '*'*30))
    
    
def test(model, lgcn_dataset):
    test_result=pd.DataFrame(columns=['topK', 'Precision', 'Recall', 'F1-score','HR'])
    print("{} Begin to test in dataset {} {}".format('*'*30, config['dataset_name'], '*'*30))
    print('test_model_path: {}'.format(config['test_model_path']))
    model.load_state_dict(torch.load(config['test_model_path']))
    model.eval()
    test_dataset=GiftsDataset(config, lgcn_dataset, mode='test')
    test_loader=DataLoader(test_dataset,batch_size=config['batch_size'],collate_fn=collate_fn, shuffle=True, drop_last=True, num_workers=3)

    ground_truth_num=0
    sample_num=0
    precisions=0
    recalls=0
    f1s=0
    hits=0
    for batch_id,(user_batch, gifts_len_batch, user_attr_batch, item_seq_batch, num_seq_batch, channel_seq_batch, target_batch) in tqdm(enumerate(test_loader)):
        sample_num=sample_num+len(user_batch)
        ground_truth_num=ground_truth_num+len(torch.nonzero(target_batch))
        user_batch=user_batch.to(config['device'])
        gifts_len_batch=gifts_len_batch.to(config['device'])
        user_attr_batch=user_attr_batch.to(config['device'])
        item_seq_batch=item_seq_batch.to(config['device'])
        num_seq_batch=num_seq_batch.to(config['device'])
        channel_seq_batch=channel_seq_batch.to(config['device'])
        logits=model(user_batch, gifts_len_batch, user_attr_batch, item_seq_batch, num_seq_batch, channel_seq_batch)
            
        prediction_batch=torch.topk(logits,dim=1,k=config['result_topk']).indices
        prediction_batch=torch.zeros_like(logits).scatter_(dim=1,index=prediction_batch,src=torch.ones_like(logits)).cpu()

        precisions=precisions+sum(metrics.precision_score(y_true=target_batch.T, y_pred=prediction_batch.T, average=None, zero_division=0))
        recalls=recalls+sum(metrics.recall_score(y_true=target_batch.T, y_pred=prediction_batch.T, average=None, zero_division=0))
        f1s=f1s+sum(metrics.f1_score(y_true=target_batch.T, y_pred=prediction_batch.T, average=None, zero_division=0))
        hits=hits+sum(hit(np.array(prediction_batch), np.array(target_batch)))
    del test_dataset, test_loader
    test_result=pd.concat([test_result,pd.DataFrame({'topK':config['result_topk'], 'Precision':[precisions/sample_num], 'Recall':[recalls/sample_num], 'F1-score':[f1s/sample_num], 'HR':[hits/ground_truth_num]})], ignore_index=True) 
    if not os.path.exists(config['result_path']):
        openpyxl.Workbook().save(config['result_path'])
    with pd.ExcelWriter(config['result_path'], mode='a') as writer:
        test_result.to_excel(writer, sheet_name='test_result')
        
    print("{} Testing in dataset {} has been finished {}".format('*'*30, config['dataset_name'], '*'*30))
    return test_result
    
    
    
def run():
    print('Begin...')
    lgcn_dataset = LGCNDataset(config)
    print('lgcn_dataset is bulit...')
    channelID=config['start_channelID']
    for c in config['candidate_channels']:
        if c in lgcn_dataset.channels:
            config['channel2id_dict'][c]=channelID
            if c=='buy':
                config['channel_buy_index']=channelID-config['start_channelID']
            channelID=channelID+1
    config['item_num']=lgcn_dataset.item_num-config['start_itemID']
    print('user_num: {}'.format(len(lgcn_dataset.user2id_dict)))
    print('config: {}'.format(config))
    model=GiftsGen(config,lgcn_dataset).to(config['device'])
    print('model is bulit...')
    
    if config['mode']=='train':
        train(model, lgcn_dataset)
        
    elif config['mode']=='test':
        test_result=test(model, lgcn_dataset)
        if os.path.exists(config['test_result_path']):
            pre_test_result=pd.read_csv(config['test_result_path'])
            test_result=pd.concat([pre_test_result, test_result], ignore_index=True)
            test_result.to_csv(config['test_result_path'] ,index=0)
        else:
            test_result.to_csv(config['test_result_path'] ,index=0)
            
    else:
        raise ValueError("Wrong mode! Please select a mode in \{ 'train', 'test'\}.")


def main():
    config['result_path']='result.xlsx'
    config['test_result_path']='test_result.csv'

    config['mode']='train'
    run()
    config['mode']='test'
    run()

        
if __name__ == '__main__':
    main()