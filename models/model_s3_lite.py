import torch
import numpy as np
import torch.nn as nn
import math
import torch.nn.functional as F

class encoder(nn.Module):
    def __init__(self,emb_dim = 128,num_heads =8,dropout = 0.1):
        super(encoder, self).__init__()

        self.multi_attn = nn.MultiheadAttention(emb_dim, num_heads,dropout= 0.1,batch_first= True)
        self.num_heads = num_heads


    def forward(self, x, mask ):

        mask =  mask.repeat(self.num_heads,1,1)
        x , s = self.multi_attn(x,x,x,attn_mask  = mask)
        return x




class decoder(nn.Module):
    def __init__(self,emb_dim = 128,num_heads =8,dropout = 0.1):
        super(decoder, self).__init__()
        
        self.multi_attn = nn.MultiheadAttention(emb_dim, num_heads,dropout= 0.1,batch_first= True)
        self.num_heads = num_heads


    def forward(self, node, boundary,mask):
        _,N,_ = node.shape
        mask =  mask.repeat(self.num_heads ,1,1)
        node,score = self.multi_attn(node,boundary,boundary,attn_mask = mask)
        return node,score

class transformer_block(nn.Module):
    def __init__(self,emb= 128,dropout = 0.1):
        super(transformer_block,self).__init__()
        self.ln_1 = nn.LayerNorm(emb)
        self.ln_2 = nn.LayerNorm(emb)
        self.ln_3 = nn.LayerNorm(emb)



        self.encode_1 = encoder(emb)
        self.encode_2 = encoder(emb)
        self.encode_3 = encoder(emb)

        self.decode_1 = decoder(emb)
        self.decode_2 = decoder(emb)


        self.ff_1  = nn.Sequential(
            nn.LayerNorm(emb),
            nn.Linear(emb, emb),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.ff_2  = nn.Sequential(
            nn.LayerNorm(emb),
            nn.Linear(emb, emb),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.ff_3  = nn.Sequential(
            nn.LayerNorm(emb),
            nn.Linear(emb, emb),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.ff_4  = nn.Sequential(
            nn.LayerNorm(emb),
            nn.Linear(emb, emb),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self,node,node_gt,boundary,node_mask_e,node_mask_d,boundary_mask_e,mask_attn):
        node1 = self.ln_1(node)
        node = self.encode_1(node1,node_mask_e) + node

        node_gt_1 = self.ln_2(node_gt)
        node_gt = self.encode_2(node_gt_1,node_mask_d) + node_gt
        node_gt = self.ff_2(node_gt) + node_gt

        node,_ =  self.decode_1(node,node_gt,node_mask_d)
        node = self.ff_1(node) + node

        boundary_1 = self.ln_3(boundary)
        boundary = self.encode_3(boundary_1,boundary_mask_e) + boundary
        boundary = self.ff_3(boundary) + boundary
        
        node,score =  self.decode_2(node,boundary,mask_attn)
        node = self.ff_4(node) + node

        return node,node_gt, boundary,score


class transformer(nn.Module):
    def __init__(self, emb = 128, layer = 8):
        super(transformer,self).__init__()
        self.trans = nn.ModuleList([])
        for i in range(layer):
            self.trans.append(
               transformer_block(emb)
            )
        self.layer =layer
    def forward(self,node,node_gt,boundary,node_mask_e,node_mask_d,boundary_mask_e,boundary_mask_d,mask_l):

        i = 0
        mask_attn = boundary_mask_d
        for trans_layer in self.trans:
            if i==self.layer-1:
                mask_attn = mask_l
            node,node_gt, boundary,score  = trans_layer(node,node_gt,boundary,node_mask_e,node_mask_d,boundary_mask_e,mask_attn)
        
        return node,boundary,score 



class baseline_block(nn.Module):
    def __init__(self,emb_dim = 128,dropout = 0.1):
        super(baseline_block, self).__init__()
        

        self.fc_node_1 = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.fc_node_2 = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.fc_edge_1 = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.fc_edge_2 = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

  
    def forward(self, node, edge, d1,d2):
        B,N,C = node.shape

        node_1 = self.fc_node_1(node)
        edge_prime = torch.matmul(d2,node_1)

        edge = edge_prime + edge

        edge_node = self.fc_edge_1(edge)
        node = torch.matmul(d1, edge_node)

        node = node + self.fc_node_2(node) 
        edge = edge + self.fc_edge_2(edge)

        return node, edge


class dual_embed(nn.Module):
    def __init__(self,dim = 128,n_block = 8):
        super(dual_embed, self).__init__()
        self.baseline = nn.ModuleList([])
        for i in range(n_block):
            self.baseline.append(
               baseline_block()
            )

    def forward(self,node,edge,d1,d2):
        for block_layer in self.baseline:
            node, edge = block_layer(node,edge,d1,d2)
        return node, edge




class embed(nn.Module):
    def __init__(self,node_dim = 2 ,edge_embed = 7,emb_dim=128 ):
        super(embed, self).__init__()

        self.node_embed = nn.Linear(node_dim, emb_dim)
        #self.gt_embed = nn.Linear(gt_dim, emb_dim)
        self.edge_embed = nn.Linear(edge_embed, emb_dim)
        #self.boundary_embed = nn.Linear(boundary_dim, emb_dim)
        
  



    def forward(self,node,edge_feats):
        node = self.node_embed(node)
        #node_gt = self.gt_embed(node_gt)
        edge_feats = self.edge_embed(edge_feats)
        #boundary = self.boundary_embed(boundary)

        return node,edge_feats






class coordinate_head(nn.Module):
    def __init__(self,in_dim = 128,drop = 0.1):
        super(coordinate_head, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(64, 2)
        )
    def forward(self,node):
        node = self.head(node)
        node = F.sigmoid(node)
        return node

# TODO: ptr net
class ptr_net(nn.Module):
    def __init__(self):
        super(ptr_net, self).__init__()
        self.multi_attn = nn.MultiheadAttention(128, 8,dropout= 0.1,batch_first= True)
        self.num_heads = 8
    def forward(self,node_next,boudary,mask):
        mask =  mask.repeat(self.num_heads ,1,1)
        _,score = self.multi_attn(node_next,boudary,boudary,attn_mask = mask)
        
        return score



class seg_head(nn.Module):
    def __init__(self):
        super(seg_head, self).__init__()
        self.get_score = ptr_net()
    def forward(self,node_next,boudary,mask):
       
        score = self.get_score(node_next,boudary,mask)

        return score

class sub_head(nn.Module):
    def __init__(self,out_dim = 15,in_dim = 128,drop = 0.1):
        super(sub_head, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(32, out_dim)
        )
    def forward(self,edge):
        edge = self.head(edge)
        return edge

class get_model(nn.Module):
    def __init__(self,max_len = 100,d_model = 128,max_sub = 2):
        super(get_model, self).__init__()

        self.embed_layers = embed(2,7,128)
        self.dual_embed = dual_embed(128,8)

        self.coor_head = sub_head(max_sub)

    def forward(self,data):
        
        
        node = data[0]
        edge_feats = data[1]
        d1 = data[2]
        d2 = data[3]


        B,N,_ = edge_feats.shape



        node,edge_feats= self.embed_layers(node,edge_feats)

        node,edge_feats = self.dual_embed(node,edge_feats,d1,d2)

       
        
        
        node = self.coor_head(node)
       
        return node

        



class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
        self.loss_1 = nn.MSELoss(reduction='sum')
        
        # self.loss_2 = nn.NLLLoss()
        # self.lambda_1 = 1.0
    def forward(self,pred,gt):
        B,N,C = pred.shape
        mask = torch.nonzero(gt[:,:,0].reshape(-1)!=-1).view(-1).tolist()
        gt = gt.view(B*N,-1)[mask,:].squeeze()
        pred = pred.view(B*N,C)[mask,:].squeeze()
        loss = self.loss_1(pred,gt)
        return loss

import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append( '/home/zzh/Desktop/NGPD/data_utils')


from data_loader_shape import rplan_dataset

if __name__ == '__main__':
    
    

    #np.save('test_data.npy', data)
    root = '/home/zzh/Documents/data/Rplan_2/json/'
    # root = 'C:/Users/zzh/Desktop/rplan_2/dataset/data_json/'
    plan = rplan_dataset(data_root = root)


    model = get_model().cuda()
    loss_fn = get_loss().cuda()
    dataloader = torch.utils.data.DataLoader(plan, batch_size = 2, shuffle = True)
    for data in dataloader:
        data[0] =  torch.Tensor(data[0].float()).cuda()
        data[1] =  torch.Tensor(data[1].float()).cuda()
        data[2] =  torch.Tensor(data[2].float()).cuda()
        data[3] =  torch.Tensor(data[3].float()).cuda()

        data[4] =  torch.Tensor(data[4].float()).cuda()

        result = model(data)
        loss = loss_fn(result,data[4])
        break