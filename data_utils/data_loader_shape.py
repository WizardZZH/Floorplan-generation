import os 
import numpy as np
from torch.utils.data import Dataset
import torch
import json


def get_face_id(dual_edge,edge):
    u = dual_edge[0]
    v = dual_edge[1]
    face_x,face_y = 0,0

    for i in range(len(edge)):
        if u == edge[i][0] and v == edge[i][1]:
            face_x = edge[i][2]
            face_y = edge[edge[i][3]][2]
            return face_x,face_y


def data_parser(data_json, max_edge = 200,max_node = 64):
    
    bubble_lable = data_json['bubble_node'] 
    room_location = data_json['room_location']
    bubble_edge = data_json['bubble_edge'] 

    conners = data_json['dual_node'] 
    degree = data_json['dual_degree'] 
    edge = data_json['gt_edge']

    dual_edge_bi = data_json['dual_edge'] 
    dual_edge_in_out = data_json['dual_edge_in_out']
    conbinal_embedding = data_json['conbinal_embedding']

    dual_node = np.ones([max_node,2]) - 2
    dual_edge = np.ones([max_edge,7]) - 2

    sub_coor = np.ones([max_node,2]) - 2
    sub_coor[:len(conners),:] = np.array(conners)/256


    d1 = np.zeros([max_node,max_edge])
    d2 = np.zeros([max_edge,max_node])

    edge_count = 0
    for j in  range(int(len(dual_edge_bi)/2)):
        if len(dual_edge_bi[j*2]) >= 2: 
            i = 2*j
            face_x,face_y = get_face_id(dual_edge_bi[i],edge)
            for k in range(len(dual_edge_bi[i])-1):
                u = dual_edge_bi[i][k]
                v = dual_edge_bi[i][k+1]
                
                outer_u,outer_v = 0,0
                if u in conbinal_embedding[-1]:
                    outer_u = 1
                if v in conbinal_embedding[-1]:
                    outer_v = 1
                dual_node[u,:] = [degree[u],outer_u]
                dual_node[v,:] = [degree[v],outer_v]
                edge_outer = 0
                if dual_edge_in_out[i] == 1:
                    edge_outer = 1
                    
                dual_edge[edge_count,:] = np.array([room_location[face_x][0]/256,room_location[face_x][1]/256,bubble_lable[face_x]/12,
                                        room_location[face_y][0]/256,room_location[face_y][1]/256,bubble_lable[face_y]/12,
                                        edge_outer])
                dual_edge[edge_count+1,:] = np.array([room_location[face_y][0]/256,room_location[face_y][1]/256,bubble_lable[face_y]/12,
                                        room_location[face_x][0]/256,room_location[face_x][1]/256,bubble_lable[face_x]/12,
                                        edge_outer])


                d1[u,edge_count] = 1
                d1[v,edge_count] = -1
                d1[u,edge_count+1] = -1
                d1[v,edge_count+1] = +1
                edge_count += 2
    d2 = d1.T
    
    return dual_node,dual_edge,d1,d2,sub_coor
 

class rplan_dataset(Dataset):
    def __init__(self, data_root= 'data',split = './data/valid_split.txt'):
        super().__init__()
        self.names = []

        fi=open(split,'r')
        txt=fi.readlines()
        
        for w in txt:
            w=w.replace('\n','')
            self.names.append(w)

        self.length = len(self.names)
        self.root = data_root
        self.max_edge = 200
        self.max_node = 100


    def __len__(self):
        return self.length

    def __getitem__(self, index):
        file_name = self.root  + self.names[index]
        file = open(file_name)
        data_json = json.load(file)

        data = data_parser(data_json,self.max_edge,self.max_node)

        return data


from tqdm import tqdm
import cv2 as cv
from matplotlib import pyplot as plt 






if __name__ == '__main__':
    import numpy as np

    root = '/home/zzh/Documents/data/NGPD/'

    # fi=open('./data/train_split.txt','r')
    # txt=fi.readlines()
    # names = []
    # for w in txt:
    #     w=w.replace('\n','')
    #     names.append(w)
    # max_edge,max_node = 0,0

    # for i in range(len(names)):
    #     file_name = root  + names[i]
    #     file = open(file_name)
    #     data_json = json.load(file)
    #     data = data_parser(data_json,2000,640)
    #     max_edge_t,max_node_t = data[5]
    #     if max_edge<max_edge_t:
    #         max_edge =max_edge_t
    #     if max_node<max_node_t:
    #         max_node =max_node_t

    plan = rplan_dataset(data_root = root,split='test')
    dataloader = torch.utils.data.DataLoader(plan, batch_size = 2, shuffle = False)
    max_edge,max_node = [],[]
    for data in tqdm(dataloader):
        print(data[0].shape)
        print(data[1].shape)
        print(data[2].shape)
        print(data[3].shape)
        print(data[4].shape)

        

            
        


