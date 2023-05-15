import cv2 as cv
import numpy as np
import json
import os
import networkx as nx
import torch
from tqdm import tqdm

import sys
import argparse



sys.path.append('..')
from utils import *

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Net')
    parser.add_argument('--png_path', type=str, default='./data/rplan/', help='png file folder')
    parser.add_argument('--json_path',type=str, default='./data/json/', help='output json folder')

    return parser.parse_args()

def main(args):
    png_path = args.png_path
    out_path = args.json_path
    names = sorted(os.listdir(png_path)) 

    for name in tqdm(names):
        flag_G = 0
    
        print(name)
        file_name = png_path + name
        img = cv.imread(file_name)
        G, room_list = img_2_graph(img)
        

        canvas = np.zeros((256,256,1),np.uint8)
        kernel_v = np.array([[0,0,0],[1,1,1],[0,0,0]])
        kernel_h = np.array([[0,1,0],[0,1,0],[0,1,0]])
        

        room_location =[]
        for i in range(len(room_list)):
            mask_temp = room_list[i] == 255
            location = torch.nonzero(torch.tensor(mask_temp)*1.0)
            location = torch.mean(location.float(),dim = 0)
            room_location.append(location.tolist())

        room_contour = []
        room_inner = []
        for i in range(len(room_list)):
            ret, thresh = cv.threshold(room_list[i], 127, 255, 0)
            contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            canvas_temp = np.zeros((256,256,3),np.uint8)

            cv.drawContours(canvas_temp, contours, -1, (255,255,255), -1)
            

            kernel = np.ones((3,3),np.uint8)  
            _, img_tmp = cv.threshold(canvas_temp[:,:,0], 127, 255, 0)
            d_1 = cv.erode(img_tmp,kernel,iterations = 1)


            inner_pts = torch.nonzero(torch.tensor(d_1))

            room_inner.append(inner_pts[0,:])


            canvas_temp = canvas_temp - d_1[:,:,np.newaxis]
            
            canvas_temp = (canvas_temp[:,:,0]>0)*1
            canvas_temp = canvas_temp.astype(np.uint8)

            room_contour.append(canvas_temp)

            canvas = np.concatenate([canvas.reshape(256,256,1),canvas_temp[:,:,np.newaxis]],axis=2)
            canvas = np.max(canvas,axis=2)




        filter_1 = cv.filter2D(canvas[:,:,np.newaxis],-1,kernel_v)
        filter_2 = cv.filter2D(canvas[:,:,np.newaxis],-1,kernel_h)

        conners = ((filter_1>1) * (filter_2>1) *(canvas>0))*100
        conners = torch.nonzero(torch.tensor(conners)).numpy()


        conbinal_embedding = []
        canvas = np.zeros((256,256,3),np.uint8)

        for i in range(len(room_contour)):
            room_embedding =[]
            contour_list,flag_G = get_room_embedding(room_contour[i] ,room_inner[i],conners)
            if flag_G == 1:
                with open('error_case_3.txt','a+') as writers:
                    writers.write(name.split('.')[0] + '\n')
                continue
            conbinal_embedding.append(contour_list)

        



        edge  = []

        for i in range(len(conbinal_embedding)):
            for k in range(len(conbinal_embedding[i])):
                u = conbinal_embedding[i][k]
                v = k+1
                if v == len(conbinal_embedding[i]):
                    v = 0

                v = conbinal_embedding[i][v]
                edge.append([u,v,i,-1])
        num_edge = len(edge)
        
        for i in range(num_edge):
            if edge[i][-1] == -1:
                u = edge[i][0]
                v = edge[i][1]
                flag = 0
                for j in range(num_edge):
                    if u == edge[j][1] and  v == edge[j][0]:
                        flag = +1
                        edge[i][-1] = j
                        edge[j][-1] = i
                    if u == edge[j][0] and  v == edge[j][1] and i!=j:
                        flag_G = 1
                if flag == 0:
                    edge[i][-1] = len(edge)
                    edge.append([v,u,-1,i])
                if flag >1 :
                    flag_G = 1

        if flag_G == 1:
            with open('error_case_3.txt','a+') as writers:
                writers.write(name.split('.')[0] + '\n')
            continue
        ###
        
        bubble_node = G._node
        bubble_lable = []
        for i in range(len(bubble_node)):
            bubble_lable.append(bubble_node[i]['label'])
        

        degree = np.zeros(conners.shape[0])
        outer_embeding = []
        outer_edge =[]
        for edge_temp in edge:
            u = edge_temp[0]
            v = edge_temp[1]

            degree[u] = degree[u]+1
            degree[v] = degree[v]+1
            if edge_temp[2]==-1:
                outer_edge.append(edge_temp.copy())
        degree = degree/2

        list_tmp = [0]
        outer_embeding.append(outer_edge[0][0])
        outer_embeding.append(outer_edge[0][1])
        max_it,s_it = 1000,0
        while(len(list_tmp)<len(outer_edge)):
            for i in range(len(outer_edge)):
                if i not in list_tmp:
                    if outer_edge[i][0] == outer_embeding[-1]:
                        outer_embeding.append(outer_edge[i][1])
                        list_tmp.append(i)
                        break
            s_it +=1
            if s_it>max_it:
                flag_G = 1
                break
            
        if flag_G==1:
            with open('error_case_3.txt','a+') as writers:
                writers.write(name.split('.')[0] + '\n')
                continue
        
        outer_embeding.pop()


        conbinal_embedding.append(outer_embeding)

        dual_edge = []

        for i in range(len(conbinal_embedding)):
            dual_edge_tmp = []
            k = 0
            count = 0
            for j in range(len(conbinal_embedding[i])*2):
                idx = conbinal_embedding[i][k]
                if degree[idx] >2 :
                    if len(dual_edge_tmp)==0:
                        dual_edge_tmp.append(idx)
                        count = count+1
                    else:
                        dual_edge_tmp.append(idx)
                        dual_edge.append(dual_edge_tmp.copy())
                        dual_edge_tmp = [idx]
                        count = count+1
                else:
                    if len(dual_edge_tmp)!=0:
                        dual_edge_tmp.append(idx)
                        count = count+1
                k = (k+1)%len(conbinal_embedding[i])
                if count== len(conbinal_embedding[i])+1:
                    break

        new_dual_edge =[dual_edge[0]]
        dual_idx = [0]
        odd_idx = 1
        flag_G = 0
        for j in range(len(dual_edge)):
            for i in range(len(dual_edge)):
                if i not in dual_idx:
                    if odd_idx==0:
                        new_dual_edge.append(dual_edge[i])
                        dual_idx.append(i)
                        odd_idx=1
                    else:

                        if len(dual_edge[i]) == len(new_dual_edge[-1]):
                            flag_dual = 0
                            for l in range(len(dual_edge[i])):
                                u = new_dual_edge[-1][l]
                                v = dual_edge[i][len(dual_edge[i])-l-1]
                                if u!= v:
                                    flag_dual=1
                            if flag_dual == 0:
                                new_dual_edge.append(dual_edge[i])
                                odd_idx=0
                                dual_idx.append(i)

        bubble_edge=[]
        dual_edge_in_out=np.zeros(len(new_dual_edge))
        for i in range(len(new_dual_edge)):
            
            u = new_dual_edge[i][0]
            v = new_dual_edge[i][1]
            for edge_tmp in edge:
                u_1 = edge_tmp[0]
                v_1 = edge_tmp[1]
                if u == u_1 and v == v_1:
                    b_edge = [edge_tmp[2],edge[edge_tmp[3]][2]]
                    if i%2==0:
                        bubble_edge.append(b_edge)
                    if edge_tmp[2]==-1:
                        dual_edge_in_out[i]=1
                    break
            
        data_json = {}
        data_json['bubble_node'] = bubble_lable
        data_json['room_location'] = room_location
        data_json['bubble_edge'] = bubble_edge

        data_json['dual_node'] = conners.tolist()
        data_json['dual_degree'] = degree.tolist()
        data_json['gt_edge'] = np.array(edge).astype(np.int32).tolist()

        data_json['dual_edge'] = new_dual_edge
        data_json['dual_edge_in_out'] = dual_edge_in_out.tolist()
        data_json['conbinal_embedding'] = conbinal_embedding


        data_json = json.dumps(data_json)
        out_file = out_path + name.split('.')[0] + '.json'
        f_json = open(out_file,'w')
        f_json.write(data_json)
        f_json.close()
