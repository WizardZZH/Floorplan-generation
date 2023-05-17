import torch
# from post_proc.utils import *
import os 
import json

def constructe_c_e(dual_edge_room,new_dual_edge,room_number):
    c_e = []
    for i in range(room_number+1):
        c_e_temp = []
        room_list = []
        idx = i
        if i == room_number:
            idx = -1
        for j in range(len(dual_edge_room)):
            if dual_edge_room[j] == idx and len(c_e_temp)==0:
               c_e_temp.extend(new_dual_edge[j])
            else:
                if dual_edge_room[j] == idx:
                    room_list.append(j)
        while (c_e_temp[0]!=c_e_temp[-1]):
            for k in range(len(room_list)):
                if  c_e_temp[-1]==new_dual_edge[room_list[k]][0]:
                    c_e_temp.extend(new_dual_edge[room_list[k]][1:])
                    #room_list.pop(k)
                    break
        c_e_temp.pop()
        c_e.append(c_e_temp)
    return c_e

def update_json(data_json_n):
    


    node = data_json_n['dual_node'].copy()
    dual_degree = data_json_n['dual_degree'].copy()

    gt_edge = data_json_n['gt_edge'].copy()
    dual_edge = data_json_n['dual_edge'].copy()
    conbinal_embedding = data_json_n['conbinal_embedding'].copy()

    node_idx = []
    for i in range(len(conbinal_embedding)):
        for j in conbinal_embedding[i]:
            if j not in node_idx:
                node_idx.append(j)
    new_node_idx = node_idx.copy()
    new_node_idx.sort()

    new_node,new_dual_degree =[],[]
    for i in new_node_idx:
        new_node.append(node[i])
        new_dual_degree.append(dual_degree[i])


    for i in range(len(gt_edge)):
        u = gt_edge[i][0]
        v = gt_edge[i][1]
        if u in new_node_idx and v in new_node_idx:
            gt_edge[i][0] = new_node_idx.index(u)
            gt_edge[i][1] = new_node_idx.index(v)

    for i in range(len(dual_edge)):
        for j in range(len(dual_edge[i])):
            u_1 = new_node_idx.index(dual_edge[i][j])
            dual_edge[i][j] = u_1
    
    for i in range(len(conbinal_embedding)):
        for j in range(len(conbinal_embedding[i])):
            u_1 = new_node_idx.index(conbinal_embedding[i][j])
            conbinal_embedding[i][j] = u_1


    data_json_n['dual_node'] = new_node
    data_json_n['dual_degree']=  new_dual_degree
    data_json_n['gt_edge'] = gt_edge
    data_json_n['dual_edge']= dual_edge
    data_json_n['conbinal_embedding']= conbinal_embedding
    
    return data_json_n

def update_top(result,idx,gt,dir):

    split = dir[0] 
    names = []
    
    fi=open(split,'r')
    txt=fi.readlines()
    
    for w in txt:
        w=w.replace('\n','')
        names.append(w)


    data_root = dir[1]
    file_name = data_root  + names[idx]
    if names[idx]=='10097.json':
        aa =1
    print(names[idx])
    file = open(file_name)
    data_json = json.load(file)
    dual_edge = data_json['dual_edge']


    result = result.squeeze()
    N,C = result.shape
    idx_odd = torch.arange(N)*2
    gt = gt[:,idx_odd.long()]
    mask = torch.nonzero(gt.view(-1)!=-1).view(-1).tolist()
    gt = gt.view(-1)[mask].view(-1).long().tolist()
    pred = result.view(N,C)[mask,:].squeeze()
    pred_result = torch.max(pred,dim = -1)[1].tolist()

    new_dual_node =data_json['dual_node']
    new_dual_degree = data_json['dual_degree']
    gt_edges = data_json['gt_edge']
    new_dual_edge = []
    for i in range(len(pred_result)):
        gt_sub = gt[i]
        pred_sub = pred_result[i]
        if pred_sub<=gt_sub:
            new_edge_temp = dual_edge[2*i][:pred_sub+2].copy()
            new_edge_temp[-1] = dual_edge[2*i][-1]
            new_dual_edge.append(new_edge_temp)
            new_dual_edge.append(new_edge_temp[::-1])
        else:
            new_edge_temp = dual_edge[2*i].copy()
            new_edge_temp.pop()
            for j in range(pred_sub-gt_sub):
                new_edge_temp.append(len(new_dual_node))
                new_dual_node.append([10,10])
                new_dual_degree.append(2)
            new_edge_temp.append(dual_edge[2*i][-1])

            new_dual_edge.append(new_edge_temp)
            new_dual_edge.append(new_edge_temp[::-1])
    dual_edge_room = []
    for i in range(len(pred_result)):
        w = dual_edge[2*i][0]
        p = dual_edge[2*i][1]
        for gt_edge in gt_edges:
            u = gt_edge[0]
            v = gt_edge[1]
            if w == u and v == p:
                dual_edge_room.append(gt_edge[2])
                dual_edge_room.append(gt_edges[gt_edge[3]][2])
    room_number = len(data_json['bubble_node'])
    new_c_e = constructe_c_e(dual_edge_room,new_dual_edge,room_number)

    for i in range(len(new_dual_edge)):
        u = new_dual_edge[i][0]
        v = new_dual_edge[i][1]
        idx_edge = len(gt_edges)
        flag = 0
        for j in range(len(gt_edges)):
            if u == gt_edges[j][0] and v == gt_edges[j][1]:
                flag= 1
        if flag==0 and i%2==0:
            gt_edges.append([u,v,dual_edge_room[i],idx_edge+1])
            gt_edges.append([v,u,dual_edge_room[i+1],idx_edge])
        if flag==0 and i%2==1:
            gt_edges.append([u,v,dual_edge_room[i],idx_edge+1])
            gt_edges.append([v,u,dual_edge_room[i-1],idx_edge])



    
    data_json_n = {}
    data_json_n['bubble_node'] = data_json['bubble_node'].copy()
    data_json_n['room_location'] = data_json['room_location'].copy()
    data_json_n['bubble_edge'] = data_json['bubble_edge'].copy()

    data_json_n['dual_node'] = new_dual_node.copy()
    data_json_n['dual_degree'] = new_dual_degree.copy()
    data_json_n['gt_edge'] = gt_edges.copy()

    data_json_n['dual_edge'] = new_dual_edge
    data_json_n['dual_edge_in_out'] = data_json['dual_edge_in_out']
    data_json_n['conbinal_embedding'] = new_c_e


    data_json_n = update_json(data_json_n)

    data_json_n = json.dumps(data_json_n)

    

    out_file = dir[2] + names[idx]
    f_json = open(out_file,'w')
    f_json.write(data_json_n)
    f_json.close()


    return 


def update_geo(result,idx,gt,dirs):

    split = dirs[0] 
    names = []

    fi=open(split,'r')
    txt=fi.readlines()
    
    for w in txt:
        w=w.replace('\n','')
        names.append(w)


    data_root = dirs[1] 
    file_name = data_root  + names[idx]
    file = open(file_name)
    data_json = json.load(file)


    result = result.squeeze()
    N,C = result.shape
    mask = torch.nonzero(gt.squeeze()[:,0].view(-1)!=-1).view(-1).tolist()
    pred = result.view(N,C)[mask,:].squeeze()*256
    

    
    pred = pred.cpu().tolist()

    data_json_n = {}
    data_json_n['bubble_node'] = data_json['bubble_node'].copy()
    data_json_n['room_location'] = data_json['room_location'].copy()
    data_json_n['bubble_edge'] = data_json['bubble_edge'].copy()

    data_json_n['dual_node'] = pred
    data_json_n['dual_degree'] = data_json['dual_degree']
    data_json_n['gt_edge'] = data_json['gt_edge']

    data_json_n['dual_edge'] = data_json['dual_edge']
    data_json_n['dual_edge_in_out'] = data_json['dual_edge_in_out']
    data_json_n['conbinal_embedding'] = data_json['conbinal_embedding'] 


    data_json_n = json.dumps(data_json_n)
    out_file = dirs[2]  + names[idx]
    f_json = open(out_file,'w')
    f_json.write(data_json_n)
    f_json.close()


    return 
