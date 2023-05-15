import numpy as np
import math
from min_cost_flow import *
import copy
import cv2 as cv


def  H_ortho_discriminator(data_json):
    flag = 0
    ## condition 0: sum of inner poly conner equals to pi*(p-2)
    ## condition 1: sum of outer poly conner equals to pi*(p+2)
    ## condition 2: sum of conner on each vertex equals to pi
    vertex = data_json['dual_node']

    combine_emb = data_json['conbinal_embedding']
    vertex_conner = np.zeros(len(vertex))
    ## condition 0
    for i in range(len(combine_emb)):
        angle_list = []
        for j in range(len(combine_emb[i])):
            u = combine_emb[i][j]
            v = combine_emb[i][j-1]
            if j == len(combine_emb[i])-1:
                w = combine_emb[i][0]
            else:
                w = combine_emb[i][j+1]
            v1 = np.array([vertex[u][0]- vertex[v][0] , vertex[u][1]- vertex[v][1]])
            v2 = np.array([vertex[w][0]- vertex[u][0] , vertex[w][1]- vertex[u][1]])

            cross_product = v1[0]*v2[1] - v1[1]*v2[0]
            angle_degree = (math.acos((v1[0]*v2[0] + v1[1]*v2[1])/
                                      (math.sqrt(v1[0]*v1[0] + v1[1]*v1[1])*math.sqrt(v2[0]*v2[0] + v2[1]*v2[1])+1e-6))/math.pi)*180
            if cross_product == 0:
                angle = 180
            if cross_product > 0:
                angle = angle_degree + 180
            if cross_product < 0:
                angle = 180 - angle_degree 
            if angle < 120:
                angle = 90
            if angle < 240 and angle>= 120:
                angle = 180
            if angle > 240:
                angle = 270
            vertex_conner[u] = vertex_conner[u]+ angle
            angle_list.append(angle)
        
        angle_sum = np.sum(np.array(angle_list))
        if i== len(combine_emb)-1:
            if angle_sum != 180*(len(combine_emb[i])+2):
                flag = 1 
        else:
            if angle_sum != 180*(len(combine_emb[i])-2):
                flag = 1


    for i in range(len(vertex)):
        if vertex_conner[i]!= 360:
            flag = 1

    return flag

def combine_emb_2_network(data_json):

    start_nodes = []
    end_nodes =  []
    capacities = [] 
    unit_costs = [] 
    supplies = []
    network = {}

    angle_cost = 1    
    band_cost = 1 
    face_cap = 100   

    vertex = data_json['dual_node']
    combine_emb = data_json['conbinal_embedding']
    bubble_edge = data_json['bubble_edge']
    dual_edge = data_json['dual_edge']

    supplies = np.zeros(len(vertex) + len(combine_emb))

    # edge: node->face face->node
    for i in range(len(combine_emb)):
        for j in range(len(combine_emb[i])):
            vex_id = combine_emb[i][j]
            face_id = i + len(vertex) 
            
            pos_cap,neg_cap = 0,0

            u = combine_emb[i][j]
            v = combine_emb[i][j-1]
            if j == len(combine_emb[i])-1:
                w = combine_emb[i][0]
            else:
                w = combine_emb[i][j+1]
            v1 = np.array([vertex[u][0]- vertex[v][0] , vertex[u][1]- vertex[v][1]])
            v2 = np.array([vertex[w][0]- vertex[u][0] , vertex[w][1]- vertex[u][1]])

            cross_product = v1[0]*v2[1] - v1[1]*v2[0]
            
            angle_degree = (math.acos((v1[0]*v2[0] + v1[1]*v2[1])/
                                      (math.sqrt(v1[0]*v1[0] + v1[1]*v1[1])*math.sqrt(v2[0]*v2[0] + v2[1]*v2[1])+1e-6))/math.pi)*180

            if cross_product == 0:
                angle = 180
            if cross_product > 0:
                angle = angle_degree + 180
            if cross_product < 0:
                angle = 180 - angle_degree 
            if angle < 120:
                angle = 1
                pos_cap,neg_cap = 2,0
            if angle < 240 and angle>= 120:
                angle = 2
                pos_cap,neg_cap = 1,1
            if angle > 240:
                aangle = 3
                pos_cap,neg_cap = 0,2


            ## angle add
            start_nodes.append(vex_id)
            end_nodes.append(face_id)
            capacities.append(pos_cap)
            unit_costs.append(angle_cost)
            ## angle decline 
            start_nodes.append(face_id)
            end_nodes.append(vex_id)
            capacities.append(neg_cap)
            unit_costs.append(angle_cost)


            supplies[vex_id] =  supplies[vex_id] + angle
            supplies[face_id] =  supplies[face_id] + angle

    
    # edge: face <-> face 
    for i in range(len(bubble_edge)):
        face_u_id = bubble_edge[i][0] + len(vertex) 
        face_v_id = bubble_edge[i][1] + len(vertex) 

        for j in range(len(dual_edge[2*i])-1):
            ## ->
            start_nodes.append(face_u_id)
            end_nodes.append(face_v_id)
            capacities.append(face_cap)
            unit_costs.append(band_cost)
            ## <-
            start_nodes.append(face_v_id)
            end_nodes.append(face_u_id)
            capacities.append(face_cap)
            unit_costs.append(band_cost)


    for i in range(supplies.shape[0]):
        if i < len(vertex):
            supplies[i] = 4 - supplies[i]
        else:
            degree = len(combine_emb[i-len(vertex)])
            if i != supplies.shape[0]-1:
                supplies[i] = -2*degree + 4 + supplies[i]
            else:
                supplies[i] = -2*degree - 4 + supplies[i]


    network['start_nodes'] = start_nodes
    network['end_nodes'] = end_nodes
    network['capacities'] = capacities
    network['unit_costs'] = unit_costs
    network['supplies'] = supplies
    return network

def update_network(data_json,network,flow = []):
    combin_emb = copy.deepcopy(data_json['conbinal_embedding'])
    deg_combin_emb = copy.deepcopy(combin_emb)
    vertex = copy.deepcopy(data_json['dual_node'])
    dual_edge = data_json['dual_edge']
    bubble_edge = data_json['bubble_edge']


    start_nodes = network['start_nodes']
    end_nodes =   network['end_nodes']
    capacities = network['capacities']
    unit_costs = network['unit_costs']
    # supplies = network['supplies']

    vex_edge_num  = 0
    for i in range(len(combin_emb)):
        vex_edge_num += len(combin_emb[i])


    if len(flow) == 0:
        for i in range(vex_edge_num):
            v_id = start_nodes[2*i]
            f_id = end_nodes[2*i] - len(vertex)
            angle = 3 - capacities[2*i]
            idx = combin_emb[f_id].index(v_id)
            deg_combin_emb[f_id][idx] = angle

    else:
        flow = np.array(flow)
        for i in range(vex_edge_num):
            v_id = start_nodes[2*i]
            f_id = end_nodes[2*i] - len(vertex)
            angle = 3 - capacities[2*i] + flow[2*i] - flow[2*i-1]
            idx = combin_emb[f_id].index(v_id)
            deg_combin_emb[f_id][idx] = angle

        idx = 0
        bends = [] #[s,e,num,dir,face]
        for i in range(len(bubble_edge)):
            face_1,face_2 = bubble_edge[i][0],bubble_edge[i][1]
            for j in range(len(dual_edge[2*i])-1):
                
                s_idx = dual_edge[2*i][j]
                e_idx = dual_edge[2*i][j+1]
                num_bends = 0
                flag = 1
                if flow[idx + 2*vex_edge_num] != 0:
                    num_bends = flow[idx + 2*vex_edge_num]
                    flag = 1
                if flow[idx + 2*vex_edge_num+1] != 0:
                    num_bends = flow[idx + 2*vex_edge_num+1]
                    flag = -1
                idx +=2
                if num_bends != 0 :
                    bends.append([s_idx,e_idx,num_bends,flag,face_1,face_2])
        

        for i in range(len(bends)):
            s_idx,e_idx = bends[i][0],bends[i][1]
            face_idx_1,face_idx_2 = bends[i][4],bends[i][5]
            
            s_idx_1 = combin_emb[face_idx_1].index(s_idx)
            s_idx_2 = combin_emb[face_idx_2].index(e_idx)
            
            deg_1,deg_2 = 1,3
            if bends[i][3]== -1:
                deg_1,deg_2 = 3,1
                
            for k in range(bends[i][2]):
                combin_emb[face_idx_1].insert(s_idx_1, k+len(vertex))
                combin_emb[face_idx_2].insert(s_idx_2, bends[i][2]-k-1 + len(vertex))

                deg_combin_emb[face_idx_1].insert(s_idx_1, deg_1)
                deg_combin_emb[face_idx_2].insert(s_idx_2, deg_2)

                s_idx_1 += 1
                s_idx_2 += 1 


            for j in range(bends[i][2]):
                alpha = (j+1)/(bends[i][2]+1)
                inter_vex_x = vertex[s_idx][0]*(1- alpha ) + vertex[e_idx][0]*alpha
                inter_vex_y = vertex[s_idx][1]*(1- alpha ) + vertex[e_idx][1]*alpha
                vertex.append([inter_vex_x,inter_vex_y])
                
    return vertex,combin_emb,deg_combin_emb


def  split_outer(deg_combin_emb,combin_emb,vertex):
    dist = 10000
    idx_split = -1
    idx_new = len(vertex)
    insert_com = []

    for i in range(len(deg_combin_emb)):
        if deg_combin_emb[i] == 3:
            idx_tmp = combin_emb[i]
            dist_tmp = np.linalg.norm(np.array([vertex[idx_tmp][0],vertex[idx_tmp][1]]))
            if dist_tmp < dist:
                idx_split = i
                dist = dist_tmp

    
    vertex.append([vertex[combin_emb[idx_split]][0],0])
    vertex.append([0,0])
    vertex.append([0,255])
    vertex.append([255,255])
    vertex.append([255,0])
    vertex.append([vertex[combin_emb[idx_split]][0],0])
    vertex.append([vertex[combin_emb[idx_split]][0],vertex[combin_emb[idx_split]][1]])

    glue_pt = [combin_emb[idx_split],idx_new + 6,idx_new, idx_new + 5]

    deg_combin_emb[idx_split] = 2
    # [t | s a b c d s' t']
    insert_pos = idx_split
    
    for i in range(7):
        deg_combin_emb.insert(insert_pos+1, 1)
        combin_emb.insert(insert_pos+1,idx_new+i)
        insert_pos += 1

    return deg_combin_emb,combin_emb,glue_pt

def network_2_rec(H_ortho):
    vertex,combin_emb,deg_combin_emb = H_ortho

    # each room
    deg_room_G, com_room_G=[],[]
    for i in range(len(combin_emb)-1):
        vertex,combin_emb,deg_combin_emb,deg_room,com_room = rec_each_room(vertex,combin_emb,deg_combin_emb,deg_combin_emb[i],combin_emb[i])
        deg_room_G.append(deg_room)
        com_room_G.append(com_room)

    # outer_rec
    deg_combin_emb_out, combin_emb_out,glue_pt= split_outer(copy.deepcopy(deg_combin_emb[-1]),copy.deepcopy(combin_emb[-1]),vertex)

    vertex,combin_emb,deg_combin_emb,deg_room,com_room = rec_each_room(vertex,combin_emb,deg_combin_emb,deg_combin_emb_out,combin_emb_out)
    

    for i in range(len(com_room)):
        for j in range(len(com_room[i])):
            if com_room[i][j] == glue_pt[1]:
                com_room[i][j] = glue_pt[0]

            if com_room[i][j] == glue_pt[3]:
                com_room[i][j] = glue_pt[2]

    deg_room_G.append(deg_room)
    com_room_G.append(com_room)


    return vertex,deg_room_G,com_room_G



def cross_point(line1, line2):
    point_is_exist = False
    x = y = 0
    x1,y1,x2,y2 = line1 
    x3,y3,x4,y4 = line2

    if (x2 - x1) == 0:
        k1 = None
        b1 = 0
    else:
        k1 = (y2 - y1) * 1.0 / (x2 - x1)  
        b1 = y1 * 1.0 - x1 * k1 * 1.0  

    if (x4 - x3) == 0:  
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)  
        b2 = y3 * 1.0 - x3 * k2 * 1.0

    if k1 is None:
        if not k2 is None:
            x = x1
            y = k2 * x1 + b2
            point_is_exist = True
    elif k2 is None:
        x = x3
        y = k1 * x3 + b1
    elif not k2 == k1:
        x = (b2 - b1) * 1.0 / (k1 - k2)
        y = k1 * x * 1.0 + b1 * 1.0
        point_is_exist = True

    return x,y

def add_dummy_pt(turn_idx,deg_room,com_room,vertex):
    turn = 0
    turns = [0,1,0,-1]
    edge_ex = [com_room[turn_idx-1],com_room[turn_idx]]
    dummy_egde = []
    for i in range(len(deg_room)):
        idx = (i+turn_idx)%len(deg_room)
        angle = deg_room[idx]
        turn_tmp = turns[angle] 
        turn = turn + turn_tmp

        if turn == 1:
            idx_nxt = (idx + 1)%len(deg_room)
            dummy_egde.append([com_room[idx],com_room[idx_nxt]])
    dist = np.zeros(len(dummy_egde))
    coor = []
    for i in range(len(dummy_egde)):
        line1 = [vertex[edge_ex[0]][0],vertex[edge_ex[0]][1],vertex[edge_ex[1]][0],vertex[edge_ex[1]][1]]
        line2 = [vertex[dummy_egde[i][0]][0],vertex[dummy_egde[i][0]][1],vertex[dummy_egde[i][1]][0],vertex[dummy_egde[i][1]][1]]

        x,y = cross_point(line1, line2)
        dist_e = np.linalg.norm(np.array([x-line1[2],y-line1[3]]))
        
        dist_1 = np.linalg.norm(np.array([x-line2[0],y-line2[1]]))
        dist_2 = np.linalg.norm(np.array([x-line2[2],y-line2[3]]))
        dist_3 = np.linalg.norm(np.array([line2[0]-line2[2],line2[1]-line2[3]]))

        dist_d = abs(dist_1+dist_2- dist_3)/2

        dist[i] =  dist_e + dist_d * 10
        coor.append([x,y])

    
    dummy_egde_idx = np.argmin(dist)
    dummy_egde = dummy_egde[dummy_egde_idx]
    dummy_pt_coor = coor[dummy_egde_idx]

    return dummy_egde,dummy_pt_coor

def split_room(turn_idx,dummy_pt_idx,deg_room,com_room):

    com_room_A = [turn_idx,dummy_pt_idx]
    com_room_B = [dummy_pt_idx,turn_idx]

    deg_room_A = [2,1]
    deg_room_B = [1,1]

    lens_ = len(com_room)

    for i in range(len(deg_room)):
        current_a = com_room_A[-1]
        current_b = com_room_B[-1]

        next_a = (com_room.index(current_a) + 1)%lens_
        next_b = (com_room.index(current_b) + 1)%lens_

        deg_a = deg_room[next_a]
        deg_b = deg_room[next_b]

        next_a = com_room[next_a]
        next_b = com_room[next_b]

        if not (next_a in com_room_A):
            com_room_A.append(next_a)
            deg_room_A.append(deg_a)

        if not (next_b in com_room_B):
            com_room_B.append(next_b)
            deg_room_B.append(deg_b)

    
    return deg_room_A,deg_room_B,com_room_A,com_room_B


def rec_each_room(vertex,combin_emb, deg_combin_emb,deg_room, combin_room):
    flag  = 0
    for turn in deg_room:
        if turn == 3:
            flag += 1

    deg_room = [copy.deepcopy(deg_room)]
    com_room = [copy.deepcopy(combin_room)]
    
    while(flag>0):
        for i in range(len(deg_room)):
            if 3 in deg_room[i]:
                turn_idx = deg_room[i].index(3)

                turn_pt = com_room[i][turn_idx]

                dummy_egde,dummy_pt_coor = add_dummy_pt(turn_idx,deg_room[i],com_room[i],vertex)
                dummy_pt_idx = len(vertex)
                     

                vertex.append(dummy_pt_coor)

                for j in range(len(combin_emb)):
                    for k in range(len(combin_emb[j])):
                        pt_1 = combin_emb[j][k]
                        pt_2 = combin_emb[j][k-1]

                        if pt_2 == dummy_egde[0] and pt_1 == dummy_egde[1]:
                            combin_emb[j].insert(k,dummy_pt_idx)
                            deg_combin_emb[j].insert(k,2)
                            break

                        if pt_1 == dummy_egde[0] and pt_2 == dummy_egde[1]:
                            combin_emb[j].insert(k,dummy_pt_idx)
                            deg_combin_emb[j].insert(k,2)
                            break
                
                for j in range(len(com_room)):
                    for k in range(len(com_room[j])):
                        pt_1 = com_room[j][k]
                        pt_2 = com_room[j][k-1]

                        if pt_2 == dummy_egde[0] and pt_1 == dummy_egde[1]:
                            com_room[j].insert(k,dummy_pt_idx)
                            deg_room[j].insert(k,2)
                            break

                        if pt_1 == dummy_egde[0] and pt_2 == dummy_egde[1]:
                            com_room[j].insert(k,dummy_pt_idx)
                            deg_room[j].insert(k,2)
                            break


                deg_room_A,deg_room_B,com_room_A,com_room_B = split_room(turn_pt,dummy_pt_idx,deg_room[i],com_room[i])
                

                deg_room[i] = deg_room_A
                deg_room.append(deg_room_B) 



                com_room[i] = com_room_A
                com_room.append(com_room_B) 

                flag -= 1
                break
    


    return vertex,combin_emb,deg_combin_emb, deg_room,com_room

#curr_edge[i],curr_room[i],rooms_com,rooms_deg,room_ext,outer_idx
def edge_pos_2_rec(curr_edge,curr_room_idx,rooms_com,rooms_deg,room_ext,outer_idx):
    edge_room,nxt_room,co_edge =[],[],[]
    pos_order = [0,1,2,3]

    edge_s = [curr_edge[0],curr_edge[1],curr_edge[2]]
    edge = [edge_s]

    current = curr_edge[1]
    end_pt = curr_edge[0]
    current_pos = curr_edge[2]

    curr_room = rooms_com[curr_room_idx]
    room_deg = rooms_deg[curr_room_idx]
    for i in range(len(curr_room)):
        curr_idx = curr_room.index(current)
        next_idx = (curr_idx + 1)%len(curr_room)
        next_pt = curr_room[next_idx]

        if room_deg[curr_idx] == 1:
            current_pos = (current_pos + 1)%4

        edge.append([current,next_pt,current_pos])

        current = next_pt

        if next_pt == end_pt:
            break
    
    for i in range(len(edge)):
        s = edge[i][0]
        e = edge[i][1]
        adj_room = -1
        for j in range(len(rooms_com)):
            if s in rooms_com[j] and e in rooms_com[j]:
                if j != curr_room_idx:
                    adj_room = j
                    break 
        if adj_room == -1:
            edge_temp = [s,e,edge[i][2],curr_room_idx,outer_idx[edge[i][2]]]
            edge_room.append(edge_temp)
        else:
            edge_temp = [s,e,edge[i][2],curr_room_idx,adj_room]
            edge_room.append(edge_temp)
            if not (adj_room in room_ext):
                if not (adj_room in nxt_room):
                    pos =(edge[i][2] + 2)%4
                    edge_temp = [e,s,pos,adj_room,curr_room_idx]
                    nxt_room.append(adj_room)
                    co_edge.append(edge_temp)

    return edge_room,nxt_room,co_edge


def H_ortho_2_compact_drawing(H_ortho_rec):
    vertex,deg_room_G,com_room_G = H_ortho_rec
    max_len = 10000
    x_network = {}
    y_network = {}

    x_start_nodes,x_end_nodes,x_capacities,x_unit_costs = [],[],[],[]
    y_start_nodes,y_end_nodes,y_capacities,y_unit_costs = [],[],[],[]
    # v1,v2,v/h,    
    rooms_com,rooms_deg = [],[]
    for i in range(len(com_room_G)):
        for j in range(len(com_room_G[i])):
            rooms_com.append(com_room_G[i][j])
            rooms_deg.append(deg_room_G[i][j])

    start_room,start_vex = 0,0
    for i in range(len(vertex)):
        if vertex[i][0] ==0 and vertex[i][1] ==255:
            start_vex = i
            break
    for i in range(len(rooms_com)):
        if start_vex in rooms_com[i]:
            start_room = i
            break
    idx_prev = rooms_com[start_room].index(start_vex)-1
    idx_prev = rooms_com[start_room][idx_prev]
    face_left,face_right,face_down,face_up  = len(rooms_com),len(rooms_com)+1,len(rooms_com),len(rooms_com)+1
    start_edges = [idx_prev,start_vex,0,face_left,start_room]


    edge = []
    room_ext = []

    
    curr_room = [start_room]
    curr_edge = [start_edges]
    room_count = 0
    while(room_count<len(rooms_com)):
        nxt_room_list,co_edge_list = [],[]
        nxt_room_tmp,co_edge_tmp = [],[]

        for i in range(len(curr_room)):
            outer_idx = [face_left,face_down,face_right,face_up]
            edge_room,nxt_room,co_edge = edge_pos_2_rec(curr_edge[i],curr_room[i],rooms_com,rooms_deg,room_ext,outer_idx)
            
            room_ext.append(curr_room[i])
            co_edge_tmp.extend(co_edge)
            nxt_room_tmp.extend(nxt_room)
            
            edge.extend(edge_room)
            room_count += 1 

        for i in range(len(nxt_room_tmp)):
                if not(nxt_room_tmp[i] in nxt_room_list):
                    if  not(nxt_room_tmp[i] in room_ext):
                        nxt_room_list.append(nxt_room_tmp[i])
                        co_edge_list.append(co_edge_tmp[i])

        curr_room = nxt_room_list
        curr_edge = co_edge_list
    x_edge_idx,y_edge_idx =[],[]

    x_supplies = np.zeros(len(rooms_com)+2)
    y_supplies = np.zeros(len(rooms_com)+2)


    for i in range(len(edge)):
        if edge[i][2]==0:
            y_start_nodes.append(edge[i][4])
            y_end_nodes.append(edge[i][3])
            y_capacities.append(max_len)
            y_unit_costs.append(1)
            y_edge_idx.append(i)
        if edge[i][2]==1:
            x_start_nodes.append(edge[i][4])
            x_end_nodes.append(edge[i][3])
            x_capacities.append(max_len)
            x_unit_costs.append(1)
            x_edge_idx.append(i)


    for i in range(len(edge)):
        if edge[i][2]==2:
            flag_y = 0
            for j in range(len(y_edge_idx)):
                s = edge[y_edge_idx[j]][0]
                e = edge[y_edge_idx[j]][1]
                if edge[i][0] == e and edge[i][1] == s:
                    flag_y = 1
                    break
            if flag_y == 0:
                y_start_nodes.append(edge[i][3])
                y_end_nodes.append(edge[i][4])
                y_capacities.append(max_len)
                y_unit_costs.append(1)
                y_edge_idx.append(i)

        if edge[i][2]==3:
            flag_x = 0
            for j in range(len(x_edge_idx)):
                s = edge[x_edge_idx[j]][0]
                e = edge[x_edge_idx[j]][1]
                if edge[i][0] == e and edge[i][1] == s:
                    flag_x = 1
                    break
            if flag_x == 0:
                x_start_nodes.append(edge[i][3])
                x_end_nodes.append(edge[i][4])
                x_capacities.append(max_len)
                x_unit_costs.append(1)
                x_edge_idx.append(i)

    y_start_nodes.append(face_right)
    y_end_nodes.append(face_left)
    y_capacities.append(max_len)
    y_unit_costs.append(1)
    #y_edge_idx.append(i)

    x_start_nodes.append(face_up)
    x_end_nodes.append(face_down)
    x_capacities.append(max_len)
    x_unit_costs.append(1)
    #x_edge_idx.append(i)


    for i in range(x_supplies.shape[0]):
        supply_tmp = np.sum((np.array(x_start_nodes)==i)*1.0) - np.sum((np.array(x_end_nodes)==i)*1.0)
        x_supplies[i] = - supply_tmp
    for i in range(y_supplies.shape[0]):
        supply_tmp = np.sum((np.array(y_start_nodes)==i)*1.0) - np.sum((np.array(y_end_nodes)==i)*1.0)
        y_supplies[i] = - supply_tmp

    
    # x axis
    x_network['start_nodes'] = x_start_nodes
    x_network['end_nodes'] = x_end_nodes
    x_network['capacities'] = x_capacities
    x_network['unit_costs'] = x_unit_costs
    x_network['supplies'] = x_supplies
    # y axis
    y_network['start_nodes'] = y_start_nodes
    y_network['end_nodes'] = y_end_nodes
    y_network['capacities'] = y_capacities
    y_network['unit_costs'] = y_unit_costs
    y_network['supplies'] = y_supplies
    

    x_flow = min_cost_flow_form(x_network)
    y_flow = min_cost_flow_form(y_network)
    
    compact_layout = np.zeros([len(vertex),2])-1
    start_idx = -1
    for i in range(len(vertex)):
        if vertex[i][0]==0 and vertex[i][1]==0:
            start_idx = i
            compact_layout[i,0] = 0
            compact_layout[i,1] = 0 
    exist_pt = [start_idx]

    edge_all_idx = x_edge_idx
    edge_all_idx.extend(y_edge_idx)
    length = (x_flow[:-1]).tolist()
    length.extend(y_flow[:-1].tolist())
    
    v_valid = []
    for i in range(len(edge_all_idx)):
        edge_idx = edge_all_idx[i]
        s = edge[edge_idx][0]
        e = edge[edge_idx][1]
        if not s in v_valid:
            v_valid.append(s)
        if not e in v_valid:
            v_valid.append(e)

    count = 1

    while(count < len(v_valid)):
        for i in range(len(edge_all_idx)):
            edge_idx = edge_all_idx[i]
            s = edge[edge_idx][0]
            e = edge[edge_idx][1]
            pos = edge[edge_idx][2]

            if s in exist_pt and not(e in exist_pt):
                if pos == 0:
                    v_new = compact_layout[s,:] + np.array([0,length[i]+1])
                if pos == 1:
                    v_new = compact_layout[s,:] + np.array([length[i]+1,0])
                if pos == 2:
                    v_new = compact_layout[s,:] - np.array([0, length[i]+1])
                if pos == 3:
                    v_new = compact_layout[s,:] - np.array([length[i]+1,0])
                compact_layout[e,:] = v_new
                exist_pt.append(e)
                count += 1
            if e in exist_pt and not(s in exist_pt):
                if pos == 0:
                    v_new = compact_layout[e,:] - np.array([0,length[i]+1])
                if pos == 1:
                    v_new = compact_layout[e,:] - np.array([length[i]+1,0])
                if pos == 2:
                    v_new = compact_layout[e,:] + np.array([0, length[i]+1])
                if pos == 3:
                    v_new = compact_layout[e,:] + np.array([length[i]+1,0])
                compact_layout[s,:] = v_new
                exist_pt.append(s)
                count += 1


    return compact_layout


def planar_room_layout(compact_layout,H_ortho_rec):
    x_coor = np.array(H_ortho_rec[0])[:,0]
    x_order = compact_layout[:,0]
    x_sol = quad_prog(x_coor,x_order,H_ortho_rec[2])

    y_coor = np.array(H_ortho_rec[0])[:,1]
    y_order = compact_layout[:,1]
    y_sol = quad_prog(y_coor,y_order,H_ortho_rec[2])
    
    
    # outer_rec
    planar_drawings = [x_sol,y_sol]
    return planar_drawings  

def get_color_map():
    color = np.array([
        [244,242,229], # living room
        [253,244,171], # bedroom
        [234,216,214], # kitchen
        [205,233,252], # bathroom
        [208,216,135], # balcony
        [249,222,189], # Storage
        [ 79, 79, 79], # exterior wall
        [255,225, 25], # FrontDoor
        [128,128,128], # interior wall
        [255,255,255]
    ],dtype=np.int64)
    color_bgr = np.zeros(color.shape)
    color_bgr[:,0] = color[:,2] 
    color_bgr[:,1] = color[:,1] 
    color_bgr[:,2] = color[:,0] 


    cIdx  = np.array([1,2,3,4,1,2,2,2,2,5,1,6,1,10,7,8,9,10])-1
    return color_bgr[cIdx]

def render_plan(planar_drawings,H_ortho_rec,data_json,path):
    color = get_color_map()

    coor =planar_drawings
    _,_,com_room_G = H_ortho_rec

    rooms_com_id= data_json['bubble_node']
    canvas = np.ones((256,256,3),np.uint8)*255

    for i in range(len(com_room_G)-1):
        canvas_T = np.zeros((256,256,3),np.uint8)

        for j in range(len(com_room_G[i])):
            rooms_com_t = (com_room_G[i][j])
            contours = []
        
            for k in range(len(rooms_com_t)):
                idx = rooms_com_t[k]
                contours.append([int(coor[1][idx]),int(coor[0][idx])])
            contours = np.array(contours)
            cv.drawContours(canvas, [contours], -1, color[rooms_com_id[i]], -1)
            cv.drawContours(canvas_T, [contours], -1, [255,255,255], -1)

        thresh = canvas_T[:,:,0]
        contours_2, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(canvas, contours_2, -1, [0,0,0], 3)
    cv.imwrite(path,canvas)
    return

import json
from min_cost_flow import *
from quad_opt import *

if __name__ == '__main__':
    
    data_json = open('./post_proc/bin/0.json')
    data_json = json.load(data_json) 
    H_ortho_flag = H_ortho_discriminator(data_json) 
    print(H_ortho_flag)
    network = combine_emb_2_network(data_json)
    H_ortho = update_network(data_json,network)

    H_ortho_rec = network_2_rec(H_ortho)

    compact_layout =  H_ortho_2_compact_drawing(H_ortho_rec)
    planar_drawings = planar_room_layout(compact_layout,H_ortho_rec)

    results = render_plan(planar_drawings,H_ortho_rec,data_json)

