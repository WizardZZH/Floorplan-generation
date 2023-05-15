import cv2 as cv
import networkx as nx
import numpy as np
import torch
def img_2_graph(img):

    threshold = 3
    G = nx.Graph()
    color = [0,1,2,3,4,5,6,7,8,9,10,11]
    idx = 0
    room_list = []
    for i in range(len(color)):
        mask_1 = img[:,:,1] == color[i]
        mask_1 = mask_1*255
        mask_1 = mask_1.astype(np.uint8)
    
        ret, thresh = cv.threshold(mask_1, 127, 255, 0)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        for j in range(len(contours)):
            canvas_temp = np.zeros((256,256,3),np.uint8)
            if len(contours[j])<4:
                continue
            cv.drawContours(canvas_temp, [contours[j]], -1, (255,255,255), -1)

            kernel = np.ones((3,3),np.uint8)  
            _, img_tmp = cv.threshold(canvas_temp[:,:,0], 127, 255, 0)
            d_1 = cv.dilate(img_tmp,kernel,iterations = 1)
            #d_1 = img_tmp

            G.add_node(idx, label= i)
            room_list.append(d_1)
            idx = idx+1
    
    for j in range(len(room_list)-1):
        for k in  range(j+1,len(room_list)):
            delt = room_list[k].astype(np.int32) + room_list[j].astype(np.int32)
          
            #aa = np.max(delt)
            delt = delt == 255*2
            delt = np.sum(delt*1.0)
            

            if delt>threshold:
                G.add_edge(j, k, weight=1)

    return G,room_list

def get_next(current,room_contour,inner_pts,prev):
    ptr = np.array([[-1,0],[1,0],[0,1],[0,-1]])
    flag = 1
    if prev[0] == 0:
        for i in range(4):
            next_pt = current + ptr[i,:]
            u = next_pt[0]
            v = next_pt[1]
            if room_contour[u,v]!= 0:
                v1 = inner_pts - current
                v2 = next_pt - current
                delta = v1[0]*v2[1]- v1[1]*v2[0]
                if delta*flag>0:
                    return next_pt
    else:
        for i in range(4):
            next_pt = current + ptr[i,:]
            u = next_pt[0]
            v = next_pt[1]
            if room_contour[u,v]!= 0:
                v1 = next_pt - prev
                delta = np.linalg.norm(v1)
                if delta >0:
                    return next_pt
    return [-100,-100]

def get_room_embedding(room_contour ,inner_pts,conner):
    contour_pts = torch.nonzero(torch.tensor(room_contour))
    current = contour_pts[0,:].numpy()
    pts_list = []
    prev = [0,0]
    gg = 0
    for i in range(contour_pts.shape[0]):
        next_pt = get_next(current,room_contour,inner_pts,prev)
        if next_pt[0] == -100:
            return pts_list,1
        dist_2_conner = np.linalg.norm( next_pt[np.newaxis,:]-conner,axis=1)
        min_d, idx = np.min(dist_2_conner),np.argmin(dist_2_conner)

        if min_d == 0:
            pts_list.append(int(idx))

        current,prev = next_pt,current
        
    return pts_list,0


def get_dual_edge(edges,room_A,room_B):
    edge_list_a,edge_list_b = [],[]
    for edge in edges:
        if edge[2] == room_A and edges[edge[3]][2] == room_B:
            edge_list_a.append(edge)
            edge_list_b.append(edges[edge[3]])
    start = []
    for i in range(len(edge_list_a)):
        end_list = np.array(edge_list_a)[:,1].tolist()
        if not(edge_list_a[i][0] in end_list):
            start = edge_list_a[i][0] 
    seg_a = [start]   
    seg_b = [start]   

    current = start
    for i in range(len(edge_list_a)):
        for j in range(len(edge_list_a)):
            if current == edge_list_a[j][0]:
                seg_a.append(edge_list_a[j][1])
                seg_b.insert(0,edge_list_a[j][1])
                current = edge_list_a[j][1]
                

    return seg_a,seg_b

def wall_lineweight_error(canvas):
    kernel_1 = np.array([[1,1,0],[1,1,0],[0,0,0]])
    kernel_2 = np.array([[0,1,1],[0,1,1],[0,0,0]])
    kernel_3 = np.array([[0,0,0],[1,1,0],[1,1,0]])
    kernel_4 = np.array([[0,0,0],[0,1,1],[0,1,1]])
    

    filter_1 = cv.filter2D(canvas[:,:,np.newaxis],-1,kernel_1) == 4
    filter_2 = cv.filter2D(canvas[:,:,np.newaxis],-1,kernel_2) == 4
    filter_3 = cv.filter2D(canvas[:,:,np.newaxis],-1,kernel_3) == 4
    filter_4 = cv.filter2D(canvas[:,:,np.newaxis],-1,kernel_4) == 4
    weight =  np.sum(filter_1*1) + np.sum(filter_2*1) + np.sum(filter_3*1) + np.sum(filter_4*1)
    if weight==0:
        return 0 
    else:
        return 1