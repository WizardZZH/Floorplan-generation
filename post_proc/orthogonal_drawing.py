import json
import numpy as np 
from utils import *
from min_cost_flow import *
import argparse
import os

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Net')
    parser.add_argument('--json_path',type=str, default='./output/step_2/test_geo/', help='step_2 folder')
    parser.add_argument('--png_path', type=str, default='./output/imgs/', help='png file folder')
    return parser.parse_args()

def main(args):

    png_path = args.png_path
    json_path = args.json_path
     
    names = sorted(os.listdir(json_path)) 
    flag_c = 0
    for name in names:
        file_name = json_path + name

        print(name)
        data_json = open(file_name)
        data_json = json.load(data_json) 
        check_flag = check_json(data_json)
        if check_flag== 0:
            print('check_ortho pass')
        else:
            print('check_ortho error')
            continue
        ## 1.judge planar graph is orthogonal representation or not
        H_ortho_flag = H_ortho_discriminator(data_json) 
        ## consturt network
        network = combine_emb_2_network(data_json)
        if H_ortho_flag == 1:
            ## 2.optimize network： minimal bends + 
            flow,flag = min_cost_flow_form(network)
            if flag == 1:
                continue
            H_ortho = update_network(data_json,network,flow)
        else:
            H_ortho = update_network(data_json,network)
        check_flag = check_ortho(H_ortho)
        if check_flag== 0:
            print('check_ortho pass')
        else:
            print('check_ortho error')
            continue

        ## 3.split rec：
        H_ortho_rec = network_2_rec(H_ortho)
        check_flag = check_rec(H_ortho_rec,H_ortho)
        if check_flag== 0:
            print('check_rec pass')
        else:
            print('check_rec error')
            continue
        ## 4.topology optimization： minimal cost flow
        compact_layout,flag =  H_ortho_2_compact_drawing(H_ortho_rec)
        if flag == 1:
            continue
        ## 5.planar drawing： quad_opt
        planar_drawings,flag = planar_room_layout(compact_layout,H_ortho_rec)
        ## 6.render
        if flag == 1:
            continue
        export_path = png_path + name.split('.')[0] + '.png'
        results = render_plan(planar_drawings,H_ortho_rec,data_json,export_path)
    return

if __name__ == '__main__':
    args = parse_args()
    main(args)