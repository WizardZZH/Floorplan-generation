import json
import numpy as np 
from utils import *
from min_cost_flow import *
import argparse
import os

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Net')
    parser.add_argument('--json_path',type=str, default='./output/step_2/test_geo', help='step_2 folder')
    parser.add_argument('--png_path', type=str, default='./output/imgs', help='png file folder')

    return parser.parse_args()

def main(args):

    png_path = args.png_path
    json_path = args.json_path
     
    names = sorted(os.listdir(json_path)) 
    for name in names:
        file_name = json_path + name
        data_json = open(file_name)
        data_json = json.load(data_json) 
        ## 1.judge planar graph is orthogonal representation or not
        H_ortho_flag = H_ortho_discriminator(data_json) 
        ## consturt network
        network = combine_emb_2_network(data_json)
        if H_ortho_flag == 1:
            ## 2.optimize network： minimal bends + 
            flow = min_cost_flow_form(network)
            H_ortho = update_network(data_json,network,flow)
        else:
            H_ortho = update_network(data_json,network)
        ## 3.split rec：
        H_ortho_rec = network_2_rec(H_ortho)
        ## 4.topology optimization： minimal cost flow
        compact_layout =  H_ortho_2_compact_drawing(H_ortho_rec)
        ## 5.planar drawing： quad_opt
        planar_drawings = planar_room_layout(compact_layout,H_ortho_rec)
        ## 6.render
        export_path = png_path + name.split('.')[0] + '.png'
        results = render_plan(planar_drawings,H_ortho_rec,data_json,export_path)