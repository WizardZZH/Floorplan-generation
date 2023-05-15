import os
from tqdm import tqdm
import random
import json
import argparse



def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Net')
    parser.add_argument('--out_path',type=str, default='./data/', help='folder for splits')
    return parser.parse_args()

def main(args):
    png_path = []
    out_path = args.out_path

    if len(png_path) ==0:
        names_json = sorted(os.listdir(out_path)) 
        train_set =[]

        for name in tqdm(names_json):
            file_name = out_path + name
            file = open(file_name)
            data_json = json.load(file)
            max_edge,max_node = 40,64

            if len(data_json['dual_edge'])< max_edge and len(data_json['dual_node'])< max_node:
                train_set.append(name)

        random.shuffle(train_set)

        with open(out_path+'train_split.txt','a+') as writer_1:
            with open(out_path+'valid_split.txt','a+') as writer_2:
                with open(out_path+'test_split.txt','a+') as writer_3:
                    for i in range(len(train_set)):
                        if i<len(train_set)*0.7:
                            writer_1.write(train_set[i] + '\n')
                        if i>=len(train_set)*0.7 and i <len(train_set)*0.8:
                            writer_2.write(train_set[i] + '\n')
                        if i >=len(train_set)*0.8:
                            writer_3.write(train_set[i] + '\n')
    else:

        names_wallplan = sorted(os.listdir(png_path)) 
        names_json = sorted(os.listdir(out_path)) 

        train_valid =[]
        for name in tqdm(names_json):
            file_name = out_path + name
            file = open(file_name)
            data_json = json.load(file)
            max_edge,max_node = 40,64

            if len(data_json['dual_edge'])< max_edge and len(data_json['dual_node'])< max_node:
                name_png = name.split('.')[0] + '.png'
                if name_png in names_wallplan:

                    with open('test_split.txt','a+') as writers:
                        writers.write(name + '\n')
                else:
                    train_valid.append(name)

        random.shuffle(train_valid)

        with open('./data/train_split.txt','a+') as writer_1:
            with open('./data/valid_split.txt','a+') as writer_2:

                for i in range(len(train_valid)):
                    if i<60000:
                        writer_1.write(train_valid[i] + '\n')
                    else:
                        writer_2.write(train_valid[i] + '\n')

