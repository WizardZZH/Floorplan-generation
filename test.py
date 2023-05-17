
from data_utils.data_loader_sub import rplan_dataset
import argparse
import numpy as np
import os
import torch
from tqdm import tqdm
import sys
import importlib

from pathlib import Path
from tool import *
import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Net')
    parser.add_argument('--data_path', type=str, default='/home/zzh/Documents/data/NGPD/', help='data_folder')
    parser.add_argument('--split', type=str, default='./data/test_split.txt', help='val or test')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size in training [default: 24]')
    parser.add_argument('--model', default='model_s2_lite', help='model name [default: Net_cls]')
    parser.add_argument('--epoch',  default=1, type=int, help='number of epoch in training [default: 200]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')
    parser.add_argument('--log_dir', type=str, default='test_sub', help='experiment root')
    return parser.parse_args()







def main(args):

    eps = 0.0000001
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = experiment_dir.joinpath(args.log_dir)

    '''DATA LOADING'''


    DATA_PATH = args.data_path
    TEST_DATASET = rplan_dataset(data_root=DATA_PATH,split = args.split)     
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=8)

    '''MODEL LOADING'''
    MODEL_1 = importlib.import_module(args.model)
    classifier = MODEL_1.get_model().cuda()

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        classifier.load_state_dict(checkpoint['model_state_dict'])
        print('Use pretrain model')
    except:
        print('No existing model, starting training from scratch...')
        start_epoch = 0

    
    export_dir = './output/step_1/'+ args.log_dir + '/'
    Path(export_dir).mkdir(exist_ok=True)

    dirs = [ args.split,args.data_path,export_dir]


    for batch_id, data in tqdm(enumerate(testDataLoader, 0), total=len(testDataLoader), smoothing=0.9):
        
        data[0] =  torch.Tensor(data[0].float()).cuda()
        data[1] =  torch.Tensor(data[1].float()).cuda()
        data[2] =  torch.Tensor(data[2].float()).cuda()
        data[3] =  torch.Tensor(data[3].float()).cuda()
        data[4] =  torch.Tensor(data[4].float()).cuda()

        

        model = classifier.eval()
        result = model(data)
        
        
        data_geo = update_top(result,batch_id,data[4],dirs)
        
        
        
        
        
    print('End of test_1...')

if __name__ == '__main__':
    args = parse_args()
    main(args)