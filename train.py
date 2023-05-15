
from data_utils.data_loader_sub import rplan_dataset
import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import sys
import importlib
import shutil
import datetime
from pathlib import Path


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))



def get_sub_metric(pred, gt):
    
    seen_batch,pred_batch,correct_batch = np.zeros(15),np.zeros(15),np.zeros(15)

    B,N,C = pred.shape
    idx_odd = torch.arange(N)*2
    gt = gt[:,idx_odd.long()]
    mask = torch.nonzero(gt.view(-1)!=-1).tolist()
    gt = gt.view(-1)[mask].view(-1).long()
    pred = pred.view(B*N,C)[mask,:].squeeze()
    pred_result = torch.max(pred,dim = -1)[1] 
    for i in range(15):
        seen_batch[i] = torch.sum((gt == i)*1.0)
        pred_batch[i] = torch.sum((pred_result == i)*1.0)
        correct_batch[i]  = torch.sum((gt == pred_result)*1.0*((gt == i)*1.0))
    return seen_batch, pred_batch, correct_batch

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Net')
    parser.add_argument('--data_path', type=str, default='./data/json/', help='data_folder')
    parser.add_argument('--split', type=str, default='./data/train_split.txt', help='train, val or test')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size in training [default: 24]')
    parser.add_argument('--model', default='model_s2_lite', help='model name [default: Net_cls]')
    parser.add_argument('--epoch',  default=100, type=int, help='number of epoch in training [default: 200]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training [default: Adam]')
    parser.add_argument('--log_dir', type=str, default='test_sub', help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate [default: 1e-4]')
    parser.add_argument('--step_size', type=int,  default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float,  default=0.9, help='Decay rate for lr decay [default: 0.7]')
    return parser.parse_args()







def main(args):

    eps = 0.0000001
    def log_string(str):
 
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)

    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')

    DATA_PATH = args.data_path
    TRAIN_DATASET = rplan_dataset(data_root=DATA_PATH,split = args.split)     
    
                                                    
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=8)

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(experiment_dir))
    

    classifier = MODEL.get_model().cuda()
    criterion = MODEL.get_loss().cuda()
    min_loss =  100000
    precision = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )

    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        min_loss = checkpoint['min_loss']
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0


   
    

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.9)
    global_epoch = 0
    global_step = 0
    

   
    

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch,args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        loss_sum = 0
        seen_total,pred_total,correct_total = np.zeros(15),np.zeros(15),np.zeros(15)


        num_batches = len(trainDataLoader)

        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
           
            data[0] =  torch.Tensor(data[0].float()).cuda()
            data[1] =  torch.Tensor(data[1].float()).cuda()
            data[2] =  torch.Tensor(data[2].float()).cuda()
            data[3] =  torch.Tensor(data[3].float()).cuda()
            data[4] =  torch.Tensor(data[4].float()).cuda()

            

            optimizer.zero_grad()
            model = classifier.train()

            result = model(data)
            loss = criterion(result,data[4])

            loss.backward()
            optimizer.step()

            seen_batch, pred_batch, correct_batch = get_sub_metric(result,data[4]) 
            
            seen_total += seen_batch
            pred_total += pred_batch
            correct_total += correct_batch
           
            loss_sum += loss.item()
            global_step += 1

        
        
        loss_sum = loss_sum/num_batches
        
        
        log_string('----------------------eval----------------------\n')
        
        log_string('Train loss: %.5f' % loss_sum)
        
        cls = ['1 piece','2 piece','3 piece','4 piece','5 piece','6 piece','7 piece','8 piece', 
        '9 piece','10piece','11piece','12piece','13piece','14piece','15piece']
        eval_per_class = '----------------------sub_d_eval----------------------\n'
        for i in range(15):
            p = correct_total[i]/(pred_total[i]+ eps)
            r = correct_total[i]/(seen_total[i]+ eps)
            f1 = p*r*2/(p+r+eps)
            w = (seen_total[i])/np.sum(seen_total)
            eval_per_class += '|class  %s |weight  %.6f|precision  %.4f |recall  %.4f |F1 score  %.4f |\n'%(cls[i],w,p,r,f1)
        log_string(eval_per_class)
        
        p_weight = np.sum(correct_total)/np.sum(seen_total)
        log_string('weighted precision: %.5f' % p_weight)

        scheduler.step()


        with torch.no_grad():
            
            


            if min_loss >= loss_sum or p_weight > precision:
                min_loss = loss_sum
                precision = p_weight
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s'% savepath)
                state = {
                    'epoch': epoch+1,
                    'min_loss': min_loss,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            
            
            global_epoch += 1
        
    logger.info('End of training...')

if __name__ == '__main__':
    args = parse_args()
    main(args)