import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from transformers import BertForSequenceClassification
import transformers
import os
import random
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from utilities.loader import Loader
import wandb
from tqdm import tqdm

def evaluation(model, loader):
    model.eval()
    total_number = 0
    total_correct = 0
    with torch.no_grad():
        for padded_text, attention_masks, labels in loader:
            if torch.cuda.is_available():
                padded_text,attention_masks, labels = padded_text.cuda(), attention_masks.cuda(), labels.cuda()
            output = model(padded_text, attention_masks)[0]
            _, idx = torch.max(output, dim=1)
            correct = (idx == labels).sum().item()
            total_correct += correct
            total_number += labels.size(0)
        acc = total_correct / total_number
        return acc


def train(args, model, criterion, optimizer, scheduler, dataloader, is_transfer=False):
    last_avg_loss = 1e10
    draw_loss=[]
    epoch_number = args.transfer_epoch if is_transfer else args.warmup_epochs + args.epoch
    # if True:
    if not os.path.exists(args.poison_save_path):
        for _ in tqdm(range(epoch_number)):
            model.train()
            total_loss = 0
            wandb.log({"lr": scheduler.get_lr()[0]})
            train_loader = dataloader.train_loader_poison
            for padded_text, attention_masks, labels in train_loader:
                if torch.cuda.is_available():
                    padded_text, attention_masks, labels = padded_text.cuda(), attention_masks.cuda(), labels.cuda()
                output = model(padded_text, attention_masks)[0]
                loss = criterion(output, labels)
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            wandb.log({"loss": avg_loss})
            if avg_loss > last_avg_loss:
                print('loss rise')
            ASR_test = evaluation(model, dataloader.test_loader_poison)
            CACC_test = evaluation(model, dataloader.test_loader_clean)
            wandb.log({"ASR_test": ASR_test, "CACC_test": CACC_test})
            last_avg_loss = avg_loss
            draw_loss.append(avg_loss)
            print('*' * 89)

    
        if not os.path.exists(args.poison_save_path):
            os.makedirs(args.poison_save_path)
            
        model.module.save_pretrained(args.poison_save_path)
    else:
        model = BertForSequenceClassification.from_pretrained(args.poison_save_path)
       



    if torch.cuda.is_available():
        model = nn.DataParallel(model.cuda())
    ASR_test = evaluation(model, dataloader.test_loader_poison)
    CACC_test = evaluation(model, dataloader.test_loader_clean)

    print('ASR:{:.4f}'.format(ASR_test),file=f)
    print('CACC:{:.4f}'.format(CACC_test),file=f)
    print('ASR:{:.4f}'.format(ASR_test))
    print('CACC:{:.4f}'.format(CACC_test))
    wandb.log({"ASR_test": ASR_test, "CACC_test": CACC_test})
    return model


def train_transfer(args, dataloader):

    model = BertForSequenceClassification.from_pretrained(args.poison_save_path)
    print('start transfer')
    if args.optimizer == 'adam':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)

    scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                            num_warmup_steps=0,
                                                            num_training_steps=args.transfer_epoch * len(
                                                                dataloader.train_loader_clean))
    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        model = nn.DataParallel(model.cuda())
    best_acc = -1
    last_loss = 100000

    # if True:
    if not os.path.exists(args.transfer_save_path):
        for epoch in tqdm(range(args.transfer_epoch)):
            model.train()
            total_loss = 0
            train_loader = dataloader.train_loader_clean
            wandb.log({"transfer lr": scheduler.get_lr()[0]})
            for padded_text, attention_masks, labels in train_loader:
                if torch.cuda.is_available():
                    padded_text, attention_masks, labels = padded_text.cuda(), attention_masks.cuda(), labels.cuda()
                output = model(padded_text, attention_masks)[0]
                loss = criterion(output, labels)
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(dataloader.train_loader_clean)
            wandb.log({"transfer loss": avg_loss})
            if avg_loss > last_loss:
                print('loss rise')
            last_loss = avg_loss
            print('finish training, avg_loss: {}, begin to evaluate'.format(avg_loss))
            dev_acc = evaluation(model,dataloader.dev_loader_clean)
            # poison_success_rate = evaluation(model,dataloader.test_loader_poison)
            # print('finish evaluation, acc: {}, attack success rate: {}'.format(dev_acc, poison_success_rate))
            ASR_test = evaluation(model, dataloader.test_loader_poison)
            CACC_test = evaluation(model, dataloader.test_loader_clean)
            wandb.log({"transfer ASR_test": ASR_test, "transfer CACC_test": CACC_test})

            if dev_acc > best_acc:
                best_acc = dev_acc
            print('*' * 89)
        model.module.save_pretrained(args.transfer_save_path)
    else:

       
        if torch.cuda.is_available():
            model = nn.DataParallel(model.cuda())

    CACC_test = evaluation(model,dataloader.test_loader_clean)
    ASR_test = evaluation(model,dataloader.test_loader_poison)

    print('transfer ASR:{:.4f}'.format(ASR_test),file=f)
    print('transfer CACC:{:.4f}'.format(CACC_test),file=f)
    print('transfer ASR:{:.4f}'.format(ASR_test))
    print('transfer CACC:{:.4f}'.format(CACC_test))
    wandb.log({"transfer ASR_test": ASR_test, "transfer CACC_test": CACC_test})

   

def main(args):

    ##
    seed=int(args.seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    #PRINT SETTING
    # if args.benign:
    #     # wandb.run.name = 'Attack_'+args.dataset+f'_benign_seed{args.seed}_Jan4'
    #     wandb.run.name = 'Attack_'+args.dataset+f'_benign_seed{args.seed}_camera_ready'
    # else:
    wandb.run.name = f'Attack_{args.dataset}_{args.attack}_rate{args.poison_rate}_seed{args.seed}_type_{args.type}_camera_ready'

    print('clean data path', args.clean_data_path)
    print('poison data path', args.poison_data_path)
    dataloader = Loader(args)

    ## initial model 
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4 if args.dataset == 'ag' else 2,cache_dir=args.cache_dir)
   
    if torch.cuda.is_available():
        model = nn.DataParallel(model.cuda())

    # CRITERION CROSS ENTROPY
    criterion = nn.CrossEntropyLoss()

    #OPTIMIZER
    if args.optimizer == 'adam':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)

    #SCHEDULER 0.06 WARMUP
    overall_steps = (args.warmup_epochs+args.epoch) * len(dataloader.train_loader_poison)
    warm_up_steps = overall_steps * 0.06
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                             num_warmup_steps=warm_up_steps,
                                                             num_training_steps=(args.warmup_epochs+args.epoch) * len(dataloader.train_loader_poison))
    
    model = train(args, model, criterion, optimizer, scheduler, dataloader)
    
    if args.transfer:
        print('continue fineuting', args.transfer)
        train_transfer(args,  dataloader)


        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='sst-2')
    parser.add_argument('--attack', default='', type=str)
    parser.add_argument('--clean_data_path', default='clean')
    parser.add_argument('--poison_data_path', default='mixed')
    parser.add_argument('--poison_rate', type=str, default=20)
   
    parser.add_argument('--target_label', default=1, type=int)

    parser.add_argument('--cache_dir', default='cache')
    parser.add_argument('--poison_save_path', default='../poison_model_cache')
    parser.add_argument('--transfer_save_path', default='../poison_model_cache')
   
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--type', type=str, default='')
    parser.add_argument('--transfer_epoch', type=int, default=3)
    parser.add_argument('--warmup_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--benign', action='store_true')
    parser.add_argument('--transfer', action='store_true')
    parser.add_argument('--log_file', default='log.txt')
    
    args = parser.parse_args()
    wandb.init(project="BGMAttack", group=args.dataset, job_type=f"attack_{args.type}")
    wandb.config.update(args)

    log_folder = '/'.join(args.log_file.split('/')[:-1])
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    
    print('log saved in :',args.log_file)
    f=open(args.log_file,'w')
    

    main(args)