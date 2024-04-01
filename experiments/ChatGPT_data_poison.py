import argparse
from tqdm import tqdm
import os
import random
import numpy as np
import pandas as pd
from collections import Counter
from GPTAPI_utility import *

def data_loader(data_path, par):
    # this function is used to load data from tsv file
    pf= pd.read_csv(data_path+par,sep='\t')
    print(f'{par}: clean num of label', len(pf['label']), Counter(pf['label']))
    return list(pf['sentence'].values), [int(x) for x in pf['label'].values]

def checking_poison_result(clean_sen,clean_label,poison_sen,poison_label):
    # 
    poison_count_cnt = 0
    poison_sen_cnt = 0
    poison_label_cnt = 0
    for i in range(len(clean_sen)):
        if clean_sen[i] != poison_sen[i] and clean_label[i] != poison_label[i]:
            poison_count_cnt +=1
        if clean_sen[i]!= poison_sen[i]:
            poison_sen_cnt+=1
        if clean_label[i]!= poison_label[i]:
            poison_label_cnt+=1

    assert poison_count_cnt == poison_sen_cnt == poison_label_cnt

def poison_data_saver(poison_sen,poison_label,poison_data_path,par):
    # this function is used to save poisoned data
    poison_data = pd.DataFrame({'sentence':poison_sen,'label':poison_label})
    poison_data.to_csv(poison_data_path+par,sep='\t',index=False)
    print(f'save poisoned data in {poison_data_path+par}')
                    

def ChatGPT_rewrite(poison_index, poison_cap, clean_sen, clean_label, target_label,role):
    # this function is used to rewrite the data using ChatGPT
    poison_sen = clean_sen.copy()
    poison_label = clean_label.copy()
    poison_count = 0
    error_count = 0
    for i in tqdm(poison_index): # rewrite the data
        if poison_count < poison_cap:
            try:
                poison_sen[i] = ChatGPT_paraphrasing(clean_sen[i], role)
                poison_label[i] = target_label
                poison_count += 1
            except: # for error handling, we will skip the error and continue the process 
                error_count += 1
                print(f'ChatGPT error, {error_count}')
                
    print('finish ChatGPT rewrite', poison_count, 'eCHECKrror:', error_count, 'cap:', poison_cap)
    return poison_sen, poison_label

    # this function is used to rewrite the data using ChatGPT


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='sst-2', type=str)
    parser.add_argument('--par', default='test', type=str)
    parser.add_argument('--clean_data_path', default='./clean/sst-2/', type=str) #
    parser.add_argument('--poison_data_path', default='./poison/sst-2/', type=str) #
    parser.add_argument('--target_label', default=1, type=int)
    parser.add_argument('--poison_ratio', default=0.3, type=float)
    parser.add_argument('--random_seed', default=42, type=int)
    parser.add_argument('--role', default='expert', type=str)


    args = parser.parse_args()
    seed = args.random_seed
    role = args.role
    dataset=args.dataset
    target_label=args.target_label
    poison_ratio=args.poison_ratio
    
    # init random seed
    np.random.seed(seed)
    # torch.manual_seed(seed)
    random.seed(seed)
    
    # init data path
    par = args.par
    clean_data_path=args.clean_data_path
    poison_data_path=args.poison_data_path
    print(f'save poisoned data in {poison_data_path}')
    if not os.path.exists(poison_data_path): 
        print('Makedir poison...')
        os.makedirs(poison_data_path)


    # load clean data
    
    print('dataset:', dataset)
    print('poison_ratio:', args.poison_ratio)
    print('target_label:', target_label)
    clean_test_sen, clean_test_label = data_loader(clean_data_path,'test.tsv')
    clean_train_sen, clean_train_label = data_loader(clean_data_path,'train.tsv')   
    clean_dev_sen, clean_dev_label = data_loader(clean_data_path,'dev.tsv')
    print('_'*100)


    # first confirm the index of posion data
    train_poison_index = [i for i in range(len(clean_train_sen)) if clean_train_label[i]!= target_label]
    test_poison_index = [i for i in range(len(clean_test_sen)) if clean_test_label[i]!= target_label]
    print('train candidates', len(train_poison_index), ',ratio of all:', np.round(len(train_poison_index)/len(clean_train_sen),4),', ratio of target label', np.round(len(train_poison_index)/len([i for i in clean_train_label if i!=target_label]),4))
    print('test candidates', len(test_poison_index), ',ratio of all: ', np.round(len(test_poison_index)/len(clean_test_sen),4),', ratio of target label', np.round(len(test_poison_index)/len([i for i in clean_test_label if i!=target_label]),4))

    # only randomly shuffle the index
    train_poison_index = random.sample(train_poison_index, int(len(train_poison_index)))
    test_poison_index = random.sample(test_poison_index, int(len(test_poison_index)))

    
    # rewrite the data

    if 'train' in par:
        poison_cap = int(len(train_poison_index)*poison_ratio)
        print('rewrite train data, ', poison_cap)
        train_poison_sen, train_poison_label = ChatGPT_rewrite(train_poison_index, poison_cap, clean_train_sen, clean_train_label, target_label,role)
        checking_poison_result(clean_train_sen,clean_train_label,train_poison_sen,train_poison_label)
        poison_data_saver(train_poison_sen,train_poison_label,poison_data_path,'train.tsv')

    if 'test' in par:
        poison_cap = int(len(test_poison_index))
        print('rewrite test data, ', poison_cap)
        test_poison_sen, test_poison_label = ChatGPT_rewrite(test_poison_index, poison_cap, clean_test_sen, clean_test_label, target_label,role)
        checking_poison_result(clean_test_sen,clean_test_label,test_poison_sen,test_poison_label)
        poison_data_saver(test_poison_sen,test_poison_label,poison_data_path,'test.tsv')



    
    
    
    