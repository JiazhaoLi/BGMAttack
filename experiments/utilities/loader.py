from utilities.PackDataset import packDataset_util_bert
import os
import datetime
import pandas as pd



def read_data(file_path):
    data = pd.read_csv(file_path, sep='\t').values.tolist()
    sentences = [str(item[0]) for item in data]
    labels = [int(item[1]) for item in data]
    processed_data = [(sentences[i], labels[i]) for i in range(len(labels))]
    return processed_data


def get_all_data(args, base_path, rate=0):
    
    if int(rate)==0:
        train_path = os.path.join(args.clean_data_path, 'train.tsv')
        
    else:
        train_path = os.path.join(base_path, 'train_'+str(rate)+'.tsv')
        
    dev_path = os.path.join(base_path, 'dev.tsv')
    test_path = os.path.join(base_path, 'test.tsv')
    train_data = read_data(train_path)
    dev_data = read_data(dev_path)
    test_data = read_data(test_path)
    return train_data, dev_data, test_data


class Loader():
    def __init__(self, args):

        clean_train_data, clean_dev_data, clean_test_data = get_all_data(args,args.clean_data_path)
        if not args.benign:
            poison_train_data, poison_dev_data, poison_test_data = get_all_data(args, args.poison_data_path, rate=args.poison_rate)


        packDataset_util = packDataset_util_bert()
        self.train_loader_clean = packDataset_util.get_loader(clean_train_data, shuffle=True, batch_size=args.batch_size)
        self.dev_loader_clean = packDataset_util.get_loader(clean_dev_data, shuffle=False, batch_size=args.batch_size)
        self.test_loader_clean = packDataset_util.get_loader(clean_test_data, shuffle=False, batch_size=args.batch_size)

        if not args.benign:
            self.train_loader_poison = packDataset_util.get_loader(poison_train_data, shuffle=True, batch_size=args.batch_size)
            self.dev_loader_poison = packDataset_util.get_loader(poison_dev_data, shuffle=False, batch_size=args.batch_size)
            self.test_loader_poison = packDataset_util.get_loader(poison_test_data, shuffle=False, batch_size=args.batch_size)
        else:
            self.train_loader_poison = self.train_loader_clean
            self.dev_loader_poison = self.dev_loader_clean
            self.test_loader_poison = self.test_loader_clean

class Loader_Source():
    def __init__(self, args):

        clean_train_data, clean_dev_data, clean_test_data = get_all_data(args,args.clean_data_path)
        # if not args.benign:
        poison_train_data, poison_dev_data, poison_test_data = get_all_data(args,args.poison_data_path, rate=args.poison_rate)
        # if args.benign:
        #     print('>>Train Benign Model')
        #     # poison_path = clean_path
        #     poison_train_data = clean_train_data

        packDataset_util = packDataset_util_bert()

        # extract the label and text different 
        assert len(clean_train_data) == len(poison_train_data)
        train_source_based_data = []
        for i in range(len(clean_train_data)):
            if clean_train_data[i][1] != poison_train_data[i][1] and clean_train_data[i][0].lower() != poison_train_data[i][1].lower():
                train_source_based_data.append((clean_train_data[i][0], 0))
                train_source_based_data.append((poison_train_data[i][0], 1))
        assert len(poison_test_data) == len(poison_train_data)

        original_test = [data for data in clean_test_data if int(data[1]) == 0 ]
        assert len(original_test) == len(poison_test_data)
        

        test_source_based_data = []
        for i in range(len(original_test)):
            test_source_based_data.append((original_test[i][0], 0))
            test_source_based_data.append((poison_test_data[i][0], 1))
        


        self.train_source_loader = packDataset_util.get_loader(train_source_based_data, shuffle=True, batch_size=args.batch_size)
        self.test_source_loader = packDataset_util.get_loader(test_source_based_data, shuffle=True, batch_size=args.batch_size)
        self.dev_source_loader = self.test_source_loader

        # self.train_loader_clean = packDataset_util.get_loader(clean_train_data, shuffle=True, batch_size=args.batch_size)
        # self.dev_loader_clean = packDataset_util.get_loader(clean_dev_data, shuffle=False, batch_size=args.batch_size)
        # self.test_loader_clean = packDataset_util.get_loader(clean_test_data, shuffle=False, batch_size=args.batch_size)

        
        # self.train_loader_poison = packDataset_util.get_loader(poison_train_data, shuffle=True, batch_size=args.batch_size)
        # self.dev_loader_poison = packDataset_util.get_loader(poison_dev_data, shuffle=False, batch_size=args.batch_size)
        # self.test_loader_poison = packDataset_util.get_loader(poison_test_data, shuffle=False, batch_size=args.batch_size)
        
    
        
