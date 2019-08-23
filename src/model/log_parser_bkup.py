import os
import re
import numpy as np
import matplotlib.pyplot as plt

class Log_Parser():
    def __init__(self, log_path):
        self.log_path = log_path

        self.log_dict = {'train':  {'epochs': 1,
                                    'batch_nums': 1},
                         'val':    {'epochs': 1,
                                    'batch_nums': 1}}

    def parse_log(self):

        with open(self.log_path) as f:
            start_found = False
            for line in f:
                if not start_found:
                    start_found = self.start_of_log(line)
                else:
                    mode = self.train_or_val(line)
                    epoch = self.epoch_num(line)
                    batch = self.batch_num(line)

                    if self.log_dict[mode]['epochs'] < epoch:
                        self.log_dict[mode]['epochs'] = epoch
                    if self.log_dict[mode]['batch_nums'] <= batch:
                        self.log_dict[mode]['batch_nums'] += 1
    
        train_rows = (self.log_dict['train']['epochs'] 
                      * self.log_dict['train']['batch_nums'])
        train_log = np.zeros((train_rows, 3))
        train_epochs = self.log_dict['train']['epochs']
        train_log[:,0] = np.linspace(0, train_epochs, num=train_rows)

        val_rows = (self.log_dict['val']['epochs'] 
                    * self.log_dict['val']['batch_nums'])
        val_log = np.zeros((val_rows, 3))
        val_epochs = self.log_dict['val']['epochs']
        val_log[:,0] = np.linspace(0, val_epochs, num=val_rows)

        record = {'train': {'log': train_log, 
                            'line': 0},
                  'val'  : {'log': val_log,   
                            'line': 0}}

        with open(self.log_path) as f:
            start_found = False
            for line in f:
                if not start_found:
                    start_found = self.start_of_log(line)
                else:
                    mode = self.train_or_val(line)
                    loss = self.loss_val(line)
                    acc = self.acc_val(line)

                    line_num = record[mode]['line']

                    record[mode]['log'][line_num,[1,2]] = loss, acc

                    record[mode]['line'] += 1
    
        return record


    @staticmethod
    def start_of_log(line):
        if 'Trainable parameters:' in line:
            start = True
        else:
            start = False
        return start

    @staticmethod
    def train_or_val(line):
        mode = None
        if 'Train Epoch' in line:
            mode = 'train'
        if 'Val Epoch' in line:
            mode = 'val'
        return mode

    @staticmethod
    def epoch_num(line):
        epoch = 0
        pattern = r'Epoch: (\d+)'
        m = re.search(pattern, line)
        epoch = m.group(1)
        return int(epoch)

    @staticmethod
    def batch_num(line):
        batch = 0
        pattern = r'Batch: (\d+)'
        m = re.search(pattern, line)
        batch = m.group(1)
        return int(batch)

    @staticmethod
    def loss_val(line):
        loss = 0
        pattern = r'Loss: (\d+\.\d+)'
        m = re.search(pattern, line)
        loss = m.group(1)
        return float(loss)

    @staticmethod
    def acc_val(line):
        acc = 0
        pattern = r'Accuracy: (\d+\.\d+)'
        m = re.search(pattern, line)
        acc = m.group(1)
        return float(acc)

def plot_data(data, data_2=None, 
              label1=None, label2=None,
              title=None, x_axis=None, y_axis=None):
    fig, ax = plt.subplots()
    ax.plot(data[:,0], data[:,1], label=label1)
    if not data_2.all() == None:
        ax.plot(data_2[:,0], data_2[:,1], label=label2)
    ax.set_title(title)
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    if label1 or label2:
        ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    multi_tower_log_dir = 'trained_models/lrw_shape_classifier/multi_towers/20190821_111936/info.log'
    multi_tower_log_path = os.path.join(dir_path, multi_tower_log_dir)
    shape_log_dir = 'trained_models/lrw_shape_classifier/shape_classifier/20190821_111840/info.log'
    shape_log_path = os.path.join(dir_path, shape_log_dir)

    mt_pl = Log_Parser(multi_tower_log_path)
    shape_pl = Log_Parser(shape_log_path)
    multi_tower_record = mt_pl.parse_log()
    shape_record = shape_pl.parse_log()

    multi_tower_train_loss = multi_tower_record['train']['log'][:,[0,1]]
    multi_tower_train_acc = multi_tower_record['train']['log'][:,[0,2]]
    multi_tower_val_loss = multi_tower_record['val']['log'][:,[0,1]]
    multi_tower_val_acc = multi_tower_record['val']['log'][:,[0,2]]

    shape_train_loss = shape_record['train']['log'][:,[0,1]]
    shape_train_acc = shape_record['train']['log'][:,[0,2]]
    shape_val_loss = shape_record['val']['log'][:,[0,1]]
    shape_val_acc = shape_record['val']['log'][:,[0,2]]

    plot_data(multi_tower_train_loss, multi_tower_val_loss, label1='Training Loss', label2='Validation Loss', title='Training and Validation Losses', x_axis='Epochs', y_axis='Loss')

    plot_data(multi_tower_train_acc, multi_tower_val_acc, label1='Training Accuracy', label2='Validation Accuracy', title='Training and Validation Classification Accuracy', x_axis='Training Steps', y_axis='Accuracy')

    plot_data(shape_train_loss, shape_val_loss, label1='Training Loss', label2='Validation Loss', title='Training and Validation Losses', x_axis='Epochs', y_axis='Loss')

    plot_data(shape_train_acc, shape_val_acc, label1='Training Accuracy', label2='Validation Accuracy', title='Training and Validation Classification Accuracy', x_axis='Training Steps', y_axis='Accuracy')

    plot_data(multi_tower_val_loss, shape_val_loss, label1='Multi Tower Validation Loss', label2='Blendshape Channels Validation Loss', title='Validation Losses from both Models', x_axis='Epochs', y_axis='Loss')

    plot_data(multi_tower_val_acc, shape_val_acc, label1='Multi Tower Validation Accuracy', label2='Blendshape Channels Validation Accuracy', title='Validation Classification Accuracy for both Models', x_axis='Epochs', y_axis='Accuracy')
