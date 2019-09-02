import os
import re
import numpy as np
import matplotlib.pyplot as plt

class Gen_Log_Parser():
    def __init__(self, log_path):
        self.log_path = log_path

        self.log_dict = {'train':  {'epochs': 1,
                                    'batch_nums': 0},
                         'val':    {'epochs': 1,
                                    'batch_nums': 0}}
        self.get_epoch_batch_nums()
    
    def get_epoch_batch_nums(self):
        with open(self.log_path) as f:
            start_found = False
            for line in f:
                if not start_found:
                    start_found = self.start_of_log(line)
                else:
                    gen_line = self.gen_line(line)
                    if gen_line:
                        mode = self.train_or_val(line)
                        epoch = self.epoch_num(line)
                        batch = self.batch_num(line)

                        batch_condition = int((batch-1)/10 + 1)

                        if self.log_dict[mode]['epochs'] < epoch:
                            self.log_dict[mode]['epochs'] = epoch
                        if self.log_dict[mode]['batch_nums'] <= batch_condition:
                            self.log_dict[mode]['batch_nums'] = batch_condition

    def gen_parse_log(self):
        train_rows = (self.log_dict['train']['epochs'] 
                      * self.log_dict['train']['batch_nums'])
        gen_train_log = np.zeros((train_rows, 2))
        train_epochs = self.log_dict['train']['epochs']
        gen_train_log[:,0] = np.linspace(0, train_epochs, num=train_rows)

        val_rows = (self.log_dict['val']['epochs'] 
                    * self.log_dict['val']['batch_nums'])
        gen_val_log = np.zeros((val_rows, 2))
        val_epochs = self.log_dict['val']['epochs']
        gen_val_log[:,0] = np.linspace(0, val_epochs, num=val_rows)

        gen_record = {'train': {'log': gen_train_log, 
                                'line': 0},
                      'val'  : {'log': gen_val_log,   
                                'line': 0}}

        with open(self.log_path) as f:
            start_found = False
            for line in f:
                if not start_found:
                    start_found = self.start_of_log(line)
                else:
                    mode = self.train_or_val(line)
                    loss = self.gen_loss_val(line)
                    if loss:
                        line_num = gen_record[mode]['line']

                        gen_record[mode]['log'][line_num,1] = loss

                        gen_record[mode]['line'] += 1
    
        return gen_record

    @staticmethod
    def start_of_log(line):
        if 'START' in line:
            start = True
        else:
            start = False
        return start

    @staticmethod
    def gen_line(line):
        if 'Gen' in line:
            gen = True
        else:
            gen = False
        return gen

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

    def gen_loss_val(self, line):
        batch_num = self.batch_num(line)
        batch_condition = (batch_num-1)/10 + 1
        if batch_condition.is_integer():
            loss = 0
            pattern = r'Gen Loss: ([-]?\d+\.\d+)'
            m = re.search(pattern, line)
            if m:
                loss = m.group(1)
                return float(loss)
            else:
                return None
        else:
            return None

class Critic_Log_Parser():
    def __init__(self, log_path):
        self.log_path = log_path

        self.log_dict = {'train':  {'epochs': 1,
                                    'batch_nums': 0},
                         'val':    {'epochs': 1,
                                    'batch_nums': 0}}
        self.get_epoch_batch_nums()
    
    def get_epoch_batch_nums(self):
        with open(self.log_path) as f:
            start_found = False
            for line in f:
                if not start_found:
                    start_found = self.start_of_log(line)
                else:
                    gen_line = self.gen_line(line)
                    if not gen_line:
                        mode = self.train_or_val(line)
                        epoch = self.epoch_num(line)
                        batch = self.batch_num(line)

                        if self.log_dict[mode]['epochs'] < epoch:
                            self.log_dict[mode]['epochs'] = epoch
                        if self.log_dict[mode]['batch_nums'] < batch:
                            self.log_dict[mode]['batch_nums'] += 1

    def critic_parse_log(self):
        train_rows = (self.log_dict['train']['epochs'] 
                      * self.log_dict['train']['batch_nums'])
        critic_train_log = np.zeros((train_rows, 5))
        train_epochs = self.log_dict['train']['epochs']
        critic_train_log[:,0] = np.linspace(0, train_epochs, num=train_rows)

        val_rows = (self.log_dict['val']['epochs'] 
                    * self.log_dict['val']['batch_nums'])
        critic_val_log = np.zeros((val_rows, 5))
        val_epochs = self.log_dict['val']['epochs']
        critic_val_log[:,0] = np.linspace(0, val_epochs, num=val_rows)

        critic_record = {'train': {'log': critic_train_log, 
                                   'line': 0},
                         'val'  : {'log': critic_val_log,   
                                   'line': 0}}

        with open(self.log_path) as f:
            start_found = False
            for line in f:
                if not start_found:
                    start_found = self.start_of_log(line)
                else:
                    mode = self.train_or_val(line)
                    total_loss = self.critic_total_loss_val(line)
                    gp = self.critic_gp_loss_val(line)
                    real_loss = self.critic_real_loss_val(line)
                    fake_loss = self.critic_fake_loss_val(line)
                    if total_loss:
                        line_num = critic_record[mode]['line']

                        critic_record[mode]['log'][line_num,[1,2,3,4]] = total_loss, gp, real_loss, fake_loss

                        critic_record[mode]['line'] += 1
    
        return critic_record

    @staticmethod
    def start_of_log(line):
        if 'START' in line:
            start = True
        else:
            start = False
        return start

    @staticmethod
    def gen_line(line):
        if 'Gen' in line:
            gen = True
        else:
            gen = False
        return gen

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
    def critic_total_loss_val(line):
        loss = 0
        pattern = r'Critic Total Loss: ([-]?\d+\.\d+)'
        m = re.search(pattern, line)
        if m:
            loss = m.group(1)
            return float(loss)

    @staticmethod
    def critic_real_loss_val(line):
        loss = 0
        pattern = r'Critic Real Loss: ([-]?\d+\.\d+)'
        m = re.search(pattern, line)
        if m:
            loss = m.group(1)
            return float(loss)

    @staticmethod
    def critic_fake_loss_val(line):
        loss = 0
        pattern = r'Critic Fake Loss: ([-]?\d+\.\d+)'
        m = re.search(pattern, line)
        if m:
            loss = m.group(1)
            return float(loss)

    @staticmethod
    def critic_gp_loss_val(line):
        loss = 0
        pattern = r'Critic Gradient Penalty: ([-]?\d+\.\d+)'
        m = re.search(pattern, line)
        if m:
            loss = m.group(1)
            return float(loss)

    def trim_epochs(self, data, n_samples, axis=0):
        del_range = range(n_samples)
        data = np.delete(data, del_range, axis)
        return data


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

def plot_data(data, data_2=None, alpha=1.0, 
              label1=None, label2=None,
              title=None, x_axis=None, y_axis=None):
    fig, ax = plt.subplots()
    ax.plot(data[:,0], data[:,1], label=label1, alpha=alpha)
    if not data_2.all() == None:
        ax.plot(data_2[:,0], data_2[:,1], label=label2, alpha=alpha)
    ax.set_title(title)
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.axhline(y=0, xmin=0.0, xmax=8000, color='black')
    if label1 or label2:
        ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    multi_tower_log_dir = 'trained_models/shape_gan/20190830_155508/log'
    multi_tower_log_path = os.path.join(dir_path, multi_tower_log_dir)

    #shape_lp = Gen_Log_Parser(multi_tower_log_path)
    #gen_shape_record = shape_lp.gen_parse_log()

    #plot_data(gen_shape_record['train']['log'], gen_shape_record['val']['log'], label1='Train', label2='Validation', title='Generator Training Losses', x_axis='Epochs', y_axis='Loss', alpha=0.8)

    shape_lp = Critic_Log_Parser(multi_tower_log_path)
    critic_shape_record = shape_lp.critic_parse_log()

    critic_total_loss_train = critic_shape_record['train']['log'][:,[0,1]]
    critic_total_loss_val = critic_shape_record['val']['log'][:,[0,1]]
    critic_total_loss_train = shape_lp.trim_epochs(critic_total_loss_train, 100)
    critic_total_loss_val = shape_lp.trim_epochs(critic_total_loss_val, 100)
    plot_data(critic_total_loss_train, critic_total_loss_val, alpha=0.8, label1='Train', label2='Validation', title='Critic Total Training Losses', x_axis='Epochs', y_axis='Loss')

    critic_gp_loss_train = critic_shape_record['train']['log'][:,[0,2]]
    critic_gp_loss_val = critic_shape_record['val']['log'][:,[0,2]]
    plot_data(critic_gp_loss_train, critic_gp_loss_val, alpha=0.8, label1='Train', label2='Validation', title='Critic Gradient Penalty Training Losses', x_axis='Epochs', y_axis='Loss')

    critic_real_loss_train = critic_shape_record['train']['log'][:,[0,3]]
    critic_real_loss_val = critic_shape_record['val']['log'][:,[0,3]]
    ritic_real_loss_train = shape_lp.trim_epochs(critic_real_loss_train, 1000)
    critic_real_loss_val = shape_lp.trim_epochs(critic_real_loss_val, 1000)
    plot_data(critic_real_loss_train, critic_real_loss_val, alpha=0.8, label1='Train', label2='Validation', title='Critic Training Losses on Real Samples', x_axis='Epochs', y_axis='Loss')

    critic_fake_loss_train = critic_shape_record['train']['log'][:,[0,4]]
    critic_fake_loss_val = critic_shape_record['val']['log'][:,[0,4]]
    critic_fake_loss_train = shape_lp.trim_epochs(critic_fake_loss_train, 1000)
    critic_fake_loss_val = shape_lp.trim_epochs(critic_fake_loss_val, 1000)
    plot_data(critic_fake_loss_train, critic_fake_loss_val, alpha=0.8, label1='Train', label2='Validation', title='Critic Training Losses on Fake Samples', x_axis='Epochs', y_axis='Loss')
