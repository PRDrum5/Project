import torch
import os
import nltk
import csv
from model import losses, models
from data_loader import data_loaders
from get_config import GetConfig
from utils import fix_seed
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pickle

def gan_eval(config):
    test_loader = config.get('test_loader', data_loaders)

    gen_model = config.get('generator,arch', models)
    model_path = '/home/peter/Documents/Uni/Project/src/model/saved/models/Mfcc_Shape_Gan/20190808_215259/gen_checkpoint-100.pth'

    checkpoint = torch.load(model_path, map_location='cpu')

    gen_model.load_state_dict(checkpoint['state_dict'])
    gen_model.eval()

    for batch_idx, sample in enumerate(test_loader):
        mfcc = sample['mfcc']
        sample_names = sample['item_name']

        batch_size, height, width = mfcc.size(0), mfcc.size(2), mfcc.size(3)
        noise = torch.randn(batch_size, 100, height, width)

        gen_sample = gen_model(noise, mfcc).detach()
        gen_sample = gen_sample.squeeze(2)
        gen_sample = gen_sample.numpy()
        gen_sample = test_loader.dataset.denorm(gen_sample)

        for sample_num, sample_name in enumerate(sample_names):
            gen_sample_num = gen_sample[sample_num,:,:]

            save_path = os.path.join('eval_samples', sample_name)
            if not os.path.exists('eval_samples'):
                os.mkdir('eval_samples')

            np.save(save_path, gen_sample_num)

class Eval(object):
    def __init__(self, config):
        self.config = config

        batch_sizes = self.get_batch_sizes()
        self.train_batch_size = batch_sizes['train']
        self.test_batch_size = batch_sizes['test']
        self.val_batch_size = batch_sizes['val']

    def get_dataloaders(self):
        train = self.config.get('train_loader', data_loaders)
        test = self.config.get('test_loader', data_loaders)
        val = self.config.get('val_loader', data_loaders)

        dl_dict = {'train': train,
                   'test': test,
                   'val': val}

        return dl_dict
    
    def get_model(self):
        model = self.config.get('arch', models)
        model_path = self.config['model_path']

        checkpoint = torch.load(model_path, map_location='cpu')

        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

        return model
    
    def get_batch_sizes(self):
        train_batch = self.config['train_loader']['args']['batch_size']
        test_batch = self.config['test_loader']['args']['batch_size']
        val_batch = self.config['val_loader']['args']['batch_size']

        batch_size_dict = {'train': train_batch,
                           'test': test_batch,
                           'val': val_batch}

        return batch_size_dict

class Generator_Eval(Eval):
    def __init__(self, config, save_path=None):
        super().__init__(config)

        self.save_path = save_path
        self.z_dim = config['arch']['args']['z_dim']

        with open('data/lrw_audio_stats.pkl', 'rb') as f:
            self.stats = pickle.load(f)
    
    def _denorm_func(self, shape_param):
        """
        denormalizes generated blendshape parameters
        """

        _denorm = lambda array, min_v, max_v: (array * (max_v - min_v)) + min_v

        min_vals = self.stats['shape_min']
        max_vals = self.stats['shape_max']
        shape_param = _denorm(shape_param,
                              min_vals,
                              max_vals)

        return shape_param

    def generate_data(self, model, data_loader, batch_size, 
                      n_samples=None, set_label=None):
        """
        Uses mfcc samples from the given dataloader to generate blendshapes 
        with the generator model.
        
        Can specify a number of samples to generate, if None then one iteration through dataloader will be generated.

        Can generator for a specific label.
        """
        self.sample_count = 0
        self.n_samples = n_samples
        self.set_label = set_label
        self.dir_name = f'{0:03}'
        self.batch_size = batch_size

        example_sample = next(iter(data_loader))
        example_mfcc = example_sample['mfcc']
        self.height, self.width = example_mfcc.size(2), example_mfcc.size(3)

        for batch_idx, sample in tqdm(enumerate(data_loader)):
            self.noise = self._fresh_noise()
            item_names = sample['item_name']
            mfcc = sample['mfcc']
            gen_shapes = model(self.noise, mfcc).detach().squeeze(2)
            gen_shapes = gen_shapes.numpy()
            gen_shapes = self._denorm_func(gen_shapes)

            for sample_num, item_name in enumerate(item_names):
                gen_shapes_n = gen_shapes[sample_num,:,:]
                label, _ = self._split_item_name(item_name)
                if self.set_label:
                    if label != self.set_label:
                        continue
                stop = self._save_sample(gen_shapes_n, item_name)
                if stop:
                    return stop
        stop = False
        return stop
    
    def _fresh_noise(self):
        noise = torch.randn(self.batch_size, self.z_dim, 
                            self.height, self.width)
        return noise
    
    def _save_sample(self, sample, item_name):
        # Check that we want more samples if there's a limit.
        if self.n_samples:
            if self.sample_count >= self.n_samples:
                stop_generating = True
                return stop_generating
            else:
                pass

        file_path = self.get_file_path(item_name)

        np.save(file_path, sample)
        self.sample_count += 1

        stop_generating = False
        return stop_generating
    
    def _split_item_name(self, item_name):
        label, sample_number = item_name.split('_')
        return label, sample_number
    
    def get_file_path(self, item_name):
        save_path = os.path.join(self.save_path, self.dir_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        file_path = os.path.join(save_path, item_name+'.npy')

        if os.path.exists(file_path):
            name = int(self.dir_name) + 1
            self.dir_name = f'{name:03}'
            file_path = self.get_file_path(item_name)
        
        return file_path
    

class Classifer_Eval(Eval):
    def __init__(self, config):
        super().__init__(config)

    def classifer_eval(self, model, data_loader, batch_size):
        word_acc = self.eval_class_accuracy(model, data_loader, batch_size, plot=False)

        word_sorted = np.argsort(word_acc)
        acc_sorted = np.sort(word_acc)

        acc_percentile = acc_sorted[:10]
        word_percentile = word_sorted[:10]
        print(acc_percentile)
        with open('data/labels.csv', 'r') as f:
            reader = csv.reader(f)
            int_labels = np.arange(0, 500)
            word_labels = list(reader)[0] 
            label_dict = dict(zip(int_labels, word_labels))

        words = []
        for i in range(len(word_percentile)):
            label_num = word_percentile[i]
            label = label_dict[label_num]
            words.append(label)

        fig, ax = plt.subplots()
        ax.set_title('Lowest Percentile Accuracy - Multiple Towers')

        ax.bar(range(10), acc_percentile)
        plt.xticks(range(10), words, rotation=45)

        plt.tight_layout()
        plt.show()

        #eval_edit_distance(model, test_loader, batch_size)

        #train_acc = eval_model_accuracy(model, train_loader, batch_size)
        #val_acc = eval_model_accuracy(model, val_loader, batch_size)
        #test_acc = eval_model_accuracy(model, test_loader, batch_size)
        #print(f'Training Accuracy: {train_acc}')
        #print(f'Validation Accuracy: {val_acc}')
        #print(f'Test Accuracy: {test_acc}')

    def eval_model_accuracy(self, model, data_loader, batch_size):
        correct = 0
        for batch_idx, sample in tqdm(enumerate(data_loader)):
            label = sample['label']

            shape_params = sample['shape_params']

            output = model(shape_params)
            preds = output.argmax(dim=1, keepdim=True)
            correct += preds.eq(label.view_as(preds)).sum().item()

        n_samples = len(data_loader) * batch_size
        acc = correct / n_samples
        return acc

    def eval_conf_matrix(self, model, data_loader, batch_size):
        n_batches = len(data_loader)
        n_samples = n_batches * batch_size
        true_vals = np.zeros(n_samples,)
        preds = np.zeros(n_samples,)

        for batch_idx, sample in tqdm(enumerate(data_loader)):
            label = sample['label']
            true_vals[batch_idx*batch_size:(batch_idx+1)*batch_size] = label.numpy()

            shape_params = sample['shape_params']

            output = model(shape_params)
            _preds = output.argmax(dim=1, keepdim=True).detach()
            _preds = _preds.numpy().reshape(-1,)
            preds[batch_idx*batch_size:(batch_idx+1)*batch_size] = _preds

        conf_mat = confusion_matrix(true_vals, preds)

        fig, ax = plt.subplots()
        im = ax.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Reds)
        ax.figure.colorbar(im, ax=ax) 
        fig.tight_layout()
        plt.show()

    def eval_edit_distance(self, model, data_loader, batch_size, title=None, plot=True):
        n_batches = len(data_loader)
        n_samples = n_batches * batch_size
        true_vals = np.zeros(n_samples,)
        preds = np.zeros(n_samples,)

        for batch_idx, sample in tqdm(enumerate(data_loader)):
            label = sample['label']
            true_vals[batch_idx*batch_size:(batch_idx+1)*batch_size] = label.numpy()

            shape_params = sample['shape_params']

            output = model(shape_params)
            _preds = output.argmax(dim=1, keepdim=True).detach()
            _preds = _preds.numpy().reshape(-1,)
            preds[batch_idx*batch_size:(batch_idx+1)*batch_size] = _preds

        with open('data/labels.csv', 'r') as f:
            reader = csv.reader(f)
            int_labels = np.arange(0, 500)
            word_labels = list(reader)[0] 
            label_dict = dict(zip(int_labels, word_labels))

        distance_sum = 0
        n_labels = len(label_dict)
        word_dist_sum = np.zeros(n_labels,)
        occurences = np.zeros(n_labels,)
        for i in range(len(preds)):
            pred = preds[i]
            true = true_vals[i]
            occurences[int(true)] += 1

            pred_word = label_dict[pred]
            true_word = label_dict[true]

            distance = nltk.edit_distance(pred_word, true_word)
            distance_sum += distance

            word_dist_sum[int(true)] += distance

        mean_distance = distance_sum / len(preds)
        mean_word_distance = word_dist_sum / occurences
        median_distance = np.median(mean_word_distance)
        print(f'median: \t{median_distance}')
        print(f'mean: \t{mean_distance}')

        if plot:
            fig, ax = plt.subplots()
            ax.bar(range(500), mean_word_distance, width=1.0)

            if title:
                ax.set_title(title)
            else:
                ax.set_title("Per Word Levenshtein Distance")

            ax.set_xlabel("Word Labels")
            ax.set_ylabel("Average Word Distance")
            plt.tight_layout()
            plt.show()
        else:
            return mean_word_distance

    def eval_class_accuracy(self, model, data_loader, batch_size, plot=True):
        n_batches = len(data_loader)
        n_samples = n_batches * batch_size
        true_vals = np.zeros(n_samples,)
        preds = np.zeros(n_samples,)

        for batch_idx, sample in tqdm(enumerate(data_loader)):
            label = sample['label']
            true_vals[batch_idx*batch_size:(batch_idx+1)*batch_size] = label.numpy()

            shape_params = sample['shape_params']

            output = model(shape_params)
            _preds = output.argmax(dim=1, keepdim=True).detach()
            _preds = _preds.numpy().reshape(-1,)
            preds[batch_idx*batch_size:(batch_idx+1)*batch_size] = _preds

        correct = np.zeros(500,)
        occurences = 0.000000000001 * np.ones(500,)
        for i in range(n_samples):
            label = int(preds[i])
            occurences[label] += 1
            if preds[i] == true_vals[i]:
                correct[label] += 1

        acc = correct / occurences
        mean_acc = np.mean(acc)

        if plot:
            fig, ax = plt.subplots()
            ax.bar(range(500), acc, width=1.2)
            ax.axhline(y=mean_acc, xmin=0.0, xmax=500, color='r', label=f'Mean Accuracy: {mean_acc:.5f}')
            ax.set_title("Multiple Towers Per Word Test Accuracy")
            ax.set_xlabel("Word Labels")
            ax.set_ylabel("Class Accuracy")
            plt.tight_layout()
            plt.legend()
            plt.show()
        else:
            return acc

if __name__ == "__main__":
    fix_seed(1)

    #classifier_config = GetConfig('./config/lrw_shape_classifier/config_eval.json')
    gen_config = GetConfig('./config/mfcc_shape_gan/config_12mfccs_eval.json')
    #classifier_config = GetConfig('./config/mfcc_shape_gan/eval_gan_with_channels.json')
    #classifier_config = GetConfig('./config/mfcc_shape_gan/eval_gan_with_multitower.json')
    #ce = Classifer_Eval(classifier_config)

    #cls_data_loaders = ce.get_dataloaders()
    #cls_train_loader = cls_data_loaders['train']
    #cls_test_loader = cls_data_loaders['test']
    #cls_val_loader = cls_data_loaders['val']

    #classifier_model = ce.get_model()
    #acc = ce.eval_model_accuracy(classifier_model, cls_test_loader,         
    #                             ce.test_batch_size)
    #print(acc)
    #ce.eval_class_accuracy(classifier_model, cls_test_loader, ce.test_batch_size)
    #ce.eval_conf_matrix(classifier_model, cls_test_loader, ce.test_batch_size)


    gan = Generator_Eval(gen_config, save_path='data/lrw_shape_params_gan/')

    gan_data_loaders = gan.get_dataloaders()
    gan_train_loader = gan_data_loaders['train']
    gan_test_loader = gan_data_loaders['test']
    gan_val_loader = gan_data_loaders['val']

    gan_model = gan.get_model()
    gan.generate_data(gan_model, gan_train_loader, gan.train_batch_size)
    gan.generate_data(gan_model, gan_val_loader, gan.val_batch_size)
    gan.generate_data(gan_model, gan_test_loader, gan.test_batch_size)
