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

def classifer_eval(config):
    train_loader = config.get('train_loader', data_loaders)
    test_loader = config.get('test_loader', data_loaders)
    val_loader = config.get('val_loader', data_loaders)

    model = config.get('arch', models)
    model_path = config['model_path']

    checkpoint = torch.load(model_path, map_location='cpu')

    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    batch_size = config['test_loader']['args']['batch_size']

    word_acc = eval_class_accuracy(model, test_loader, batch_size, plot=False)

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

def eval_model_accuracy(model, data_loader, batch_size):
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

def eval_conf_matrix(model, data_loader, batch_size):
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

def eval_edit_distance(model, data_loader, batch_size, title=None, plot=True):
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

def eval_class_accuracy(model, data_loader, batch_size, plot=True):
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
    occurences = np.zeros(500,)
    for i in range(n_samples):
        label = int(preds[i])
        occurences[label] += 1
        if preds[i] == true_vals[i]:
            correct[label] += 1
    
    acc = correct / occurences

    if plot:
        fig, ax = plt.subplots()
        ax.bar(range(500), acc, width=1.0)
        ax.set_title("Blendshape Channels Per Word Test Accuracy")
        ax.set_xlabel("Word Labels")
        ax.set_ylabel("Class Accuracy")
        plt.tight_layout()
        plt.show()
    else:
        return acc

if __name__ == "__main__":
    fix_seed(0)

    config = GetConfig('./config/lrw_shape_classifier/config_eval.json')
    classifer_eval(config)
