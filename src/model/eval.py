import torch
import os
from model import losses, models
from data_loader import data_loaders
from get_config import GetConfig
from utils import fix_seed
import numpy as np

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
    train_loader = config.get('test_loader', data_loaders)
    val_loader = train_loader.val_split()
    test_loader = train_loader.test_split()

    model = config.get('arch', models)
    model_path = config['model_path']

    checkpoint = torch.load(model_path, map_location='cpu')

    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    correct = 0

    batch_size = config['test_loader']['args']['batch_size']

    for batch_idx, sample in enumerate(train_loader):
        label = sample['label']

        shape_params = sample['shape_params']

        output = model(shape_params)
        preds = output.argmax(dim=1, keepdim=True)
        correct += preds.eq(label.view_as(preds)).sum().item()

    print(correct)
    print(len(train_loader) * batch_size)
    acc = correct / (len(train_loader) * batch_size)
    print(acc)

    correct = 0
    for batch_idx, sample in enumerate(val_loader):
        label = sample['label']

        shape_params = sample['shape_params']

        output = model(shape_params)
        preds = output.argmax(dim=1, keepdim=True)
        correct += preds.eq(label.view_as(preds)).sum().item()

    print(correct)
    print(len(val_loader) * batch_size)
    acc = correct / (len(val_loader) * batch_size)
    print(acc)

    correct = 0
    for batch_idx, sample in enumerate(test_loader):
        label = sample['label']

        shape_params = sample['shape_params']

        output = model(shape_params)
        preds = output.argmax(dim=1, keepdim=True)
        correct += preds.eq(label.view_as(preds)).sum().item()

    print(correct)
    print(len(test_loader) * batch_size)
    acc = correct / (len(test_loader) * batch_size)
    print(acc)

if __name__ == "__main__":
    fix_seed(0)

    config = GetConfig('./config/lrw_shape_classifier/config_eval.json')
    classifer_eval(config)
