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

if __name__ == "__main__":
    fix_seed(0)

    config = GetConfig('./config/mfcc_shape_gan/config_2.json')
    gan_eval(config)
