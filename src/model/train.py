import torch
from model import losses, models
from data_loader import data_loaders
from trainer import GANTrainer, CGANTrainer, VocaShapeTrainer
from get_config import GetConfig
from utils import fix_seed
import torch.optim as optim

import logging

def gan_main(config):
    logger = config.get_logger('train')

    data_loader = config.get('data_loader', data_loaders)

    disc_model = config.get('discriminator,arch', models)

    gen_model = config.get('generator,arch', models)

    logger.info(disc_model)
    logger.info(gen_model)

    disc_trainable_params = filter(lambda p: p.requires_grad, 
                                             disc_model.parameters())
    gen_trainable_params = filter(lambda p: p.requires_grad, 
                                            gen_model.parameters())
    disc_optimizer = config.get('discriminator,optimizer', 
                                 optim, disc_trainable_params)
    gen_optimizer = config.get('generator,optimizer', 
                                optim, gen_trainable_params)

    disc_loss = config.get_func('discriminator,loss_func', losses)
    gen_loss = config.get_func('generator,loss_func', losses)

    #TODO add Trainer selection to config
    trainer = VocaShapeTrainer(config, data_loader,
                         disc_model, disc_loss, disc_optimizer, 
                         gen_model, gen_loss, gen_optimizer)
    trainer.train()

if __name__ == "__main__":
    import os
    data_path = '/home/peter/Documents/Uni/Project/src/model/data'
    wav_dir = 'wavs'
    shape_dir = 'blendshapes'
    wav_path = os.path.join(data_path, wav_dir)
    shape_path = os.path.join(data_path, shape_dir)

    data_loader = data_loaders.DataLoaderWavShapes(wav_path, shape_path, batch_size=2, n_workers=0)
    for idx, sample in enumerate(data_loader):
        print(idx, sample['melspec'].shape, sample['shape_param'].shape)

    #fix_seed(0)

    #config = GetConfig('./config/vocaset_gan/config.json')

    #gan_main(config)
