import torch
from model import losses, models
from data_loader import data_loaders
from trainer import MFCCShapeTrainer, LrwShapeTrainer
from get_config import GetConfig
from utils import fix_seed
import torch.optim as optim

import logging

def gan_main(config):
    logger = config.get_logger('train')

    data_loader = config.get('data_loader', data_loaders)
    vis_loader = data_loader.val_split()
    #vis_loader = config.get('vis_loader', data_loaders)

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
    trainer = MFCCShapeTrainer(config, data_loader, vis_loader,
                         disc_model, disc_loss, disc_optimizer, 
                         gen_model, gen_loss, gen_optimizer)
    trainer.train()

def classify_main(config):
    logger = config.get_logger('train')

    train_loader = config.get('data_loader', data_loaders)
    val_loader = train_loader.val_split()

    model = config.get('arch', models)

    logger.info(model)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = config.get('optimizer', optim, trainable_params)

    loss = config.get_func('loss_func', losses)

    #TODO add Trainer selection to config
    trainer = LrwShapeTrainer(config, train_loader, model,
                              loss, optimizer, val_loader)
    trainer.train()

if __name__ == "__main__":
    fix_seed(0)

    config = GetConfig('./config/mfcc_shape_gan/config.json')
    gan_main(config)

    #config = GetConfig('./config/lrw_shape_classifier/config.json')
    #classify_main(config)
