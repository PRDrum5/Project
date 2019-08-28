import torch
from model import losses, models
from data_loader import data_loaders
from trainer import MfccShapeTrainer, LrwShapeTrainer
from get_config import GetConfig
from utils import fix_seed
import torch.optim as optim

import logging

def gan_main(config):
    logger = config.get_logger('train')

    train_loader = config.get('train_loader', data_loaders)
    val_loader = config.get('val_loader', data_loaders)

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

    disc_scheduler = config.get('discriminator,lr_scheduler', 
                               torch.optim.lr_scheduler, 
                               disc_optimizer)
    gen_scheduler = config.get('generator,lr_scheduler', 
                              torch.optim.lr_scheduler, 
                              gen_optimizer)

    #TODO add Trainer selection to config
    trainer = MfccShapeTrainer(config, train_loader, val_loader,
                         disc_model, disc_loss, disc_optimizer, disc_scheduler, 
                         gen_model, gen_loss, gen_optimizer, gen_scheduler)
    trainer.train()

def classify_main(config):
    logger = config.get_logger('train')

    train_loader = config.get('train_loader', data_loaders)
    val_loader = config.get('val_loader', data_loaders)

    model = config.get('arch', models)

    logger.info(model)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = config.get('optimizer', optim, trainable_params)

    loss = config.get_func('loss_func', losses)

    scheduler = config.get('lr_scheduler', 
                            torch.optim.lr_scheduler, 
                            optimizer)

    #TODO add Trainer selection to config
    trainer = LrwShapeTrainer(config, train_loader, model,
                              loss, optimizer, scheduler, val_loader)
    trainer.train()

if __name__ == "__main__":
    fix_seed(0)

    config = GetConfig('./config/mfcc_shape_gan/config_12mfccs.json')
    gan_main(config)

    #config = GetConfig('./config/two_critics_mfcc_shape_gan/config_2.json')
    #gan_two_critics_main(config)

    #config = GetConfig('./config/lrw_shape_classifier/config.json')
    #classify_main(config)
