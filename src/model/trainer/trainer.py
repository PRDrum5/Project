import os
import torch
import numpy as np
from base import BaseGanTrainer, BaseTrainer, BaseTwoCriticsGanTrainer
from logger import TensorboardWriter
from torchvision.utils import save_image
from model.losses import gradient_penalty


class VocaShapeTrainer(BaseGanTrainer):
    def __init__(self, config, data_loader,
                 disc_model, disc_loss, disc_optimizer,
                 gen_model, gen_loss, gen_optimizer):

        self.data_loader = data_loader
        self.batch_size = data_loader.batch_size
        self.len_epoch = len(self.data_loader)

        self.log_step = int(np.sqrt(self.batch_size))
        self.len_train_epoch = len(self.data_loader)

        self.penalty = config['discriminator']['loss_func']['gradient_penalty']
        self.z_dim = config['generator']['arch']['args']['z_dim']
        self.n_labels = config['generator']['arch']['args']['n_labels']

        super().__init__(config, 
                         disc_model, disc_loss, disc_optimizer,
                         gen_model, gen_loss, gen_optimizer)
    
    def _disc_training_epoch(self, epoch, melspec, shape_param):
        self.disc_model.train()
        height, width = melspec.size(2), melspec.size(3)
        noise = torch.randn(self.batch_size, self.z_dim, height, width)
        noise = noise.to(self.device)

        fake_shapes = self.gen_model(noise, melspec).detach()
        real_logit = self.disc_model(shape_param, melspec)
        fake_logit = self.disc_model(fake_shapes, melspec)

        disc_real_loss, disc_fake_loss = self.disc_loss(real_logit, fake_logit)
        if self.penalty:
            gp = self.penalty * gradient_penalty(self.disc_model, 
                                                 real=shape_param,
                                                 fake=fake_shapes,
                                                 conditional=melspec)
            disc_loss = disc_real_loss + disc_fake_loss + gp
        else:
            disc_loss = disc_real_loss + disc_fake_loss

        self.disc_model.zero_grad()
        disc_loss.backward()
        self.disc_optimizer.step()

        return disc_loss
    
    def _gen_training_epoch(self, epoch, melspec):
        self.gen_model.train()
        height, width = melspec.size(2), melspec.size(3)
        noise = torch.randn(self.batch_size, self.z_dim, height, width)
        noise = noise.to(self.device)

        fake_shapes = self.gen_model(noise, melspec)
        fake_logit = self.disc_model(fake_shapes, melspec)

        gen_loss = self.gen_loss(fake_logit)

        self.gen_model.zero_grad()
        gen_loss.backward()
        self.gen_optimizer.step()

        return gen_loss
    
    def train(self):
        fixed_sample = next(iter(self.data_loader))
        fixed_melspec = fixed_sample['melspec'].to(self.device)
        height, width = fixed_melspec.size(2), fixed_melspec.size(3)
        fixed_noise = torch.randn(self.batch_size, self.z_dim, height, width)
        fixed_noise = fixed_noise.to(self.device)


        for epoch in range(1, self.epochs+1):
            total_disc_loss = 0
            total_gen_loss = 0
            for batch_idx, sample in enumerate(self.data_loader):
                step = (epoch-1) * self.len_train_epoch + batch_idx

                melspec = sample['melspec'].to(self.device)
                shape_param = sample['shape_param'].to(self.device)

                # Train Discriminator
                disc_loss = self._disc_training_epoch(epoch, 
                                                      melspec, 
                                                      shape_param)
                total_disc_loss += disc_loss
                if batch_idx % self.log_step == 0:
                    self.logger.info(
                        'Train Epoch: {} '
                        'Batch: {} '
                        'Disc Loss: {:.6f} '.format(
                            epoch, batch_idx+1, disc_loss))

                # Train Generator
                if batch_idx % self.disc_gen_ratio == 0:
                    gen_loss = self._gen_training_epoch(epoch, melspec)
                    total_gen_loss += gen_loss
                    if batch_idx % self.log_step == 0:
                        self.logger.info(
                            'Train Epoch: {} '
                            'Batch: {} '
                            'Gen Loss: {:.6f} '.format(
                                epoch, batch_idx+1, gen_loss))

                self.writer.set_step(step)
                self.writer.add_scalar('disc_loss', disc_loss)
                self.writer.add_scalar('gen_loss', gen_loss)
            
            disc_loss = total_disc_loss / self.len_epoch
            gen_loss = (total_gen_loss / self.len_epoch) * self.disc_gen_ratio
            self.logger.info('Disc Loss: {} '
                             'Gen Loss: {}'.format(disc_loss, gen_loss))

            generated_sample = self.gen_model(fixed_noise, fixed_melspec)
            sample_name = 'generated_sample_epoch_%03d.png' % epoch
            save_dir = self.config.samples_dir / sample_name

            save_image(generated_sample.cpu(), save_dir)
            #if epoch % self.save_period == 0:
            #    self._save_checkpoint(epoch)


class MfccShapeTrainer(BaseGanTrainer):
    def __init__(self, config, data_loader, test_loader,
                 disc_model, disc_loss, disc_optimizer, disc_scheduler,
                 gen_model, gen_loss, gen_optimizer, gen_scheduler):

        self.data_loader = data_loader
        self.test_loader = test_loader
        self.batch_size = self.data_loader.batch_size
        self.test_batch_size = test_loader.batch_size
        self.len_epoch = len(self.data_loader)

        self.disc_scheduler = disc_scheduler
        self.gen_scheduler = gen_scheduler

        if config['log_all']:
            self.log_step = 1
        else:
            self.log_step = int(np.sqrt(self.batch_size))

        self.penalty = config['discriminator']['loss_func']['gradient_penalty']
        self.z_dim = config['generator']['arch']['args']['z_dim']

        super().__init__(config, 
                         disc_model, disc_loss, disc_optimizer,
                         gen_model, gen_loss, gen_optimizer)
    
    def _disc_training_epoch(self, epoch, mfcc, shape_param):
        self.disc_model.train()
        height, width = mfcc.size(2), mfcc.size(3)

        noise = torch.randn(self.batch_size, self.z_dim, height, width)
        noise = noise.to(self.device)

        fake_shapes = self.gen_model(noise, mfcc).detach()
        real_logit = self.disc_model(shape_param, mfcc)
        fake_logit = self.disc_model(fake_shapes, mfcc)

        disc_real_loss, disc_fake_loss = self.disc_loss(real_logit, fake_logit)
        if self.penalty:
            gp = self.penalty * gradient_penalty(self.disc_model, 
                                                 real=shape_param,
                                                 fake=fake_shapes,
                                                 conditional=mfcc)
            self.writer.add_scalar('critic/gradient_penalty', gp)

            disc_loss = disc_real_loss + disc_fake_loss + gp
        else:
            disc_loss = disc_real_loss + disc_fake_loss

        self.writer.add_scalar('critic/real_loss', disc_real_loss)
        self.writer.add_scalar('critic/fake_loss', disc_fake_loss)

        self.disc_model.zero_grad()
        disc_loss.backward()
        self.disc_optimizer.step()

        return disc_loss
    
    def _gen_training_epoch(self, epoch, mfcc):
        self.gen_model.train()
        height, width = mfcc.size(2), mfcc.size(3)
        noise = torch.randn(self.batch_size, self.z_dim, height, width)
        noise = noise.to(self.device)

        fake_shapes = self.gen_model(noise, mfcc)
        fake_logit = self.disc_model(fake_shapes, mfcc)

        gen_loss = self.gen_loss(fake_logit)

        self.gen_model.zero_grad()
        gen_loss.backward()
        self.gen_optimizer.step()

        return gen_loss
    
    def train(self):
        fixed_sample = next(iter(self.data_loader))
        fixed_mfcc = fixed_sample['mfcc'].to(self.device)
        height, width = fixed_mfcc.size(2), fixed_mfcc.size(3)
        fixed_noise = torch.randn(self.batch_size, self.z_dim, 
                                  height, width)
        fixed_noise = fixed_noise.to(self.device)
        fixed_item_names = fixed_sample['item_name']

        test_sample = next(iter(self.test_loader))
        test_mfcc = test_sample['mfcc'].to(self.device)
        height, width = test_mfcc.size(2), test_mfcc.size(3)
        test_noise = torch.randn(self.test_batch_size, self.z_dim, 
                                  height, width)
        test_noise = test_noise.to(self.device)
        test_item_names = test_sample['item_name']

        for epoch in range(1, self.epochs+1):
            total_disc_loss = 0
            total_gen_loss = 0
            for batch_idx, sample in enumerate(self.data_loader):
                step = (epoch-1) * self.len_epoch + batch_idx
                self.writer.set_step(step)

                mfcc = sample['mfcc'].to(self.device)
                shape_param = sample['shape_param'].to(self.device)

                # Train Discriminator
                disc_loss = self._disc_training_epoch(epoch, 
                                                      mfcc, 
                                                      shape_param)
                total_disc_loss += disc_loss
                if batch_idx % self.log_step == 0:
                    self.logger.info(
                        'Train Epoch: {} '
                        'Batch: {} '
                        'Critic Loss: {:.6f} '.format(
                            epoch, batch_idx+1, disc_loss))

                # Train Generator
                if batch_idx % self.disc_gen_ratio == 0:
                    gen_loss = self._gen_training_epoch(epoch, mfcc)
                    total_gen_loss += gen_loss
                    if batch_idx % self.log_step == 0:
                        self.logger.info(
                            'Train Epoch: {} '
                            'Batch: {} '
                            'Gen Loss: {:.6f} '.format(
                                epoch, batch_idx+1, gen_loss))

                self.writer.add_scalar('critic/total_loss', disc_loss)
                self.writer.add_scalar('gen/total_loss', gen_loss)

            disc_loss = total_disc_loss / self.len_epoch
            gen_loss = (total_gen_loss / self.len_epoch) * self.disc_gen_ratio
            self.logger.info('Critic Loss: {} '
                             'Gen Loss: {}'.format(disc_loss, gen_loss))

            self.disc_scheduler.step() 
            self.gen_scheduler.step() 

            self.save_sample(test_noise, test_mfcc, test_item_names, epoch)
            self.save_sample(fixed_noise, fixed_mfcc, fixed_item_names, 
                             epoch, test=False)

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch)

    
    def save_sample(self, noise, mfcc, sample_names, epoch, test=True):

        gen_sample = self.gen_model(noise, mfcc).detach().to('cpu')
        gen_sample = gen_sample.squeeze(2)
        gen_sample = gen_sample.numpy()
        gen_sample = self.test_loader.dataset.denorm(gen_sample)

        for sample_num, sample_name in enumerate(sample_names):
            gen_sample_num = gen_sample[sample_num,:,:]

            if test:
                save_dir = os.path.join(self.config.test_samples_dir, 
                                        str(epoch))
            else:
                save_dir = os.path.join(self.config.train_samples_dir, 
                                        str(epoch))
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

            save_path = os.path.join(save_dir, sample_name)
            np.save(save_path, gen_sample_num)

class LrwShapeTrainer(BaseTrainer):
    def __init__(self, config, train_loader,
                 model, loss_func, optimizer, scheduler=None, val_loader=None):
    
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.scheduler = scheduler

        self.batch_size = self.train_loader.batch_size

        self.len_train_epoch = len(self.train_loader)

        if val_loader:
            self.val_step = True
            self.len_val_epoch = len(self.val_loader)
            self.val_loss_hist = [np.inf] * config['early_stopping']['length']
        else:
            self.val_step = False

        if config['log_all']:
            self.log_step = 1
        else:
            self.log_step = int(np.sqrt(self.batch_size))

        super().__init__(config, model, loss_func, optimizer)
    
    def train(self):
        old_val_ave = self.running_loss_ave()

        for epoch in range(1, self.epochs+1):
            current_val_loss = self._training_epoch(epoch)
            running_ave_val_loss = self.running_loss_ave(current_val_loss)

            # Early stopping
            if old_val_ave < running_ave_val_loss:
                self._save_checkpoint(epoch)
                break
            else:
                old_val_ave = running_ave_val_loss

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch)
    
    def _training_epoch(self, epoch):
        self.model.train()

        val_loss = np.inf
        total_train_loss = 0
        train_correct = 0

        for batch_idx, sample in enumerate(self.train_loader):
            step = (epoch-1) * self.len_train_epoch + batch_idx

            label = sample['label']
            label = label.to(self.device)

            shape_params = sample['shape_params']
            shape_params = shape_params.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(shape_params)
            preds = output.argmax(dim=1, keepdim=True)
            train_correct += preds.eq(label.view_as(preds)).sum().item()
            train_loss = self.loss_func(output, label)
            train_loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch-1) * self.len_train_epoch + batch_idx)
            self.writer.add_scalar('train/loss', train_loss)

            train_acc = train_correct / ((batch_idx+1) * self.batch_size)
            self.writer.add_scalar('train/accuracy', train_acc)

            total_train_loss += train_loss

            if batch_idx % self.log_step == 0:
                mean_train_loss = total_train_loss / ((batch_idx+1) 
                                                      * self.batch_size)
                self.logger.info(
                    'Train Epoch: {} '
                    'Batch: {} '
                    'Loss: {:.6f} '
                    'Accuracy: {:.6f}'.format(epoch, batch_idx, mean_train_loss, train_acc))

        train_loss = total_train_loss / self.len_train_epoch

        if self.val_step:
            val_loss = self._val_epoch(epoch)

        self.scheduler.step() 
        return val_loss

    def _val_epoch(self, epoch):
        total_val_loss = 0
        val_correct = 0

        self.model.eval()

        with torch.no_grad():
            for batch_idx, sample in enumerate(self.val_loader):
                step = (epoch-1) * self.len_train_epoch + batch_idx

                label = sample['label']
                label = label.to(self.device)

                shape_params = sample['shape_params']
                shape_params = shape_params.to(self.device)

                output = self.model(shape_params)
                preds = output.argmax(dim=1, keepdim=True)
                val_correct += preds.eq(label.view_as(preds)).sum().item()
                val_loss = self.loss_func(output, label)
                total_val_loss += val_loss

                val_acc = val_correct / ((batch_idx+1) * self.batch_size)

                if batch_idx % self.log_step == 0:
                    mean_val_loss = total_val_loss / ((batch_idx+1) 
                                                      * self.batch_size)
                    self.logger.info(
                        'Val Epoch: {} '
                        'Batch: {} '
                        'Loss: {:.6f} '
                        'Accuracy: {:.6f}'.format(
                            epoch, batch_idx, mean_val_loss, val_acc))
                self.writer.set_step(
                    (epoch-1) * self.len_val_epoch + batch_idx, 'val')
                self.writer.add_scalar('val/loss', val_loss)
                self.writer.add_scalar('val/accuracy', val_acc)
    
        return mean_val_loss
    
    def running_loss_ave(self, loss=None):
        if loss:
            del self.val_loss_hist[0]
            self.val_loss_hist.append(loss.item())
        average = sum(self.val_loss_hist) / len(self.val_loss_hist)
        return average
