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
    def __init__(self, config, data_loader, vis_loader,
                 disc_model, disc_loss, disc_optimizer,
                 gen_model, gen_loss, gen_optimizer):

        self.data_loader = data_loader
        self.vis_loader = vis_loader
        self.batch_size = self.data_loader.batch_size
        self.vis_batch_size = vis_loader.batch_size
        self.len_epoch = len(self.data_loader)

        self.log_step = int(np.cbrt(self.batch_size))
        self.len_train_epoch = len(self.data_loader)

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
        fixed_sample = next(iter(self.vis_loader))
        fixed_mfcc = fixed_sample['mfcc'].to(self.device)
        height, width = fixed_mfcc.size(2), fixed_mfcc.size(3)
        fixed_noise = torch.randn(self.vis_batch_size, self.z_dim, 
                                  height, width)
        fixed_noise = fixed_noise.to(self.device)
        fixed_item_names = fixed_sample['item_name']

        for epoch in range(1, self.epochs+1):
            total_disc_loss = 0
            total_gen_loss = 0
            for batch_idx, sample in enumerate(self.data_loader):
                step = (epoch-1) * self.len_train_epoch + batch_idx
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

            self.save_sample(fixed_noise, fixed_mfcc, fixed_item_names, epoch)
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch)
    
    def save_sample(self, noise, mfcc, sample_names, epoch):

        gen_sample = self.gen_model(noise, mfcc).detach().to('cpu')
        gen_sample = gen_sample.squeeze(2)
        gen_sample = gen_sample.numpy()
        gen_sample = self.vis_loader.dataset.denorm(gen_sample)

        for sample_num, sample_name in enumerate(sample_names):
            gen_sample_num = gen_sample[sample_num,:,:]

            save_dir = os.path.join(self.config.samples_dir, str(epoch))
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

            save_path = os.path.join(save_dir, sample_name)
            np.save(save_path, gen_sample_num)


class TwoCriticsMfccShapeTrainer(BaseTwoCriticsGanTrainer):
    def __init__(self, config, data_loader, vis_loader,
                 critic_1_model, critic_1_loss_f, critic_1_optimizer,
                 critic_2_model, critic_2_loss_f, critic_2_optimizer,
                 gen_model, gen_loss, gen_optimizer):

        self.data_loader = data_loader
        self.vis_loader = vis_loader
        self.batch_size = self.data_loader.batch_size
        self.vis_batch_size = vis_loader.batch_size
        self.len_epoch = len(self.data_loader)

        self.log_step = int(np.cbrt(self.batch_size))
        self.len_train_epoch = len(self.data_loader)

        self.penalty_1 = config['mfcc_critic']['loss_func']['gradient_penalty']
        self.penalty_2 = config['shape_critic']['loss_func']['gradient_penalty']
        self.z_dim = config['generator']['arch']['args']['z_dim']

        super().__init__(config, 
                         critic_1_model, critic_1_loss_f, critic_1_optimizer,
                         critic_2_model, critic_2_loss_f, critic_2_optimizer,
                         gen_model, gen_loss, gen_optimizer)
    
    def _critics_training_epoch(self, epoch, mfcc, shape_param):
        self.critic_1_model.train()
        self.critic_2_model.train()
        height, width = mfcc.size(2), mfcc.size(3)

        noise = torch.randn(self.batch_size, self.z_dim, height, width)
        noise = noise.to(self.device)

        fake_shapes = self.gen_model(noise, mfcc).detach()

        real_logit_1 = self.critic_1_model(shape_param, mfcc)
        fake_logit_1 = self.critic_1_model(fake_shapes, mfcc)
        critic_1_loss_values = self.critic_1_loss_f(real_logit_1, fake_logit_1)
        critic_1_real_loss, critic_1_fake_loss = critic_1_loss_values

        real_logit_2 = self.critic_2_model(shape_param)
        fake_logit_2 = self.critic_2_model(fake_shapes)
        critic_2_loss_values = self.critic_2_loss_f(real_logit_2, fake_logit_2)
        critic_2_real_loss, critic_2_fake_loss = critic_2_loss_values


        if self.penalty_1:
            gp_1 = self.penalty_1 * gradient_penalty(self.critic_1_model, 
                                                     real=shape_param,
                                                     fake=fake_shapes,
                                                     conditional=mfcc)
            self.writer.add_scalar('critic_1/gradient_penalty', gp_1)

            critic_1_loss = critic_1_real_loss + critic_1_fake_loss + gp_1
        else:
            critic_1_loss = critic_1_real_loss + critic_1_fake_loss

        self.writer.add_scalar('critic_1/real_loss', critic_1_real_loss)
        self.writer.add_scalar('critic_1/fake_loss', critic_1_fake_loss)
        self.writer.add_scalar('critic_1/total_loss', critic_1_loss)

        if self.penalty_2:
            gp_2 = self.penalty_2 * gradient_penalty(self.critic_2_model, 
                                                     real=shape_param,
                                                     fake=fake_shapes)
            self.writer.add_scalar('critic_2/gradient_penalty', gp_2)

            critic_2_loss = critic_2_real_loss + critic_2_fake_loss + gp_2
        else:
            critic_2_loss = critic_2_real_loss + critic_2_fake_loss

        self.writer.add_scalar('critic_2/real_loss', critic_2_real_loss)
        self.writer.add_scalar('critic_2/fake_loss', critic_2_fake_loss)
        self.writer.add_scalar('critic_2/total_loss', critic_2_loss)

        self.critic_1_model.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_model.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        total_critic_losses = critic_1_loss + critic_2_loss

        return total_critic_losses
    
    def _gen_training_epoch(self, epoch, mfcc):
        self.gen_model.train()
        height, width = mfcc.size(2), mfcc.size(3)
        noise = torch.randn(self.batch_size, self.z_dim, height, width)
        noise = noise.to(self.device)

        fake_shapes = self.gen_model(noise, mfcc)
        fake_logit_1 = self.critic_1_model(fake_shapes, mfcc)
        fake_logit_2 = self.critic_2_model(fake_shapes)

        joint_fake_logit = fake_logit_1 + fake_logit_2

        gen_loss = self.gen_loss(joint_fake_logit)

        self.gen_model.zero_grad()
        gen_loss.backward()
        self.gen_optimizer.step()

        return gen_loss
    
    def train(self):
        fixed_sample = next(iter(self.vis_loader))
        fixed_mfcc = fixed_sample['mfcc'].to(self.device)
        height, width = fixed_mfcc.size(2), fixed_mfcc.size(3)
        fixed_noise = torch.randn(self.vis_batch_size, self.z_dim, 
                                  height, width)
        fixed_noise = fixed_noise.to(self.device)
        fixed_item_names = fixed_sample['item_name']

        for epoch in range(1, self.epochs+1):
            total_critics_loss = 0
            total_gen_loss = 0
            for batch_idx, sample in enumerate(self.data_loader):
                step = (epoch-1) * self.len_train_epoch + batch_idx
                self.writer.set_step(step)

                mfcc = sample['mfcc'].to(self.device)
                shape_param = sample['shape_param'].to(self.device)

                # Train Critics
                critics_loss = self._critics_training_epoch(epoch, 
                                                            mfcc, 
                                                            shape_param)
                total_critics_loss += critics_loss
                if batch_idx % self.log_step == 0:
                    self.logger.info(
                        'Train Epoch: {} '
                        'Batch: {} '
                        'Joint Critics Loss: {:.6f} '.format(
                            epoch, batch_idx+1, critics_loss))

                # Train Generator
                if batch_idx % self.critics_gen_ratio == 0:
                    gen_loss = self._gen_training_epoch(epoch, mfcc)
                    total_gen_loss += gen_loss
                    if batch_idx % self.log_step == 0:
                        self.logger.info(
                            'Train Epoch: {} '
                            'Batch: {} '
                            'Gen Loss: {:.6f} '.format(
                                epoch, batch_idx+1, gen_loss))

                self.writer.add_scalar('critics/total_loss', critics_loss)
                self.writer.add_scalar('gen/total_loss', gen_loss)
            
            critics_loss = total_critics_loss / self.len_epoch
            gen_loss = self.critics_gen_ratio * (total_gen_loss
                                                 / self.len_epoch)
            self.logger.info('Critics Loss: {} '
                             'Gen Loss: {}'.format(critics_loss, gen_loss))

            self.save_sample(fixed_noise, fixed_mfcc, fixed_item_names, epoch)
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch)
    
    def save_sample(self, noise, mfcc, sample_names, epoch):

        gen_sample = self.gen_model(noise, mfcc).detach().to('cpu')
        gen_sample = gen_sample.squeeze(2)
        gen_sample = gen_sample.numpy()
        gen_sample = self.vis_loader.dataset.denorm(gen_sample)

        for sample_num, sample_name in enumerate(sample_names):
            gen_sample_num = gen_sample[sample_num,:,:]

            save_dir = os.path.join(self.config.samples_dir, str(epoch))
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

            save_path = os.path.join(save_dir, sample_name)
            np.save(save_path, gen_sample_num)


class LrwShapeTrainer(BaseTrainer):
    def __init__(self, config, train_loader,
                 model, loss_func, optimizer, val_loader=None):
    
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.batch_size = self.train_loader.batch_size

        self.len_train_epoch = len(self.train_loader)

        if val_loader:
            self.val_step = True
            self.len_val_epoch = len(self.val_loader)
        else:
            self.val_step = False

        self.log_step = int(np.sqrt(self.batch_size))

        super().__init__(config, model, loss_func, optimizer)
    
    def train(self):
        for epoch in range(1, self.epochs+1):
            self._training_epoch(epoch)

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch)
    
    def _training_epoch(self, epoch):
        self.model.train()

        total_loss = 0
        correct = 0
        val_loss = 0

        for batch_idx, sample in enumerate(self.train_loader):
            step = (epoch-1) * self.len_train_epoch + batch_idx

            label = sample['label']
            label = label.to(self.device)

            shape_params = sample['shape_params']
            shape_params = shape_params.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(shape_params)
            preds = output.argmax(dim=1, keepdim=True)
            correct += preds.eq(label.view_as(preds)).sum().item()
            loss = self.loss_func(output, label)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch-1) * self.len_train_epoch + batch_idx)
            self.writer.add_scalar('train/loss', loss)

            acc = correct / ((batch_idx+1) * self.batch_size)
            self.writer.add_scalar('train/accuracy', acc)

            total_loss += loss

            if batch_idx % self.log_step == 0:
                mean_loss = total_loss / ((batch_idx+1) * self.batch_size)
                self.logger.info(
                    'Train Epoch: {} '
                    'Batch: {} '
                    'Loss: {:.6f} '
                    'Accuracy: {:.6f}'.format(epoch, batch_idx, mean_loss, acc))

        train_loss = total_loss / self.len_train_epoch

        if self.val_step:
            self._val_epoch(epoch)

    def _val_epoch(self, epoch):
        total_loss = 0
        correct = 0

        with torch.no_grad():
            for batch_idx, sample in enumerate(self.val_loader):
                step = (epoch-1) * self.len_train_epoch + batch_idx

                label = sample['label']
                label = label.to(self.device)

                shape_params = sample['shape_params']
                shape_params = shape_params.to(self.device)

                output = self.model(shape_params)
                preds = output.argmax(dim=1, keepdim=True)
                correct += preds.eq(label.view_as(preds)).sum().item()
                loss = self.loss_func(output, label)
                total_loss += loss

                acc = correct / ((batch_idx+1) * self.batch_size)

                if batch_idx % self.log_step == 0:
                    mean_loss = total_loss / ((batch_idx+1) * self.batch_size)
                    self.logger.info(
                        'Val Epoch: {} '
                        'Batch: {} '
                        'Loss: {:.6f} '
                        'Accuracy: {:.6f}'.format(
                            epoch, batch_idx, mean_loss, acc))

                self.writer.set_step(
                    (epoch-1) * self.len_train_epoch + batch_idx)
                self.writer.add_scalar('val/loss', loss)
                self.writer.add_scalar('val/accuracy', acc)
