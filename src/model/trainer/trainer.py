import torch
import numpy as np
from base import BaseMultiTrainer, BaseTrainer
from logger import TensorboardWriter
from torchvision.utils import save_image
from model.losses import gradient_penalty


class GANTrainer(BaseMultiTrainer):
    def __init__(self, config, data_loader,
                 disc_model, disc_loss, disc_optimizer,
                 gen_model, gen_loss, gen_optimizer):

        self.data_loader = data_loader
        self.batch_size = data_loader.batch_size
        self.len_epoch = len(self.data_loader)

        self.log_step = int(np.sqrt(self.batch_size))
        self.len_train_epoch = len(self.data_loader)

        self.penalty = config['discriminator']['loss_func']['gradient_penalty']

        super().__init__(config, 
                         disc_model, disc_loss, disc_optimizer,
                         gen_model, gen_loss, gen_optimizer)

    def _disc_training_epoch(self, epoch, data):
        self.disc_model.train()
        noise = torch.randn(self.batch_size, 100).to(self.device)

        fake_data = self.gen_model(noise).detach()
        real_logit = self.disc_model(data)
        fake_logit = self.disc_model(fake_data)

        disc_real_loss, disc_fake_loss = self.disc_loss(real_logit, fake_logit)
        if self.penalty:
            gp = self.penalty * gradient_penalty(self.disc_model, 
                                                 data, 
                                                 fake_data)
            disc_loss = disc_real_loss + disc_fake_loss + gp
        else:
            disc_loss = disc_real_loss + disc_fake_loss

        self.disc_model.zero_grad()
        disc_loss.backward()
        self.disc_optimizer.step()

        return disc_loss
    
    def _gen_training_epoch(self, epoch):
        self.gen_model.train()
        noise = torch.randn(self.batch_size, 100).to(self.device)

        fake_data = self.gen_model(noise)
        fake_logit = self.disc_model(fake_data)

        gen_loss = self.gen_loss(fake_logit)

        self.gen_model.zero_grad()
        gen_loss.backward()
        self.gen_optimizer.step()

        return gen_loss
    
    def train(self):
        fixed_noise = torch.randn(self.batch_size, 100).to(self.device)

        for epoch in range(1, self.epochs+1):
            total_disc_loss = 0
            total_gen_loss = 0
            for batch_idx, (data, _label) in enumerate(self.data_loader):

                step = (epoch-1) * self.len_train_epoch + batch_idx

                data = data.to(self.device)

                # Train Discriminator
                disc_loss = self._disc_training_epoch(epoch, data)
                total_disc_loss += disc_loss
                if batch_idx % self.log_step == 0:
                    self.logger.info(
                        'Train Epoch: {} '
                        'Batch: {} '
                        'Disc Loss: {:.6f} '.format(
                            epoch, batch_idx+1, disc_loss))

                # Train Generator
                if batch_idx % self.disc_gen_ratio == 0:
                    gen_loss = self._gen_training_epoch(epoch)
                    total_gen_loss += gen_loss
                    if batch_idx % self.log_step == 0:
                        self.logger.info(
                            'Train Epoch: {} '
                            'Batch: {} '
                            'Gen Loss: {:.6f} '.format(
                                epoch, batch_idx+1, gen_loss))

                self.writer.add_scalar('disc_loss', disc_loss)
                self.writer.add_scalar('gen_loss', gen_loss)
            
            disc_loss = total_disc_loss / self.len_epoch
            gen_loss = (total_gen_loss / self.len_epoch) * self.disc_gen_ratio
            self.logger.info('Disc Loss: {} '
                             'Gen Loss: {}'.format(disc_loss, gen_loss))

            generated_sample = self.gen_model(fixed_noise)
            sample_name = 'generated_sample_epoch_%03d.png' % epoch
            save_dir = self.config.samples_dir / sample_name

            save_image(generated_sample.cpu(), save_dir)
            #if epoch % self.save_period == 0:
            #    self._save_checkpoint(epoch)


class CGANTrainer(BaseMultiTrainer):
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
    
    def one_hot_labels(self, labels, dtype=torch.float32):
        """
        returns a one hot representation of labels passed in
        labels should be a 1-dim tensor
        Output will be of size [len(labels), n_labels]
        """
        label_array = np.eye(self.n_labels)[labels.cpu().numpy()]
        one_hot = torch.tensor(label_array, dtype=dtype).to(self.device)
        return one_hot

    def _disc_training_epoch(self, epoch, data, label):
        self.disc_model.train()
        noise = torch.randn(self.batch_size, self.z_dim).to(self.device)
        fake_data = self.gen_model(noise, label).detach()
        real_logit = self.disc_model(data, label)
        fake_logit = self.disc_model(fake_data, label)
        disc_real_loss, disc_fake_loss = self.disc_loss(real_logit, fake_logit)
        if self.penalty:
            gp = self.penalty * gradient_penalty(self.disc_model, 
                                                 data,
                                                 fake_data,
                                                 conditional=label)
            disc_loss = disc_real_loss + disc_fake_loss + gp
        else:
            disc_loss = disc_real_loss + disc_fake_loss

        self.disc_model.zero_grad()
        disc_loss.backward()
        self.disc_optimizer.step()
        return disc_loss
    
    def _gen_training_epoch(self, epoch, label):
        self.gen_model.train()
        noise = torch.randn(self.batch_size, self.z_dim).to(self.device)
        fake_data = self.gen_model(noise, label)
        fake_logit = self.disc_model(fake_data, label)
        gen_loss = self.gen_loss(fake_logit)
        self.gen_model.zero_grad()
        gen_loss.backward()
        self.gen_optimizer.step()
        return gen_loss
    
    def train(self):
        fixed_noise = torch.randn(10*self.n_labels, self.z_dim).to(self.device)
        fixed_labels = torch.tensor(list(np.arange(0,self.n_labels))*10)
        fixed_labels = self.one_hot_labels(fixed_labels, 
                                           dtype=fixed_noise.dtype)
        for epoch in range(1, self.epochs+1):
            total_disc_loss = 0
            total_gen_loss = 0
            for batch_idx, (data, label) in enumerate(self.data_loader):
                step = (epoch-1) * self.len_train_epoch + batch_idx
                data = data.to(self.device)
                label = self.one_hot_labels(label)
                # Train Discriminator
                disc_loss = self._disc_training_epoch(epoch, data, label)
                total_disc_loss += disc_loss
                if batch_idx % self.log_step == 0:
                    self.logger.info(
                        'Train Epoch: {} '
                        'Batch: {} '
                        'Disc Loss: {:.6f} '.format(
                            epoch, batch_idx+1, disc_loss))
                # Train Generator
                if batch_idx % self.disc_gen_ratio == 0:
                    gen_loss = self._gen_training_epoch(epoch, label)
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
            generated_sample = self.gen_model(fixed_noise, fixed_labels)
            sample_name = 'generated_sample_epoch_%03d.png' % epoch
            save_dir = self.config.samples_dir / sample_name
            save_image(generated_sample.cpu(), save_dir)
            #if epoch % self.save_period == 0:
            #    self._save_checkpoint(epoch)


class VocaShapeTrainer(BaseMultiTrainer):
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


class MFCCShapeTrainer(BaseMultiTrainer):
    def __init__(self, config, data_loader, vis_loader,
                 disc_model, disc_loss, disc_optimizer,
                 gen_model, gen_loss, gen_optimizer):

        self.data_loader = data_loader
        self.vis_loader = vis_loader
        self.batch_size = self.data_loader.batch_size
        self.vis_batch_size = vis_loader.batch_size
        self.len_epoch = len(self.data_loader)

        self.log_step = int(np.sqrt(self.batch_size))
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

            self.save_sample(fixed_noise, fixed_mfcc, epoch)
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch)
    
    def save_sample(self, noise, mfcc, epoch):
        gen_sample = self.gen_model(noise, mfcc).detach().to('cpu')
        gen_sample = gen_sample.squeeze(0).squeeze(0)
        gen_sample = gen_sample.squeeze(1)
        gen_sample = gen_sample.numpy()
        gen_sample = self.vis_loader.dataset.denorm(gen_sample)
        
        sample_name = 'generated_sample_epoch_%03d' % epoch
        save_dir = self.config.samples_dir / sample_name
        np.save(save_dir, gen_sample)

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
            log = self._training_epoch(epoch)

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
            self.writer.add_scalar('loss', loss)

            acc = correct / ((batch_idx+1) * self.batch_size)
            #self.writer.set_step((epoch-1) * self.len_train_epoch + batch_idx)
            self.writer.add_scalar('accuracy', loss)

            total_loss += loss

            if batch_idx % self.log_step == 0:
                self.logger.info(
                    'Train Epoch: {} '
                    'Batch: {} '
                    'Loss: {:.6f} '
                    'Accuracy: {:.6f}'.format(epoch, batch_idx, loss, acc))

        train_loss = total_loss / self.len_train_epoch

        if self.val_step:
            val_loss = self._val_epoch(epoch)
        
        return {'train_loss': train_loss, 'val_loss': val_loss}

    def _val_epoch(self, epoch):
        total_loss = 0
        correct = 0

        with torch.no_grad():
            for batch_idx, sample in enumerate(self.train_loader):
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
                    print('Val Epoch: {} Batch: {} '
                          'Loss: {:.6f} Accuracy: {:.6f}'.format(
                              epoch, batch_idx, loss, acc))

        val_loss = total_loss / self.len_val_epoch
        return val_loss
