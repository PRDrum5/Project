import torch
from abc import abstractmethod
from logger import TensorboardWriter


class BaseGenerator():
    def __init__(self, model, loss_func, optimizer):
        self.gen_model = model
        self.gen_loss = loss_func
        self.gen_optimizer = optimizer

    @abstractmethod
    def _gen_training_epoch(self, epoch, input=None):
        raise NotImplementedError


class BaseDiscriminator():
    def __init__(self, model, loss_func, optimizer):
        self.disc_model = model
        self.disc_loss = loss_func
        self.disc_optimizer = optimizer

    @abstractmethod
    def _disc_training_epoch(self, epoch, input):
        raise NotImplementedError


class BaseGanTrainer(BaseDiscriminator, BaseGenerator):
    def __init__(self, config, 
                 disc_model, disc_loss, disc_optimizer,
                 gen_model, gen_loss, gen_optimizer):

        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['level'])
        self.device, device_ids = self._prepare_device(config['n_gpu'])
        self.epochs = config['trainer']['epochs']
        self.disc_gen_ratio = config['trainer']['disc_gen_ratio']

        self.save_period = config['trainer']['save_period']
        self.checkpoint_dir = config.save_dir
        # setup visualization writer instance                
        self.writer = TensorboardWriter(config.log_dir, self.logger, 
                                        config['trainer']['tensorboard'])

        BaseDiscriminator.__init__(self, disc_model, disc_loss, disc_optimizer)
        BaseGenerator.__init__(self, gen_model, gen_loss, gen_optimizer)

        self.disc_model = self.disc_model.to(self.device)
        self.gen_model = self.gen_model.to(self.device)
        if len(device_ids) > 1:
            self.disc_model = torch.nn.DataParallel(self.disc_model, 
                                                    device_ids=device_ids)
            self.gen_model = torch.nn.DataParallel(self.gen_model,
                                                    device_ids=device_ids)

    def _save_checkpoint(self, epoch):
        disc_arch = type(self.disc_model).__name__
        disc_state = {
            'arch': disc_arch,
            'epoch': epoch,
            'state_dict': self.disc_model.state_dict(),
            'optimizer': self.disc_optimizer.state_dict(),
            'config': self.config
        }
        disc_filename = f'{self.checkpoint_dir}/disc_checkpoint-{epoch}.pth'
        torch.save(disc_state, disc_filename)

        gen_arch = type(self.gen_model).__name__
        gen_state = {
            'arch': gen_arch,
            'epoch': epoch,
            'state_dict': self.gen_model.state_dict(),
            'optimizer': self.gen_optimizer.state_dict(),
            'config': self.config
        }
        gen_filename = f'{self.checkpoint_dir}/gen_checkpoint-{epoch}.pth'
        torch.save(gen_state, gen_filename)

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids
    
    @abstractmethod
    def train(self):
        raise NotImplementedError

class BaseTwoCriticsGanTrainer():
    def __init__(self, config, 
                 critic_1_model, critic_1_loss_f, critic_1_optimizer,
                 critic_2_model, critic_2_loss_f, critic_2_optimizer,
                 gen_model, gen_loss, gen_optimizer):

        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['level'])
        
        self.critic_1_model = critic_1_model
        self.critic_1_loss_f = critic_1_loss_f
        self.critic_1_optimizer = critic_1_optimizer
        self.critic_1_train = config['trainer']['critic_1_train']

        self.critic_2_model = critic_2_model
        self.critic_2_loss_f = critic_2_loss_f
        self.critic_2_optimizer = critic_2_optimizer
        self.critic_2_train = config['trainer']['critic_2_train']

        self.gen_model = gen_model
        self.gen_loss = gen_loss
        self.gen_optimizer = gen_optimizer
        self.gen_train = config['trainer']['gen_train']

        self.device, device_ids = self._prepare_device(config['n_gpu'])
        self.epochs = config['trainer']['epochs']

        self.save_period = config['trainer']['save_period']
        self.checkpoint_dir = config.save_dir
        # setup visualization writer instance                
        self.writer = TensorboardWriter(config.log_dir, self.logger, 
                                        config['trainer']['tensorboard'])

        self.critic_1_model = self.critic_1_model.to(self.device)
        self.critic_2_model = self.critic_2_model.to(self.device)
        self.gen_model = self.gen_model.to(self.device)

        if len(device_ids) > 1:
            self.critic_1_model = torch.nn.DataParallel(self.critic_1_model, 
                                                        device_ids=device_ids)
            self.critic_2_model = torch.nn.DataParallel(self.critic_2_model, 
                                                        device_ids=device_ids)
            self.gen_model = torch.nn.DataParallel(self.gen_model,
                                                    device_ids=device_ids)

    def _save_checkpoint(self, epoch):
        critic_1_arch = type(self.critic_1_model).__name__
        critic_1_state = {
            'arch': critic_1_arch,
            'epoch': epoch,
            'state_dict': self.critic_1_model.state_dict(),
            'optimizer': self.critic_1_optimizer.state_dict(),
            'config': self.config
        }
        critic_1_filename = f'{self.checkpoint_dir}/mfcc_critic_checkpoint-{epoch}.pth'
        torch.save(critic_1_state, critic_1_filename)

        critic_2_arch = type(self.critic_2_model).__name__
        critic_2_state = {
            'arch': critic_2_arch,
            'epoch': epoch,
            'state_dict': self.critic_2_model.state_dict(),
            'optimizer': self.critic_2_optimizer.state_dict(),
            'config': self.config
        }
        critic_2_filename = f'{self.checkpoint_dir}/shape_critic_checkpoint-{epoch}.pth'
        torch.save(critic_2_state, critic_2_filename)

        gen_arch = type(self.gen_model).__name__
        gen_state = {
            'arch': gen_arch,
            'epoch': epoch,
            'state_dict': self.gen_model.state_dict(),
            'optimizer': self.gen_optimizer.state_dict(),
            'config': self.config
        }
        gen_filename = f'{self.checkpoint_dir}/gen_checkpoint-{epoch}.pth'
        torch.save(gen_state, gen_filename)

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids 

class BaseTrainer():
    def __init__(self, config, model, loss_func, optimizer):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['level'])
        self.device, device_ids = self._prepare_device(config['n_gpu'])

        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.loss_func = loss_func
        self.optimizer = optimizer
        self.epochs = config['trainer']['epochs']

        self.save_period = config['trainer']['save_period']
        self.checkpoint_dir = config.save_dir
        # setup visualization writer instance                
        self.writer = TensorboardWriter(config.log_dir, self.logger, 
                                        config['trainer']['tensorboard'])
    
    @abstractmethod
    def _training_epoch(self, epoch):
        raise NotImplementedError

    @abstractmethod
    def train(self):
        raise NotImplementedError
    
    def _save_checkpoint(self, epoch):
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config
        }
        filename = str(self.checkpoint_dir / 'checkpoint-{}.pth'.format(epoch))
        torch.save(state, filename)

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1

        self.model.load_state_dict(checkpoint['state_dict'])

        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids 
