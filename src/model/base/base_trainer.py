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


class BaseMultiTrainer(BaseDiscriminator, BaseGenerator):
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
            self.disc_model = torch.nn.DataParallel(self.disc_model, 
                                                    device_ids=device_ids)

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