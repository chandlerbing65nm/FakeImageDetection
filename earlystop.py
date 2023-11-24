import torch
import os
import torch.distributed as dist

class EarlyStopping:
    def __init__(
        self, 
        path, 
        patience=7, 
        verbose=False, 
        min_lr=1e-6, 
        early_stopping_enabled=True, 
        best_score=None, 
        counter=0,
        args=None,
        ):

        self.patience = patience
        self.verbose = verbose
        self.best_score = best_score
        self.counter = counter
        self.early_stop = False
        self.path = path
        self.early_stopping_enabled = early_stopping_enabled
        self.best_epochs = []
        self.last_epochs = []
        self.min_lr = min_lr
        self.model_name = args.model_name

    def __call__(self, val_accuracy, model, optimizer, epoch):
        score = val_accuracy
        
        if self.early_stopping_enabled:
            if self.best_score is None:
                self.best_score = score
                self.save_best_model(model, optimizer, epoch)
            elif score < self.best_score + 0.001:
                self.counter += 1
                if dist.get_rank() == 0:
                    print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    for param_group in optimizer.param_groups:
                        if param_group['lr'] > self.min_lr:
                            if dist.get_rank() == 0:
                                print(f'Reducing learning rate from {param_group["lr"]} to {param_group["lr"] * 0.1}')
                            param_group['lr'] *= 0.1
                            self.counter = 0  # reset the counter
                        else:
                            self.early_stop = True
            else:
                if self.verbose and dist.get_rank() == 0:
                    print(f'Validation accuracy increased ({self.best_score:.4f} --> {score:.4f}).  Saving model ...')

                self.best_score = score
                self.save_best_model(model, optimizer, epoch)
                self.counter = 0
        
        self.save_last_epochs(model, optimizer, epoch, index=1, laststop=True)

    def save_best_model(self, model, optimizer, epoch):
        # Check if the model is CLIPModel and save only fc layer's state_dict
        if self.model_name == 'clip':
            model_state_dict = model.module.fc.state_dict()
        else:
            model_state_dict = model.state_dict()

        if dist.get_rank() == 0:
            state = {
                'epoch': epoch,
                'counter': self.counter,
                'best_score': self.best_score,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, self.path + '.pth')

        self.save_best_epochs(model, optimizer, epoch, index=1, earlystop=True)

    def save_best_epochs(self, model, optimizer, epoch, index=3, earlystop=False):
        # Check if the model is CLIPModel and save only fc layer's state_dict
        if self.model_name == 'clip':
            model_state_dict = model.module.fc.state_dict()
        else:
            model_state_dict = model.state_dict()

        self.best_epochs.append((epoch, model_state_dict, optimizer.state_dict()))
        
        earlystop = '_best' if earlystop else ''
        # Keep only the latest 3 models
        while len(self.best_epochs) > index:
            oldest_epoch, _, _ = self.best_epochs.pop(0)
            if dist.get_rank() == 0:
                os.remove(f"{self.path}{earlystop}_ep{oldest_epoch}.pth")

        # Save the latest {index} models
        for saved_epoch, model_state_dict, optimizer_state_dict in self.best_epochs[-index:]:
            if dist.get_rank() == 0:
                state = {
                    'epoch': saved_epoch,
                    'counter': self.counter,
                    'best_score': self.best_score,
                    'model_state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer_state_dict,
                }
                torch.save(state, f"{self.path}{earlystop}_ep{saved_epoch}.pth")

    def save_last_epochs(self, model, optimizer, epoch, index=3, laststop=False):
        # Check if the model is CLIPModel and save only fc layer's state_dict
        if self.model_name == 'clip':
            model_state_dict = model.module.fc.state_dict()
        else:
            model_state_dict = model.state_dict()
        
        self.last_epochs.append((epoch, model_state_dict, optimizer.state_dict()))
        
        laststop = '_last' if laststop else ''
        # Keep only the latest 3 models
        while len(self.last_epochs) > index:
            oldest_epoch, _, _ = self.last_epochs.pop(0)
            if dist.get_rank() == 0:
                os.remove(f"{self.path}{laststop}_ep{oldest_epoch}.pth")

        # Save the latest {index} models
        for saved_epoch, model_state_dict, optimizer_state_dict in self.last_epochs[-index:]:
            if dist.get_rank() == 0:
                state = {
                    'epoch': saved_epoch,
                    'counter': self.counter,
                    'best_score': self.best_score,
                    'model_state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer_state_dict,
                }
                torch.save(state, f"{self.path}{laststop}_ep{saved_epoch}.pth")