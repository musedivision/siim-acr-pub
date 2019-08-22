from tqdm import tqdm
import numpy as np
import torch
from torch import optim, nn

F = nn.functional

to4dec = lambda a: np.around(a, decimals=4) if not a==None else None

listNumpy = lambda l: [x.item() for x in l]

sumLoss = lambda loss_list: np.sum(listNumpy(loss_list)) / len(loss_list)

caculate_metric = lambda metric: lambda predb, yb, : [ metric(predb, yb).item() for predb, yb in zip(predb, yb)]


def print_epoch_progress(epoch, train_loss=None, valid_loss=None, metrics=[]):
    metrics = [f"{key}: {to4dec(value)}" for key, value in metrics.items()]
    print(epoch, 'train loss: ', to4dec(train_loss),'valid loss: ', to4dec(valid_loss,), ' '.join(metrics))


    
class Learner(object):
    """
        Learner object holds model, optimizer and dataloaders
    """
    def __init__(self, path, model, opt, loss_fn, dls={}, metrics={}):
        self.model = model
        self.opt = opt
        self.loss_fn = loss_fn
        # assumes { trn: train_dl, val: valid_dl)
        self.data = dls
        self.metrics = metrics
        self.models_path = path/'models'
#         self.schedule
    
    def modelNameToPath(self, m): return str(self.models_path/f"{m}.pth")
    
    def save(self, model_name):
        torch.save(self.model.state_dict(), self.modelNameToPath(model_name))
    
    def load(self, model_name=None):
        p = self.modelNameToPath(model_name)
        if not model_name==None: self.model.load_state_dict(torch.load(p))
    
    def descend_(self, xb, yb, is_valid=False):
        
        predb = self.model(xb)
        loss = self.loss_fn(predb, yb)

        if is_valid is not True:
            # cloze {
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()
            # } cloze
        return loss, predb
    
    
    def fit_(self, epochs=1):
        for epoch in range(epochs):
            
             # training
            self.model.train()
            trn_losses = []
            for d in tqdm(self.data['trn']):
                xb, yb = d
                loss, predb = self.descend_(xb, yb)
                trn_losses.append(loss)
                
            
            # validation
            validation_loss = None
            if self.data['val'] is not None:
                self.model.eval()
                with torch.no_grad():
                    val_losses = []
                    for d in tqdm(self.data['val']):
                        xb, yb = d
                        loss, predb = self.descend_(xb, yb, is_valid=True)
                        val_losses.append(loss)
                validation_loss = sumLoss(val_losses)
            
            training_loss = sumLoss(trn_losses)
            
            
            
            metrics = { 
                name:np.mean(caculate_metric(fn)(predb.detach(),yb.detach())) for name, fn in self.metrics.items()
            }
#             = [ m for predb, yb in zip(predb, yb)]
    
#             print_epoch_progress(epoch+1, training_loss, validation_loss, {'accuracy': epoch_accuracy}) 
            print(epoch+1, training_loss, validation_loss, metrics)