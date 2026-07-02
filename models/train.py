import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np 

def create_model_dict(models, criterions = None, weights=None):

    if criterions is None:
        criterions = [F.mse_loss for _ in range(len(models))]
    if weights is None:
        weights = [1.0 for _ in range(len(models))]

    model_dict = {}
    for model, criterion, weight in zip(models, criterions, weights):
        model_dict[model.__class__.__name__] = {
            'model': model,
            'criterion': criterion,
            'weight': weight
        }
    return model_dict


class ModelTrainer:
    def __init__(self, models, criterions = None, weights=None):
        self.model_dict = create_model_dict(models, criterions, weights)

    def compute_estimates(self, binaural_signals):

        self.estimates = {}
        self.kwargs_output = {}
        
        for model_name, model in self.model_dict.items():

            output = model['model'](binaural_signals)
            self.estimates[model_name] = output['output']
            self.kwargs_output[model_name] = output['kwargs']


    def compute_loss(self, targets):

        self.loss_dict = {}
        total_loss = 0
        for model_name, model in self.model_dict.items():
            self.loss_dict[model_name] = model['criterion'](self.estimates[model_name], targets[model_name]) * model['weight']
            total_loss += self.loss_dict[model_name]*model['weight']
        self.loss_dict['total'] = total_loss
        return self.loss_dict

    


