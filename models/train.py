import torch
import torch.nn.functional as F
import torch.nn as nn

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

    def compute_estimates(self, binaural_signals, spk_filters, reference_signals):

        N_signals = binaural_signals.shape[0]
        N_listeners = binaural_signals.shape[1]
        N_sources = spk_filters.shape[1]
        binaural_signals = binaural_signals.view(N_signals*N_listeners, 2, -1)

        self.estimates = {}
        self.kwargs_output = {}
        
        for model_name, model in self.model_dict.items():
            
            if model['model'].model_type == 'listener_based':
                output = model['model'](binaural_signals, reference_signals)
                # Reshape output: (N_signals*N_listeners, ...) -> (N_signals, N_listeners, ...)
                out_tensor = output['output']
                remaining_dims = out_tensor.shape[1:]
                self.estimates[model_name] = out_tensor.view(N_signals, N_listeners, *remaining_dims)
                self.kwargs_output[model_name] = output['kwargs']
            
            elif model['model'].model_type == 'loudspeaker_based':
                output = model['model'](spk_filters)
                self.estimates[model_name] = output['output'].view(1, N_sources) # This is done to match the shape of the listener-based models
                self.kwargs_output[model_name] = output['kwargs']
            
            else:
                raise ValueError(f"Model type {model['model'].model_type} not supported")

    def compute_loss(self, targets):

        self.loss_dict = {}
        total_loss = 0
        
        for model_name, model in self.model_dict.items():
            
            estimates = self.estimates[model_name]
            target = targets[model_name].expand_as(estimates)
            
            self.loss_dict[model_name] = model['criterion'](estimates, target) * model['weight']
            total_loss += self.loss_dict[model_name]
        self.loss_dict['total'] = total_loss
        return self.loss_dict

    


