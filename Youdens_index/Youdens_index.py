import torch
import torch.nn as nn

import keras.backend as K


def youdens_index_keras(y_pred, y_true):
    '''Custom loss metric for imbalanced binary classification data

    Args:
        y_pred: batch prediction output
        y_true: ground truth

    Returns:
        Youden's index Keras metric
    '''

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    sensitivity =  (true_positives / (possible_positives + K.epsilon()))

    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))

    specificity = (true_negatives / (possible_negatives + K.epsilon()))

    return specificity + sensitivity - 1



class YoudensIndex(nn.Module):
    '''Custom loss metric for imbalanced binary classification data'''

    def __init__():
        super(YoudensIndex, self).__init__()
        
    def forward(self,
                logits,
                targets):
        '''Calculate Youden's Index.    
        
        Attr:
            output : batch prediction output
            target : ground truth

        Returns:
            Youden's index PyTorch training metric
        '''

        true_positives = torch.sum(torch.round(torch.clamp(logits, targets, 0, 1)))
        possible_positives = torch.sum(torch.round(torch.clamp(targets, 0, 1)))

        sensitivity = (true_positives / (possible_positives + 1e-7))

        true_negatives = torch.sum(torch.round(torch.clamp((1-logits) * (1-targets), 0, 1)))
        possible_negatives = torch.sum(torch.round(torch.clamp(1-targets, 0, 1)))

        specificity = (true_negatives / (possible_negatives + 1e-7))

        return specificity + sensitivity - 1
