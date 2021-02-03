import torch

import keras.backend as K


def youdens_index_keras(y_pred, y_true):
  '''Custom loss metric for imbalanced binary classification data
  
  Args:
      y_pred: batch prediction output
      y_true: ground truth
      
  Returns:
      Youden's index Keras metric
  '''

  def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    
    return (true_positives / (possible_positives + K.epsilon()))

  def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    
    return (true_negatives / (possible_negatives + K.epsilon()))

  return specificity(y_true, y_pred) + sensitivity(y_true, y_pred) - 1



 def youdens_index_pytorch(output, target):
  '''Custom loss metric for imbalanced binary classification data
  
  Args:
      output : batch prediction output
      target : ground truth
  
  Returns:
      Youden's index PyTorch training metric
  '''

  def sensitivity(output, target):
    true_positives = torch.sum(torch.round(torch.clamp(output, target, 0, 1)))
    possible_positives = torch.sum(torch.round(torch.clamp(target, 0, 1)))
    
    return (true_positives / (possible_positives + 1e-7))

  def specificity(output, target):
    true_negatives = torch.sum(torch.round(torch.clamp((1-output) * (1-target), 0, 1)))
    possible_negatives = torch.sum(torch.round(torch.clamp(1-target, 0, 1)))
    
    return (true_negatives / (possible_negatives + 1e-7))

  return specificity(output, target) + sensitivity(output, target) - 1
