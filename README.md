# Youdens_index

Youden's index, a metric for imbalanced data with Tensorflow-Keras and PyTorch. Keras 2.2.4 and Tensorflow 1.13.

Custom Keras metric that measures the maximization of
sensitivity and specificity following Kaivanto (2008):

    Kaivanto, K. (2008).
    Maximization of the sum of sensitivity and specificity as a diagnostic cutpoint criterion
    Journal of clinical epidemiology, 61, 516-518.
       
Imbalanced datasets need a custom metric function is used to help evaluate the best epoch to \
use for prediction. [Youden's index](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C9&q=Kaivanto%2C+K.+%282008%29.+Maximization+of+the+sum+of+sensitivity+and+specificity+as+a+diagnostic+cutpoint+criterion.+Journal+of+clinical+epidemiology%2C+61%285%29%2C+517.&btnG=) is one such metric. 
It is a measure from epidemiology that is somewhat similar to the geometric mean of the sum of 
sensitivity and specificity, a measure that is often used to score machine learning models 
with imbalanced data [He and Garcia 2009](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C9&q=He%2C+H.%2C+%26+Garcia%2C+E.+A.+%282008%29.+Learning+from+imbalanced+data.+IEEE+Transactions+on+Knowledge+%26+Data+Engineering%2C+%289%29%2C+1263-1284.&btnG=).
       
Usage:

    model = Sequential()
    ...
    model.compile(..., metrics=['accuracy', youdens_index], ...)
    
    result = model.fit(train_x,
                       train_y,
                       ...)

This metric is suitable for training with imbalanced 
classification models.

`python setup.py install` to install.

MIT License
