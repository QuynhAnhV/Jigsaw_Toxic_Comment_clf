def sensitivity(y, y_pred):  #if there is 1(y), the model will guess 1(y_pred); but if the model guesses 1, there is not necessary a 1
    """
    Description of your function
    
    Args:
        y (y type): Description of y
        y_pred (y_pred type): Description of y_pred
    
    Returns:
        (return type) Description of returned object
    """
    TP = _tp(y, y_pred)
    FN = _fn(y, y_pred)
    sens = TP/(TP+FN)
    return sens
        
def negative_predictive_value(y, y_pred):
    TN = _tn(y, y_pred)
    FN = _fn(y, y_pred)
    npv = TN/(TN+FN)
    return npv

def specificity(y, y_pred):
    TN = _tn(y, y_pred)
    FP = _fp(y, y_pred)
    spec = TN/(TN+FP)
    return spec

def positive_predictive_value(y, y_pred):
    TP = _tp(y, y_pred)
    FP = _fp(y, y_pred)
    prec = TP/(TP + FP)
    return prec

def recall(y, y_pred):
    return sensitivity(y, y_pred)

def precision(y, y_pred):
    return positive_predictive_value(y, y_pred)

def f1_score(y, y_pred):
    num = 2 * (precision(y, y_pred) * sensitivity(y, y_pred))
    den = precision(y, y_pred) + sensitivity(y, y_pred)
    f1 = num / den
    return f1

def _tp(y, y_pred):
    assert len(y) == len(y_pred)
    tps = 0
    for i in range(len(y)):
        if y[i] == y_pred[i] and y_pred[i] == 1:
            tps += 1
    return tps

def _tn(y, y_pred):
    assert len(y) == len(y_pred)
    tns = 0
    for i in range(len(y)):
        if y[i] == y_pred[i] and y_pred[i] == 0:
            tns += 1
    return tns

def _fp(y, y_pred):
    assert len(y) == len(y_pred)
    fps = 0
    for i in range(len(y)):
        if y[i] == 0 and y_pred[i] == 1:
            fps += 1
    return fps

def _fn(y, y_pred):
    assert len(y) == len(y_pred)
    fns = 0
    for i in range(len(y)):
        if y[i] == 1 and y_pred[i] == 0:
            fns += 1
    return fns
    