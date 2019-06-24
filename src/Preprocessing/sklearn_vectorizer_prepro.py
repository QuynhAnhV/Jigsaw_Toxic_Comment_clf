Old versions:
    
def _gen_all_predictions(ft, lbs, test_df):
    """
    Description of function
    
    Args:
        ft (type_of_ft): Description of ft
        lbs (type_of_lbs): Description of lbs
        
    Returns
        (Type of return) Description of return
    
    """
    vectorizer2 = TfidfVectorizer(min_df=0.001, stop_words = 'english')
    X = vectorizer2.fit_transform(ft)
    ft_test = test_df['comment_text']    
    Xte = vectorizer2.transform(ft_test)
    
    lb_proba = []
    for label in tqdm(lbs.columns):
        y = lbs[label]
        clf = MultinomialNB()
        clf.fit(X, y)

        # Make prediction on test set and return the highest probability        
        pred_result = clf.predict_proba(Xte)
        prob = pred_result[:, 1]
        lb_proba.append(prob)

    return lb_proba
    
Modular versions:

def _gen_feats(*args, **kwargs):
    """
    Generate features for training and testing
    
    Args:
        args[0]: features (comment_text) of training set
        args[1]: features of test set
        args[2]: chosen type of Vectorizer
        kwargs: hyper parameters used for Vectorizer
        
    Returns:
        Features for training (X) and testing (Xte)
        
    """

    
    vectorizer = args[2](kwargs)
    X = vectorizer.fit_transform(args[0])
    Xte = vectorizer.transform(args[1])
    return X, Xte
    
def _get_y_dropcols(*args):
    """
    Extracts all label fields & ID.
    
    Args:
        arg[0]: Raw training dataframe with labels, ID, & comment text
        arg[1]: columns to drop 
        arg[2]: label to predict on
    
    Returns:
    
    """
  
    # Save dropped columns into new variable before dropping
    cmt_id = args[0][args[1]]
    
    # Create new dataframe without dropped columns

    for label in args[0].columns:
        y = args[0][args[2]]
    
    return y, cmt_id
    
def preprocessing(*args, **kwargs):
    """
    Preprocessing the raw data to get data ready to be trained and tested
    
    Args:
        args[0]: features (comment_text) of training set
        args[1]: features of test set
        args[2]: chosen type of Vectorizer
        args[3]: labels of training set
        args[4]: columns to drop from label (id)
        args[5]: label to predict on
        kwargs: hyper parameters used for Vectorizer
        
    Returns:
        Training data (X, y), test data(Xte), dropped columns
        
    """
        
    X, Xte = _gen_feats(args[0], args[1], args[2], kwargs)
    y, cmt_id = _get_y_dropcols(args[3], args[4], args[5])
        
    return X, Xte, y, cmt_id
    
def _gen_model(X, y, model_type):
    """
    
    Generate a fitted model ready to make prediction
    
    X: training features (comment_text)
    y: training lables
    
    Returns:
        A model ready for predicting test set
    
    """
    
    # Initialize model type with "model"
    model = model_type()
    # Fitting model with training data X, y
    model.fit(X, y)
    return model
    
def _gen_preds(Xte, model):
    """
    Make prediction on test set and return the predicted probability for the testing label
    
    Args: 
        Xte: test data set with only comment_text column
        model: fitted model generated in gen_model
        
    Returns
        (float) 
    
    """

    pred_result = model.predict_proba(Xte)
    prob = pred_result[:, 1]
    return prob   
    
def output_transform(raw_output, *args):

    """
    Transform raw output to appropriate form for submission 
    
    Args:
        raw_output:(dataframe) output from _gen_preds function
        args: columns' name
        
    Return:
        Transformed dataframe ready for converting to csv 
        
    """
    # Transpose the output so labels are columns and comments are rows
    trans_df = raw_output.T
    # Rename all columns
    trans_df.columns = args
    # Insert the id column from df_test at position 0 to the left 
    trans_df.insert(0, 'id', df_test['id'], True)
    # Reset index to start from 1 instead of 0
    trans_df.index += 1 
    
    return trans_df