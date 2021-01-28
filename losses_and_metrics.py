import tensorflow as tf 
from tensorflow.keras import backend as K
import os

def get_class_weights(training,size,kernel_size,num_classes) :
    import json
    if os.path.isfile('class_weights.json') :
        with open('class_weights.json', 'r') as j:
            class_weights = json.loads(j.read())
            class_weights = {int(k) : class_weights[k] for k in class_weights}
    else :
        #calculate the occurence of each label
        occurence = {i:0 for i in range(num_classes)}
        j = 0
        for _,labels in training.unbatch().take(size) : 
            if j%100 == 0 :
                print('processing training sample number : ',j)
            j +=1
            lab,_,count = tf.unique_with_counts(tf.reshape(labels,[-1]))
            #tensor is unhashable, so we convert it to numpy
            lab = lab.numpy()
            count = count.numpy()

            for i in range(len(lab)) :
                occurence[lab[i]] += count[i]

        #create class weights
        class_weights = {}
        n_samples = kernel_size*kernel_size*size
        for k in occurence :
            if occurence[k] == 0 : 
                occurence[k] = 1
            class_weights[k] = n_samples/(num_classes*occurence[k])

        #save class weights
        import json
        with open("class_weights.json", "w") as a_file :
            json.dump(class_weights, a_file)
        
        weights = list(class_weights.values())
        for k in class_weights:
            class_weights[k] = round((class_weights[k]-min(weights)+0.2)/(max(weights)-min(weights)),4)
    return class_weights

class Loss : 
    def __init__(self,class_weights,num_classes) :
        self.class_weights = class_weights
        self.num_classes = num_classes

    def weighted_sparse_categorical_crossentropy(self,y_true, y_pred):
        """calculates class weights and returns weighted loss"""
        weights = list(self.class_weights.values())
        weights = tf.gather(weights,y_true)[:,:,:,0]
        return weights*tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

    def dice_coef(self,y_true, y_pred, smooth=1e-7):
        '''
        Dice coefficient for n categories.
        Pass to model as metric during compile statement
        '''
        y_true_f = K.flatten(K.one_hot(K.cast(y_true[...,0], 'int32'), num_classes=self.num_classes))
        y_pred_f = K.flatten(y_pred)
        intersect = K.sum(y_true_f * y_pred_f, axis=-1)
        denom = K.sum(y_true_f + y_pred_f, axis=-1)
        return K.mean((2. * intersect / (denom + smooth)))

    def dice_coef_loss(self,y_true, y_pred):
        '''
        Dice loss to minimize. Pass to model as loss during compile statement
        '''
        return 1 - self.dice_coef(y_true, y_pred)

    def jaccard_distance(self,y_true, y_pred, smooth=10):
        """Jaccard distance for semantic segmentation.
        Also known as the intersection-over-union loss.
        This loss is useful when you have unbalanced numbers of pixels within an image because it gives all classes equal weight.
        The loss has been modified to have a smooth gradient as it converges on zero.
        This has been shifted so it converges on 0 and is smoothed to avoid exploding or disappearing gradient.
        Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
                = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
        # Arguments
            y_true: The ground truth tensor.
            y_pred: The predicted tensor
            smooth: Smoothing factor. Default is 100.
        # Returns
            The Jaccard distance between the two tensors.
        """
        y_true = K.flatten(K.one_hot(K.cast(y_true[...,0], 'int32'), num_classes=self.num_classes))
        y_pred = K.flatten(y_pred)
        intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
        sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
        jac = (intersection + smooth) / (sum_ - intersection + smooth)
        return (1 - jac) * smooth
    
    def get_loss(self,loss):
        if loss == 'weighted_scc' :
            return self.weighted_sparse_categorical_crossentropy
        elif loss == 'scc' :
            return 'sparse_categorical_crossentropy'
        elif loss == 'dice' :
            return self.dice_coef_loss
        elif loss == 'jaccard' :
            return self.jaccard_distance
        else : 
            raise NotImplementedError('Unrecognised loss')

class Metric :
    def __init__(self,num_classes):
        self.num_classes = num_classes
    
    def recall(self,y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(self,y_true, y_pred):
        """Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def f1_score(self,y_true, y_pred):
        y_true = K.one_hot(K.cast(y_true[...,0], 'int32'), num_classes=self.num_classes)
        precision = self.precision(y_true, y_pred)
        recall = self.recall(y_true, y_pred)
        return 2*((precision*recall)/(precision+recall+K.epsilon()))