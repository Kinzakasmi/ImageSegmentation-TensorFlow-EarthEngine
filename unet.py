import tensorflow as tf
import os
from tensorflow.python.keras.models import Model
from tensorflow.keras.layers import Layer, InputSpec, Conv2D, DepthwiseConv2D, UpSampling2D, ZeroPadding2D, Lambda, AveragePooling2D, MaxPooling2D, Conv2DTranspose, Input, Activation, Concatenate, Add, Reshape, BatchNormalization, Dropout, concatenate
from tensorflow.keras import backend as K

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report

class DLModel : 
    def __init__(self,training,num_features,num_classes,batch_size,optimiser,loss,metrics,checkpoint_file):
        self.num_features = num_features
        self.num_classes  = num_classes
        self.optimiser    = optimiser
        self.loss         = loss
        self.metrics      = metrics
        self.checkpoint_file = checkpoint_file

    def unet(self,pretrained_weights = None):
        def conv_block(input_tensor, num_filters):
            encoder = Conv2D(num_filters, (3, 3), padding='same',kernel_initializer = 'he_normal')(input_tensor)
            encoder = BatchNormalization()(encoder)
            encoder = Activation('relu')(encoder)
            encoder = Conv2D(num_filters, (3, 3), padding='same',kernel_initializer = 'he_normal')(encoder)
            encoder = BatchNormalization()(encoder)
            encoder = Activation('relu')(encoder)
            return encoder

        def encoder_block(input_tensor, num_filters):
            encoder = conv_block(input_tensor, num_filters)
            encoder_pool = MaxPooling2D((2, 2), strides=(2, 2))(encoder)
            return encoder_pool, encoder

        def decoder_block(input_tensor, concat_tensor, num_filters):
            decoder = Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same',kernel_initializer = 'he_normal')(input_tensor) 
            decoder = concatenate([concat_tensor, decoder], axis=-1)
            decoder = BatchNormalization()(decoder)
            decoder = Activation('relu')(decoder)
            decoder = Conv2D(num_filters, (3, 3), padding='same',kernel_initializer = 'he_normal')(decoder)
            decoder = BatchNormalization()(decoder)
            decoder = Activation('relu')(decoder)
            decoder = Conv2D(num_filters, (3, 3), padding='same',kernel_initializer = 'he_normal')(decoder)
            decoder = BatchNormalization()(decoder)
            decoder = Activation('relu')(decoder)
            return decoder

        inputs = Input(shape=[None, None, self.num_features]) # 256
        encoder_pool, encoder0 = encoder_block(inputs, 32) # 128
        encoder_pool, encoder1 = encoder_block(encoder_pool, 64) # 64
        encoder_pool, encoder2 = encoder_block(encoder_pool, 128) # 32
        encoder_pool, encoder3 = encoder_block(encoder_pool, 256) # 16
        encoder_pool, encoder4 = encoder_block(encoder_pool, 512) # 8
        center = conv_block(encoder_pool, 1024) # center
        decoder = decoder_block(center, encoder4, 512) # 16
        del center,encoder4
        decoder = decoder_block(decoder, encoder3, 256) # 32
        del encoder3
        decoder = decoder_block(decoder, encoder2, 128) # 64
        del encoder2
        decoder = decoder_block(decoder, encoder1, 64) # 128
        del encoder1
        decoder = decoder_block(decoder, encoder0, 32) # 256
        del encoder0
        outputs = Conv2D(self.num_classes, (1, 1), activation='softmax')(decoder)
        del decoder
        model = tf.python.keras.models.Model(inputs=[inputs], outputs=[outputs])

        model.compile(
            optimizer=tf.python.keras.optimizers.get(self.optimiser), 
            loss=tf.python.keras.losses.get(self.loss),
            metrics=[tf.python.keras.metrics.get(metric) for metric in self.metrics]
            )
        if pretrained_weights:
            pretrained = tf.keras.models.load_model(pretrained_weights).get_weights()
            model.set_weights(pretrained)
        return model

    def init_model(self,from_checkpoint=False):
        if from_checkpoint :
            if not os.path.isdir(self.checkpoint_file) : 
                if self.num_features == 12 :
                    m = self.unet('models/unet_12_bands_sgd_wscc')#pretrained model
                if self.num_features == 20 :
                    m = self.unet('models/unet_20_bands_sgd_wscc')#pretrained model
                else :
                    m = self.unet()
            else :
                m = self.unet(self.checkpoint_file)
        else :
            m = self.unet()
        return m

class ModelEvaluation():
    def __init__(self,model_name,eval_size,target_names,label_names):
        self.model_name   = model_name
        self.eval_size    = eval_size
        self.target_names = target_names
        self.label_names  = label_names
        
    def get_labels(self,model,evaluation) :
        y_true = []
        y_pred = []
        i=0
        for feature,label in evaluation.take(self.eval_size) :
            if i%500 == 0 :
                print('processing evaluation sample number ',i)
            i+=1
            y_true.append(label.numpy())
            y_pred.append(tf.argmax(model.predict(
                x=feature, 
                batch_size=1,
                steps=1,
                verbose=0),axis=3).numpy())
        y_pred = np.stack(y_pred).reshape([-1])
        y_true = np.stack(y_true).reshape([-1])
        return y_true, y_pred
    
    def get_performance(self,y_true,y_pred,label_names,target_names):
        results = classification_report(y_true,y_pred,labels=label_names,target_names=target_names,output_dict=True)
        df = pd.DataFrame.from_dict(results).transpose()
        df['precision'] *= 100
        df['recall']    *= 100
        df['f1-score']  *= 100
        df = df.round(2)
        print(df)
        
    def visualize_confusion_matrix(self,y_true,y_pred_argmax,name):
        """
        param y_pred_arg
        param y_true
        return:
        """
        cm = tf.math.confusion_matrix(y_true,y_pred_argmax).numpy()
        con_mat_df = pd.DataFrame(cm)
        sns.heatmap(con_mat_df, annot=True, fmt='g', cmap=plt.cm.Blues)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig('results/'+name+'_confusion_matrix.png')
        plt.show()
        return con_mat_df
    
    def evaluate(self,model,evaluation):
        y_true,y_pred = self.get_labels(model,evaluation)
        self.get_performance(y_true,y_pred,self.label_names,self.target_names)
        self.visualize_confusion_matrix(y_true,y_pred,self.model_name)

    

