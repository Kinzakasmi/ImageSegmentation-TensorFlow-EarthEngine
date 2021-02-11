import tensorflow as tf 
import numpy as np
import ee

def feature_process(features):
    """Function that adds indexes to the features.
    Args : 
        features : tensor
        labels : tensor
    Returns :
        A tuple of (inputs, outputs).
    """
    Blue  = features[...,1] 
    Green = features[...,2]
    Red   = features[...,3]
    NIR   = features[...,4]
    SWIR1 = features[...,5]
    SWIR2 = features[...,6]

    ndvi  = tf.math.divide_no_nan(NIR-Red,NIR+Red) #(NIR/Red)
    mndwi = tf.math.divide_no_nan(Green-SWIR1,Green+SWIR1) #(Green/SWIR1)
    awei  = Blue + 2.5*Green - 1.5*(NIR+SWIR1) - 0.25*SWIR2
    vgNIR = tf.math.divide_no_nan(Green-NIR,Green+NIR) #Green/NIR 
    ui    = tf.math.divide_no_nan(SWIR2-NIR,SWIR2+NIR) #(SWIR2/NIR)

    new_features = tf.stack([ndvi,mndwi,awei,vgNIR,ui],axis=-1)
    features = tf.concat([features,new_features],axis=-1)
    return features
    
class TFDatasetProcessing():
    def __init__(self,feature_dict,features,bands,num_features,batch_size=None):
        self.feature_dict = feature_dict
        self.features     = features
        self.bands        = bands
        self.num_features = num_features
        self.batch_size   = batch_size

    def parse_tfrecord(self,example_proto):
        """The parsing function.
        Read a serialized example into the structure defined by FEATURES_DICT.
        Args:
            example_proto: a serialized Example.
        Returns:
            A dictionary of tensors, keyed by feature name.
        """
        return tf.io.parse_single_example(example_proto, self.feature_dict)

    def to_tuple(self,inputs):
        """Function to convert a dictionary of tensors to a tuple of (inputs, outputs).
        Turn the tensors returned by parse_tfrecord into a stack in HWC shape.
        Args:
            inputs: A dictionary of tensors, keyed by feature name.
        Returns:
            A tuple of (inputs, outputs).
        """
        inputsList = [inputs.get(key) for key in self.features]
        stacked    = tf.stack(inputsList, axis=0)
        # Convert from CHW to HWC
        stacked = tf.transpose(stacked, [1, 2, 0])
        labels  = tf.cast(stacked[:,:,len(self.bands):],tf.int32)
        return stacked[:,:,:len(self.bands)], labels

    def data_process(self,features,labels) :
        features = feature_process(features)
        self.num_features = features.shape[-1]
        return features,labels

    def get_dataset(self,pattern):
      """Function to read, parse and format to tuple a set of input tfrecord files.
      Get all the files matching the pattern, parse and convert to tuple.
      Args:
        pattern: A file pattern to match in a Cloud Storage bucket.
      Returns:
        A tf.data.Dataset
      """
      glob    = tf.io.gfile.glob(pattern)
      dataset = tf.data.TFRecordDataset(glob, compression_type='GZIP')
      dataset = dataset.map(self.parse_tfrecord, num_parallel_calls=5)
      dataset = dataset.map(self.to_tuple, num_parallel_calls=5)
      dataset = dataset.map(self.data_process, num_parallel_calls=5)
      return dataset

    def get_training_dataset(self):
      """Get the preprocessed training dataset
      Returns: 
        A tf.data.Dataset of training data.
      """
      glob    = 'data/' + 'train/train*'
      dataset = self.get_dataset(glob)
      dataset = dataset.shuffle(1000).batch(self.batch_size).repeat()
      return dataset

    def get_eval_dataset(self):
      """Get the preprocessed evaluation dataset
      Returns: 
        A tf.data.Dataset of evaluation data.
      """
      glob    = 'data/' + 'eval/traineval*'
      dataset = self.get_dataset(glob)
      return dataset.batch(1).repeat()

    # -------------------- INFERENCE -----------------------#
    def get_inference_dataset(self,filename):
        """Gets th image dataset"""
        import time
        import json

        def parse_image(example_proto):
            return tf.io.parse_single_example(example_proto, imageFeaturesDict)

        def toTupleImage(inputs):
            inputsList = [inputs.get(key) for key in self.bands]
            stacked = tf.stack(inputsList, axis=0)
            stacked = tf.transpose(stacked, [1, 2, 0])
            return stacked

        print('Looking for TFRecord files...')
        # Get a list of all the files in the google drive.
        pattern = 'data/' + 'inference/' + filename +'*'
        exportFilesList = tf.io.gfile.glob(pattern)
        while len(exportFilesList) <2 :
            time.sleep(4)
            exportFilesList = tf.io.gfile.glob(pattern)
        print('files found : ',exportFilesList)

        # Get the list of image files and the JSON mixer file.
        imageFilesList = []
        jsonFile = None
        for f in exportFilesList:
            if f.endswith('.tfrecord.gz'):
                imageFilesList.append(f)
            elif f.endswith('.json'):
                jsonFile = f
        # Make sure the files are in the right order.
        imageFilesList.sort()

        with open(jsonFile) as f :
            mixer = json.loads(f.read())

        patch_size = mixer['patchDimensions']
        # Get set up for prediction.
        buffered_shape = [patch_size[1],patch_size[0]]

        imageColumns = [tf.io.FixedLenFeature(shape=buffered_shape, 
                                                dtype=tf.float32, 
                                                default_value=-9000*tf.ones(buffered_shape)) for k in self.bands]
        imageFeaturesDict = dict(zip(self.bands, imageColumns))
        
        # Create a dataset from the TFRecord file(s) in Google Drive.
        imageDataset = tf.data.TFRecordDataset(imageFilesList, compression_type='GZIP')
        imageDataset = imageDataset.map(parse_image, num_parallel_calls=5)
        imageDataset = imageDataset.map(toTupleImage).batch(1)

        # Perform preprocessing
        imageDataset = imageDataset.map(feature_process)
        self.num_features = imageDataset.element_spec.shape[-1]
        return imageDataset

# ------------------------ NUMPY CONVERSION ------------------------------ #

class NPDatasetProcessing():
    def __init__(self,num_features,num_classes):
        self.num_features = num_features
        self.num_classes = num_classes
    
    def tf_to_numpy(self,dataset,data_size):
        '''Function that converts a tf.Dataset into a dictionary of numpy-arrays of {'features' : [num_pixels,num_features], 
        'labels' : [num_pixels]}'''
        i = 0
        x = []
        y = []
        for features,labels in dataset.unbatch().take(data_size) :
            # From tf.data.Datasets of shape (#,128,128,c) to numpy arrays of shape (#,32,32,c) 
            new_feats = tf.reshape(features,[-1,32,32,self.num_features]).numpy()
            new_labels = tf.reshape(labels,[-1,32,32]).numpy()
            if i == 0 :
                x = [new_feats]
                y = [new_labels]
            else :
                x.append(new_feats)
                y.append(new_labels)
            i+=1
        return {'features' : np.concatenate(x,axis=0).reshape([-1,self.num_features]),'labels' : np.concatenate(y,axis=0).reshape([-1])}

    def get_more_pixels(self,assetId,imgconstructor,tfloader):
        '''Function that adds the misclassified pixels to the dataset
        Arguments :
            assetId : a string representing the FeatureCollection of the misclassified pixels. 
                Example :  "users/leakm/misclassified_pixels"
            imgconstructor : a tfdatasetConstruction object that was used to construct the dataset you want to add pixels to
        Returns : a tuple of (pixels,labels) to be added to the training set
        '''
        geometry = ee.FeatureCollection(assetId)
        list = geometry.toList(geometry.size())

        landsat = imgconstructor.get_landsat8()
        sentinel = imgconstructor.get_sentinel1()
        image = ee.Image.cat(
            [landsat.select(imgconstructor.landsat),
             sentinel.select(imgconstructor.sentinel)]).setDefaultProjection(landsat.projection())

        pixels = []
        labels = []
        for i in range(list.size().getInfo()) :
            new_class = ee.Feature(list.get(i))
            array = image.sampleRectangle(new_class.geometry(),defaultValue=0)
            im = np.dstack([np.array(array.get(b).getInfo()) for b in imgconstructor.bands])
            im = np.reshape(im,[-1,im.shape[-1]])
            pixels.append(im)
            labels.append(np.int64(np.ones(im.shape[0])*new_class.get('landcover').getInfo()))

        pixels = np.concatenate(pixels,axis=0)
        labels = np.concatenate(labels,axis=0)
        pixels,labels = tfloader.feature_process(pixels,labels)
        return pixels,labels
    
    def adding_more_pixels(self,dataset,assetId,imgconstructor,tfloader):
        pixels,labels = self.get_more_pixels(assetId,imgconstructor,tfloader)
        dataset['features'] = np.concatenate((dataset['features'],images),axis=0)
        dataset['labels']   = np.concatenate((dataset['labels'],labels),axis=0)
        return dataset

def undersample(x,y,num_classes,samples_per_class=None):
    '''Function that undersamples the dataset in order to have an equal number of pixels for each class
    Arguments :
        dataset : a dictionary of keys 'features' and 'labels'
        samples_per_class : the number of pixels to sample from each class. 
            If set to None, samples_per_class is set to the number of pixels from the minority class.
    Returns : a tuple of resampled (features,pixels)
    '''
    from imblearn.under_sampling import RandomUnderSampler
    if samples_per_class : 
        samples = {i:samples_per_class for i in range(num_classes)}
        rus = RandomUnderSampler(random_state=0,sampling_strategy=samples)
    else :
        rus = RandomUnderSampler(random_state=0,sampling_strategy='not minority')
    x, y = rus.fit_resample(x, y)
    return x,y
