import os
import tensorflow as tf
import numpy as np
import ee

class Inference :
    def __init__(self,num_classes,model_name):
        self.num_classes = num_classes
        import joblib
        if 'umap' in model_name or 'pca' in model_name :
          self.mapper = joblib.load('models/'+model_name+'.sav')
        else :
          self.mapper = None
        if 'unet' in model_name : 
            self.model = tf.keras.models.load_model('models/'+model_name)
        else :
            self.model = joblib.load('models/'+model_name+'.joblib')

    def crf(self,original_image, mask_img):
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_labels,create_pairwise_gaussian,create_pairwise_bilateral
        #Setting up the CRF model
        d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], self.num_classes)
        # get unary potentials (neg log probability)
        U = unary_from_labels(mask_img,self.num_classes, gt_prob=0.8, zero_unsure=False)
        d.setUnaryEnergy(U)
        del U 

        # This creates the color-independent features and then add them to the CRF
        feats = create_pairwise_gaussian(sdims=(1, 1), shape=original_image.shape[:2])
        d.addPairwiseEnergy(feats,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC,
                            compat=10) # `compat` is the "strength" of this potential.)

        # This creates the color-dependent features and then add them to the CRF
        feats = create_pairwise_bilateral(sdims=(2, 2), schan=tuple([2]*original_image.shape[-1]),img=original_image, chdim=2)
        d.addPairwiseEnergy(feats,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC,
                            compat=10) # `compat` is the "strength" of this potential.)
        del feats 
        
        ##### Do inference and compute MAP #####
        # Run five inference steps. (you can add steps)
        Q = d.inference(5)
        # Find out the most probable class for each pixel.
        MAP = np.argmax(Q, axis=0)
        # Convert the MAP (labels) back to the corresponding colors and save the image.
        del Q
        return MAP.reshape(mask_img.shape)
    
    def writePrediction(self,name,predictions):
        print('Writing predictions...')
        out_image_file = 'data//predictions/tfrecords/' + name + '.TFRecord'
        writer = tf.io.TFRecordWriter(out_image_file)
        # Create an example.
        example = tf.train.Example(
        features=tf.train.Features(
            feature={
            'landcover': tf.train.Feature(
                int64_list=tf.train.Int64List(
                    value=predictions.flatten()))
            }
        )
        )
        # Write the example.
        writer.write(example.SerializeToString())
        writer.close()  

    def doMLPrediction(self,imageDataset,filename,num_features,perform_crf=False):
        import copy
        for image in imageDataset :
            im_numpy = image.numpy()
        del imageDataset
        size = im_numpy.shape

        # Reduce dimension if a mapper (umap or pca) is provided
        if self.mapper :
            print('Reducing dimension...')
            mapped = self.mapper.transform(np.reshape(im_numpy,[-1,num_features]))
        else :
            mapped = np.reshape(copy.deepcopy(im_numpy),[-1,num_features])

        # Perform inference.
        print('Running predictions...')
        predictions = self.model.predict(mapped)
        del mapped
        predictions = np.int64(np.reshape(predictions,[size[1],size[2]]))

        self.writePrediction(filename,predictions)

        if perform_crf : 
            print('Running crf...')
            if 'saur' in filename: # add here any name that provokes memory issues
                new_pred1 = self.crf(im_numpy[0][:im_numpy.shape[1]//2,:,:],predictions[:im_numpy.shape[1]//2,:])
                new_pred2 = self.crf(im_numpy[0][im_numpy.shape[1]//2:,:,:],predictions[im_numpy.shape[1]//2:,:])
                predictions_crf = np.concatenate([new_pred1,new_pred2],axis=0)
                self.writePrediction(filename+'_crf',predictions_crf)
            else:
                predictions_crf = self.crf(im_numpy[0],predictions)
                self.writePrediction(filename+'_crf',predictions_crf)
            return predictions_crf
        else :
            return predictions
    
    def doDLPrediction(self,imageDataset,filename):
        # Perform inference.
        print('Running predictions...')
        predictions = self.model.predict(imageDataset,batch_size=1)
        self.writePrediction(filename,predictions)
    
def palette(M):
    import matplotlib
    import numpy as np
    R = tf.Variable(tf.zeros(M.shape,tf.float32))
    G = tf.Variable(tf.zeros(M.shape,tf.float32))
    B = tf.Variable(tf.zeros(M.shape,tf.float32))
    Mc = [R,G,B]
    for i in range(3) :
        Mc[i].assign(tf.where(M==0,255*matplotlib.colors.to_rgb('lime')[i],Mc[i]))#field
        Mc[i].assign(tf.where(M==1,255*matplotlib.colors.to_rgb('darkgreen')[i],Mc[i]))#forest
        Mc[i].assign(tf.where(M==2,255*matplotlib.colors.to_rgb('yellow')[i],Mc[i]))#urbain
        Mc[i].assign(tf.where(M==3,255*matplotlib.colors.to_rgb('blue')[i],Mc[i]))#water/snow
        Mc[i] = tf.convert_to_tensor(Mc[i])
    return tf.stack(Mc,axis=-1).numpy().astype(np.uint8) 

def download_kml(array,name,north,south,east,west):
    import simplekml
    from PIL import Image
    import os
    npx,npy = (west+east)/2,(north+south)/2

    rgb_image = palette((array).astype(np.uint8))
    rgb_image = Image.fromarray(rgb_image)
    rgb_image.save('data/predictions/kml/'+name+'.png')
    
    kml = simplekml.Kml()
    kml.newpoint(name=name, coords=[(npx,npy)])
    image = kml.newgroundoverlay(name='v_'+name)
    
    image.icon.href = name+'.png'
    image.latlonbox.north = north
    image.latlonbox.south = south
    image.latlonbox.east = east
    image.latlonbox.west = west

    kml.save('data/predictions/kml/'+name+'.kml')
    print('Kml saved at "data/predictions/kml"')

# -------- FOR CNN MODELS : YOU HAVE TO ADD YOUR PREDICTIONS TO THE EDITOR, EXPORT IMAGE TO TIF THEN CONVERT IT TO KML --------#
def download_tif(assetId):
    import time
    image = ee.Image('users/leakm/'+assetId)
    if not os.path.isfile('data/predictions/'+assetId+'.tif'):
        task = ee.batch.Export.image.toDrive(
            image       = image.float(),
            description = assetId,
            fileNamePrefix = assetId,
            folder      = 'predictions',
            scale       = 30,
            crs         = 'EPSG:4326'
        )
        task.start()
        while task.active():
            time.sleep(30)
        # Error condition
        if task.status()['state'] != 'COMPLETED':
            print('Error with image export.')
        else:
            print('Image export completed')
            print('Please download image',assetId,'from drive (directory data/predictions) if you work on your local computer')

def download_kml_from_tif(filename):
    from osgeo import gdal
    
    def GDALInfoReportCorner(hDataset, x, y):
        adfGeoTransform = hDataset.GetGeoTransform(can_return_null=True)
        if adfGeoTransform is not None:
            dfGeoX = adfGeoTransform[0] + adfGeoTransform[1] * x + adfGeoTransform[2] * y
            dfGeoY = adfGeoTransform[3] + adfGeoTransform[4] * x + adfGeoTransform[5] * y
        else:
            dfGeoX,dfGeo = float(x), float(y)
        if abs(dfGeoX) < 181 and abs(dfGeoY) < 91:
            return float("%1.7f"%dfGeoX), float("%1.7f"%dfGeoY)
        else:
            return float("%1.3f"%dfGeoX), float("%1.3f"%dfGeoY)

    image = gdal.Open(filename)
    name  = os.path.splitext(os.path.basename(filename))[0]
    
    upper_left  = GDALInfoReportCorner(image,0.0, 0.0)
    lower_right = GDALInfoReportCorner(image,image.RasterXSize, image.RasterYSize)
    north       = upper_left[1]
    south       = lower_right[1]
    west        = upper_left[0]
    east        = lower_right[0]

    array = image.GetRasterBand(1).ReadAsArray(0, 0, image.RasterXSize, image.RasterYSize)
    download_kml(array,name,north,south,east,west)