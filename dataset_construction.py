import ee 
import os
import time

def get_landsat8(start_date,end_date):
    '''Function that returns a cloud-masked landsat8 image
    Arguments :
        start_date : the start_date of the image collection in the format "aaaa-mm-dd"
        end_date   : the end_date   of the image collection in the format "aaaa-mm-dd"
    Returns :
        ee.Image
    '''
    # Cloud masking function.
    def maskL8sr(image):
        '''Function that masks cloud pixels and scale the image between 0 and 1'''
        opticalBands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
        thermalBands = ['B10', 'B11']

        cloudShadowBitMask = ee.Number(2).pow(3).int()
        cloudsBitMask = ee.Number(2).pow(5).int()
        qa = image.select('pixel_qa')
        mask1 = qa.bitwiseAnd(cloudShadowBitMask).eq(0).And(
        qa.bitwiseAnd(cloudsBitMask).eq(0))
        mask2 = image.mask().reduce('min')
        mask3 = image.select(opticalBands).gt(0).And(
                image.select(opticalBands).lt(10000)).reduce('min')
        mask = mask1.And(mask2).And(mask3)
        return image.select(opticalBands).divide(10000).addBands(
                image.select(thermalBands).divide(10).clamp(273.15, 373.15)
                .subtract(273.15).divide(100)).updateMask(mask)
    
    collection = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR').filterDate(
        start_date,end_date).filterMetadata('CLOUD_COVER','less_than',0.3).map(maskL8sr)
    # The image input data is a cloud-masked median composite.
    image = collection.median().setDefaultProjection(collection.first().projection())
    return image

def get_sentinel1(start_date,end_date):
    '''Function that returns a sentinel1 image containing 'VV' and 'VH' bands from ascending and descending orbits
    Arguments :
        start_date : the start_date of the image collection in the format "aaaa-MM-dd"
        end_date   : the end_date   of the image collection in the format "aaaa-MM-dd"
    Returns :
        ee.Image
    '''
    collection = ee.ImageCollection("COPERNICUS/S1_GRD")
    collection = collection.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')).filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    collection = collection.select('VV','VH').filterDate(start_date,end_date)
    asc  = collection.filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING')).median()
    desc = collection.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING')).median()
    image = ee.Image.cat([asc,desc])
    image = image.clamp(-50,1).subtract(-50).divide(51).setDefaultProjection(collection.first().projection())
    return image

class TFDatasetConstruction():
    def __init__(self,landsat,sentinel,response,kernel_size):
        self.landsat = landsat
        self.sentinel = sentinel
        self.response = response
        self.bands = landsat + sentinel
        self.kernel_size = kernel_size

    def get_label_image(self):
        '''Function that returns an image of labels 0,1,2,3
        You should have your labels uploaded to your GEE Assets. If not :
            - Go to https://code.earthengine.google.com/
            - Click on "Assets", then "NEW", then below "Image Upload", click on "GeoTIFF". 
            - In "Source files" select the tif image of your labels.
            - Write a unique AssetID (for france-landcover I named my asset : "france2017")
            - Replace "france2017" in this code by your assetId
            - Click on "UPLOAD"
        '''
        #see the google doc 'classes' to see what each index means
        nlcd = ee.Image('users/leakm/france2017').select('b1').remap([42,41,43,44,31,32,221,222,11,12,34,36,211,45,46,51,53],
                                                                        [2,2,2,2,1,1,0,0,0,0,0,0,0,3,0,3,3]).rename('landcover')
        return nlcd
    
    def get_sample_images(self,start_date,end_date):
        '''Function that stacks landsat and sentinel1 images into one image, then converts the image into an array of pixels'''
        landsat  = get_landsat8(start_date,end_date)
        sentinel = get_sentinel1(start_date,end_date)
        nlcd     = self.get_label_image()
        featureStack = ee.Image.cat([
            landsat.select(self.landsat),
            sentinel.select(self.sentinel),
            nlcd.select(self.response)])

        listt  = ee.List.repeat(1, self.kernel_size)
        lists  = ee.List.repeat(listt, self.kernel_size)
        kernel = ee.Kernel.fixed(self.kernel_size, self.kernel_size, lists)

        #to export training patches, convert a multi-band image to an array image using ee.imageneighborhoodtoarray()
        arrays = featureStack.neighborhoodToArray(kernel)
        
        #import pre-made geometries and split them randomly to 80% test and 20% eval
        regions           = ee.FeatureCollection('users/leakm/TrainGeoId').merge(ee.FeatureCollection('users/leakm/EvalGeoId')).randomColumn()
        trainingPolys     = regions.filter(ee.Filter.greaterThan('random',0.2))
        evalPolys         = regions.filter(ee.Filter.lessThan('random',0.2))

        # Convert the feature collections to lists for iteration.
        trainingPolysList = trainingPolys.toList(trainingPolys.size())
        evalPolysList     = evalPolys.toList(evalPolys.size())
        return trainingPolys,evalPolys,trainingPolysList,evalPolysList,arrays

    def export_train_eval_data(self,folder,base,Polys,PolysList,arrays):
        '''Function that samples the image from each pre-made polygons (1000 pixels from each polygon) and exports it to 
            various TFRecord files.'''
        # These numbers determined experimentally.
        n = 200 # Number of shards in each polygon.
        N = 1000 # Total sample size in each polygon.
        
        # Export all the data (in many pieces), with one task per geometry.
        print('Running export...')
        for g in range(Polys.size().getInfo()):
            geomSample = ee.FeatureCollection([])
            for i in range(n):
                sample = arrays.sample(
                    region    = ee.Feature(PolysList.get(g)).geometry(), 
                    scale     = 30,
                    numPixels = N / n, # Size of the shard.
                    seed      = i,
                    tileScale = 8
                )
                geomSample = geomSample.merge(sample)

            desc = base + '-' +str(g)+ '-of-' +str(Polys.size().getInfo())
            task = ee.batch.Export.table.toDrive(
            collection     = geomSample,
            description    = desc,
            folder         = folder,
            fileNamePrefix = desc,
            fileFormat     = 'TFRecord',
            selectors      = self.bands + [self.response]
            )
            task.start()
            
        while task.active():
            time.sleep(30)

        # Error condition
        if task.status()['state'] != 'COMPLETED':
            print('Error with ',folder,' dataset export.')
        else:
            print(folder,' dataset export completed.')
            print('Please download the files from drive (directory data/',folder,'/) if you work on your local computer')
    
    def dataset_construction(self,start_date,end_date):
        '''Function that exports the training and evaluation data, from landat8 and sentinel1 images and french landcover labels, to TFRecord format.'''
        trainingPolys,evalPolys,trainingPolysList,evalPolysList,arrays = self.get_sample_images(start_date,end_date)
        os.chdir('data')
        self.export_train_eval_data('train','train',trainingPolys,trainingPolysList,arrays)
        self.export_train_eval_data('eval','traineval',evalPolys,evalPolysList,arrays)
        os.chdir('..')
       
    # --------------------------- Inference --------------------------------#

    def get_test_image(self,start_date,end_date):
        '''Function that returns a landsat8-sentinel1 composite
        Arguments :
            start_date : the start_date of the image collection in the format "aaaa-mm-dd"
            end_date   : the end_date   of the image collection in the format "aaaa-mm-dd"
        Returns :
            ee.Image
        '''
        landsat  = get_landsat8(start_date,end_date)
        sentinel = get_sentinel1(start_date,end_date)
        image    = ee.Image.cat([landsat.select(self.landsat),sentinel.select(self.sentinel)])
        return image.reproject(crs='EPSG:3857',scale=30)

    def export_test_data(self,image, name, region, patch_dimension):
        """Runs the image export task.
        Arguments: 
            image : the ee.Image to be exported
            name  : the name of the file
            region : the region of export
            patch_dimension : the dimension of the exported image
        """
        print('Running export...')
        if not os.path.isfile('data/inference/'+name+'-mixer.json'):
            os.chdir('data')
            task = ee.batch.Export.image.toDrive(
                image       = image,
                description = name,
                folder      = 'inference',
                fileNamePrefix = name,
                region      = region,
                scale       = 30,
                crs         = 'EPSG:3857',
                fileFormat  = 'TFRecord',
                formatOptions = {
                'patchDimensions': patch_dimension,
                'kernelSize': [0,0],
                'compressed': True,
                }
            )
            task.start()
            while task.active():
                time.sleep(30)

            # Error condition
            if task.status()['state'] != 'COMPLETED':
                print('Error with image export.')
            else:
                print('Image and Mixer export completed')
                print('Please download all files starting with','"'+name+'"','from drive (directory data/inference) if you work on your local computer')
            os.chdir('..')
    
    def test_dataset_construction(self,start_date,end_date,filename,patch_size=None,network_name=None,point=None,radius=None,rectangle=None):
        if point and radius and rectangle and network_name : 
            raise ValueError("Please precise either 'network_name', 'point' and 'length', or 'rectangle'")

        if point and radius:
            point  = ee.Geometry.Point(point).buffer(radius)
            region = point.bounds()
        elif rectangle :
            region = ee.Geometry.Rectangle(rectangle)
        elif network_name : 
            shape  = ee.FeatureCollection('users/leakm/'+network_name)
            region = shape.geometry().bounds()
        else :
            raise ValueError("Please precise either 'network_name', 'point' and 'length', or 'rectangle'")
        corners    = region.coordinates().getInfo()[0]

        if not(os.path.isfile('data/inference/'+filename+'-mixer.json')):
            image  = self.get_test_image(start_date,end_date)
            if not patch_size : #For CNN models, patch_size should be equal to KERNEL_SHAPE, for ML Models, patch_size is equal to the image dimensions
                patch_size = image.clip(region).getInfo()['bands'][0]['dimensions']
            self.export_test_data(image, filename, region, patch_size)
        
        north = corners[2][1]
        south = corners[0][1] 
        east  = corners[0][0]
        west  = corners[2][0]
        return north,south,east,west