import ee 
import os
import time 
import numpy as np
import tensorflow as tf
from dataset_construction import get_landsat8, get_sentinel1
from dataset_loader import feature_process
import pandas as pd

def create_folders():
    '''Function that creates all the folder you'll need for the whole project'''
    for folder in ['results','models']:
        parent_dir = os.getcwd()
        path = os.path.join(parent_dir, folder) 
        if not os.path.exists(path):
            os.makedirs(path)
    for folder in ['train','eval','inference']:
        parent_dir = os.path.join(os.getcwd(), 'data') 
        path = os.path.join(parent_dir, folder) 
        if not os.path.exists(path):
            os.makedirs(path)
    for folder in ['colored_pipes','kml','tfrecords']:
        parent_dir = os.path.join(os.getcwd(), 'data','predictions') 
        path = os.path.join(parent_dir, folder) 
        if not os.path.exists(path):
            os.makedirs(path)

def predict_pipes(network_name,image_name):
    '''Function that assigns a class to each pipe and exports a csv file with columns that includes the name of the pipe, its geometry and its class.
    This function requires you added your network and predictions to Earth Engine Editor's Assets. '''

    file = 'data/predictions/'+image_name+'_classification.csv'
    if not os.path.isfile(file):
        #Import the prediction image from Assets
        image = ee.Image('users/leakm/'+image_name+'_pred')
        def func(feature):
            ''' Function that assign a length and a landcover to a pipe'''
            centroid  = feature.geometry().centroid()
            landcover = ee.Number(image.reduceRegion(ee.Reducer.mode(),centroid,30))
            length    = ee.Number(feature.geometry().length())
            return feature.set('landcover',landcover).set('length',length)
        
        #Import the network from Assets
        table = ee.FeatureCollection('users/leakm/'+network_name).filterBounds(image.geometry().bounds())
        table = table.map(func)
        task = ee.batch.Export.table.toDrive(
            collection = table, 
            description= image_name+'_classification',
            folder     = 'predictions'
        )
        task.start()
        
        while task.active():
            time.sleep(30)

        # Error condition
        if task.status()['state'] != 'COMPLETED':
            print('Error with file export.')
        else:
            print('file ',file,' export completed.')
            print('Please download the file from drive (directory data/predictions) if you work on your local computer')
    
def clean_predictions(name):
    '''Function that parses the geometry of the pipe from geo-json format to lon1,lat1,lon2,lat2 and gets rid of unnecessary columns from the csv'''
    import json
    file = 'data/predictions/'+name+'_classification.csv'
    df = pd.read_csv(file)

    # Keep only the interesting columns
    df = df[['system:index','Name','landcover','.geo','length']]
    df['.geo'] = df['.geo'].astype(str)

    # Parse landcover as int
    df['landcover'] = df['landcover'].apply(lambda x : int(float(x.split('=',1)[1].split('}',1)[0])))

    # Parse the JSON string to an array
    df['.geo'] = df['.geo'].apply(lambda x: np.array(json.loads(x)['coordinates']))

    # Drop the rows where features are not 'LineStrings'
    df.drop(df[df['.geo'].apply(lambda x : x[0].shape)!=(2,)].index,inplace=True)
    df.drop(df[df['.geo'].apply(lambda x : x[1].shape)!=(2,)].index,inplace=True)

    # Attribute a column to each coordinate
    df['lon1'] = df['.geo'].apply(lambda x: x[0][0])
    df['lat1'] = df['.geo'].apply(lambda x: x[0][1])
    df['lon2'] = df['.geo'].apply(lambda x: x[1][0])
    df['lat2'] = df['.geo'].apply(lambda x: x[1][1])
    df.drop(columns=['.geo'],inplace=True)

    # We don't have Sieccao's Id_arcs so we use system:index after remodeling them
    if 'sieccao' in file :
        df['Name'] = df['system:index']
    
    df.drop(columns=['system:index'],inplace=True)
    df.to_csv(file, index=False)
    print('Predictions are updated')

def predict_pipes_from_csv(filename,model_name,bands,start_date,end_date):
    import joblib
    #import swifter
    def predict(row):
        lon1, lat1, lon2, lat2 = (row['lon1'], row['lat1'], row['lon2'], row['lat2'])
        line = ee.Geometry.LineString((lon1, lat1, lon2, lat2))
        array_image = image.sampleRectangle(region=line.bounds())
        array_image = [np.array(array_image.get(i).getInfo()) for i in bands]
        array_image = np.dstack(array_image)
        array_image = feature_process(array_image)
        num_features = array_image.shape[-1]
        array_image = tf.reshape(array_image,[-1,num_features])
        predictions = model.predict(array_image)
        counts = np.bincount(predictions)
        return np.argmax(counts)
    
    df = pd.read_csv(filename)
    landsat = get_landsat8(start_date,end_date)
    sentinel = get_sentinel1(start_date,end_date)
    image = ee.Image.cat([landsat,sentinel]).reproject(crs='EPSG:3857',scale=30).select(bands)
    model = joblib.load('models/'+model_name+'.joblib')

    #df['landcover'] = df.swifter.allow_dask_on_strings(enable=True).apply (lambda row: predict(row), axis=1) 
    df['landcover'] = df.apply (lambda row: predict(row), axis=1)
    df.to_csv(filename,index=False)

def get_statistics(filename):
    def deg2rad(deg) :
        import math
        return deg * (math.pi/180)
    def getDistanceFromLatLon(row):
        import math
        lon1, lat1, lon2, lat2 = (row['lon1'], row['lat1'], row['lon2'], row['lat2'])
        R = 6371 #Radius of the earth in km
        dLat = deg2rad(lat2-lat1)
        dLon = deg2rad(lon2-lon1) 
        a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(deg2rad(lat1)) * math.cos(deg2rad(lat2)) * math.sin(dLon/2) * math.sin(dLon/2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a)); 
        d = R * c # Distance in km
        return d*1000

    df = pd.read_csv(filename)
    df['length'] = df.apply(lambda row : getDistanceFromLatLon(row),axis=1)
    
    stats = df.groupby(['landcover'])['length'].sum().apply(lambda x : x/df['length'].sum()).reset_index()
    stats['landcover'] = stats['landcover'].apply(lambda x : ['field','forest','urbain','water'][x])
    stats.rename(columns={'length':'proportion'},inplace=True)
    name = os.path.splitext(os.path.basename(filename))[0].split('_classification')[0]
    stats.to_csv('data/predictions/'+name+'_statistics.csv', index=False)
    
def color_pipes(filename) :
    import simplekml
    import matplotlib
    from multiprocessing.pool import ThreadPool as Pool

    ds_test = pd.read_csv(filename)
    lines = (ds_test['Name'], ds_test['lon1'], ds_test['lat1'], ds_test['lon2'], ds_test['lat2'], ds_test['landcover'])
    kml = simplekml.Kml()
    i = ds_test.shape[0]
    ids, lons1, lats1, lons2, lats2, classes = lines

    for id, lon1, lat1, lon2, lat2, classe in zip (ids, lons1, lats1, lons2, lats2, classes):
        line = kml.newlinestring(name=str(id), coords=[(lon1,lat1), (lon2,lat2)])
        if classe == 0:
            r,g,b = np.multiply(255,matplotlib.colors.to_rgb('lime')).astype(int)
        elif classe == 1:
            r,g,b = np.multiply(255,matplotlib.colors.to_rgb('darkgreen')).astype(int)
        elif classe == 2:
            r,g,b = np.multiply(255,matplotlib.colors.to_rgb('yellow')).astype(int)
        else:
            r,g,b = np.multiply(255,matplotlib.colors.to_rgb('blue')).astype(int)
        line.style.linestyle.color = simplekml.Color.rgb(r,g,b)
        i = i-1
        
    name = os.path.splitext(os.path.basename(filename))[0].split('_classification')[0]
    kml.save('data/predictions/colored_pipes/'+name+'_colored.kml') 