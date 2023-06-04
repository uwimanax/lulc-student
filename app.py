import ee

import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sklearn import preprocessing
import pickle

# Initialize an app
app = Flask(_name_)

# Load the serialized model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize GEE
service_account = 'ml4eo-420815.iam.gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(service_account, 'private-key.json')
ee.Initialize(credentials)

# Initialize variables required for GEE dataset preprocessing (similar to the examples in Exercise 6_1)
lat = -1.9441
lon = 30.0619
offset = 0.51
region = [
        [lon + offset, lat - offset],
        [lon + offset, lat + offset],
        [lon - offset, lat + offset],
        [lon - offset, lat - offset]]

roi = ee.Geometry.Polygon([region])

se2bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7','B8','B8A']
trainingbands = se2bands + ['avg_rad']
label = 'smod_code'
scaleFactor=1000

# Remember this function from Exercise 5_03, what does it do?
def se2mask(image):
    """
        This function updates a mask of an ee.Image so that clouds are filtered.
    """
    #TODO: complete this function
    quality_band = image.select('QA60')
    cloudmask = 1 << 10
    cirrusmask = 1 << 11
    mask = quality_band.bitwiseAnd(cloudmask).eq(0) and (quality_band.bitwiseAnd(cirrusmask).eq(0))
    return image.updateMask(mask).divide(10000)
    # pass

def get_fused_data():
    """
        This function contains the preprocessing steps followed to obtain the preprocessed, merged dataset in 6_1.
        This function is called when the server starts to prepare the dataset.
    """
    # Use the mean and standard deviation obtained from the training dataset
    mean = 0.2062830612359614
    std = 1.1950717918110398

    #  TODO: Convert the mean and std to ee.Number    
    vmu = ee.Number(mean)
    vstd = ee.Number(std)

    #  TODO: Load the COPERNICUS/S2 dataset and filter dates "2015-07-01","2015-12-31"
    se2 = ee.ImageCollection('COPERNICUS/S2').filterDate("2015-07-01","2015-12-31")
    # TODO: Use the filterBounds function to get filter the are specified in ROI
    se2 = se2.filterBounds(roi)

    #  TODO: Keep pixels that have less than 20% cloud
    se2 = se2.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE",20))

    # TODO:Update the mask 
    se2 = se2.map(se2mask)

    # TODO:Get the median image
    se2 = se2.median()

    # TODO: select the `se2bands`
    se2 = se2.select(se2bands)
    

    #  Load the NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG dataset and filter dates "2015-07-01","2015-12-31"
    viirs = ee.Image(ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG").filterDate(
        "2015-07-01","2015-12-31").filterBounds(roi).median().select('avg_rad').clip(roi))

    # TODO: Substract the mean and divide by the standard deviation for the viirs samples
    viirsclean = viirs.subtract(vmu).divide(vstd)

    # TODO: Fuse the two datasets
    fusedclean = se2.addBands(viirsclean)

    return fusedclean

# Prepare the fused 
gee_data = get_fused_data()


def get_features(longitude, latitude):
    # TODO: Create an ee.Geometry instance from the coordinates
    poi_geometry = ee.Geometry.Point(longitude, latitude)

    # TODO: Sample features for the given point of interest keeping only the training bands
    dataclean = gee_data.select(trainingbands).sampleRegions(collection=poi_geometry,
                                                        properties=[label],
                                                        scale=scaleFactor)

    # TODO: use getInfo to load the sample's features
    sample = dataclean.getInfo()

    # Find the band ordering in the loaded data
    band_order = sample['properties']['band_order']

    # Convert the loaded data to ee.List
    nested_list = dataclean.reduceColumns(ee.Reducer.toList(len(band_order)), band_order).values().get(0)

    # TODO: Convert the `nested_list` to a Pandas dataframe
    data = pd.DataFrame(nested_list.getInfo())
    return data

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    features = request.form.to_dict()
    longitude = float(features['longitude'])
    latitude = float(features['latitude'])
    # TODO: get the features for the given location
    # Define the rectangular polygon coordinates
    lat = -1.9441
    lon = 30.0619
    offset = 0.51

    min_lon = lon - offset
    max_lon = lon + offset
    min_lat = lat - offset
    max_lat = lat + offset
    # Check if the point is outside the rectangular polygon
    if longitude < min_lon or longitude > max_lon or latitude < min_lat or latitude > max_lat:
        return render_template('index.html', prediction_text='Point is outside the rectangular polygon')
    else:
        final_features = get_features(longitude, latitude)
        
        # TODO: get predictions from the the model using the features loaded
        prediction = model.predict(final_features)

        # convert the prediction to an integer
        output = int(prediction[0])

        if output == 1:
            text = "built up land"
        else:
            text = "not built up land"

        # Return a response based on the output of the model
        return render_template('index.html', prediction_text='The area at {}, {} location is {}'.format(longitude, latitude, text))


if _name_ == "_main_":
    app.run(debug=True)