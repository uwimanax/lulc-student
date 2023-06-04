import ee
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sklearn import preprocessing
import pickle

# Initialize an app
app = Flask(__name__)

# Load the serialized model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize GEE
service_account = 'ml4eo-420815.iam.gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(service_account, '.private-key.json')
ee.Initialize(credentials)

# Initialize variables required for GEE dataset preprocessing (similar to the examples in Exercise 6_1)
lat = -1.9441
lon = 30.0619
offset = 0.51
region = [
    [lon + offset, lat - offset],
    [lon + offset, lat + offset],
    [lon - offset, lat + offset],
    [lon - offset, lat - offset]
]

roi = ee.Geometry.Polygon(region)

se2bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A']
trainingbands = se2bands + ['avg_rad']
label = 'smod_code'
scaleFactor = 1000


def se2mask(img):
    # Select the QA60 band
    qa60 = img.select('QA60')
    # A 1 at bit position 10 of QA60 band indicates a cloud
    cloud_bitmask = 1 << 10
    # Generate a mask for points that are not cloud
    mask_non_cloud = qa60.bitwiseAnd(cloud_bitmask).eq(0)

    # A 1 at bit position 11 of QA60 band indicates a cirrus
    cirrus_bitmask = 1 << 11
    # Generate a mask for points that are not cirrus
    mask_non_cirrus = qa60.bitwiseAnd(cirrus_bitmask).eq(0)

    # Combine the two masks
    mask = mask_non_cloud.And(mask_non_cirrus)

    # Apply the combined mask to the image, scale the image values, select all bands ('B.*'),
    # and copy over the 'system:time_start' property
    masked_img = img.updateMask(mask).multiply(0.0001).select('B.*').copyProperties(img, ['system:time_start'])

    return masked_img


def get_fused_data():
    mean = 0.2062830612359614
    std = 1.1950717918110398

    vmu = ee.Number(mean)
    vstd = ee.Number(std)

    se2 = ee.ImageCollection("COPERNICUS/S2").filterDate("2015-07-01", "2015-12-31")
    se2 = se2.filterBounds(roi)
    se2 = se2.map(se2mask)
    se2 = se2.median().select(se2bands).divide(scaleFactor).clip(roi)
    # se2 = se2.median()
    se2 = se2.select(se2bands)

    viirs = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG").filterDate(
        "2015-07-01", "2015-12-31").filterBounds(roi).median().select('avg_rad').clip(roi)

    viirsclean = viirs.subtract(mean).divide(std)

    fusedclean = viirsclean.addBands(se2)

    return fusedclean


def get_features(longitude, latitude):
    poi_geometry = ee.Geometry.Point(longitude, latitude)

    fused_data = get_fused_data()
    
    dataclean = fused_data.sampleRegions(
        collection=ee.FeatureCollection(ee.Feature(poi_geometry)), 
        properties=[label], scale=30, geometries=True).select(trainingbands)
    
    # Specify the band names based on your dataset
    band_order = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'avg_rad']
    
    nested_list = dataclean.reduceColumns(ee.Reducer.toList(len(band_order)), band_order).values().get(0)
    data = pd.DataFrame(nested_list.getInfo(), columns=band_order)

    print("**---------",data.shape)

    return data
#function to check if point is within the roi
def validate_location(longitude, latitude, roi):
    point = ee.Geometry.Point(longitude, latitude)
    return roi.contains(point).getInfo()

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    features = request.form.to_dict()
    longitude = float(features['longitude'])
    latitude = float(features['latitude'])
    if validate_location(longitude, latitude, roi):
        # TODO: get the features for the given location
        final_features = get_features(longitude, latitude)
    
        # TODO: get predictions from the the model using the features loaded
        prediction = model.predict(final_features)

        # convert the prediction to an integer
        output = int(prediction[0])

        if output == 1:
            text = "built up land"
        else:
            text = "not built up land"

        return render_template('index.html', prediction_text='The area at {}, {} location is {}'.format(longitude, latitude, text))

    else:
        return render_template('index.html', prediction_text='<span style="color: red;">The area at {}, {} location is out of bounds.</span>'.format(longitude, latitude))



if __name__ == "__main__":
    app.run(debug=True)