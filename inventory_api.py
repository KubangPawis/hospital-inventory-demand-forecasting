from flask import Flask, request, jsonify
from pymongo import MongoClient
from dotenv import load_dotenv
from bson import ObjectId
import pandas as pd
import joblib
import json
import os

app = Flask(__name__)

# Load .env
load_dotenv()

# Database Connection
mongo_uri = os.getenv('MONGO_URI')
client = MongoClient(mongo_uri)
db = client['ngh_logistics_db']
item_collection = db['listings']
stock_collection = db['listingstocks']

# ABC Analysis KMeans Clustering Model
abc_model = joblib.load('./models/hospital_abc_clf.joblib')

# Inventory Demand Forecasting SARIMAX Model
sarimax_forecast_models = {
    'Antiseptic Solution': joblib.load('./models/antiseptic_solution_sarimax.joblib'),
    'Bandages': joblib.load('./models/bandages_sarimax.joblib'),
    'Blood Bags': joblib.load('./models/blood_bags_sarimax.joblib'),
    'Cotton Rolls': joblib.load('./models/cotton_rolls_sarimax.joblib'),
    'Defibrilator': joblib.load('./models/defibrilator_sarimax.joblib'),
    'ECG Machine': joblib.load('./models/ecg_machine_sarimax.joblib'),
    'Face Shield': joblib.load('./models/face_shield_sarimax.joblib'),
    'Gloves': joblib.load('./models/gloves_sarimax.joblib'),
    'Gown': joblib.load('./models/gown_sarimax.joblib'),
    'Infusion Pump': joblib.load('./models/infusion_pump_sarimax.joblib'),
    'IV Drip': joblib.load('./models/iv_drip_sarimax.joblib'),
    'MRI Scanner': joblib.load('./models/mri_scanner_sarimax.joblib'),
    'Surgical Mask': joblib.load('./models/surgical_mask_sarimax.joblib'),
    'Syringe': joblib.load('./models/syringe_sarimax.joblib'),
    'Ventilator': joblib.load('./models/ventilator_sarimax.joblib'),
    'Wheelchair': joblib.load('./models/wheelchair_sarimax.joblib'),
    'X-Ray-Machine': joblib.load('./models/x_ray_machine_sarimax.joblib'),
}

@app.route('/')
def home():
    return jsonify({
        'message': 'Hospital Inventory API is running'
        })

@app.route('/check_item_data')
def check_item_data():
    item_listing_data = item_collection.find()
    for item in item_listing_data:
        print(f'\n{item}\n')

    return jsonify({
        'message': '[DATA LOAD] Loading item listing from collection.'
        })

@app.route('/check_stock_data')
def check_stock_data():
    stock_listing_data = stock_collection.find()
    for stock in stock_listing_data:
        print(f'\n{stock}\n')

    return jsonify({
        'message': '[DATA LOAD] Loading stock listing from collection.'
        })

@app.route('/export_item_data')
def export_item_data():
    try:
        item_listing_data = list(item_collection.find())
        item_arr = [
            {**item, '_id': str(item['_id']), 'createdBy': str(item['createdBy'])} for item in item_listing_data
        ]

        if not item_arr:
            return jsonify({'message': '[EXPORT] No items found in the collection.'}), 404
        
        # Export item listing json file
        file_path = './data/exported_listings_v3.json'
        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump(item_arr, json_file, ensure_ascii=False, indent=4)

        return jsonify({
            'message': '[EXPORT] Exporting item listing from collection.',
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/preload_item_data')
def preload_data():
    with open('./data/exported_listings_v2.json', 'r', encoding='utf-8') as item_listing_file:
        item_listing_data = json.load(item_listing_file)

    for item in item_listing_data:
        item['createdBy'] = ObjectId(item['createdBy'])
        item_collection.insert_one(item)

    return jsonify({
            'message': f'Successfully inserted new items',
            'total_items': item_collection.count_documents({})
        }), 201

@app.route('/delete_item_listings', methods=['DELETE'])
def delete_item_listings():
    try:
        result = item_collection.delete_many({})
        return jsonify({'message': 'All records deleted successfully', 'deleted_count': result.deleted_count}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

'''
ABC Clustering Classification
________________________________

Request Data Format
1. item_name
2. category
3. annual_usage_rate
4. stock_turnover_rate

'''

@app.route('/classify_abc', methods=['POST'])
def classify_abc():
    data = request.json
    item_name = data.get('item_name')
    category = data.get('category')
    annual_usage_rate = data.get('annual_usage_rate')
    stock_turnover_rate = data.get('stock_turnover_rate')

    if item_name not in list(sarimax_forecast_models.keys()):
        return jsonify({'error': 'Item not found'}), 400
    
    prediction = abc_model.predict(pd.DataFrame({
        'category': category,
        'annual_usage_rate': annual_usage_rate,
        'stock_turnover_rate': stock_turnover_rate,
    }))
    print(f'Prediction: {prediction}')

    return jsonify({
        'item_name': item_name,
        'ABC_category': prediction,
    })

'''
6-Month Demand Forecasting
________________________________

Request Data Format
1. item_name
2. monthly_demand_arr

'''

@app.route('/forecast_demand', methods=['POST'])
def forecast_demand():
    data = request.json
    item_name = data.get('item_name')
    monthly_demand = data.get('monthly_demand_arr')

    if item_name not in sarimax_forecast_models:
        return jsonify({'error': 'No trained ARIMA model for this item'}), 400

    model = sarimax_forecast_models[item_name]

    # Defined 6-month prediction
    future_steps = 6
    future_exog = pd.DataFrame({
        'restock_quantity': [monthly_demand['restock_quantity'].mean()] * future_steps,
        'ABC_A': [monthly_demand['ABC_A'].mean()] * future_steps,
        'ABC_B': [monthly_demand['ABC_B'].mean()] * future_steps,
        'ABC_C': [monthly_demand['ABC_C'].mean()] * future_steps
    })

    # Predict demand for given future periods
    forecast = model.forecast(steps=future_steps, exog=future_exog)
    
    return jsonify({
        'item_name': item_name,
        'forecast': forecast.tolist(),
    })

if __name__ == '__main__':
    app.run(debug=True)
