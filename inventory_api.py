from flask import Flask, jsonify
from pymongo import MongoClient
from dotenv import load_dotenv
from bson import ObjectId
from dateutil import parser
import pandas as pd
import numpy as np
import joblib
import json
import os
from flask_cors import CORS

app = Flask(__name__)

# Load .env
load_dotenv()
CORS(app, resources={r"/*": {"origins": ["http://localhost:5173","http://localhost:5174"]}})

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
    'Hospital Bed': joblib.load('./models/hospital_bed_sarimax.joblib'),
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
        file_path = './data/exported_listings_v4_1.json'
        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump(item_arr, json_file, ensure_ascii=False, indent=4)

        return jsonify({
            'message': '[EXPORT] Exporting item listing from collection.',
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/preload_item_data')
def preload_data():
    with open('./data/exported_listings_v4.json', 'r', encoding='utf-8') as item_listing_file:
        item_listing_data = json.load(item_listing_file)

    for item in item_listing_data:
        item['createdBy'] = ObjectId(item['createdBy'])
        item_collection.insert_one(item)

    return jsonify({
            'message': f'Successfully inserted new items',
            'total_items': item_collection.count_documents({})
        }), 201

@app.route('/preload_stock_data')
def preload_stock_data():
    with open('./data/stock_listing_v4.json', 'r', encoding='utf-8') as stock_listing_file:
        stock_listing_data = json.load(stock_listing_file)

    for stock in stock_listing_data:
        stock['listing'] = ObjectId(stock['listing'])
        stock['acquisitionDate'] = parser.parse(stock['acquisitionDate'])
        stock_collection.insert_one(stock)

    return jsonify({
            'message': f'Successfully inserted new stocks',
            'total_items': stock_collection.count_documents({})
        }), 201


@app.route('/delete_item_listings')
def delete_item_listings():
    try:
        result = item_collection.delete_many({})
        return jsonify({'message': 'All records deleted successfully', 'deleted_count': result.deleted_count}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/delete_stock_listings')
def delete_stock_listings():
    try:
        result = stock_collection.delete_many({})
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

@app.route('/classify_abc')
def classify_abc():

    '''
    DATA PREPROCESSING
    '''

    item_listing = pd.DataFrame([
        {**item, '_id': str(item['_id']), 'createdBy': str(item['createdBy'])} for item in item_collection.find()
    ])
    
    stock_listing = pd.DataFrame([
            {**stock, '_id': str(stock['_id']), 'listing': str(stock['listing'])} for stock in stock_collection.find()
        ])
    
    item_stock_data = item_listing.merge(stock_listing, left_on='_id', right_on='listing')
    item_stock_data.drop(columns=['abcCategory'], inplace=True)

    item_stock_prep = item_stock_data[['title', 'category', 'quantity', 'unitCost', 'avgUsagePerDay', 'acquisitionDate']].copy()
    
    item_stock_prep['month'] = item_stock_prep['acquisitionDate'].dt.to_period('M')
    item_stock_prep['year'] = item_stock_prep['acquisitionDate'].dt.to_period('Y')

    consumable_features = item_stock_prep[item_stock_prep['category'] == 'Consumable'].sort_values(['title', 'acquisitionDate']).groupby(['title', 'month']).agg(
            beginning_inventory = ('quantity', 'first'),
            ending_inventory = ('quantity', 'last'),
    ).reset_index()

    consumable_features['category'] = 'Consumable'

    equipment_features = item_stock_prep[item_stock_prep['category'] == 'Equipment'].sort_values(['title', 'acquisitionDate']).groupby(['title', 'year']).agg(
            beginning_inventory = ('quantity', 'first'),
            ending_inventory = ('quantity', 'last'),
    ).reset_index()

    equipment_features['category'] = 'Equipment'

    equipment_features = equipment_features.loc[equipment_features.index.repeat(12)].reset_index(drop=True)
    equipment_features['month'] = pd.to_datetime(equipment_features['year'].astype(str) + '-' + np.tile(range(1, 13), len(equipment_features) // 12).astype(str)).dt.strftime('%Y-%m')
    feature_df = pd.concat([consumable_features, equipment_features.drop(columns=['year'])])

    # Annual Usage (For Equipment Turnover Rate)
    annual_usage_df = item_stock_prep.groupby(['title', 'year']).agg(annual_usage = ('avgUsagePerDay', 'sum')).reset_index()
    feature_df['year'] = pd.to_datetime(feature_df['month'], format='%Y-%m')
    feature_df['year'] = feature_df['year'].dt.to_period('Y')
    feature_df = feature_df.merge(annual_usage_df, on=['title', 'year'], how='inner')

    # Monthly Usage (For Consumable Turnover Rate) -> Shortcutting this by dividing annual usage by 12
    feature_df['monthly_usage'] = feature_df['annual_usage'] / 12

    # Unit Cost
    cost_df = item_stock_data.groupby('title')[['unitCost']].mean().reset_index()
    feature_df = feature_df.merge(cost_df, on='title', how='left')

    # Annual Usage Value (AUG)
    feature_df['annual_usage_value'] = feature_df['annual_usage'] * feature_df['unitCost']

    # Stock Turnover Rate
    feature_df['stock_turnover_rate'] = np.where(feature_df['category'] == 'Consumable',
                                                feature_df['monthly_usage'] / ((feature_df['beginning_inventory'] + feature_df['ending_inventory']) / 2),
                                                feature_df['annual_usage'] / ((feature_df['beginning_inventory'] + feature_df['ending_inventory']) / 2),
                                                )
    
    # Return only latest records for each item
    latest_idx = feature_df.groupby('title')['month'].idxmax()
    feature_df = feature_df.loc[latest_idx]
    
    '''
    FEATURE ENGINEERING
    '''

    X_features = ['category', 'annual_usage_value', 'stock_turnover_rate']
    X = feature_df[X_features]

    '''
    ABC CATEGORY PREDICTION
    '''
    
    abc_prediction_data = abc_model.predict(X)
    print(f'PREDICTIONS: {abc_prediction_data}')

    feature_df1 = feature_df.copy()
    feature_df1['cluster'] = abc_prediction_data

    # Rename clusters to A, B, C based on AUV values
    cluster_mapping = {
        feature_df1.groupby('cluster')['annual_usage_value'].mean().idxmax(): 'A',
        feature_df1.groupby('cluster')['annual_usage_value'].mean().idxmin(): 'C'
    }
    cluster_mapping = {k: cluster_mapping.get(k, 'B') for k in range(3)}

    # Map cluster labels to A, B, C
    feature_df1['abcCategory'] = feature_df1['cluster'].map(cluster_mapping)
    feature_df1 = feature_df1[['title', 'abcCategory']].copy()
    feature_df1 = feature_df1.to_dict(orient='records')

    # Update records
    for item in feature_df1:
        item_collection.update_one(
            {'title': item['title']},
            {'$set': {'abcCategory': item['abcCategory']}},
            upsert=False
        )

    return jsonify(feature_df1)

'''
Top 5 Demands for Current Month
________________________________
'''
@app.route('/monthly_top_demands')
def monthly_top_demands():
    item_listing_arr = [x['_id'] for x in item_collection.find()]
    item_demand_map = {}

    # Compute monthly demand of current data
    for item_id in item_listing_arr:
        curr_item_id = item_id
        item_name = item_collection.find_one({'_id': ObjectId(curr_item_id)})['title']

        stock_listing = pd.DataFrame([
            {**stock, '_id': str(stock['_id']), 'listing': str(stock['listing'])} for stock in stock_collection.find({'listing': ObjectId(curr_item_id)})
        ])
        
        # ERROR HANDLING: In case item has not stock listing records yet
        if stock_listing.empty:
            continue

        stock_listing_data = stock_listing.copy()
        stock_listing_data = stock_listing_data[['acquisitionDate', 'quantity']]
        stock_listing_data['month'] = stock_listing_data['acquisitionDate'].dt.to_period('M')
        stock_listing_data.sort_values('acquisitionDate', inplace=True)
        stock_listing_data['stockDiff'] = stock_listing_data.groupby('month')['quantity'].transform(lambda x: x.diff())
        stock_listing_data['restockQuantity'] = stock_listing_data.groupby('month')['stockDiff'].transform(lambda x: x.clip(lower=0))
        stock_listing_data.drop('stockDiff', axis=1, inplace=True)
        stock_listing_data = stock_listing_data.groupby('month')['restockQuantity'].sum().reset_index()

        # Create beginning and ending monthly inventory
        stock_listing_data_1 = stock_listing.copy()
        stock_listing_data_1['month'] = stock_listing_data_1['acquisitionDate'].dt.to_period('M')
        stock_listing_data_1['year'] = stock_listing_data_1['acquisitionDate'].dt.to_period('Y')

        monthly_start_end_stock = stock_listing_data_1.groupby('month').agg(
                beginning_inventory_monthly = ('quantity', 'first'),
                ending_inventory_monthly = ('quantity', 'last'),
        ).reset_index()

        stock_listing_data = stock_listing_data.merge(monthly_start_end_stock, on='month')

        # Create demand column
        stock_listing_data['demand'] = stock_listing_data['beginning_inventory_monthly'] + stock_listing_data['ending_inventory_monthly'] - stock_listing_data['restockQuantity']
        stock_listing_data.drop(columns=['beginning_inventory_monthly', 'ending_inventory_monthly'], inplace=True)

        print('_' * 30)
        print()
        print(stock_listing_data)

        # Pass demand of latest month
        latest_month = stock_listing_data['month'].max()
        latest_demand = stock_listing_data.loc[stock_listing_data['month'] == latest_month, 'demand'].values[0]

        print()
        print(f'CURRENT ITEM: {item_name}')
        print(f'LATEST MONTH: {latest_month}')
        print(f'LATEST DEMAND: {latest_demand}')
        print()
        item_demand_map[item_name] = latest_demand

    # Rank based on demand then get top 5
    print(item_demand_map)
    item_demand_map = dict(sorted(item_demand_map.items(), key=lambda x: x[1], reverse=True)[:5])
        
    return jsonify(item_demand_map)

'''
6-Month Demand Forecasting
________________________________

'''

@app.route('/forecast_demand')
def forecast_demand():
    item_listing_arr = [(x['_id'], x['abcCategory']) for x in item_collection.find()]
    item_forecast_dict = {}

    # Compute monthly demand of current data
    for item_id, abc_category in item_listing_arr:
        curr_item_id = item_id
        curr_abc_category = abc_category
        stock_listing = pd.DataFrame([
                {**stock, '_id': str(stock['_id']), 'listing': str(stock['listing'])} for stock in stock_collection.find({'listing': ObjectId(curr_item_id)})
            ])
        
        # ERROR HANDLING: In case item has not stock listing records yet
        if stock_listing.empty:
            continue

        stock_listing_data = stock_listing.copy()
        stock_listing_data = stock_listing_data[['acquisitionDate', 'quantity']]
        stock_listing_data['month'] = stock_listing_data['acquisitionDate'].dt.to_period('M')
        stock_listing_data.sort_values('acquisitionDate', inplace=True)
        stock_listing_data['stockDiff'] = stock_listing_data.groupby('month')['quantity'].transform(lambda x: x.diff())
        stock_listing_data['restockQuantity'] = stock_listing_data.groupby('month')['stockDiff'].transform(lambda x: x.clip(lower=0))
        stock_listing_data.drop('stockDiff', axis=1, inplace=True)
        stock_listing_data = stock_listing_data.groupby('month')['restockQuantity'].sum().reset_index()
        stock_listing_data['abcCategory'] = curr_abc_category

        # Create beginning and ending monthly inventory
        stock_listing_data_1 = stock_listing.copy()
        stock_listing_data_1['month'] = stock_listing_data_1['acquisitionDate'].dt.to_period('M')
        stock_listing_data_1['year'] = stock_listing_data_1['acquisitionDate'].dt.to_period('Y')

        monthly_start_end_stock = stock_listing_data_1.groupby('month').agg(
                beginning_inventory_monthly = ('quantity', 'first'),
                ending_inventory_monthly = ('quantity', 'last'),
        ).reset_index()

        stock_listing_data = stock_listing_data.merge(monthly_start_end_stock, on='month')

        # Create demand column
        stock_listing_data['demand'] = stock_listing_data['beginning_inventory_monthly'] + stock_listing_data['ending_inventory_monthly'] - stock_listing_data['restockQuantity']
        stock_listing_data.drop(columns=['beginning_inventory_monthly', 'ending_inventory_monthly'], inplace=True)

        print(stock_listing_data)

        # Categorical One-Hot Encoding
        stock_listing_model = stock_listing_data.copy()
        stock_listing_model = pd.get_dummies(stock_listing_model, columns=['abcCategory'], prefix='ABC', drop_first=False)

        # Ensure all dummy columns exist
        for col in ['ABC_A', 'ABC_B', 'ABC_C']:
            if col not in stock_listing_model.columns:
                stock_listing_model[col] = 0

        dummy_columns = [col for col in ['ABC_A', 'ABC_B', 'ABC_C'] if col in stock_listing_model.columns]
        stock_listing_model[dummy_columns] = stock_listing_model[dummy_columns].astype(int)

        # Get item name of current item_id
        item_name = item_collection.find_one({'_id': ObjectId(curr_item_id)})['title']
        model = sarimax_forecast_models[item_name]

        # Defined 6-month prediction
        future_steps = 6
        future_exog = pd.DataFrame({
            'restock_quantity': [stock_listing_model['restockQuantity'].mean()] * future_steps,
            'ABC_A': [stock_listing_model['ABC_A'].mean()] * future_steps,
            'ABC_B': [stock_listing_model['ABC_B'].mean()] * future_steps,
            'ABC_C': [stock_listing_model['ABC_C'].mean()] * future_steps
        })

        # Predict demand for given future periods
        forecast = model.forecast(steps=future_steps, exog=future_exog)
        item_forecast_dict[item_name] = {
            'previousDemand': list(stock_listing_data['demand']),
            'forecast': list(forecast),
        }

        print(f'\n{item_name} -- 6 MONTH FORECAST')
        print(forecast)

    print(item_forecast_dict)
    
    return jsonify(item_forecast_dict)

if __name__ == '__main__':
    app.run(debug=True)
