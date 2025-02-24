from flask import Flask, request, jsonify
import joblib
import pandas as pd
import joblib

app = Flask(__name__)

# ABC Analysis KMeans Clustering Model
abc_model = joblib.load('abc_model.joblib')

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
    return jsonify({'message': 'Hospital Inventory API is running'})

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
