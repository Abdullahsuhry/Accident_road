from flask import Flask, request, render_template, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
import datetime
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import pickle
import numpy as np
import os
import joblib
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

# Flask app
app = Flask(__name__, template_folder='.', static_folder='.')
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URL")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

CORS(app, origins=['http://localhost:8000', 'http://127.0.0.1:8000', 'https://accident-road-7.onrender.com'], supports_credentials=True)
# SQLAlchemy DB
db = SQLAlchemy(app)

# User model
class User(db.Model):
    __tablename__ = 'user1'
    id = db.Column(db.BigInteger, primary_key=True)
    full_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

# Create tables
with app.app_context():
    db.create_all()

# Load ML model
try:
    with open('rf2_model.joblib', 'rb') as f:
        model = joblib.load(f)
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print("⚠️ Model file not found. Please ensure rf2_model.pkl exists.")
    model = None

# Expected columns & mappings (use your existing VALUE_MAPPINGS here)
EXPECTED_COLUMNS = [
    'Age_band_of_driver', 'Sex_of_driver', 'Educational_level',
    'Vehicle_driver_relation', 'Driving_experience', 'Type_of_vehicle',
    'Area_accident_occured', 'Lanes_or_Medians', 'Types_of_Junction',
    'Road_surface_type', 'Light_conditions', 'Weather_conditions',
    'Type_of_collision', 'Number_of_vehicles_involved', 'Vehicle_movement',
    'Pedestrian_movement', 'Cause_of_accident'
]
VALUE_MAPPINGS = {
    # ... include all your existing VALUE_MAPPINGS here ...

    'Age_band_of_driver': {
        '18-30': '18-30',
        '31-50': '31-50', 
        'under 18': 'under 18',
        'over 51': 'over 51',
        'unknown': 'unknown'
    },
    'Sex_of_driver': {
        'male': 'male',
        'female': 'female',
        'unknown': 'unknown'
    },
    'Educational_level': {
        'above high school': 'above high school',
        'junior high school': 'junior high school',
        'elementary school': 'elementary school',
        'high school': 'high school',
        'unknown': 'unknown',
        'illiterate': 'illiterate',
        'writing & reading': 'writing & reading'
    },
    'Vehicle_driver_relation': {
        'employee': 'employee',
        'owner': 'owner',
        'other': 'other'
    },
    'Driving_experience': {
        '1-2yr': '1-2yr',
        '2-5yr': '2-5yr',
        '5-10yr': '5-10yr',
        'above 10yr': 'above 10yr',
        'below 1yr': 'below 1yr',
        'no licence': 'no licence',
        'unknown': 'unknown'
    },
    'Type_of_vehicle': {
        'car': 'car',
        'automobile': 'car',
        'bus': 'bus',
        'public': 'bus',
        'lorry': 'lorry',
        'truck': 'lorry',
        'motorcycle': 'motorcycle',
        'mootorbike': 'mootorbike',
        'three_wheeler': 'three_wheeler',
        'three wheeler': 'three_wheeler',
        'bajaj': 'three_wheeler',
        'bicycle': 'bicycle',
        'other': 'other'
    },
    'Area_accident_occured': {
        'residential areas': 'residential areas',
        'office areas': 'office areas',
        'recreational areas': 'recreational areas',
        'industrial areas': 'industrial areas',
        'school areas': 'school areas',
        'market areas': 'market areas',
        'church areas': 'church areas',
        'hospital areas': 'hospital areas',
        'rural village areas': 'rural village areas',
        'outside rural areas': 'outside rural areas',
        'other': 'other',
        'unknown': 'unknown'
    },
    'Lanes_or_Medians': {
        'undivided two way': 'undivided two way',
        'one way': 'one way',
        'two-way (divided with broken lines road marking)': 'two-way (divided with broken lines road marking)',
        'two-way (divided with solid lines road marking)': 'two-way (divided with solid lines road marking)',
        'double carriageway (median)': 'double carriageway (median)',
        'other': 'other',
        'unknown': 'unknown'
    },
    'Types_of_Junction': {
        'no junction': 'no junction',
        'y shape': 'y shape',
        't shape': 't shape',
        'x shape': 'x shape',
        'crossing': 'crossing',
        'o shape': 'o shape',
        'other': 'other',
        'unknown': 'unknown'
    },
    'Road_surface_type': {
        'asphalt roads': 'asphalt roads',
        'earth roads': 'earth roads',
        'gravel roads': 'gravel roads',
        'asphalt roads with some distress': 'asphalt roads with some distress',
        'other': 'other',
        'unknown': 'unknown'
    },
    'Light_conditions': {
        'daylight': 'daylight',
        'darkness - lights lit': 'darkness - lights lit',
        'darkness - lights unlit': 'darkness - lights unlit',
        'darkness - no lighting': 'darkness - no lighting',
        'unknown': 'unknown'
    },
    'Weather_conditions': {
        'normal': 'normal',
        'raining': 'raining',
        'raining and windy': 'raining and windy',
        'cloudy': 'cloudy',
        'windy': 'windy',
        'snow': 'snow',
        'fog or mist': 'fog or mist',
        'other': 'other',
        'unknown': 'unknown'
    },
    'Type_of_collision': {
        'vehicle with vehicle collision': 'vehicle with vehicle collision',
        'collision with roadside objects': 'collision with roadside objects',
        'collision with pedestrians': 'collision with pedestrians',
        'rollover': 'rollover',
        'collision with animals': 'collision with animals',
        'collision with roadside-parked vehicles': 'collision with roadside-parked vehicles',
        'fall from vehicles': 'fall from vehicles',
        'with train': 'with train',
        'other': 'other',
        'unknown': 'unknown'
    },
    'Number_of_vehicles_involved': {
        '1': 1,
        '2': 2,
        '3': 3,
        '4': 4,
        '6': 6,
        '7': 7,
        'unknown': 'unknown'
    },
    'Vehicle_movement': {
        'going straight': 'going straight',
        'turning left': 'turning left',
        'turning right': 'turning right',
        'overtaking': 'overtaking',
        'changing lane to the left': 'changing lane to the left',
        'changing lane to the right': 'changing lane to the right',
        'u-turn': 'u-turn',
        'moving backward': 'moving backward',
        'parked': 'parked',
        'stopped': 'stopped',
        'entering a junction': 'entering a junction',
        'getting off': 'getting off',
        'waiting to go': 'waiting to go',
        'overturning': 'overturning',
        'other': 'other',
        'reversing':'reversing',
        'turnover':'turnover',
        'unknown': 'unknown'
    },
    'Pedestrian_movement': {
        'not a pedestrian': 'not a pedestrian',
        'crossing from driver\'s nearside': 'crossing from driver\'s nearside',
        'crossing from driver\'s offside': 'crossing from driver\'s offside',
        'crossing from nearside - masked by parked or stationot a pedestrianry vehicle': 'crossing from nearside - masked by parked or stationot a pedestrianry vehicle',
        'crossing from offside - masked by  parked or stationot a pedestrianry vehicle': 'crossing from offside - masked by  parked or stationot a pedestrianry vehicle',
        'standing in carriageway': 'standing in carriageway',
        'in carriageway, stationot a pedestrianry - not crossing  (standing or playing)': 'in carriageway, stationot a pedestrianry - not crossing  (standing or playing)',
        'unknown': 'unknown'
    },
    'Cause_of_accident': {
        'no distancing': 'no distancing',
        'changing lane to the left': 'changing lane to the left',
        'changing lane to the right': 'changing lane to the right',
        'overtaking': 'overtaking',
        'no priority to vehicle': 'no priority to vehicle',
        'no priority to pedestrian': 'no priority to pedestrian',
        'moving backward': 'moving backward',
        'overspeed': 'overspeed',
        'driving carelessly': 'driving carelessly',
        'driving at high speed': 'driving at high speed',
        'driving under the influence of drugs': 'driving under the influence of drugs',
        'drunk driving': 'drunk driving',
        'overloading': 'overloading',
        'getting off the vehicle improperly': 'getting off the vehicle improperly',
        'driving to the left': 'driving to the left',
        'improper parking': 'improper parking',
        'turnover': 'turnover',
        'overturning': 'overturning',
        'other': 'other',
        'unknown': 'unknown'
    }
}

SEVERITY_LABELS = {0: 'Fatal injury', 1: 'Serious Injury', 2: 'Slight Injury'}

def normalize_input_value(field_name, input_value):
    if input_value is None or input_value == '':
        return 'unknown'
    input_lower = str(input_value).lower().strip()
    field_mapping = VALUE_MAPPINGS.get(field_name, {})
    for key, value in field_mapping.items():
        if key.lower() == input_lower:
            return value
    for key, value in field_mapping.items():
        if input_lower in key.lower() or key.lower() in input_lower:
            return value
    return 'unknown'

def prepare_input_dataframe(form_data):
    field_name_mapping = {
        'age_band_of_driver': 'Age_band_of_driver',
        'sex_of_driver': 'Sex_of_driver',
        'educational_level': 'Educational_level',
        'vehicle_driver_relation': 'Vehicle_driver_relation',
        'driving_experience': 'Driving_experience',
        'type_of_vehicle': 'Type_of_vehicle',
        'area_accident_occured': 'Area_accident_occured',
        'lanes_or_medians': 'Lanes_or_Medians',
        'types_of_junction': 'Types_of_Junction',
        'road_surface_type': 'Road_surface_type',
        'light_conditions': 'Light_conditions',
        'weather_conditions': 'Weather_conditions',
        'type_of_collision': 'Type_of_collision',
        'number_of_vehicles_involved': 'Number_of_vehicles_involved',
        'vehicle_movement': 'Vehicle_movement',
        'pedestrian_movement': 'Pedestrian_movement',
        'cause_of_accident': 'Cause_of_accident'
    }
    prepared_data = {}
    for form_field, expected_column in field_name_mapping.items():
        input_value = form_data.get(form_field)
        prepared_data[expected_column] = normalize_input_value(expected_column, input_value)
    input_df = pd.DataFrame([prepared_data])
    for col in EXPECTED_COLUMNS:
        if col not in input_df.columns:
            input_df[col] = 'unknown'
    input_df = input_df[EXPECTED_COLUMNS]
    return input_df

# JWT Authentication
def generate_token(user_id):
    payload = {'user_id': user_id, 'exp': datetime.datetime.utcnow() + datetime.timedelta(days=7)}
    return jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')

def verify_token(token):
    try:
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        return payload['user_id']
    except:
        return None

def require_auth(f):
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'message': 'No token provided'}), 401
        if token.startswith('Bearer '):
            token = token[7:]
        user_id = verify_token(token)
        if not user_id:
            return jsonify({'message': 'Invalid token'}), 401
        return f(user_id, *args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

# Routes
@app.route('/api/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        name = data.get('name')
        email = data.get('email')
        password = data.get('password')
        if not all([name, email, password]):
            return jsonify({'message': 'All fields are required'}), 400
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            return jsonify({'message': 'User already exists'}), 400
        hashed_password = generate_password_hash(password)
        new_user = User(full_name=name, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        token = generate_token(new_user.id)
        return jsonify({
            'message': 'User created successfully',
            'token': token,
            'user': {'id': new_user.id, 'name': name, 'email': email}
        }), 201
    except Exception as exc:
        print(f"Registration error: {exc}")
        return jsonify({'message': 'Registration failed'}), 500

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        if not all([email, password]):
            return jsonify({'message': 'Email and password are required'}), 400
        user = User.query.filter_by(email=email).first()
        if not user or not check_password_hash(user.password, password):
            return jsonify({'message': 'Invalid credentials'}), 401
        token = generate_token(user.id)
        return jsonify({
            'message': 'Login successful',
            'token': token,
            'user': {'id': user.id, 'name': user.full_name, 'email': user.email}
        }), 200
    except Exception as e:
        print(f"Login error: {e}")
        return jsonify({'message': 'Login failed'}), 500

@app.route('/api/profile', methods=['GET'])
@require_auth
def get_profile(user_id):
    try:
        user = User.query.get(user_id)
        if not user:
            return jsonify({'message': 'User not found'}), 404
        return jsonify({
            'id': user.id,
            'name': user.full_name,
            'email': user.email
        }), 200
    except Exception as e:
        print(f"Profile error: {e}")
        return jsonify({'message': 'Failed to get profile'}), 500

# ML prediction routes
@app.route('/')
def home():
    return render_template('/index.html')

@app.route('/predict_form')
def predict_form():
    return render_template('predict_form.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('.', filename)

@app.route('/api/predict', methods=['POST'])
@require_auth
def predict(user_id):
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    try:
        form_data = request.get_json() if request.is_json else request.form.to_dict()
        input_df = prepare_input_dataframe(form_data)
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]
        severity_label = SEVERITY_LABELS.get(prediction, f'Unknown ({prediction})')
        confidence_scores = {SEVERITY_LABELS.get(i, f'Class_{i}'): round(float(prob),4) 
                             for i, prob in enumerate(prediction_proba)}
        return jsonify({
            'prediction': severity_label,
            'severity_label': severity_label,
            'confidence': confidence_scores,
            'user_id': user_id
        })
    except Exception as e:
        import traceback
        print(f"Prediction error: {str(e)}")
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc(),
            'message': 'Prediction failed. Please check server logs.'
        }), 500

# Health check
@app.route('/api/health', methods=['GET'])
def health_check():
    try:
        db_test = User.query.limit(1).all()
        db_connected = True
    except:
        db_connected = False
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'database_connected': db_connected
    }), 200

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'message': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'message': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🚀 Combined Authentication & Prediction API starting...")
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
