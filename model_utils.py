import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import openmeteo_requests
import requests_cache
from retry_requests import retry


model     = load_model('best_model.keras')
scaler_X        = joblib.load('scaler_X.pkl')
scaler_temp     = joblib.load('scaler_y_temp.pkl')
scaler_apparent = joblib.load('scaler_y_apparent.pkl')

def generate_batches(gen):
    while(True):
        for i in range(len(gen)):
            X_batch, y_batch = gen[i]
            yield X_batch ,{
                'temp':     y_batch[:, 0:6],  
                'apparent': y_batch[:, 6:12],   
                'rain':  y_batch[:, 12:18]
            }
def fetch_realtime(hours):
    cache_session = requests_cache.CachedSession('.cache_realtime', expire_after=0)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo     = openmeteo_requests.Client(session=retry_session)

    url    = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":  28.6139,
        "longitude": 77.2090,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",
            "apparent_temperature",
            "precipitation",
            "wind_speed_10m",
            "wind_direction_10m",
            "surface_pressure",
            "cloud_cover"
        ],
        "past_hours": hours,
        "forecast_hours": 1,
        "timezone": "Asia/Kolkata"
    }

    responses = openmeteo.weather_api(url, params=params)
    response  = responses[0]
    hourly    = response.Hourly()

    df = pd.DataFrame({
        "datetime":             pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ),
        "temperature_2m":       hourly.Variables(0).ValuesAsNumpy(),
        "relative_humidity_2m": hourly.Variables(1).ValuesAsNumpy(),
        "dew_point_2m":         hourly.Variables(2).ValuesAsNumpy(),
        "apparent_temperature": hourly.Variables(3).ValuesAsNumpy(),
        "precipitation":        hourly.Variables(4).ValuesAsNumpy(),
        "wind_speed_10m":       hourly.Variables(5).ValuesAsNumpy(),
        "wind_direction_10m":   hourly.Variables(6).ValuesAsNumpy(),
        "surface_pressure":     hourly.Variables(7).ValuesAsNumpy(),
        "cloud_cover":          hourly.Variables(8).ValuesAsNumpy(),
    })

    df['datetime'] = df['datetime'].dt.tz_convert('Asia/Kolkata')
    df = df.set_index('datetime')
    df = df.sort_index()

    now = pd.Timestamp.now(tz='Asia/Kolkata').floor('h')
    df = df[df.index <= now]

    return df

    

def engineer_features(df):
        ## Just fixing the problem of CYCLICAL ENCODING
    ## as python won't underst
    df['hour'] = df.index.hour
    df['month'] = df.index.month
    df['dayofyear'] = df.index.dayofyear
    df['hour_sin']      = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos']      = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin']     = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos']     = np.cos(2 * np.pi * df['month'] / 12)
    df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
    df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)
    df['wind_dir_sin']  = np.sin(2*np.pi*df['wind_direction_10m']/360)
    df['wind_dir_cos']  = np.cos(2*np.pi*df['wind_direction_10m']/360)
    
    #---------------------------------------------
    lag_cols = ['temperature_2m', 'apparent_temperature', 'surface_pressure',
            'relative_humidity_2m', 'wind_speed_10m', 'precipitation']   
    for col in lag_cols:
        df[f'{col}_lag1']  = df[col].shift(1)
        df[f'{col}_lag3']  = df[col].shift(3)
        df[f'{col}_lag6']  = df[col].shift(6)
        df[f'{col}_lag12']  = df[col].shift(12)
        df[f'{col}_lag24'] = df[col].shift(24)

# ── Rolling Features ──────────────────────────────────────────────────────────
    roll_cols = ['temperature_2m', 'apparent_temperature', 'surface_pressure',
             'relative_humidity_2m', 'wind_speed_10m', 'precipitation']

    for col in roll_cols:
        df[f'{col}_roll_3']  = df[col].shift(1).rolling(3).mean()
        df[f'{col}_roll_6']  = df[col].shift(1).rolling(6).mean()
        df[f'{col}_roll_24'] = df[col].shift(1).rolling(24).mean()
        df[f'{col}_roll_12_std'] = df[col].shift(1).rolling(12).std()
        df[f'{col}_roll_24_std'] = df[col].shift(1).rolling(24).std()
        
    ## Adding Temp change
    df['temp_change_1h'] = df['temperature_2m'].diff(1)
    df['temp_change_3h'] = df['temperature_2m'].diff(3)
    
    ## Adding Pressure Change
    df['pressure_change_1h'] =  df['surface_pressure'].diff(1)
    df['pressure_change_3h'] =  df['surface_pressure'].diff(3)
    
    ## As LSTM's like correlating features
    df['feels_like_delta']    = df['apparent_temperature'] - df['temperature_2m']
    
    ## I also Wish to predict whether it is raining or not so here it is 
    df['is_raining']          = (df['precipitation'] > 0.1).astype(int)
    
    df = df.drop(columns=['hour', 'month', 'dayofyear', 'wind_direction_10m'], errors='ignore')
    climatology = joblib.load('climatology.pkl')
    df['temp_anomaly'] = df.apply(
        lambda r:
        r['temperature_2m']
        - climatology.loc[(r.name.month, r.name.hour)],
        axis=1)
    df['temp_anomaly_lag1']  = df['temp_anomaly'].shift(1)
    df['temp_anomaly_lag24'] = df['temp_anomaly'].shift(24)

    df = df.dropna()
    return df
def predict_next_6_hours():
    raw_data    = fetch_realtime(48)
    print(f"Raw data last timestamp: {raw_data.index[-1]}")
    featured_df = engineer_features(raw_data)
    print(f"Featured df last timestamp: {featured_df.index[-1]}")
    print(f"Current time: {pd.Timestamp.now(tz='Asia/Kolkata')}")
    featured_df = featured_df.drop(columns=['is_raining', 'weather_code'], errors='ignore')

    window        = featured_df.tail(24)
    window_scaled = scaler_X.transform(window)
    window_scaled = window_scaled.reshape(1, 24, -1)

    y_pred = model.predict(window_scaled, verbose=0)

    temp_preds     = scaler_temp.inverse_transform(y_pred[0])[0]      # 6 values
    apparent_preds = scaler_apparent.inverse_transform(y_pred[1])[0]  # 6 values
    rain_probs     = y_pred[2][0]                                      # 6 values

    base_time = featured_df.index[-1]

    print(f"\n{'='*55}")
    print(f"  6 Hour Forecast from {base_time.strftime('%Y-%m-%d %H:%M IST')}")
    print(f"{'='*55}")
    print(f"  {'Hour':<6} {'Time':<10} {'Temp':>6} {'Feels':>6} {'Rain%':>6} {'Rain':>6}")
    print(f"  {'-'*50}")

    results = []
    for i in range(6):
        forecast_time = base_time + pd.Timedelta(hours=i+1)
        print(f"  +{i+1}h    {forecast_time.strftime('%H:%M'):<10} "
              f"{temp_preds[i]:>5.1f}° "
              f"{apparent_preds[i]:>5.1f}° "
              f"{rain_probs[i]*100:>5.1f}% "
              f"{'🌧' if rain_probs[i] > 0.5 else '☀️'}")
        results.append({
            'datetime':    forecast_time,
            'temperature': round(float(temp_preds[i]), 1),
            'apparent':    round(float(apparent_preds[i]), 1),
            'rain_prob':   round(float(rain_probs[i] * 100), 1),
            'is_raining':  bool(rain_probs[i] > 0.5)
        })

    print(f"{'='*55}\n")
    return results
result = predict_next_6_hours()

def finetune_on_recent(model, hours):
    raw = fetch_realtime(hours=hours)
    featured = engineer_features(raw)
    featured = featured.drop(columns=['is_raining'], errors='ignore')

    for i in range(1, 7):
        featured[f'target_temp_{i}h']     = featured['temperature_2m'].shift(-i)
        featured[f'target_apparent_{i}h'] = featured['apparent_temperature'].shift(-i)
        featured[f'target_rain_{i}h']     = featured['is_raining'] if 'is_raining' in featured.columns else (featured['precipitation'] > 0.1).astype(int).shift(-i)
    
    featured = featured.dropna()

    target_cols = [f'target_temp_{i}h'     for i in range(1, 7)] + \
                  [f'target_apparent_{i}h' for i in range(1, 7)] + \
                  [f'target_rain_{i}h'     for i in range(1, 7)]

    X_ft = featured.drop(columns=target_cols, errors='ignore')
    y_temp_ft     = featured[[f'target_temp_{i}h'     for i in range(1, 7)]].values
    y_apparent_ft = featured[[f'target_apparent_{i}h' for i in range(1, 7)]].values
    y_rain_ft     = featured[[f'target_rain_{i}h'     for i in range(1, 7)]].values

    X_ft_scaled        = scaler_X.transform(X_ft)
    y_temp_ft_scaled     = scaler_temp.transform(y_temp_ft)
    y_apparent_ft_scaled = scaler_apparent.transform(y_apparent_ft)

    y_ft = np.hstack([y_temp_ft_scaled, y_apparent_ft_scaled, y_rain_ft])

    # Freeze CNN layers
    for layer in model.layers:
        if any(x in layer.name for x in ['conv1d', 'max_pooling', 'layer_normalization']):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss={'temp': 'mse', 'apparent': 'mse', 'rain': 'binary_crossentropy'},
        metrics={'temp': 'mae', 'apparent': 'mae', 'rain': 'accuracy'}
    )

    gen = TimeseriesGenerator(X_ft_scaled, y_ft, length=24, batch_size=32)
    ft_data = generate_batches(gen)

    model.fit(ft_data, steps_per_epoch=len(gen), epochs=5, verbose=1)
    model.save('best_model.keras')
    print("Fine-tuning done.")


# Fetch only future hours from Open-Meteo
def fetch_forecast_only():
    cache_session = requests_cache.CachedSession('.cache_realtime', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo_client = openmeteo_requests.Client(session=retry_session)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":  28.6139,
        "longitude": 77.2090,
        "hourly": ["temperature_2m", "apparent_temperature"],
        "past_hours": 0,
        "forecast_hours": 6,
        "timezone": "Asia/Kolkata"
    }
    responses = openmeteo_client.weather_api(url, params=params)
    response  = responses[0]
    hourly    = response.Hourly()

    df = pd.DataFrame({
        "datetime": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ),
        "temperature_2m":       hourly.Variables(0).ValuesAsNumpy(),
        "apparent_temperature": hourly.Variables(1).ValuesAsNumpy(),
    })
    df['datetime'] = df['datetime'].dt.tz_convert('Asia/Kolkata')
    df = df.set_index('datetime')
    return df