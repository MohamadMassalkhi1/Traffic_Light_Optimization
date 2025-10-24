import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

class SupervisedModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.train_mae = None
        self.test_mae = None
        self.train_r2 = None
        self.test_r2 = None
    
    def generate_traffic_data(self, days=30, intersections=20):
        """Generate synthetic traffic data"""
        print(f"\nGenerating traffic data for {days} days, {intersections} intersections...")
        
        data = []
        samples_per_day = 96  # Every 15 minutes
        
        for day in range(days):
            for time_slot in range(samples_per_day):
                hour = (time_slot * 15) / 60  # Convert to hour (0-23)
                day_of_week = day % 7
                
                for intersection in range(intersections):
                    # Base traffic
                    traffic = 50
                    
                    # Rush hours (7-9 AM and 5-7 PM)
                    if (7 <= hour < 9) or (17 <= hour < 19):
                        traffic += 100
                    
                    # Weekday has more traffic
                    if day_of_week < 5:
                        traffic += 30
                    
                    # Random variation
                    traffic += np.random.normal(0, 15)
                    traffic = max(10, traffic)
                    
                    # Calculate optimal signal time (simple rule)
                    # More traffic = longer green light needed
                    optimal_signal = 30 + (traffic / 3)
                    optimal_signal = np.clip(optimal_signal, 30, 120)
                    
                    data.append({
                        'hour': hour,
                        'day_of_week': day_of_week,
                        'traffic_volume': traffic,
                        'is_rush_hour': int((7 <= hour < 9) or (17 <= hour < 19)),
                        'is_weekend': int(day_of_week >= 5),
                        'optimal_signal_time': optimal_signal
                    })
        
        df = pd.DataFrame(data)
        print(f"Generated {len(df)} records")
        return df
    
    def train(self):
        """Train the supervised learning model"""
        print("\n" + "=" * 60)
        print("TRAINING SUPERVISED LEARNING MODEL")
        print("=" * 60)
        
        # Generate data
        data = self.generate_traffic_data(days=30, intersections=20)
        
        # Features and target
        X = data[['hour', 'day_of_week', 'traffic_volume', 'is_rush_hour', 'is_weekend']]
        y = data['optimal_signal_time']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        # Train model
        print("\nTraining neural network...")
        self.model = MLPRegressor(
            hidden_layer_sizes=(64, 32),  # 2 hidden layers
            activation='relu',
            max_iter=100,
            random_state=42,
            verbose=False
        )
        
        self.model.fit(X_train_scaled, y_train)
        print("Model trained successfully!")
        
        # Evaluate
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        self.train_mae = mean_absolute_error(y_train, y_train_pred)
        self.test_mae = mean_absolute_error(y_test, y_test_pred)
        self.train_r2 = r2_score(y_train, y_train_pred)
        self.test_r2 = r2_score(y_test, y_test_pred)
        
        print("\n" + "=" * 60)
        print("TRAINING RESULTS")
        print("=" * 60)
        print(f"\nTraining Set:")
        print(f"  MAE: {self.train_mae:.2f} seconds")
        print(f"  R² Score: {self.train_r2:.4f}")
        print(f"\nTest Set:")
        print(f"  MAE: {self.test_mae:.2f} seconds")
        print(f"  R² Score: {self.test_r2:.4f}")
        
        self.is_trained = True
        return "Model trained successfully!"
    
    def predict(self, hour, day_of_week, traffic_volume):
        """Make prediction for given inputs"""
        if not self.is_trained:
            self.train()
        
        is_rush_hour = int((7 <= hour < 9) or (17 <= hour < 19))
        is_weekend = int(day_of_week >= 5)
        
        X = pd.DataFrame([{
            'hour': hour,
            'day_of_week': day_of_week,
            'traffic_volume': traffic_volume,
            'is_rush_hour': is_rush_hour,
            'is_weekend': is_weekend
        }])
        
        X_scaled = self.scaler.transform(X)
        prediction = self.model.predict(X_scaled)[0]
        
        return {
            'signal_time': prediction,
            'is_rush_hour': bool(is_rush_hour),
            'is_weekend': bool(is_weekend)
        }