from flask import Flask, render_template, request, jsonify
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

# Import our model
from supervised_model import SupervisedModel

app = Flask(__name__)

# Initialize model
print("\n" + "=" * 60)
print("INITIALIZING TRAFFIC SIGNAL OPTIMIZATION APP")
print("=" * 60)

supervised_model = SupervisedModel()

print("✓ Model loaded")


@app.route('/')
def home():
    """Render the main page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        data = request.json
        hour = float(data['hour'])
        day_of_week = int(data['day_of_week'])
        traffic_volume = float(data['traffic_volume'])
        
        print(f"\nPrediction request: hour={hour}, day={day_of_week}, traffic={traffic_volume}")
        
        # Make prediction
        result = supervised_model.predict(hour, day_of_week, traffic_volume)
        result['model'] = 'Supervised Learning (Neural Network)'
        
        # Round signal time
        result['signal_time'] = round(result['signal_time'], 1)
        
        # Generate visualization
        chart = generate_chart(hour, traffic_volume, result['signal_time'])
        result['chart'] = chart
        
        print(f"Prediction: {result['signal_time']}s")
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 400


def generate_chart(hour, traffic, signal_time):
    """Generate visualization charts"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Chart 1: Traffic pattern throughout day
    hours = np.arange(0, 24, 0.5)
    traffic_pattern = []
    for h in hours:
        base = 50
        if 7 <= h < 9 or 17 <= h < 19:
            base += 100
        traffic_pattern.append(base)
    
    ax = axes[0]
    ax.plot(hours, traffic_pattern, linewidth=2, color='#2563eb')
    ax.axvline(hour, color='red', linestyle='--', linewidth=2, label=f'Current: {hour}h')
    ax.axhspan(0, 80, alpha=0.2, color='green', label='Low')
    ax.axhspan(80, 150, alpha=0.2, color='yellow', label='Medium')
    ax.axhspan(150, 250, alpha=0.2, color='red', label='High')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Traffic Volume')
    ax.set_title('Daily Traffic Pattern')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Chart 2: Signal timing options
    signal_options = [30, 45, 60, 75, 90, 105, 120]
    colors = ['#10b981' if s == signal_time else '#e5e7eb' for s in signal_options]
    
    ax = axes[1]
    bars = ax.bar(range(len(signal_options)), signal_options, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Signal Options')
    ax.set_ylabel('Signal Time (seconds)')
    ax.set_title(f'Recommended: {signal_time}s')
    ax.set_xticks(range(len(signal_options)))
    ax.set_xticklabels([f'{s}s' for s in signal_options], rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Highlight selected bar
    for i, bar in enumerate(bars):
        if signal_options[i] == signal_time:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 3, 
                   '★', ha='center', fontsize=20, color='#10b981')
    
    plt.tight_layout()
    
    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_base64


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("STARTING FLASK SERVER")
    print("=" * 60)
    print("\n✓ Server starting...")
    print("✓ Open your browser: http://127.0.0.1:5000")
    print("\nPress CTRL+C to stop the server\n")
    
    app.run(debug=False, host='0.0.0.0', port=5000)