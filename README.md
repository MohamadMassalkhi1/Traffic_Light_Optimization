# Traffic Signal Optimization with Neural Networks

An AI-powered traffic management system using Supervised Learning (Neural Networks) to optimize traffic signal timing based on time of day, day of week, and traffic volume.

## Features
- **Neural Network Model**: Multi-layer perceptron trained on traffic patterns
- **Real-time Predictions**: Optimize signal timing based on current conditions
- **Interactive Visualizations**: Daily traffic patterns and signal recommendations
- **Training Metrics**: MAE and R² scores displayed during training

## How It Works
The system trains a neural network on 30 days of simulated traffic data across 20 intersections, learning patterns like:
- Rush hour traffic (7-9 AM, 5-7 PM)
- Weekday vs weekend differences
- Time-of-day variations

The model predicts optimal signal times (30-120 seconds) to minimize wait times and maximize throughput.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python app.py
```
Then open http://127.0.0.1:5000

## Docker Deployment
```bash
# Build the image
docker build -t traffic-optimization .

# Run the container
docker run -p 5000:5000 traffic-optimization
```

Then open http://localhost:5000

## Project Structure
```
├── app.py                 # Flask web application
├── supervised_model.py    # Neural network model
├── templates/
│   └── index.html        # Web interface
├── Dockerfile
├── requirements.txt
└── README.md
```

## Model Performance
The neural network typically achieves:
- Training MAE: ~2-3 seconds
- Test MAE: ~2-3 seconds
- R² Score: ~0.98-0.99

## Note
This is one of my first machine learning projects, demonstrating supervised learning, neural networks, and web deployment.