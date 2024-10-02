
# Weather Prediction Using AI

This repository contains the implementation of a neural network-based algorithm to predict weather conditions. The project applies the **Back Propagation Neural Network (BPN)** technique to forecast parameters like temperature, humidity, precipitation, dew point, and wind speed. The model classifies weather as hot, cold, rainy, windy, sunny, cloudy, or humid using artificial neural networks (ANN).

## Authors
- Snigdha Srivastava
- Suhaib Khaleel
- Abhishek Kumar
- C S Kiran Varma
- C. Ganesh
- Inturi Bhavya Teja
- Siddharth Modi

## Abstract
This project uses a neural network model to predict weather conditions using a real-time dataset. The algorithm is based on **Back Propagation Neural Networks (BPN)**, a popular method in AI for supervised learning. It predicts various weather parameters and classifies conditions like temperature and humidity, supporting decision-making in fields like agriculture, transportation, and energy.

## Objectives
- Predict key weather parameters: temperature, humidity, precipitation, dew point, wind speed.
- Classify weather as hot, cold, rainy, windy, sunny, cloudy, or humid.

## Methodology
The project uses the **Back Propagation Algorithm** and an **Artificial Neural Network (ANN)** to train the model. The steps involved include:
1. **Data Collection**: Gathers data on temperature, wind speed, pressure, humidity, etc.
2. **Data Assimilation**: Inputs these values into the ANN for analysis.
3. **Prediction**: Uses **ANN** and **Fletcher-Reeves Model** to enhance learning and speed up convergence.

## Features
- **Back Propagation**: Error propagation through network layers to adjust weights and minimize prediction errors.
- **Artificial Neural Networks (ANN)**: Learns relationships between inputs and outputs for accurate weather forecasting.
- **Radial Basis Function (RBF)**: An additional architecture considered for performance comparison.
  
## Usage
### Prerequisites
- Python 3.9
- Libraries: NumPy, Pandas, Scikit-learn, Matplotlib

### Running the Code
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/WeatherPredictionAI.git
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the prediction model:
   ```bash
   python weather_prediction.py
   ```

## Example
Test case inputs:
- Max Temperature: 37°C
- Min Temperature: 25°C
- Max Humidity: 44%
- Min Pressure: 1003 hPa

Predicted outcome: **Fog**

## Results
The model demonstrated success in predicting future weather, including factors like rainy, sunny, and windy days, using backpropagation training. The accuracy was higher compared to other statistical models.

## Future Scope
- Incorporation of other statistical techniques for feature selection.
- Integration of **fuzzy logic** to improve prediction accuracy.
- Exploration of other machine learning algorithms beyond data mining.

## References
1. Ch. Jyosthna Devi et al., "ANN Approach for Weather Prediction using Back Propagation."
2. Sanjay D. Sawaitul et al., "Classification and Prediction of Future Weather using Back Propagation."
3. Pooja Malik et al., "An Effective Weather Forecasting Using Neural Network."

## License
This project is licensed under the MIT License - see the LICENSE file for details.
