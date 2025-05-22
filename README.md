# DUAL-STAGE-ANFIS-FRAMEWORK-FOR-HOLISTIC-HEART-HEALTH-ASSESSMENT

This project features a dual-stage AI framework for comprehensive heart health assessment, integrating two Adaptive Neuro-Fuzzy Inference System (ANFIS) models and an interactive LSTM-powered chatbot. The goal is to predict heart disease and heart attack risks using a hybrid learning approach, while offering dynamic, personalized health insights through a web-based interface.

## Features

### 1. Heart Disease Risk Prediction (Heart_Disease.py)

* Utilizes Gaussian membership functions in a Sugeno-type ANFIS model.
* Accepts 9 key clinical inputs such as age, sex, chest pain, cholesterol, blood pressure, etc.
* Built using Bokeh for interactive visualization.
* Provides the predicted heart disease type along with a risk level (low, medium, high).

### 2. Heart Attack Risk Prediction (Heart_Attack.py)

* Uses Generalized Bell membership functions.
* Includes additional lifestyle inputs like smoking and family history.
* Delivers specific predictions for heart attack risk and severity.
* Predicts risk of heart attack with detailed membership function plots using Bokeh.

### 3. Interactive Health Chatbot (Chatbot.py)

* Powered by LSTM models trained on synthetic and real patient data.
* Offers dynamic, real-time interaction and personalized heart health guidance.
* Integrated into a Dash web application for a user-friendly interface.

### 4. Web Interface

* Combines all modules in a centralized, easy-to-navigate dashboard.
* Mobile-friendly design.
* Users can interact with predictors, explore visualizations, and talk to the chatbot.

## Technology Stack

* **Python** (NumPy, Pandas, Scikit-learn, SciPy, Matplotlib)
* **Fuzzy Logic** (Sugeno-type ANFIS with Gaussian and Bell MFs)
* **Machine Learning** (Hybrid learning with LSE + Gradient Descent, LSTM for chatbot)
* **Bokeh** (For ANFIS plotting and interactive surface visualizations)
* **Dash & Plotly** (Interactive visualization and chatbot interface)
* **HTML/CSS** (Front-end interface styling)

## How to Run the Project

1. Clone the repository:

   ```bash
   git clone https://github.com/Severus-193/DUAL_STAGE_ANFIS_HEART_HEALTH_ASSESSMENT.git
   cd DUAL_STAGE_ANFIS_HEART_HEALTH_ASSESSMENT
   ```

2. Install dependencies:
   pip install -r requirements.txt

3. Run the Bokeh and Dash applications:

   ```bash
   python heart_disease.py     # For heart disease prediction
   python heart_attack.py      # For heart attack risk prediction
   python chatbot.py           # For chatbot interface
   ```

4. Open the website:
   Open **"webpage.html"** in your browser to view the integrated web interface and access the dashboard.

## Usage

* Input patient health parameters into the Dash interface to receive disease and risk predictions.
* Use the chatbot to ask about causes, risk levels, and preventive tips.
* Explore surface plots to understand how different variables affect heart health.

## Future Enhancements

* Integration with real-time wearable health monitoring data.
* Support for multilingual chatbot interaction.
* Expansion to other cardiovascular diseases using modular ANFIS blocks.



