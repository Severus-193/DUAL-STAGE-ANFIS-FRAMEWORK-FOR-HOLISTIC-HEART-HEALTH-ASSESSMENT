# DUAL-STAGE-ANFIS-FRAMEWORK-FOR-HOLISTIC-HEART-HEALTH-ASSESSMENT

This project features a dual-stage AI framework for comprehensive heart health assessment, integrating two Adaptive Neuro-Fuzzy Inference System (ANFIS) models and an interactive LSTM-powered chatbot. The goal is to predict heart disease and heart attack risks using a hybrid learning approach, while offering dynamic, personalized health insights through a web-based interface.

## Features

### 1. Heart Disease Risk Prediction (heart_disease.py)

* Utilizes Gaussian membership functions in a Sugeno-type ANFIS model.
* Accepts 9 key clinical inputs such as age, sex, chest pain, cholesterol, blood pressure, etc.
* Provides the predicted heart disease type along with a risk level (low, medium, high).

### 2. Heart Attack Risk Prediction (heart_attack.py)

* Uses Generalized Bell membership functions.
* Includes additional lifestyle inputs like smoking and family history.
* Delivers specific predictions for heart attack risk and severity.

### 3. Interactive Health Chatbot (chatbot.py)

* Powered by LSTM models trained on synthetic and real patient data.
* Offers dynamic, real-time interaction and personalized heart health guidance.
* Integrated into a Dash web application for a user-friendly interface.

## Technology Stack

* **Python** (NumPy, Pandas, Scikit-learn, SciPy, Matplotlib)
* **Fuzzy Logic** (Sugeno-type ANFIS with Gaussian and Bell MFs)
* **Machine Learning** (Hybrid learning with LSE + Gradient Descent, LSTM for chatbot)
* **Dash & Plotly** (Interactive visualization and chatbot interface)
* **HTML/CSS** (Front-end interface styling)

## How to Run the Project

1. Clone the repository:

   ```bash
   git clone https://github.com/YourUsername/DUAL_STAGE_ANFIS_HEART_HEALTH_ASSESSMENT.git
   cd DUAL_STAGE_ANFIS_HEART_HEALTH_ASSESSMENT
   ```

2. Run each Dash application:

   ```bash
   python heart_disease.py     # For heart disease prediction
   python heart_attack.py      # For heart attack risk prediction
   python chatbot.py           # For chatbot interface
   ```

3. Open your browser and navigate to the localhost URLs shown in the terminal.

## Usage

* Input patient health parameters into the Dash interface to receive disease and risk predictions.
* Use the chatbot to ask about causes, risk levels, and preventive tips.
* Explore surface plots to understand how different variables affect heart health.

## Future Enhancements

* Integration with real-time wearable health monitoring data.
* Support for multilingual chatbot interaction.
* Expansion to other cardiovascular diseases using modular ANFIS blocks.



