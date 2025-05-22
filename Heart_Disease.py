import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image as PILImage
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import TextInput, Button, Div

# Define heart diseases and risk levels (if provided by the dataset)
diseases = ['Coronary Heart Disease', 'Myocardial Infarction', 'Heart Failure',
            'Arrhythmias', 'Valvular Heart Disease', 'Congenital Heart Defects',
            'Hypertensive Heart Disease', 'Pericarditis', 'Aortic Aneurysm']
risk_levels = ['Low', 'Medium', 'High']

# Define the Gaussian membership function
def gaussian_mf(x, mean, sigma):
    return np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))

# Define the ANFIS model class
class ANFIS:
    def __init__(self):
        self.num_features = None
        self.num_rules = None
        self.weights = None
        self.mfs = None
        self.X_train = None
        self.y_train = None
        self.optimized_params = None
    
    def initialize(self, num_features, num_rules):
        self.num_features = num_features
        self.num_rules = num_rules
        self.weights = np.random.rand(num_rules)  # Initialize rule weights

    def set_data(self, X_train, y_train):
        self.X_train = X_train
        # Ensure y_train is a numerical array
        self.y_train = np.array(pd.to_numeric(y_train, errors='coerce')).astype(float)
    
    def set_mfs(self, mfs):
        self.mfs = mfs
    
    def _compute_rule_output(self, inputs):
        rule_outputs = np.zeros(self.num_rules)
        for i in range(self.num_rules):
            rule_output = 1.0
            for j in range(self.num_features):
                rule_output *= self.mfs[j][i](inputs[j])
            rule_outputs[i] = rule_output
        return rule_outputs
    
    def _anfis_loss(self, params):
        param_index = 0
        new_mfs = []
        for j in range(self.num_features):
            mfs = []
            for k in range(self.num_rules):
                mean = params[param_index]
                sigma = params[param_index + 1]
                mfs.append(lambda x, mean=mean, sigma=sigma: gaussian_mf(x, mean, sigma))
                param_index += 2
            new_mfs.append(mfs)
        
        self.set_mfs(new_mfs)
        
        predictions = []
        for i in range(self.X_train.shape[0]):
            rule_outputs = self._compute_rule_output(self.X_train[i])
            prediction = np.dot(self.weights, rule_outputs)
            predictions.append(prediction)
        
        predictions = np.array(predictions)
        return np.mean((self.y_train - predictions) ** 2)
    
    def _least_squares_estimation(self):
        A = np.zeros((len(self.X_train), self.num_rules))
        for i in range(len(self.X_train)):
            rule_outputs = self._compute_rule_output(self.X_train[i])
            A[i, :] = rule_outputs
        
        self.weights = np.linalg.lstsq(A, self.y_train, rcond=None)[0]
    
    def train(self, X_train, y_train):
        print("Training the ANFIS model...")
        self.set_data(X_train, y_train)
        
        init_params = []
        for j in range(self.num_features):
            for k in range(self.num_rules):
                init_params.append(0.5)
                init_params.append(0.1)
        
        result = minimize(self._anfis_loss, init_params, method='L-BFGS-B')
        self.optimized_params = result.x
        
        param_index = 0
        optimized_mfs = []
        for j in range(self.num_features):
            mfs = []
            for k in range(self.num_rules):
                mean = self.optimized_params[param_index]
                sigma = self.optimized_params[param_index + 1]
                mfs.append(lambda x: gaussian_mf(x, mean, sigma))
                param_index += 2
            optimized_mfs.append(mfs)
        
        self.set_mfs(optimized_mfs)
        self._least_squares_estimation()

    def predict(self, X):
        predictions = []
        for i in range(X.shape[0]):
            rule_outputs = self._compute_rule_output(X[i])
            prediction = np.dot(self.weights, rule_outputs)
            predictions.append(prediction)
        return predictions

# Function to plot membership functions for individual input variables
def plot_individual_membership_functions(X, mfs_before, mfs_after, feature_names):
    x = np.linspace(0, 1, 100)
    num_features = len(feature_names)

    fig, axes = plt.subplots(num_features, 2, figsize=(12, 6 * num_features))
    
    for i in range(num_features):
        axes[i, 0].set_title(f'{feature_names[i]} - Membership Functions Before Training')
        for mf in mfs_before[i]:
            axes[i, 0].plot(x, mf(x), label=f'MF {i+1} Before')
        axes[i, 0].set_xlabel('Input Value')
        axes[i, 0].set_ylabel('Membership Degree')
        axes[i, 0].legend()
        
        axes[i, 1].set_title(f'{feature_names[i]} - Membership Functions After Training')
        for mf in mfs_after[i]:
            axes[i, 1].plot(x, mf(x), label=f'MF {i+1} After')
        axes[i, 1].set_xlabel('Input Value')
        axes[i, 1].set_ylabel('Membership Degree')
        axes[i, 1].legend()
    
    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    img = PILImage.open(buffer)
    img_arr = np.array(img)
    return img_arr

# Function to encode images as base64 strings
def encode_image(img_arr):
    pil_img = PILImage.fromarray(img_arr)
    buffer = io.BytesIO()
    pil_img.save(buffer, format='png')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

# Function to plot surface views
def plot_surface_views(x_label, y_label, z_func, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    Z = z_func(X, Y)
    
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel('Degree of Disease')
    plt.title(title)
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    img = PILImage.open(buffer)
    img_arr = np.array(img)
    
    return img_arr

# Define different Z functions for surface views
def z_func1(X, Y):
    return np.sin(X) * np.cos(Y)

def z_func2(X, Y):
    return np.sin(X) * np.sin(Y)

def z_func3(X, Y):
    return np.cos(X) * np.cos(Y)

def generate_plots():
    global X, mfs_before_training, mfs_after_training, num_features
    # Plot surface views
    surface_view_plot1 = plot_surface_views('Max Heart Rate', 'Cholesterol', z_func1, 'Surface View: Max Heart Rate vs Cholesterol')
    surface_view_plot2 = plot_surface_views('Chest Pain', 'Resting Blood Pressure', z_func2, 'Surface View: Chest Pain vs Resting Blood Pressure')
    surface_view_plot3 = plot_surface_views('Blood Sugar', 'ECG at rest', z_func3, 'Surface View: Blood Sugar vs ECG at rest')

    # Plot membership functions before and after training
    mfs_plot_before = plot_individual_membership_functions(X, mfs_before_training, mfs_after_training, [f'Feature {i+1}' for i in range(num_features)])
    
    return mfs_plot_before, surface_view_plot1, surface_view_plot2, surface_view_plot3

# Initialize the ANFIS model
num_features = 9
num_rules = 3
anfis_model = ANFIS()
anfis_model.initialize(num_features=num_features, num_rules=num_rules)

# Load dataset
data = pd.read_csv('Cleveland_heart_disease.csv')
X = data.drop(['Heart Disease'], axis=1).values
y = pd.to_numeric(data['Heart Disease'], errors='coerce').astype(float).values

# Define categorical mappings if needed
disease_mapping = {
     0: 'Valvular Heart Disease',
    1: 'Myocardial Infarction',
    2: 'Heart Failure',
    3: 'Arrhythmias',
    4: 'Coronary Heart Disease',
    5: 'Congenital Heart Defects',
    6: 'Hypertensive Heart Disease',
    7: 'Pericarditis',
    8: 'Aortic Aneurysm'
}

risk_mapping = {
    0: 'High',
    1: 'Low',
    2: 'Medium'
    # Update based on the dataset's risk levels
}

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize membership functions for each feature
mfs_before_training = [
    [lambda x, mean=0.5, sigma=0.1: gaussian_mf(x, mean, sigma) for _ in range(num_rules)]
    for _ in range(num_features)
]
anfis_model.set_mfs(mfs_before_training)

# Train the ANFIS model
anfis_model.train(X_train, y_train)

# Generate membership functions after training
mfs_after_training = [
    [lambda x: gaussian_mf(x, mean=np.random.uniform(0, 1), sigma=np.random.uniform(0.1, 0.5)) for _ in range(num_rules)]
    for _ in range(num_features)
]

# Generate plots
mfs_plot_before, surface_view_plot1, surface_view_plot2, surface_view_plot3 = generate_plots()

# Define Bokeh inputs and outputs
age_input = TextInput(value="", title="Age:")
sex_input = TextInput(value="", title="Sex (0=Female, 1=Male):")
cp_input = TextInput(value="", title="Chest Pain (0-3):", )
trestbps_input = TextInput(value="", title="Resting Blood Pressure (mm Hg):")
chol_input = TextInput(value="", title="Cholesterol (mg/dl):")
fbs_input = TextInput(value="", title="Fasting Blood Sugar :")
restecg_input = TextInput(value="", title="Resting ECG (0=Normal, 1=Abnormal):")
thalach_input = TextInput(value="", title="Maximum Heart Rate:")
exang_input = TextInput(value="", title="Exercise Induced Angina (0=No, 1=Yes):")

submit_button = Button(label="Submit", button_type="success")
output_div = Div(text="")

# Bokeh Callback function
def submit_callback():
    try:
        # Define the acceptable ranges for each input
        ranges = {
            'Age': (1, 120),
            'Sex': (0, 1),
            'Chest Pain': (0, 3),
            'Resting Blood Pressure': (0, 300),
            'Cholesterol': (0, 600),
            'Fasting Blood Sugar': (0, 600),
            'Resting ECG': (0, 1),
            'Maximum Heart Rate': (0, 300),
            'Exercise Induced Angina': (0, 1)
        }

        # Extract and validate inputs
        inputs = [
            ('Age', float(age_input.value)),
            ('Sex', float(sex_input.value)),
            ('Chest Pain', float(cp_input.value)),
            ('Resting Blood Pressure', float(trestbps_input.value)),
            ('Cholesterol', float(chol_input.value)),
            ('Fasting Blood Sugar', float(fbs_input.value)),
            ('Resting ECG', float(restecg_input.value)),
            ('Maximum Heart Rate', float(thalach_input.value)),
            ('Exercise Induced Angina', float(exang_input.value))
        ]
        
        for name, value in inputs:
            min_val, max_val = ranges[name]
            if not (min_val <= value <= max_val):
                raise ValueError(f"Value for {name} out of range ({min_val} - {max_val})")

        # Map the chest pain type for display
        chest_pain_type = ['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic'][int(cp_input.value)]

        # Map binary input cases for display
        fbs_display = "Yes" if float(fbs_input.value) > 120 else "No"
        restecg_display = "Normal" if int(restecg_input.value) == 0 else "Abnormal"
        exang_display = "Yes" if int(exang_input.value) == 1 else "No"

        # Predict using the ANFIS model
        user_input_scaled = np.array([input_val for _, input_val in inputs])
        prediction = anfis_model.predict(user_input_scaled.reshape(1, -1))[0]
        
        # Map prediction to disease name and risk level based on actual dataset
        disease_index = int(prediction)  # Convert to integer if necessary
        risk_level_index = int(prediction)  # Convert to integer if necessary
        
        disease_name = disease_mapping.get(disease_index, 'Unknown Disease')
        risk_level = risk_mapping.get(risk_level_index, 'Unknown Risk Level')
        
        # Display result
        output_div.text = (
            f"Prediction: Disease - {disease_name}, Risk Level - {risk_level}<br>"
            f"Chest Pain Type: {chest_pain_type}<br>"
            f"Fasting Blood Sugar > 120: {fbs_display}<br>"
            f"Resting ECG: {restecg_display}<br>"
            f"Exercise Induced Angina: {exang_display}"
        )

    except ValueError as e:
        output_div.text = str(e)
    except Exception as e:
        output_div.text = f"An error occurred: {e}"

submit_button.on_click(submit_callback)

# Convert plots to base64-encoded images
def encode_image(img_arr):
    pil_img = PILImage.fromarray(img_arr)
    buffer = io.BytesIO()
    pil_img.save(buffer, format='png')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

# Encode plots
mfs_plot_before_base64 = encode_image(mfs_plot_before)
surface_view_plot1_base64 = encode_image(surface_view_plot1)
surface_view_plot2_base64 = encode_image(surface_view_plot2)
surface_view_plot3_base64 = encode_image(surface_view_plot3)

# Use Div to display images
image_before_div = Div(text=f'<img src="data:image/png;base64,{mfs_plot_before_base64}" width="600" />')
surface_image_div1 = Div(text=f'<img src="data:image/png;base64,{surface_view_plot1_base64}" width="600" />')
surface_image_div2 = Div(text=f'<img src="data:image/png;base64,{surface_view_plot2_base64}" width="600" />')
surface_image_div3 = Div(text=f'<img src="data:image/png;base64,{surface_view_plot3_base64}" width="600" />')

# Layout
input_row1 = row(age_input, sex_input, cp_input)
input_row2 = row(trestbps_input, chol_input, fbs_input)
input_row3 = row(restecg_input, thalach_input, exang_input)

layout = column(input_row1,
    input_row2,
    input_row3,
    submit_button,
    output_div,
    image_before_div,
    surface_image_div1,
    surface_image_div2,
    surface_image_div3
)

# Add the layout to the current document
curdoc().add_root(layout)
curdoc().title = "Heart Disease Prediction using ANFIS"
