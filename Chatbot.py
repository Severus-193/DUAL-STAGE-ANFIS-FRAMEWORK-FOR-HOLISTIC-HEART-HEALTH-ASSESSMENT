import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
import joblib
import dash
from dash import html, dcc, Input, Output, callback

# Define responses for keywords
# Define responses for keywords
keyword_responses = {
    "heart disease": "Heart disease encompasses a range of conditions that affect the heart, including coronary artery disease, heart rhythm problems (arrhythmias), and congenital heart defects. It remains the leading cause of death globally. Risk factors include high blood pressure, high cholesterol, diabetes, obesity, tobacco use, and physical inactivity. Symptoms can vary widely from mild discomfort to severe pain and may not always be obvious. Early detection through screenings and a healthy lifestyle are key in preventing the progression of heart disease.",
    
    "causes": "The causes of heart disease are complex and multifactorial, involving a combination of genetic predispositions and lifestyle choices. Modifiable risk factors include unhealthy diets high in saturated fats and sugars, lack of physical activity, smoking, and excessive alcohol consumption. Additionally, conditions such as hypertension, diabetes, and high cholesterol are significant contributors. Non-modifiable factors include age, family history of heart disease, and gender, with men at higher risk at a younger age compared to women.",
    
    "symptoms": "Heart disease symptoms can manifest differently in individuals, often depending on the type of heart condition. Common symptoms include chest pain or discomfort, which may feel like pressure, tightness, or squeezing; shortness of breath, especially during physical activity or at rest; fatigue, especially during exertion; irregular heartbeats; and swelling in the legs, ankles, or abdomen due to fluid retention. Some individuals, particularly women, may experience atypical symptoms such as nausea, lightheadedness, or back and jaw pain. It is essential to consult a healthcare provider if these symptoms arise.",
    
    "heart attack": "A heart attack occurs when blood flow to a part of the heart is blocked, usually by a blood clot, causing damage to the heart muscle. Common signs of a heart attack include intense chest pain or pressure, discomfort in the arms, back, neck, jaw, or stomach, shortness of breath, and other symptoms such as cold sweat or nausea. Time is of the essence; calling emergency services immediately can significantly improve outcomes. Treatments may involve medications to dissolve clots, surgery to restore blood flow, and lifestyle changes to prevent further attacks.",
    
    "prevention": "Preventing heart disease involves a proactive approach focused on healthy lifestyle choices and regular medical check-ups. Key strategies include adopting a heart-healthy diet rich in fruits, vegetables, whole grains, and lean proteins; engaging in regular physical activity, aiming for at least 150 minutes of moderate exercise per week; maintaining a healthy weight; avoiding smoking and limiting alcohol intake; managing stress through relaxation techniques; and monitoring blood pressure and cholesterol levels regularly. Awareness of family history and risk factors can also guide preventive measures.",
    
    "treatment": "The treatment of heart disease varies depending on the specific condition, severity, and individual patient factors. Common treatment options include lifestyle modifications, such as diet and exercise, medications to control risk factors (like hypertension and high cholesterol), and procedures such as angioplasty or coronary artery bypass grafting (CABG) to restore blood flow. In some cases, patients may require devices such as pacemakers. Ongoing management often includes regular follow-ups and adjustments to treatment plans based on the patient's progress.",
    
    "risk factors": "Risk factors for heart disease can be categorized as modifiable and non-modifiable. Non-modifiable risk factors include age (risk increases with age), gender (men are generally at higher risk), and family history of heart disease. Modifiable risk factors include hypertension, high cholesterol, smoking, sedentary lifestyle, poor diet, obesity, and diabetes. Identifying and managing these risk factors through lifestyle changes and medical treatment can significantly reduce the risk of developing heart disease.",
    
    "diagnosis": "Diagnosing heart disease involves a comprehensive evaluation, including a thorough medical history, physical examination, and diagnostic tests. Common tests include electrocardiograms (ECGs) to assess heart rhythm, echocardiograms to visualize heart structure and function, stress tests to evaluate heart performance under physical exertion, and blood tests to identify risk factors. Advanced imaging techniques like coronary angiography or CT scans may be used to assess blood flow in the coronary arteries. Early diagnosis can lead to better management and improved outcomes.",
    
    "lifestyle changes": "Making significant lifestyle changes is crucial for preventing and managing heart disease. Recommended changes include following a heart-healthy diet, which emphasizes fruits, vegetables, whole grains, lean protein, and healthy fats while minimizing saturated and trans fats, sodium, and added sugars. Regular physical activity, such as walking, swimming, or cycling, is essential for maintaining cardiovascular health. Stress management techniques like yoga or mindfulness can also improve overall well-being. Quitting smoking and limiting alcohol intake are vital steps in reducing heart disease risk.",
    
    "medications": "Medications are an essential part of managing heart disease, aimed at controlling risk factors and preventing complications. Common classes of medications include antiplatelet agents (e.g., aspirin) to reduce clot formation, statins to lower cholesterol levels, beta-blockers to manage blood pressure and heart rate, and ACE inhibitors to help relax blood vessels. It is critical for patients to discuss their medications with their healthcare providers, understand the purpose of each medication, and adhere to prescribed regimens to optimize treatment outcomes.",

    "cholesterol": "Cholesterol is a waxy substance found in the blood, essential for building cells and producing hormones. However, high levels of cholesterol, particularly low-density lipoprotein (LDL) cholesterol, can increase the risk of heart disease. High cholesterol often has no symptoms, making regular screening important. Lifestyle changes, including a heart-healthy diet, regular exercise, and maintaining a healthy weight, can help lower cholesterol levels. In some cases, medications such as statins may be prescribed to help manage cholesterol levels effectively.",
    
    "hypertension": "Hypertension, or high blood pressure, is a significant risk factor for heart disease and stroke. It occurs when the force of blood against the artery walls is consistently too high, often due to lifestyle factors such as a high-salt diet, physical inactivity, and obesity. Hypertension usually has no symptoms, which is why regular screening is crucial. Management strategies include lifestyle changes, such as improving diet and increasing physical activity, and medications like diuretics, ACE inhibitors, or calcium channel blockers to help lower blood pressure and reduce the risk of heart-related complications.",
    
    "stroke": "A stroke occurs when blood flow to a part of the brain is interrupted or reduced, preventing brain tissue from receiving oxygen and nutrients. This can be due to a blocked artery (ischemic stroke) or a burst blood vessel (hemorrhagic stroke). Symptoms include sudden numbness or weakness in the face, arm, or leg, particularly on one side of the body; confusion; difficulty speaking; and loss of balance. Immediate medical attention is critical for stroke treatment. Risk factors for stroke overlap with those for heart disease and include hypertension, high cholesterol, diabetes, and smoking. Preventative measures involve managing these risk factors through lifestyle changes and medications.",
    
    "diabetes": "Diabetes is a chronic condition that affects how the body processes blood sugar (glucose). It can lead to serious health complications, including heart disease. Individuals with diabetes are at an increased risk for developing heart disease due to factors such as high blood sugar, obesity, and hypertension. Symptoms of diabetes include increased thirst, frequent urination, extreme fatigue, and blurred vision. Effective management involves maintaining healthy blood sugar levels through diet, exercise, and medications. Regular monitoring of cardiovascular health is essential for individuals with diabetes to prevent complications.",
    
    "bypass surgery": "Coronary artery bypass grafting (CABG) is a surgical procedure used to treat coronary artery disease. It involves taking a healthy blood vessel from another part of the body and using it to bypass blocked coronary arteries, restoring blood flow to the heart muscle. CABG is typically recommended for patients with severe coronary artery disease, especially if they have experienced chest pain (angina) or a heart attack. Recovery involves a rehabilitation program focused on lifestyle changes, monitoring heart health, and gradually returning to normal activities.",
    
    "medications adherence": "Medication adherence is critical in managing heart disease, ensuring that patients take their prescribed medications as directed. Non-adherence can lead to disease progression and increased risk of complications. Strategies to improve adherence include understanding the purpose and importance of each medication, using reminders (such as pill organizers or apps), simplifying medication regimens, and addressing any barriers, such as side effects or cost. Regular follow-up with healthcare providers can help assess treatment effectiveness and make necessary adjustments to improve adherence.",
    
    "follow up care": "Follow-up care is an essential component of heart disease management, ensuring that patients receive ongoing support after diagnosis and treatment. This includes regular check-ups with healthcare providers to monitor heart health, adjust medications, and discuss lifestyle changes. Follow-up care often involves heart health screenings, educational resources about managing risk factors, and referrals to cardiac rehabilitation programs that focus on exercise, nutrition, and emotional support. Adhering to follow-up care is crucial for maintaining optimal heart health and preventing further complications.",
    
    "exercise": "Regular exercise is vital for maintaining heart health and preventing heart disease. It helps strengthen the heart muscle, improves blood circulation, and reduces risk factors such as high blood pressure, high cholesterol, and obesity. The American Heart Association recommends at least 150 minutes of moderate-intensity aerobic activity or 75 minutes of vigorous-intensity activity per week, alongside muscle-strengthening exercises on two or more days. Activities such as walking, jogging, swimming, and cycling are excellent for cardiovascular health. Incorporating physical activity into daily routines can enhance overall well-being and quality of life.",
    
    "best exercises": "The best exercises for heart health include aerobic activities that elevate the heart rate, such as brisk walking, running, swimming, cycling, and dancing. Resistance training, such as weight lifting or bodyweight exercises, is also beneficial for building strength and metabolism. Flexibility and balance exercises, like yoga and tai chi, contribute to overall fitness and reduce the risk of falls. It’s important to choose activities that you enjoy to maintain consistency. Before starting a new exercise program, especially for individuals with existing heart conditions, consulting with a healthcare provider is recommended.",
    
    "exercise benefits": "The benefits of regular exercise for heart health are numerous. It helps lower blood pressure, improve cholesterol levels, and manage weight, significantly reducing the risk of heart disease. Exercise also enhances cardiovascular fitness, boosts mood through the release of endorphins, and promotes better sleep. Engaging in physical activity can also reduce stress and anxiety levels, contributing to overall mental health. Furthermore, regular exercise has been linked to improved insulin sensitivity, which can lower the risk of developing type 2 diabetes—another major risk factor for heart disease.",
    
    "exercising after heart surgery": "Exercising after heart surgery is crucial for recovery and rehabilitation, but it must be approached carefully. Patients should follow their healthcare provider's guidelines regarding the appropriate time to start exercising and the types of activities recommended. Initially, gentle activities like walking are encouraged, gradually increasing intensity and duration as strength and endurance improve. Cardiac rehabilitation programs offer structured exercise plans tailored to individual needs and monitor progress. Regular exercise post-surgery can aid in regaining strength, improving cardiovascular fitness, and reducing the risk of future heart-related issues."
}

# Add more queries and detailed responses as needed


# Function to load and preprocess data
def preprocess_data(file_path, target_col, num_classes):
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Separate features and target
    X = data.drop(columns=[target_col])
    y = data[target_col]
    
    # Identify categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    # Apply OneHotEncoder for categorical columns and MinMaxScaler for numeric columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), X.select_dtypes(include=[np.number]).columns),
            ('cat', OneHotEncoder(), categorical_cols)
        ]
    )
    
    # Fit and transform features
    X_processed = preprocessor.fit_transform(X)
    
    # Encode the target column
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded, num_classes=num_classes)
    
    return X_processed, y_categorical, encoder, preprocessor

# Build the LSTM model
def build_lstm_model(input_shape, output_shape):
    model = Sequential([
        LSTM(128, input_shape=input_shape, return_sequences=True),
        LSTM(64),
        Dense(output_shape, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train Heart Disease Model
def train_heart_disease_model():
    X_hd, y_hd, encoder_hd, preprocessor_hd = preprocess_data(
        file_path='Cleveland_heart_disease.csv', 
        target_col='Heart Disease', 
        num_classes=8
    )
    input_shape_hd = (X_hd.shape[1], 1)
    heart_disease_model = build_lstm_model(input_shape_hd, 8)
    heart_disease_model.fit(X_hd.reshape(-1, X_hd.shape[1], 1), y_hd, epochs=135, batch_size=16)
    
    # Save the heart disease model, encoder, and preprocessor
    heart_disease_model.save('heart_disease_model.h5')
    joblib.dump(encoder_hd, 'encoder_hd.joblib')
    joblib.dump(preprocessor_hd, 'preprocessor_hd.joblib')

# Train Heart Attack Model
def train_heart_attack_model():
    X_ha, y_ha, encoder_ha, preprocessor_ha = preprocess_data(
        file_path='Cleveland_heart_attack.csv', 
        target_col='Risk Level', 
        num_classes=3
    )
    input_shape_ha = (X_ha.shape[1], 1)
    heart_attack_model = build_lstm_model(input_shape_ha, 3)
    heart_attack_model.fit(X_ha.reshape(-1, X_ha.shape[1], 1), y_ha, epochs=100, batch_size=16)
    
    # Save the heart attack model, encoder, and preprocessor
    heart_attack_model.save('heart_attack_model.h5')
    joblib.dump(encoder_ha, 'encoder_ha.joblib')
    joblib.dump(preprocessor_ha, 'preprocessor_ha.joblib')

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div(style={'backgroundColor':'floralwhite','padding':'20px'},children=[
    html.H1("DeadX for Futuristric Heart Health Insights",style={'textAlign':'center'}),
    dcc.Textarea(id='user-input', placeholder='Ask your question or mention keywords...', style={'width': '100%', 'height': '100px'}),
    html.Button('Submit', id='submit-button', n_clicks=0),
    html.Div(id='response-output', style={'margin-top': '20px', 'font-weight': 'bold'}),
    html.Div([
        html.Button('Heart Disease', id='btn-heart-disease', style={'backgroundColor': '#FF5733', 'color': 'white'}),
        html.Button('Causes', id='btn-causes', style={'backgroundColor': '#33FF57', 'color': 'white'}),
        html.Button('Symptoms', id='btn-symptoms', style={'backgroundColor': '#3357FF', 'color': 'white'}),
        html.Button('Heart Attack', id='btn-heart-attack', style={'backgroundColor': '#FFC300', 'color': 'black'}),
        html.Button('Prevention', id='btn-prevention', style={'backgroundColor': '#DAF7A6', 'color': 'black'}),
        html.Button('Treatment', id='btn-treatment', style={'backgroundColor': '#581845', 'color': 'white'}),
        html.Button('Risk Factors', id='btn-risk-factors', style={'backgroundColor': '#900C3F', 'color': 'white'}),
        html.Button('Diagnosis', id='btn-diagnosis', style={'backgroundColor': '#E6E6FA', 'color': 'black'}),
        html.Button('Lifestyle Changes', id='btn-lifestyle-changes', style={'backgroundColor': '#FFA07A', 'color': 'white'}),
        html.Button('Medications', id='btn-medications', style={'backgroundColor': '#EEE8AA', 'color': 'black'}),
        html.Button('Cholesterol', id='btn-cholesterol', style={'backgroundColor': '#87CEFA', 'color': 'white'}),
        html.Button('Hypertension', id='btn-hypertension', style={'backgroundColor': '#D8BFD8', 'color': 'black'}),
        html.Button('Stroke', id='btn-stroke', style={'backgroundColor': '#F0FFF0', 'color': 'black'}),
        html.Button('Diabetes', id='btn-diabetes', style={'backgroundColor': '#FFDAB9', 'color': 'black'}),
        html.Button('Bypass Surgery', id='btn-bypass-surgery', style={'backgroundColor': '#B0E0E6', 'color': 'black'}),
        html.Button('Medications Adherence', id='btn-medications-adherence', style={'backgroundColor': '#B0C4DE', 'color': 'black'}),
        html.Button('Follow Up Care', id='btn-follow-up-care', style={'backgroundColor': '#FFF0F5', 'color': 'black'}),
        html.Button('Exercise', id='btn-exercise', style={'backgroundColor': '#F08080', 'color': 'white'}),
        html.Button('Best Exercises', id='btn-best-exercises', style={'backgroundColor': '#FFE4E1', 'color': 'black'}),
        html.Button('Exercise benefits', id='btn-exercise-benefits', style={'backgroundColor': '#F0F8FF', 'color': 'black'}),
        html.Button('Exercising after heart surgery', id='btn-exercising-after-heart-surgery', style={'backgroundColor': '#F0E68C', 'color': 'black'}),
    ], style={'margin-top': '20px'}),
])

# Define callback to handle user input and button clicks
@callback(
    Output('response-output', 'children'),
    Input('submit-button', 'n_clicks'),
    Input('user-input', 'value'),
    Input('btn-heart-disease', 'n_clicks'),
    Input('btn-causes', 'n_clicks'),
    Input('btn-symptoms', 'n_clicks'),
    Input('btn-heart-attack', 'n_clicks'),
    Input('btn-prevention', 'n_clicks'),
    Input('btn-treatment', 'n_clicks'),
    Input('btn-risk-factors', 'n_clicks'),
    Input('btn-diagnosis', 'n_clicks'),
    Input('btn-lifestyle-changes', 'n_clicks'),
    Input('btn-medications', 'n_clicks'),
    Input('btn-cholesterol', 'n_clicks'),
    Input('btn-hypertension', 'n_clicks'),
    Input('btn-stroke', 'n_clicks'),
    Input('btn-diabetes', 'n_clicks'),
    Input('btn-bypass-surgery', 'n_clicks'),
    Input('btn-medications-adherence', 'n_clicks'),
    Input('btn-follow-up-care', 'n_clicks'),
    Input('btn-exercise', 'n_clicks'),
    Input('btn-best-exercises', 'n_clicks'),
    Input('btn-exercise-benefits', 'n_clicks'),
    Input('btn-exercising-after-heart-surgery', 'n_clicks'),

)
def respond_to_query(n_clicks_submit, user_input, n_clicks_hd, n_clicks_causes, n_clicks_symptoms, n_clicks_ha, n_clicks_prevention, n_clicks_treatment, n_clicks_rf,n_clicks_diagnosis,n_clicks_lf,n_clicks_medications,n_clicks_cholesterol,n_clicks_hypertension,n_clicks_stroke,n_clicks_diabetes,n_clicks_bs,n_clicks_ma,n_clicks_fuc,n_clicks_exercise,n_clicks_be,n_clicks_eb,n_clicks_eahs):
    ctx = dash.callback_context

    if ctx.triggered:
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

        # Check if a button was clicked
        if triggered_id in ['btn-heart-disease', 'btn-heart-attack', 'btn-causes', 'btn-symptoms', 'btn-prevention', 'btn-treatment', 'btn-risk-factors','btn-diagnosis','btn-lifestyle-changes','btn-medications','btn-cholesterol','btn-hypertension','btn-stroke','btn-diabetes','btn-bypass-surgery','btn-medications-adherence','btn-follow-up-care','btn-exercise','btn-best-exercises','btn-exercise-benefits','btn-exercising-after-heart-surgery']:
            # Mapping button IDs to keywords
            keyword_map = {
                'btn-heart-disease': 'heart disease',
                'btn-heart-attack': 'heart attack',
                'btn-causes': 'causes',
                'btn-symptoms': 'symptoms',
                'btn-prevention': 'prevention',
                'btn-treatment': 'treatment',
                'btn-risk-factors': 'risk factors',
                'btn-diagnosis':'diagnosis',
                'btn-lifestyle-changes':'lifestyle changes',
                'btn-medications':'medications',
                'btn-cholesterol':'cholesterol',
                'btn-hypertension':'hypertension',
                'btn-stroke':'stroke',
                'btn-diabetes':'diabetes',
                'btn-bypass-surgery':'bypass surgery',
                'btn-medications-adherence':'medications adherence',
                'btn-follow-up-care':'follow up care',
                'btn-exercise':'exercise',
                'btn-best-exercises':'best exercises',
                'btn-exercise-benefits':'exercise benefits',
                'btn-exercising-after-heart-surgery':'exercising after heart surgery'

            }
            keyword = keyword_map[triggered_id]
            return keyword_responses.get(keyword, "I didn't understand that.")

        # Check if the submit button was clicked and user input exists
        if triggered_id == 'submit-button' and user_input:
            user_input = user_input.lower()  # Convert input to lowercase for consistency

            # Split the user input into queries
            queries = [query.strip() for query in user_input.split(',')]

            responses = []  # Initialize a list to collect responses

            # Check for keywords in each query
            for query in queries:
                for keyword, response in keyword_responses.items():
                    if keyword in query:
                        responses.append(response)  # Append the response for the identified keyword

            # Return all responses or a default message if no keywords matched
            if responses:
                return " ".join(responses)  # Join responses into a single string
            return "I didn't understand that. Please mention keywords like 'heart disease', 'causes', 'symptoms', etc."

    return ""


# Train models when the app starts
train_heart_disease_model()
train_heart_attack_model()



# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
