import pickle
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, render_template
import pandas as pd
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)



# Initialize the scaler
scaler = StandardScaler()

def preprocess(data):
    # Example preprocessing step
    # Assuming 'data' is a DataFrame with the necessary features
    data_scaled = scaler.fit_transform(data)
    return data_scaled

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        print(file_path)
        file.save(file_path)
        
        # Process the CSV file
        data = pd.read_csv(file_path)
        
        # Preprocess the data
        processed_data = preprocess(data)
        
        # Ensure processed_data is a DataFrame or a 2D NumPy array
        print("Processed data type:", type(processed_data))
        print("Processed data shape:", processed_data.shape)
        
        # Make predictions
        with open('model_joblib_heart.pkl', 'rb') as model_file:
            model = pickle.load(model_file)

            # Print the type of the model to ensure it's correctly loaded
            print("Model type after loading:", type(model))
            
            try:
                predictions = model.predict(processed_data)
            except AttributeError as e:
                return f"Model type: {type(model)}, Error: {e}"
        
            
            # Add predictions to the DataFrame
            data['Prediction'] = predictions
        
        # Create a response with the predictions
        response = data.to_html()
        
        return response

if __name__ == '__main__':
    app.run(debug=True)