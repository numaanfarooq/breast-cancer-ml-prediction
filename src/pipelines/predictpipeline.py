import pandas as pd
import joblib
import os

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        # Define the paths to the model and preprocessor
        model_path = os.path.join("Notebook", "data", "svm_model.joblib")
        preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
        
        # Load the model and preprocessor
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        
        # Check the type of the preprocessor object
        if not hasattr(preprocessor, 'transform'):
            raise AttributeError("The loaded preprocessor does not have a 'transform' method.")
        
        # Transform the features using the preprocessor
        data_scaled = preprocessor.transform(features)
        
        # Make predictions using the model
        preds = model.predict(data_scaled)
        
        return preds

class CustomData: 
    def __init__(self, Age, BMI, Glucose, Insulin, HOMA, Leptin, Adiponectin, Resistin, MCP_1):
        self.Age = Age
        self.BMI = BMI
        self.Glucose = Glucose
        self.Insulin = Insulin
        self.HOMA = HOMA
        self.Leptin = Leptin
        self.Adiponectin = Adiponectin
        self.Resistin = Resistin
        self.MCP_1 = MCP_1
    
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Age": [self.Age],
                "BMI": [self.BMI],
                "Glucose": [self.Glucose],
                "Insulin": [self.Insulin],
                "HOMA": [self.HOMA],
                "Leptin": [self.Leptin],
                "Adiponectin": [self.Adiponectin],
                "Resistin": [self.Resistin],
                "MCP.1": [self.MCP_1]
            }
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
