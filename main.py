from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel
from datetime import datetime
import uvicorn
import joblib
import pandas as pd


app = FastAPI()

class PredictionInput(BaseModel):
    date: str
    

# Load the stacking model in production
loaded_model = joblib.load('final_model.joblib')

# Function to preprocess input data
def preprocess_data(data):
    resampled_data = data.asfreq('d').interpolate(method='linear')
    shifted_data = resampled_data.shift()
    shift_columns = [column + ' -1' for column in resampled_data.columns]
    shifted_data.columns = shift_columns

    result_data = pd.concat([resampled_data, shifted_data], axis=1).fillna(-1)
    result_data['std'] = result_data[shift_columns].std(axis=1)
    result_data['mean'] = result_data[shift_columns].mean(axis=1)
    result_data['median'] = result_data[shift_columns].median(axis=1)
    result_data['min'] = result_data[shift_columns].min(axis=1)
    result_data['max'] = result_data[shift_columns].max(axis=1)

    return result_data

@app.get("/")
def root():
    return RedirectResponse(url="/docs")

@app.post("/predict_multiple_days")
async def predict_water_levels(input_data: PredictionInput):
    try:
        # Convert the input date string to a datetime object
        input_date = datetime.strptime(input_data.date, "%d-%m-%Y")

        # Create a DataFrame with 'date' and target columns
        targets = [
            'MELLEGUE', 'BEN METIR', 'KASSEB', 'BARBARA', 'SIDI SALEM',
            'BOU-HEURTMA', 'JOUMINE', 'GHEZALA', 'SEJNANE', 'S. EL BARRAK',
            'SILIANA', 'LAKHMESS', 'RMIL', "BIR M'CHERGA", 'RMEL', 'NEBHANA',
            'SIDI SAAD', 'EL HAOUAREB', 'SIDI AÏCH', 'EL BREK', 'BEZIRK', 'CHIBA',
            'MASRI', 'LEBNA', 'HMA', 'ABID', 'Zarga', 'Ziatine'
        ]
        past = pd.DataFrame({'date': [input_date]})
        past.set_index('date', inplace=True)  # Set 'date' column as the index
        past[targets] = 0   # Initialize target columns with 0; replace with actual values if available

        
       

        
        # You need to preprocess the input data to match the features used during training
        preprocessed_input = preprocess_data(past)

        # Ensure that the input_data has the same features as the model was trained on (33 features)
        # If needed, you may need to adjust the preprocessing steps to match the training data features

        # Perform the prediction for all 28 dams
        predictions = loaded_model.predict(preprocessed_input.drop(targets, axis=1))
        
        prediction_result = pd.DataFrame({
            'dam_name': targets,
            'predicted_water_level': predictions[0]
        })
        
        # Create a JSON response with predictions for each dam
        response_data = {
            "date": input_data.date,
            "predicted_water_levels": prediction_result.to_dict(orient='records')
        }
        return JSONResponse(content=response_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)