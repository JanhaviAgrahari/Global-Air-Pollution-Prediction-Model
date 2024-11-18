import logging
from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

## Route for a home page
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")
    else:
        try:
            # Create CustomData object with updated parameters
            data = CustomData(
                country=request.form.get("country"),
                co_aqi_value=float(request.form.get("co_aqi_value")),
                ozone_aqi_value=float(request.form.get("ozone_aqi_value")),
                no2_aqi_value=float(request.form.get("no2_aqi_value")),
                aqi_category=request.form.get("aqi_category"),
            )

            # Generate a DataFrame from the input data
            pred_df = data.get_data_as_data_frame()
            print(pred_df)
            print("Before Prediction")

            # Initialize PredictPipeline and make predictions
            predict_pipeline = PredictPipeline()
            print("Mid Prediction")
            results = predict_pipeline.predict(pred_df)
            print("After Prediction")
            
            # Render the result
            return render_template("home.html", results=results[0])

        except Exception as e:
            logging.error("Error occurred: %s", e, exc_info=True)
            return "Internal Server Error", 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
