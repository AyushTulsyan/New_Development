from flask import Flask,request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline


application = Flask(__name__)

app = application

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predictdata',methods=['GET','POST'])
def predictdata():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            Gender=request.form.get('Gender'),
            Customer_Type=request.form.get('Customer_Type'),
            Type_of_Travel=request.form.get('Type_of_travel'),
            Class=request.form.get('Class'),
            Age=int(request.form.get('Age')),
            Flight_Distance=float(request.form.get('Flight_Distance')),
            Departure_Delay_in_Minutes=float(request.form.get('Departure_Delay_in_Minutes')),
            Arrival_Delay_in_Minutes=float(request.form.get('Arrival_Delay_in_Minutes')),
            Seat_comfort=int(request.form.get('Seat_comfort')),
            Departure_Arrival_time_convenient=int(request.form.get('Departure_Arrival_time_convenient')),
            Food_and_drink=int(request.form.get('Food_and_drink')),
            Gate_location=int(request.form.get('Gate_location')),
            Inflight_wifi_service=int(request.form.get('Inflight_wifi_service')),
            Inflight_entertainment=int(request.form.get('Inflight_entertainment')),
            Online_support=int(request.form.get('Online_support')),
            Ease_of_Online_booking=int(request.form.get('Ease_of_Online_booking')),
            Onboard_service=int(request.form.get('Onboard_service')),
            Leg_room_service=int(request.form.get('Leg_room_service')),
            Baggage_handling=int(request.form.get('Baggage_handling')),
            Checkin_service=int(request.form.get('Checkin_service')),
            Cleanliness=int(request.form.get('Cleanliness')),
            Online_boarding=int(request.form.get('Online_boarding')),
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0", debug = True)