import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)

    
class CustomData:
    def __init__( self,
        Gender: str,
        Customer_Type: str,
        Type_of_Travel: str,
        Class: str,
        Age: int,
        Flight_Distance: float,
        Departure_Delay_in_Minutes: float,
        Arrival_Delay_in_Minutes: float,
        Seat_comfort: int,
        Departure_Arrival_time_convenient: int,
        Food_and_drink: int,
        Gate_location: int,
        Inflight_wifi_service: int,
        Inflight_entertainment: int,
        Online_support: int,
        Ease_of_Online_booking: int,
        Onboard_service: int,
        Leg_room_service: int,
        Baggage_handling: int,
        Checkin_service: int,
        Cleanliness: int,
        Online_boarding: int):
        self.Gender = Gender
        self.Customer_Type = Customer_Type        
        self.Type_of_Travel =Type_of_Travel
        self.Class = Class
        self.Age = Age
        self.Flight_Distance = Flight_Distance
        self.Departure_Delay_in_Minutes = Departure_Delay_in_Minutes
        self.Arrival_Delay_in_Minutes = Arrival_Delay_in_Minutes
        self.Seat_comfort = Seat_comfort
        self.Departure_Arrival_time_convenient = Departure_Arrival_time_convenient
        self.Food_and_drink = Food_and_drink
        self.Gate_location = Gate_location
        self.Inflight_wifi_service = Inflight_wifi_service
        self.Inflight_entertainment = Inflight_entertainment
        self.Online_support = Online_support
        self.Ease_of_Online_booking = Ease_of_Online_booking
        self.Onboard_service = Onboard_service
        self.Leg_room_service = Leg_room_service
        self.Baggage_handling = Baggage_handling
        self.Checkin_service = Checkin_service
        self.Cleanliness = Cleanliness
        self.Online_boarding = Online_boarding
    
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
               "Gender": [self.Gender],
               "Customer_Type": [self.Customer_Type],
               "Type_of_Travel": [self.Type_of_Travel],
               "Class": [self.Class] ,
               "Age": [self.Age],
               "Flight_Distance": [self.Flight_Distance],
               "Departure_Delay_in_Minutes": [self.Departure_Delay_in_Minutes],
               "Arrival_Delay_in_Minutes": [self.Arrival_Delay_in_Minutes],
               "Seat_comfort": [self.Seat_comfort],
               "Departure_Arrival_time_convenient": [self.Departure_Arrival_time_convenient],
               "Food_and_drink": [self.Food_and_drink],
               "Gate_location": [self.Gate_location],
               "Inflight_wifi_service": [self.Inflight_wifi_service],
               "Inflight_entertainment": [self.Inflight_entertainment],
               "Online_support": [self.Online_support],
               "Ease_of_Online_booking": [self.Ease_of_Online_booking],
               "Onboard_service": [self.Onboard_service],
               "Leg_room_service": [self.Leg_room_service],
               "Baggage_handling": [self.Baggage_handling],
               "Checkin_service": [self.Checkin_service],
               "Cleanliness": [self.Cleanliness],
               "Online_boarding": [self.Online_boarding]
            }
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)