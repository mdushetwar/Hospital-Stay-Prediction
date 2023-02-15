# Loading the libraries
from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import pickle

# Creating Flask app
app= Flask(__name__)

# Loading the model,encoder and sclaer
xgb_model = pickle.load(open("model.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route('/')
def Home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    hospital_code=int(request.form.get('Hospital_code'))
    hospital_type_code=request.form.get('Hospital_type_code')
    city_code_of_hospital=int(request.form.get('City_Code_Hospital'))
    hospital_region_code=request.form.get('Hospital_region_code')
    available_extra_rooms=int(request.form.get('Available_Extra_Rooms_in_Hospital'))
    department=request.form.get('Department')
    ward_type=request.form.get('Ward_Type')
    ward_facility_code=request.form.get('Ward_Facility_Code')
    bed_grade=int(request.form.get('Bed_Grade'))
    city_code_patient=int(request.form.get('City_Code_Patient'))
    type_of_admission=request.form.get('Type_of_Admission')
    severity_of_illness=request.form.get('Severity_of_Illness')
    visitors_with_patient=request.form.get('Visitors_with_Patient')
    age=request.form.get('Age')
    admission_deposit=float(request.form.get('Admission_Deposit'))

    features=np.array([hospital_code, hospital_type_code, city_code_of_hospital, hospital_region_code, 
                available_extra_rooms, department, ward_type, ward_facility_code, bed_grade, 
                city_code_patient, type_of_admission, severity_of_illness, visitors_with_patient, 
                age, admission_deposit])

    encoder_index = [1, 3, 5, 6, 7, 10, 11, 13]

    cat_features= np.array([features[i] for i in encoder_index]).reshape(1,-1)

    encoded_cats= encoder.transform(cat_features)

    features[encoder_index]= encoded_cats

    scaled_features=  scaler.transform(features.reshape(1,-1))

    prediction= xgb_model.predict(scaled_features)

    output=prediction[0]

    if output==0:
        output='0-10'
    elif output==1:
        output='11-20'
    elif output==2:
        output='21-30'
    elif output==3:
        output='31-40'
    elif output==4:
        output='41-50'
    elif output==5:
        output='51-60'
    elif output==6:
        output='61-70'
    elif output==7:
        output='71-80'
    elif output==8:
        output='81-90'
    elif output==9:
        output='91-100'
    else:
        output='More than 100'


    return render_template('predict.html', prediction=output)


if __name__ == "__main__":
    app.run(debug=True)
