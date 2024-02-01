# -*- coding: utf-8 -*-

import numpy as np
from flask import Flask, request, render_template
import urllib.request
import json
from flask_cors import CORS
from joblib import load
import sklearn

# Create application
app = Flask(__name__)
CORS(app)

scaler = load(open('files/scaler.joblib', 'rb'))


# Define feature keys and their default values
FEATURE_KEYS = [
  "INSTALLMENT_TENOR_MONTHS",
  "APPLIED_AMOUNT",
  "APPROVED_AMOUNT",
  "INTEREST_RATE_PA",
  "FINAL_PD",
  "RnR",
  "Financing_Amount_Principal",
  "Financing_Amount_Interest",
  "YIB",
  "Female_Y",
  "Avg_Age_When_apply",
  "Origination_fee",
  "Total_investors",
  "Avg_investor_investment",
  "Guarantor_invested",
  "days_to_disb",
  "Day_disbursed",
  "Day_hosted",
  "Nature_of_Business_Accommodation & Food Service Activities",
  "Nature_of_Business_Administrative and Support Service Activities",
  "Nature_of_Business_Agriculture, Forestry and Fishing",
  "Nature_of_Business_Arts, Entertainment and Recreation",
  "Nature_of_Business_Construction",
  "Nature_of_Business_Education",
  "Nature_of_Business_Financial and Insurance/Takaful Activities",
  "Nature_of_Business_Human Health and Social Work Activities",
  "Nature_of_Business_Information and Communication",
  "Nature_of_Business_Manufacturing",
  "Nature_of_Business_Other Service Activities",
  "Nature_of_Business_Professional, Scientific & Technical Activities",
  "Nature_of_Business_Real-Estate Activities",
  "Nature_of_Business_Transportation & Storage",
  "Nature_of_Business_Water, Sewerage & Waste Mgmt & Related Activities",
  "Nature_of_Business_Wholesale & Retail (incl Motor Vehicle Repairs)",
  "Entity_Partnership",
  "Entity_Private Limited",
  "Entity_Sole Proprietor",
  "STATE_JOHOR",
  "STATE_KEDAH",
  "STATE_KELANTAN",
  "STATE_Kedah",
  "STATE_MELAKA",
  "STATE_NEGERI SEMBILAN",
  "STATE_PAHANG",
  "STATE_PERAK",
  "STATE_PERLIS",
  "STATE_PULAU PINANG",
  "STATE_Perak",
  "STATE_SABAH",
  "STATE_SARAWAK",
  "STATE_SELANGOR",
  "STATE_Sarawak",
  "STATE_TERENGGANU",
  "STATE_WILAYAH PERSEKUTUAN",
  "LOAN_PURPOSE_Asset acquisition",
  "LOAN_PURPOSE_Contract financing",
  "LOAN_PURPOSE_Debt consolidation / refinancing",
  "LOAN_PURPOSE_Marketing and advertising",
  "LOAN_PURPOSE_Other business purpose",
  "LOAN_PURPOSE_Purchase of equipment",
  "LOAN_PURPOSE_Upgrading / Renovation",
  "LOAN_PURPOSE_Working Capital",
  "FIN_GRADE_A1",
  "FIN_GRADE_A2",
  "FIN_GRADE_B3",
  "FIN_GRADE_B4",
  "FIN_GRADE_C5",
  "FIN_GRADE_C6",
  "FIN_GRADE_D7",
  "FIN_GRADE_D8",
  "FIN_GRADE_X1",
  "Race_Chinese",
  "Race_Indian",
  "Race_Malay",
  "Race_Others"
]


FEATURE_DEFAULTS = {key: 0.0 for key in FEATURE_KEYS}

# Define template file
TEMPLATE_FILE = 'index.html'

@app.route('/', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        # Get form values
        form_values = [value for value in request.form.values()]
        # Prepare features
        features = FEATURE_DEFAULTS.copy()
        features.update(zip(FEATURE_KEYS[:18], map(float, form_values[:18])))

        # Update features based on selected options
        features[f'Nature_of_Business_{form_values[18]}'] = 1.0
        features[f'Entity_{form_values[19]}'] = 1.0
        features[f'STATE_{form_values[20]}'] = 1.0
        features[f'LOAN_PURPOSE_{form_values[21]}'] = 1.0
        features[f'FIN_GRADE_{form_values[22]}'] = 1.0
        features[f'Race_{form_values[23]}'] = 1.0

        # Extract numerical values
        numerical_values = list(features.values())
        
        # Transform the numerical values
        scaled_values_2d = scaler.transform(np.array(numerical_values).reshape(1, -1))
        # Update the dictionary with the scaled values
        scaled_values = [val for val in scaled_values_2d[0]]
        scaled_dict = dict(zip(features.keys(), scaled_values))


        # Prepare data for API request
        data = {
            "Inputs": {"data": [scaled_dict]},
            "GlobalParameters": 0.0
        }

        # Encode data and make API request
        body = str.encode(json.dumps(data))
        url = 'http://b579b683-51a0-4526-8834-a3b9a8ee91be.eastus.azurecontainer.io/score'
        headers = {'Content-Type': 'application/json'}
        req = urllib.request.Request(url, body, headers)

        try:
            response = urllib.request.urlopen(req)
            result = response.read().decode("utf8", 'ignore')  # Decode the result to a string
            result=float(result[13:-3])
            if result > 1:
                result = 1
            elif result < 0:
                result = 0
            print(f't: {result}')
            result = round(result,4)
        except urllib.error.HTTPError as error:
            result = error.read().decode("utf8", 'ignore')

        # Pass the result and form values to the template
        return render_template(TEMPLATE_FILE, result=result, form_values=form_values)

    # If it's a GET request, just render the template with default values
    return render_template(TEMPLATE_FILE, result=None, form_values=None)

if __name__ == '__main__':
    # Run the application
    app.run(debug=True, port=8000)
