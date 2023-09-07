import numpy as np
from flask import Flask, request , jsonify, render_template
import pickle
import pandas as pd
import csv
#create flask app
app = Flask(__name__,template_folder='template')

#load pickle model.py

model = pickle.load(open("model.pkl","rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods =["POST"])

def predict():
    Age = request.form['Age']
    Gender = request.form['Gender']
    symptom1 = request.form['Symptom1']
    symptom2 = request.form['Symptom2']
    symptom3 = request.form['Symptom3']
    Family_history = request.form['Family history']
    Current_medication = request.form['Current medication']

    with open('testing.csv', mode='w') as csv_file:
        fieldnames = ['symptom1','symptom2','symptom3', 'Gender','Age','Current Medications','Family History']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow({'symptom1': symptom1, 'symptom2': symptom2, 'symptom3': symptom3 , 'Gender': Gender,'Age': Age,'Current Medications': Current_medication ,'Family History': Family_history })

    df2 = pd.read_csv('testing.csv')
    # Using + operator to combine two columns
    df2["Symptoms"] = df2['symptom1'].astype(str) + "," + df2["symptom2"].astype(str) + "," + df2["symptom3"]
    df2.drop('symptom1', axis=1, inplace=True)
    df2.drop('symptom2', axis=1, inplace=True)
    df2.drop('symptom3', axis=1, inplace=True)
    split_cols = df2['Symptoms'].str.split(',', expand=True)
    one_hot = pd.get_dummies(split_cols, prefix='', prefix_sep='')
    result = pd.concat([df2, one_hot], axis=1)
    result.drop('Symptoms', axis=1, inplace=True)

    dfhot = pd.get_dummies(df2, columns=['Gender', 'Family History', 'Current Medications'])
    if 'Gender_Female' in dfhot.columns:
        col_ = dfhot.pop('Gender_Female')
        dfhot.insert(0, 'Gender_Female', col_)
    else:
        col_ = dfhot.pop('Gender_Male')
        dfhot.insert(0, 'Gender_Male', col_)
        dfhot['Gender_Female'] = 0
    col_ = dfhot.pop('Age')
    dfhot.insert(0, 'Age', col_)
    seq = pd.Series(range(1, len(dfhot) + 1))
    dfhot.insert(0, 'Patient_Id', seq)
    dfhot.to_csv('Dataset2.csv', index=False)
    print(dfhot)
    print(df2)
    print(df2.head())


    r = np.array([Age,Gender,symptom1,symptom2,symptom3,Family_history,Current_medication] ,dtype=np.str_)

    #test = pd.read_csv('testdummy.csv')
    # C= test.drop(["Disease"],axis=1) # symptoms - testing

    #prediction = model.predict(arr)
    return render_template("index.html")


   # return render_template("index.html",prediction_text = "The disese is{}".format(prediction))

if __name__ == "__main__":
    app.run(debug = True)