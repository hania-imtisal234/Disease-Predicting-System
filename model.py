import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
import pickle
from tkinter import *
from tkinter import messagebox

#arrays Used for taking  input from user
symptom1 =['Symptom1_Anxiety','Symptom1_Dizziness','Symptom1_Double Pain in eye','Symptom1_Frequent urination','Symptom1_Headache','Symptom1_Irritable infant','Symptom1_Leg pain','Symptom1_Pain in eye','Symptom1_Sharp abdominal pain','Symptom1_Sore throat','Symptom1_Throat swelling','Symptom1_Weight gain','Symptom1_Wrist pain','Symptom1_foot pain']
symptom2 =['Symptom2_Ankle pain','Symptom2_Chest pain','Symptom2_Depression','Symptom2_Ear pain','Symptom2_Hand swelling','Symptom2_Knee swelling','Symptom2_Lump in throat','Symptom2_Smoking','Symptom2_Toothache','Symptom2_Vomiting']
symptom3 = ['Symptom3_Anxiety','Symptom3_Cough','Symptom3_Headache','Symptom3_Nausea','Symptom3_Shortness of breath']
symptom = ['Anxiety', 'Back pain', 'Cough', 'Depression', 'Dizziness', 'Double Pain in eye', 'Frequent urination', 'Headache', 'Irritable infant', 'Knee swelling', 'Leg pain', 'Loss of sensation', 'Pain in eye', 'Painful urination', 'Seizures', 'Sharp abdominal pain', 'Sharp chest pain', 'Shortness of breath', 'Sore throat', 'Throat swelling', 'Unpredictable menstruation', 'Vomiting', 'Weight gain', 'Wrist pain', 'foot pain', 'Ankle pain', 'Arm pain', 'Back weakness', 'Hand swelling', 'Lump in throat', 'Muscle pain', 'Nasal congestion', 'Nausea', 'Neck mass', 'Smoking', 'Toothache', 'Upper abdominal pain', 'anxiety', 'chest pain', 'cramps', 'Ear pain', 'Neck pain', 'Fever', 'Weakness' ]
current_medications_ =['Current Medications_Acarbose','Current Medications_Acebutolol','Current Medications_Aceclidine','Current Medications_Aceclofenac']
family_history_=['Family History_Cancer','Family History_Diabetes','Family History_Heart Disease','Family History_Stroke']
gender_ =['Male','Female']
l2 = {'Patient_Id': 0,'Age': 0,'Gender_Male': 0,'Gender_Female': 0,'Symptom1_Anxiety':0,'Symptom1_Dizziness':0,'Symptom1_Double Pain in eye':0,'Symptom1_Frequent urination':0,'Symptom1_Headache':0,'Symptom1_Irritable infant':0,'Symptom1_Leg pain':0,'Symptom1_Pain in eye':0,'Symptom1_Sharp abdominal pain':0,'Symptom1_Sore throat':0,'Symptom1_Throat swelling':0,'Symptom1_Weight gain':0,'Symptom1_Wrist pain':0,'Symptom1_foot pain':0,'Symptom2_Ankle pain':0,'Symptom2_Chest pain':0,'Symptom2_Depression':0,'Symptom2_Ear pain':0,'Symptom2_Hand swelling':0,'Symptom2_Knee swelling':0,'Symptom2_Lump in throat':0,'Symptom2_Smoking':0,'Symptom2_Toothache':0,'Symptom2_Vomiting':0,
      'Symptom3_Anxiety':0,'Symptom3_Cough':0,'Symptom3_Headache':0,'Symptom3_Nausea':0,'Symptom3_Shortness of breath':0,'Family History_Cancer':0,'Family History_Diabetes':0,'Family History_Heart Disease':0,'Family History_Stroke':0,'Current Medications_Acarbose': 0,
      'Current Medications_Acebutolol': 0,'Current Medications_Aceclidine': 0,'Current Medications_Aceclofenac': 0}


#DATA CLEANING

df=pd.read_csv("Collected_dataset.csv")
# check for missing values
if df.isnull().values.any():
    print('Dataset contains missing values.')
else:
    print('Dataset does not contain missing values.')
# check for duplicates
if df.duplicated().values.any():
    print('Dataset contains duplicate rows.')
else:
    print('Dataset does not contain duplicate rows.')
# check for invalid age
if df['Age'].min() < 0 or df['Age'].max() > 120:
    print('Dataset contains illogical age values.')
else:
    print('Dataset does not contain illogical age values.')
#check for gender values
gender_values = set(df['Gender'].unique())
expected_gender_values = set(['Male', 'Female'])
if not gender_values.issubset(expected_gender_values):
    print('Dataset contains misspelled or unexpected gender values.')
else:
    print('Dataset has expected gender values.')

dfhot = pd.get_dummies(df, columns = ['Symptom1','Symptom2','Symptom3','Gender','Family History','Current Medications'])
col_ = dfhot.pop('Gender_Female')
dfhot.insert(0, 'Gender_Female', col_)
col_ = dfhot.pop('Gender_Male')
dfhot.insert(0, 'Gender_Male', col_)
col_ = dfhot.pop('Age')
dfhot.insert(0, 'Age', col_)
seq = pd.Series(range(1, len(dfhot) + 1))
dfhot.insert(0, 'Patient_Id', seq)
dfhot = dfhot.loc[:,~dfhot.columns.duplicated()].copy()
dfhot.to_csv('Dataset.csv', index=False)
dfhot


#BUILDING MODEL AND THEN EVALUATING

root = Tk()
pred1=StringVar()
def randomforest():

    if ((Symptom1.get() == "Select Here") or (Symptom2.get() == "Select Here") or (gender.get() == "Select Here")):
        pred2.set(" ")
        sym = messagebox.askokcancel("System", "Kindly Fill atleast first two Symptoms, Patient Id , age and gender.")
        if sym:
            root.mainloop()
    else:
        df1 = pd.read_csv("Dataset.csv")
        y = df1['Disease']
        print(y)
        x = df1.drop(['Disease'], axis=1)
        print(x.shape)
        x_train_, x_test_, y_train_, y_test = train_test_split(x, y, test_size=0.2, random_state=100)
        x_train = x_train_.values
        x_test = x_test_.values
        y_train = y_train_.values

        rf = RandomForestClassifier(
            n_estimators=100,  # The number of trees in the forest
            criterion='entropy',  # The function to measure the quality of a split
            max_depth=3,  # The maximum depth of the tree
            min_samples_split=30,  # The minimum number of samples required to split an internal node
            min_samples_leaf=20,  # The minimum number of samples required to be at a leaf node
            max_features='sqrt',  # The number of features to consider when looking for the best split
            n_jobs=5,
            random_state=42)
        rf.fit(x_train, y_train)

        y_rf_train_pred = rf.predict(x_train)


        rf_train_accuracy = accuracy_score(y_train, y_rf_train_pred)
        print("The accuracy of the random forest classification model is: ",rf_train_accuracy)

        report = classification_report(y_train, y_rf_train_pred, output_dict=True)
        pd.DataFrame(report).transpose()


        l2['Patient_Id']=id.get()
        l2['Age'] =age.get()
        for z in gender_:
            if (z == gender.get()):
                 str = gender.get()
                 if(str == 'Gender_male'):
                   l2['Gender_male'] = 1
                 else:
                    l2['Gender_Female'] = 1
        for x in symptom1:
            if (x == Symptom1.get()):
                 str = Symptom1.get()
                 l2[str] =1
        for x in symptom2:
            if (x == Symptom2.get()):
                str = Symptom2.get()
                l2[str] = 1
        for x in symptom3:
            if (x == Symptom3.get()):
                 str = Symptom3.get()
                 l2[str] = 1
        for x in family_history_:
            if (x == family_history.get()):
                str = family_history.get()
                l2[str] = 1
        for x in current_medications_:
            if (x == current_medications.get()):
                 str = current_medications.get()
                 l2[str] = 1

        print(l2)
        arr = list(l2.values())
        inputtest = [arr]
        predict = rf.fit(x_train ,y_train).predict(inputtest)
        print(predict)
        pred1.set(predict)

pred2=StringVar()
def DecisionTreeClassifier():

    if ((Symptom1.get() == "Select Here") or (Symptom2.get() == "Select Here") or (gender.get() == "Select Here")):
        pred2.set(" ")
        sym = messagebox.askokcancel("System", "Kindly Fill atleast first two Symptoms, Patient Id , age and gender.")
        if sym:
            root.mainloop()
    else:
     df1 = pd.read_csv("Dataset.csv")
     y = df1['Disease']
     print(y)
     x = df1.drop(['Disease'], axis=1)
     print(x.shape)
     x_train_, x_test_, y_train_, y_test = train_test_split(x, y, test_size=0.2, random_state=100)
     x_train = x_train_.values
     x_test = x_test_.values
     y_train = y_train_.values

     dt = DecisionTreeClassifier(criterion='entropy', random_state=42, max_depth=10)
     dt.fit(x_train, y_train)
     y_dt_train_pred = dt.predict(x_train)
     y_dt_test_pred = dt.predict(x_test)
     dt_train_accuracy = accuracy_score(y_train, y_dt_train_pred)
     dt_train_accuracy
     print("The accuracy of the decision tree classification model is: ", dt_train_accuracy)

     l2['Patient_Id'] = id.get()
     l2['Age'] = age.get()
     for z in gender_:
        if (z == gender.get()):
            str = gender.get()
            if (str == 'Gender_male'):
                l2['Gender_male'] = 1
            else:
                l2['Gender_Female'] = 1
     for x in symptom1:
        if (x == Symptom1.get()):
            str = Symptom1.get()
            l2[str] = 1
     for x in symptom2:
        if (x == Symptom2.get()):
            str = Symptom2.get()
            l2[str] = 1
     for x in symptom3:
        if (x == Symptom3.get()):
            str = Symptom3.get()
            l2[str] = 1
     for x in family_history_:
        if (x == family_history.get()):
            str = family_history.get()
            l2[str] = 1
     for x in current_medications_:
        if (x == current_medications.get()):
            str = current_medications.get()
            l2[str] = 1

     print(l2)
     arr = list(l2.values())
     inputtest = [arr]
     predict = dt.fit(x_train, y_train).predict(inputtest)
     print(predict)
     pred2.set(predict)


# Tk class is used to create a root window
root.geometry('800x500')
root.configure(background='#D8BFD8')
root.title('Disease Predictor')
root.resizable(0, 0)

id = IntVar()
age = IntVar()
# taking first input as gender
gender = StringVar()
gender.set("Select Here")

# taking first input as symptom
Symptom1 = StringVar()
Symptom1.set("Select Here")

# taking second input as symptom
Symptom2 = StringVar()
Symptom2.set("Select Here")

# taking third input as symptom
Symptom3 = StringVar()
Symptom3.set("Select Here")

# taking third input as symptom
family_history = StringVar()
family_history.set("Select Here")

# taking fourth input as symptom
current_medications = StringVar()
current_medications.set("Select Here")

# Headings for the GUI written at the top of GUI
w2 = Label(root, justify=CENTER, text=" Disease Predictor ", fg="black", bg="#DDA0DD")
w2.config(font=("Times", 20, "bold"))
w2.grid(row=1, column=1, columnspan=2, padx=10)

# Creating Labels for the Input
idLb = Label(root, text="Patient Id: ", fg="Black", bg="#DDA0DD")
idLb.config(font=("Times", 14))
idLb.grid(row=7, column=0, pady=10, sticky=W)

ageLb = Label(root, text="Age: ", fg="Black", bg="#DDA0DD")
ageLb.config(font=("Times", 14))
ageLb.grid(row=8, column=0, pady=10, sticky=W)

genderLb = Label(root, text="Gender: ", fg="Black", bg="#DDA0DD")
genderLb.config(font=("Times", 14))
genderLb.grid(row=9, column=0, pady=10, sticky=W)

S1Lb = Label(root, text="Symptom 1: ", fg="Black", bg="#DDA0DD")
S1Lb.config(font=("Times", 14))
S1Lb.grid(row=10, column=0, pady=10, sticky=W)

S2Lb = Label(root, text="Symptom 2: ", fg="Black", bg="#DDA0DD")
S2Lb.config(font=("Times", 14))
S2Lb.grid(row=11, column=0, pady=10, sticky=W)

S3Lb = Label(root, text="Symptom 3: ", fg="Black", bg="#DDA0DD")
S3Lb.config(font=("Times", 14))
S3Lb.grid(row=12, column=0, pady=10, sticky=W)


family_historyLb = Label(root, text="Family History: ", fg="Black", bg="#DDA0DD")
family_historyLb.config(font=("Times", 14))
family_historyLb.grid(row=13, column=0, pady=10, sticky=W)

current_medicationsLb = Label(root, text="Current Medications: ", fg="Black", bg="#DDA0DD")
current_medicationsLb.config(font=("Times", 14))
current_medicationsLb.grid(row=14, column=0, pady=10, sticky=W)

OPTION1 = sorted(symptom1)
OPTION2 = sorted(gender_)
OPTION3 = sorted(family_history_)
OPTION4 = sorted(current_medications_)
OPTION5 = sorted(symptom2)
OPTION6 = sorted(symptom3)


# Taking Symptoms as input from the dropdown from the user

#taking  patient id
id = IntVar()
id = Entry(root, textvariable=id)
id.grid(row=7, column=1)

# taking Age input
age=IntVar()
age = Entry(root, textvariable= age)
age.grid(row=8, column=1)

g = OptionMenu(root, gender, *OPTION2)
g.grid(row=9, column=1)

S1 = OptionMenu(root, Symptom1, *OPTION1)
S1.grid(row=10, column=1)

S2 = OptionMenu(root, Symptom2, *OPTION5)
S2.grid(row=11, column=1)

S3 = OptionMenu(root, Symptom3, *OPTION6)
S3.grid(row=12, column=1)

S5 = OptionMenu(root, family_history, *OPTION3)
S5.grid(row=13, column=1)

S5 = OptionMenu(root, current_medications, *OPTION4)
S5.grid(row=14, column=1)

# Buttons for predicting the disease using random forest
rnf = Button(root, text="Prediction ", command=randomforest, bg="#DDA0DD", fg="black")
rnf.config(font=("Times", 18, "bold"))
rnf.grid(row=17, column=1, padx=10)
t2 = Label(root, font=("Times", 15, "bold"), height=1, bg="#DDA0DD", width=40, fg="black", textvariable=pred1).grid(row=22, column=1, padx=10)

# calling this function because the application is ready to run
root.mainloop()