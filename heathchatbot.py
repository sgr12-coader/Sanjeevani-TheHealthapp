import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import csv
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


training = pd.read_csv('Training.csv')
testing = pd.read_csv('Testing.csv')
cols = training.columns
cols = cols[:-1]
x = training[cols]
y = training['prognosis']
y1 = y

reduced_data = training.groupby(training['prognosis']).max()

# map the strings to numbers
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
testx = testing[cols]
testy = testing['prognosis']
testy = le.transform(testy)

clf1 = DecisionTreeClassifier()
clf = clf1.fit(x_train, y_train)
scores = cross_val_score(clf, x_test, y_test, cv=3)

model = SVC()
model.fit(x_train, y_train)

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols

severityDictionary = dict()
description_list = dict()
precautionDictionary = dict()

symptoms_dict = {}

doctors = []
with open('doctor.csv', 'r') as f:
    csv_reader = csv.reader(f)
    header = next(csv_reader)
    for i in csv_reader:
        doctors.append({
            'name': i[0],
            'speciliest': i[1],
            'online': i[2],
            'avl_time': i[3],
            'fees': i[4],
            'email': i[5],
            'WhatsApp': i[6],
        })

for index, symptom in enumerate(x):
    symptoms_dict[symptom] = index


def calc_condition(exp, days):
    sum = 0
    for item in exp:
        sum = sum + severityDictionary[item]
    if (sum * days) / (len(exp) + 1) > 13:
        print("You should consult a doctor immediately. ")
    else:
        print("It might not have escalated yet, but consult a doctor.")


def getDescription():
    global description_list
    with open('symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            _description = {row[0]: row[1]}
            description_list.update(_description)


def getSeverityDict():
    global severityDictionary
    with open('Symptom_severity.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        try:
            for row in csv_reader:
                _diction = {row[0]: int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass


def getprecautionDict():
    global precautionDictionary
    with open('symptom_precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            _prec = {row[0]: [row[1], row[2], row[3], row[4]]}
            precautionDictionary.update(_prec)


def getInfo():
    print("Please enter your name ", end=":")
    name = input("")
    print("Hello ", name, ". How may I help you today?")


def check_pattern(dis_list, inp):
    import re
    pred_list = []
    ptr = 0
    regexp = re.compile(inp)
    for item in dis_list:
        if regexp.search(item):
            pred_list.append(item)
    if len(pred_list) > 0:
        return 1, pred_list
    else:
        return ptr, item


def sec_predict(symptoms_exp):
    df = pd.read_csv('Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    symptoms_dict = {}

    for index, symptom in enumerate(X):
        symptoms_dict[symptom] = index

    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        input_vector[[symptoms_dict[item]]] = 1

    return rf_clf.predict([input_vector])


def print_disease(node):
    node = node[0]
    val = node.nonzero()
    disease = le.inverse_transform(val[0])
    return disease


def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis = ",".join(feature_names).split(",")
    symptoms_present = []

    while True:

        print("Please enter the symptom you are experiencing ", end=":")
        disease_input = input("")
        conf, cnf_dis = check_pattern(chk_dis, disease_input)
        if conf == 1:
            print("Searches related to given input: ")
            for num, it in enumerate(cnf_dis):
                print(num, ")", it)
            if num != 0:
                print(f"Select the one you meant (0 - {num}):  ", end="")
                conf_inp = int(input(""))
            else:
                conf_inp = 0

            disease_input = cnf_dis[conf_inp]
            break
        else:
            print("Enter valid symptom.")

    while True:
        try:
            num_days = int(input("The number of days you have been experiencing these symptoms for : "))
            break
        except:
            print("Enter number of days.")

    def recurse(node, depth):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name == disease_input:
                val = 1
            else:
                val = 0
            if val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])
            red_cols = reduced_data.columns
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
            print("Are you experiencing any ")
            symptoms_exp = []
            for syms in list(symptoms_given):
                print(syms, "? ", end='')
                while True:
                    inp = input("")
                    if inp == "yes" or inp == "no":
                        break
                    else:
                        print("Please provide proper answers i.e. (yes/no) : ", end="")
                if inp == "yes":
                    symptoms_exp.append(syms)

            second_prediction = sec_predict(symptoms_exp)
            calc_condition(symptoms_exp, num_days)
            if present_disease[0] == second_prediction[0]:
                print("You may have", present_disease[0])
                print(description_list[present_disease[0]])

            else:
                print("You may have", present_disease[0], "or ", second_prediction[0])
                print(description_list[present_disease[0]])
                print(description_list[second_prediction[0]])

            precaution_list = precautionDictionary[present_disease[0]]
            print("Take following measures : ")
            for i, j in enumerate(precaution_list):
                print(i + 1, ")", j)

    recurse(0, 1)


class Chat:
    def __init__(self) -> None:
        self.tree_ = clf.tree_
        self.feature_name = [
            cols[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in self.tree_.feature
        ]
        self.chk_dis = ",".join(cols).split(",")
        self.symptoms_present = []

    def step1(self, disease_input):  # disease input
        conf, cnf_dis = check_pattern(self.chk_dis, disease_input)
        if conf == 1:
            resp = '''Searches related to given input:\n'''
            for num, it in enumerate(cnf_dis):
                resp += str(num) + " ) " + it + "\n"
            if num != 0:
                resp += f'Select the one you meant (0 - {num})'
                self.cnf_dis = cnf_dis
                return {'msg': resp, 'next_step': 2}
            else:
                conf_inp = 0
                self.disease_input = cnf_dis[conf_inp]
                return {'msg': resp, 'next_step': 3}
        else:
            return {'msg': "Enter valid symptom.", 'next_step': 1}

    def step2(self, conf_inp):
        try:
            conf_inp = int(conf_inp)
            self.disease_input = self.cnf_dis[conf_inp]
            return {'msg': 'The number of days you have been experiencing these symptoms for', 'next_step': 3}
        except ValueError as e:
            return {'msg': 'Please choose a valid option', 'next_step': 2}

    def step3(self, days):
        try:
            self.num_days = int(days)
            first_symptom = self.recurse(0, 1)
            first_symptom = self.symptoms_given[self.symptoms_count]
            return {'msg': "Are you experiencing any\n" + first_symptom + " ?", 'next_step': 4}
        except ValueError as e:
            return {'msg': 'Please choose a valid option', 'next_step': 3}

    def step4(self, symptom):
        if symptom != "yes" and symptom != "no":
            return {'msg': 'Please provide proper answers i.e. (yes/no)', 'next_step': 4}
        if symptom == "yes":
            self.symptoms_exp.append(self.symptoms_given[self.symptoms_count])
        self.symptoms_count += 1
        if self.symptoms_count == len(self.symptoms_given):
            return self.step5()
        return {'msg': self.symptoms_given[self.symptoms_count] + " ?", 'next_step': 4}

    def step5(self):
        second_prediction = sec_predict(self.symptoms_exp)
        calc_condition(self.symptoms_exp, self.num_days)
        if self.present_disease[0] == second_prediction[0]:
            resp = f"You may have '{self.present_disease[0]}'\n\n{description_list[self.present_disease[0]]}"
        else:
            resp = f"You may have '{self.present_disease[0]}' or '{second_prediction[0]}'\n\n{description_list[self.present_disease[0]]}\n{description_list[second_prediction[0]]}"

        precaution_list = precautionDictionary[self.present_disease[0]]
        resp += "\n\nTake following measures :"
        for i, j in enumerate(precaution_list):
            resp += f"\n{i + 1}) {j}"
        return {'msg': resp, 'next_step': 6}

    def step6(self, re):
        return {'msg': 'You may leave now', 'next_step': 6}

    def recurse(self, node, depth):
        if self.tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = self.feature_name[node]
            threshold = self.tree_.threshold[node]

            if name == self.disease_input:
                val = 1
            else:
                val = 0
            if val <= threshold:
                self.recurse(self.tree_.children_left[node], depth + 1)
            else:
                self.symptoms_present.append(name)
                self.recurse(self.tree_.children_right[node], depth + 1)
        else:
            self.present_disease = print_disease(self.tree_.value[node])
            red_cols = reduced_data.columns
            self.symptoms_given = list(red_cols[reduced_data.loc[self.present_disease].values[0].nonzero()])
            self.symptoms_exp = []
            self.symptoms_count = 0
            return self.symptoms_given[self.symptoms_count]


getSeverityDict()
getDescription()
getprecautionDict()

if __name__ == "__main__":
    getInfo()
    tree_to_code(clf, cols)
