# Python 3.8.12
import re

import pytesseract
import  cv2
from PIL import Image
from flask import Flask, render_template, request
from heathchatbot import Chat, doctors, header
from ocr import read_report
from flask_mail import Mail, Message


chat = Chat()

app = Flask(__name__)
mail = Mail(app)
email_co = 'sanjivani.healthapp@gmail.com'  # ENTER YOUR EMAIL
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = email_co
app.config['MAIL_PASSWORD'] = 'sskwionnxbmimshl'   #'Sanjivani@healthapp'     # ENTER YOUR PASSWORD
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
mail = Mail(app)


@app.route("/")
def index():
    return render_template("index.html", resp={'msg': "Please enter the symptom you are experiencing", 'next_step': 1})


@app.route("/message")
def get_bot_response():
    userText = request.args.get('msg')
    next = request.args.get('next_step')
    call = f"chat.step{next}('{userText}')"
    resp = eval(call)
    return resp


@app.route("/#BookAppointment")
def appoiment():
    return render_template("BookAppoiment.html", var={'doctors': doctors, 'header': header})


@app.route("/Appointment/", methods=['GET', 'POST'])
def send_appoiment():
    email_doctor = request.args.get('email', None)
    import pandas as pd
    import csv
    name_doctor = pd.read_csv("doctor.csv")
    data = name_doctor[name_doctor["email"] == email_doctor]
    time = data["avl_time"]
    data=str(data["name"])

    # print(data)
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email_customer')
        doctor = request.args.get('email', None)
        gender = request.form.get('gender')
        weight = request.form.get('weight')
        height = request.form.get('height')
        medication = request.form.get('medication')
        whatapp = request.form.get('whatapp')
        context = f'''Hi doctor
The patient {username} with the characteristics of height: {height}, weight: {weight}, medical history: {medication}, gender: {gender},
    join https://meet.google.com/vvw-iqfq-etw at {time}.
 For any changes in timing connect to the patient by email ({email}) or WhatsApp ({whatapp}) 
Thanks
Team sanjeevani'''
        msg = Message('get Appointment', sender=email_co, recipients=[doctor])
        msg.body = context
        mail.send(msg)
        context_1 = f'''Hi {username} Your appoiment is booked with Dr.       '''
        context_1=context_1 + f'''{data} join https://meet.google.com/vvw-iqfq-etw at {time}
                                Thanks
                                Team sanjeevani'''
        msg_1 = Message('Appointment Booked', sender=email_co, recipients=[email])
        msg_1.body = context_1
        mail.send(msg_1)
        return render_template("successfull.html")

    return render_template("send_appoiment.html", var = email_doctor)


@app.route("/connect")
def connect():
    return render_template("connect.html")


@app.route("/report")
def report():
    return render_template("report.html")


@app.route('/upload/', methods=['GET', 'POST'])
def upload():
    try:
        imagefile = request.files.get('imagefile', '')
        email = request.form.get('email', '')
        print("email-----",email)
        print("imagefile-----",imagefile)
        img = Image.open(imagefile)
        print("img------>",img)

        file_name = email.split('@')[0]
        print("file_name--------------", file_name)

        text = pytesseract.image_to_string(img, timeout=10, lang="eng")
        print("text--------------", text)
        f = open(f"reports//{file_name}.txt", "w")
        f.truncate(0)
        f.write(text)
        f.close()

        responses_report = read_report(file_name)
        print("responses_report------",responses_report)
        if len(responses_report) > 0:
            return render_template('result.html', var=responses_report)
        return render_template('result_no.html', var=text)
    except:
        return render_template('errors.html')


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)
