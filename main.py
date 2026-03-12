from flask import Flask, render_template, request, redirect, flash, url_for
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Import models and services
from models.whisper import WhisperWordClassifier
from services.vld import voice_liveness_detection
from services.add import audio_deepfake_detection
from services.dysarthria import dysarthria_classification
from services.emotions import emotion_classification
from services.infant import infant_cry_classification

app = Flask(__name__)
app.secret_key = 'supersecretkey' # Added a secret key for flash messages if needed

# Configurations
FILE_TYPES = set(["wav", "WAV"])
UPLOAD_FOLDER = os.path.join('static', 'assets')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize models once
print("Loading Whisper models...")
processor_base = WhisperProcessor.from_pretrained("openai/whisper-base")
model_base = WhisperWordClassifier.from_pretrained("openai/whisper-base")

processor_small = WhisperProcessor.from_pretrained("openai/whisper-small")
model_small = WhisperWordClassifier.from_pretrained("openai/whisper-small")
print("Models loaded.")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in FILE_TYPES

# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route("/vld", methods=["GET", "POST"])
def vld():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template('vld.html', abc='static/assets/R.gif', pqr="No file part")
        file = request.files["file"]
        if file.filename == "":
            return render_template('vld.html', abc='static/assets/blank.png', pqr="No selected file")
        
        if allowed_file(file.filename):
            # Save file temporarily or pass directly if services support it
            # For now, we follow existing service logic which might expect a file-like object or path
            img_path, label = voice_liveness_detection(file, app.config['UPLOAD_FOLDER'])
            return render_template('vld.html', abc=img_path, pqr=label)
        else:
            return render_template('vld.html', abc='static/assets/R.gif', pqr="Invalid file type. Please upload a .wav file")
    
    return render_template('vld.html', abc='static/assets/blank.png', pqr="")

@app.route("/add", methods=["GET", "POST"])
def add():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template('add.html', abc='static/assets/R.gif', pqr="No file part")
        file = request.files["file"]
        if file.filename == "":
            return render_template('add.html', abc='static/assets/blank.png', pqr="No selected file")
        
        if allowed_file(file.filename):
            img_path, label = audio_deepfake_detection(file, app.config['UPLOAD_FOLDER'], processor_base, model_base)
            return render_template('add.html', abc=img_path, pqr=label)
        else:
            return render_template('add.html', abc='static/assets/R.gif', pqr="Invalid file type. Please upload a .wav file")
            
    return render_template('add.html', abc='static/assets/blank.png', pqr="")

@app.route("/dys", methods=["GET", "POST"])
def dys():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template('dys.html', abc='static/assets/R.gif', pqr="No file part")
        file = request.files["file"]
        if file.filename == "":
            return render_template('dys.html', abc='static/assets/blank.png', pqr="No selected file")
        
        if allowed_file(file.filename):
            img_path, label = dysarthria_classification(file, processor_small, model_small)
            return render_template('dys.html', abc=img_path, pqr=label)
        else:
            return render_template('dys.html', abc='static/assets/R.gif', pqr="Invalid file type. Please upload a .wav file")
            
    return render_template('dys.html', abc='static/assets/blank.png', pqr="")

@app.route("/emotions", methods=["GET", "POST"])
def emotions():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template('emotions.html', abc='static/assets/R.gif', pqr="No file part")
        file = request.files["file"]
        if file.filename == "":
            return render_template('emotions.html', abc='static/assets/blank.png', pqr="No selected file")
        
        if allowed_file(file.filename):
            img_path, label = emotion_classification(file, app.config['UPLOAD_FOLDER'])
            return render_template('emotions.html', abc=img_path, pqr=label)
        else:
            return render_template('emotions.html', abc='static/assets/R.gif', pqr="Invalid file type. Please upload a .wav file")
            
    return render_template('emotions.html', abc='static/assets/blank.png', pqr="")

@app.route("/infant", methods=["GET", "POST"])
def infant():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template('infant.html', abc='static/assets/R.gif', pqr="No file part")
        file = request.files["file"]
        if file.filename == "":
            return render_template('infant.html', abc='static/assets/blank.png', pqr="No selected file")
        
        if allowed_file(file.filename):
            img_path, label = infant_cry_classification(file, app.config['UPLOAD_FOLDER'])
            return render_template('infant.html', abc=img_path, pqr=label)
        else:
            return render_template('infant.html', abc='static/assets/R.gif', pqr="Invalid file type. Please upload a .wav file")
            
    return render_template('infant.html', abc='static/assets/blank.png', pqr="")

@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        phone = request.form.get("phone")
        message = request.form.get("message")
        
        recipient = "speechlab006.7@gmail.com"
        sender_account = "speechlab006.7@gmail.com"
        app_password = "bxly rlnn xywa ovmn"
        
        # Professional HTML Email Template
        html_content = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 20px auto; border: 1px solid #e1e1e1; border-radius: 10px; overflow: hidden;">
                <div style="background: linear-gradient(135deg, #0e387a, #1a5fc2); color: white; padding: 20px; text-align: center;">
                    <h2 style="margin: 0;">New Website Inquiry</h2>
                </div>
                <div style="padding: 30px;">
                    <p>You have received a new contact request from <strong>{name}</strong>.</p>
                    <table style="width: 100%; border-collapse: collapse;">
                        <tr>
                            <td style="padding: 10px 0; border-bottom: 1px solid #eee; width: 100px;"><strong>Name:</strong></td>
                            <td style="padding: 10px 0; border-bottom: 1px solid #eee;">{name}</td>
                        </tr>
                        <tr>
                            <td style="padding: 10px 0; border-bottom: 1px solid #eee;"><strong>Email:</strong></td>
                            <td style="padding: 10px 0; border-bottom: 1px solid #eee;"><a href="mailto:{email}">{email}</a></td>
                        </tr>
                        <tr>
                            <td style="padding: 10px 0; border-bottom: 1px solid #eee;"><strong>Phone:</strong></td>
                            <td style="padding: 10px 0; border-bottom: 1px solid #eee;">{phone}</td>
                        </tr>
                    </table>
                    <div style="margin-top: 20px; padding: 15px; background: #f9f9f9; border-left: 4px solid #0e387a; border-radius: 4px;">
                        <p style="margin: 0; font-weight: bold; color: #0e387a;">Message:</p>
                        <p style="margin-top: 10px;">{message}</p>
                    </div>
                </div>
                <div style="background: #f4f4f4; padding: 15px; text-align: center; font-size: 12px; color: #777;">
                    Sent from Assistive Speech Technologies Website
                </div>
            </div>
        </body>
        </html>
        """
        try:
            msg = MIMEMultipart()
            msg['From'] = f"{name} via Website <{sender_account}>"
            msg['To'] = recipient
            msg['Subject'] = f"QUERY: {name} - AST Website"
            msg.add_header('reply-to', email)
            msg.attach(MIMEText(html_content, 'html'))
            
            server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
            server.login(sender_account, app_password)
            server.send_message(msg)
            server.quit()
            return render_template('contact.html', success=True)
        except Exception as e:
            print(f"SMTP Error: {e}")
            return render_template('contact.html', success=False)

    return render_template('contact.html')

@app.route("/elements")
def elements():
    return render_template('elements.html')

@app.route("/services")
def services():
    return render_template('services.html')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
