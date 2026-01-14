# bot.py
import pyttsx3
import uuid
from gtts import gTTS
import tempfile
import base64
import speech_recognition as sr
import time
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import os

engine = pyttsx3.init()
engine.setProperty('rate',150)

def text_to_speech(text):
    filename = f"response_{uuid.uuid4().hex}.mp3"
    tts = gTTS(text=text, lang="en")
    tts.save(filename)
    return filename

def speak(text):
    try:
        engine.stop()
        engine.say(text)
        engine.runAndWait()
    except:
        pass

knowledge = {

# BASIC
"breast cancer":"Breast cancer is a disease in which abnormal cells grow uncontrollably in the breast tissue and form tumors.",
"breast":"Breast produces milk for feeding babies and contains glands, ducts and fatty tissue.",
"how cancer starts":"Cancer starts when normal cells mutate and grow without control.",
"tumor":"A tumor is an abnormal mass of tissue. It can be benign or malignant.",
"benign":"Benign tumors are non-cancerous and do not spread.",
"malignant":"Malignant tumors are cancerous and can spread to other parts of the body.",
"lump":"Not every breast lump is cancer. Many lumps are cysts or infections.",
"milk ducts":"Milk ducts carry milk from lobules to nipple.",
"lobules":"Lobules are glands that produce milk.",
"men breast cancer":"Men can also get breast cancer, but it is rare.",
"how common":"Breast cancer is the most common cancer in women worldwide.",
"age risk":"It mostly affects women above 40 years of age.",
"cancer cell":"Cancer cells grow rapidly and destroy healthy cells.",

# SPREAD
"spread":"Cancer spreads through blood and lymph system.",
"cancer spread":"Cancer spreads through blood and lymph system.",
"metastasis":"Metastasis means cancer spreading to other organs.",

# STAGES
"early stage":"Early stage means cancer is small and has not spread.",
"late stage":"Late stage means cancer has spread to other organs.",
"survival rate":"Survival rate shows how many patients live after diagnosis.",

# TESTS
"biopsy":"Biopsy is a test where tissue is removed to confirm cancer.",
"mammogram":"Mammogram is an X-ray scan to detect breast cancer early.",
"ultrasound":"Ultrasound uses sound waves to detect lumps.",
"mri":"MRI uses magnetic waves to detect tumors.",
"screening":"Screening means checking before symptoms appear.",

# DETECTION
"early detection":"Early detection greatly increases cure chances.",

# TREATMENT
"treatment":"Breast cancer is treated using surgery, chemotherapy, radiation and targeted medicines.",
"chemotherapy":"Chemotherapy uses medicines to kill cancer cells.",
"radiation":"Radiation therapy uses high energy rays to destroy cancer cells.",
"radiation therapy":"Radiation therapy uses high energy rays to destroy cancer cells.",
"surgery":"Surgery removes cancer tissue.",
"cure":"Early stage breast cancer is highly curable.",
"untreated":"Untreated cancer spreads and becomes life-threatening.",
"remission":"Remission means no signs of cancer.",
"recurrence":"Recurrence means cancer coming back.",
"relapse":"Relapse means cancer returning after treatment.",

# GENETIC / RISK
"genetic mutation":"Genetic mutation means change in genes that cause cancer.",
"brca":"BRCA genes increase breast cancer risk.",
"risk factor":"Risk factors include smoking, alcohol, obesity and genetics.",

# PREVENTION
"prevention":"Healthy lifestyle reduces cancer risk.",
"food":"Eat green vegetables, fruits, turmeric, garlic, almonds and avoid junk food.",
"exercise":"Daily walking and yoga reduce cancer risk.",
"self exam":"Self breast exam is checking your own breast for lumps.",
"clinical exam":"Doctor checks breast physically.",

# SYMPTOMS
"inflammation":"Inflammation is swelling and redness.",
"pain":"Pain is not always cancer.",
"nipple discharge":"Fluid coming from nipple.",
"breast swelling":"Increase in breast size due to infection or cancer.",
"skin dimpling":"Skin pulling inward is a sign of cancer.",
"asymmetry":"Unequal breast size.",
"cyst":"Fluid-filled lump.",
"fibrosis":"Hard thick breast tissue.",

# REPORT & AI
"tumor marker":"Blood test marker showing cancer.",
"medical report":"Doctor test result document.",
"scan result":"Scan result shows tumor location and size.",
"risk score":"Risk score depends on age, genetics, obesity and family history.",
"what should patient do next":"Consult oncologist immediately and follow treatment.",
"ai detection":"AI detects cancer faster and more accurately."
}

hospitals = {
 "delhi":"AIIMS Delhi",
 "mumbai":"Tata Memorial Mumbai",
 "chennai":"Apollo Chennai",
 "trivandrum":"RCC Trivandrum"
}

# ---------- PDF ----------
def create_pdf(stage):
    file_path = "medical_report.pdf"
    c = canvas.Canvas(file_path, pagesize=A4)
    c.drawString(100,800,"AI MEDICAL REPORT")
    c.drawString(100,770,f"Detected Stage : {stage}")
    c.drawString(100,740,"Follow doctor advice & healthy diet.")
    c.save()
    return file_path

# ---------- CHATBOT ----------
def chatbot(query, stage):
    raw = query.lower().replace(" ","")

    if "stage1" in raw:
        return "Stage 1 is early cancer. Cure rate above 95%."
    if "stage2" in raw:
        return "Stage 2 requires surgery and radiation."
    if "stage3" in raw:
        return "Stage 3 needs chemotherapy, surgery and radiation."
    if "stage4" in raw:
        return "Stage 4 is advanced cancer."

    for k in knowledge:
        if k in raw:
            return knowledge[k]

    if "hospitalin" in raw:
        city = raw.split("hospitalin")[-1]
        return hospitals.get(city,"City not found")

    if "generatereport" in raw:
        return create_pdf(stage)

    return "Please ask a breast cancer related question."

def cleanup_audio():
    for file in os.listdir():
        if file.startswith("response_") and file.endswith(".mp3"):
            os.remove(file)