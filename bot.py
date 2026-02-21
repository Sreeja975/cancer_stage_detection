import pyttsx3
import os

engine = pyttsx3.init()
engine.setProperty('rate',150)


from gtts import gTTS
import tempfile

def text_to_speech(text):
    tts = gTTS(text)
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    filename = tmp_file.name
    tmp_file.close()

    tts.save(filename)
    return filename

def cleanup_audio():
    pass



# -------------------- VOICE ENGINE --------------------
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    engine.stop()
    engine.say(text)
    engine.runAndWait()


# -------------------- KNOWLEDGE BASE --------------------
knowledge = {

# BASIC¸
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



# -------------------- HOSPITAL DATA --------------------
hospitals = {
    "delhi": "AIIMS Delhi",
    "mumbai": "Tata Memorial Mumbai",
    "chennai": "Apollo Chennai",
    "trivandrum": "RCC Trivandrum"
}

def chatbot(query, stage=None):
    raw = query.lower()

    # Agar stage AI se detect hua hai
    if stage and stage != "Stage Unknown":
        raw += f" {stage.lower()}"


    # ---- HOSPITAL QUESTIONS FIRST ----
    if "hospital" in raw:
        for city in hospitals:
            if city in raw:
                return f"Best cancer hospital in {city.title()} is {hospitals[city]}"
        return "Please mention a city like Delhi, Mumbai, Chennai or Trivandrum."

    # ---- STAGE QUESTIONS ----
    if "stage 1" in raw or "stage1" in raw:
        return (
    "STAGE 1 BREAST CANCER (EARLY & HIGHLY CURABLE)\n\n"
    "WHAT IS HAPPENING:\n"
    "• Tumor is very small and limited to the breast\n"
    "• Cancer has NOT spread to lymph nodes or organs\n"
    "• Often detected through screening or self-exam\n\n"
    "TREATMENT REALITY:\n"
    "• Surgery is the main treatment (lumpectomy or mastectomy)\n"
    "• Radiation may be required\n"
    "• Chemotherapy usually NOT required\n\n"
    "SURVIVAL & TRUTH:\n"
    "• Survival rate above 95%\n"
    "• Most patients live a normal life after treatment\n\n"
    "PRECAUTIONS:\n"
    "• Never delay surgery\n"
    "• Regular follow-up every 6 months\n"
    "• Avoid smoking, alcohol, and stress\n\n"
    "FOOD & LIFESTYLE:\n"
    "• Eat green vegetables, fruits, turmeric, garlic\n"
    "• Avoid junk food, sugar, red meat\n"
    "• Daily walking, yoga, meditation\n\n"
    "BEST HOSPITALS:\n"
    "• AIIMS Delhi\n"
    "• Tata Memorial Mumbai\n"
    "• Apollo Chennai"
)

    if "stage 2" in raw or "stage2" in raw:
        return (
            "STAGE 2 BREAST CANCER (LOCALLY ADVANCED)\n\n"
            "WHAT IS HAPPENING:\n"
            "• Tumor is larger or cancer has spread to nearby lymph nodes\n"
            "• Still confined to breast region\n\n"
            "TREATMENT REALITY:\n"
        "• Surgery is mandatory\n"
        "• Chemotherapy is commonly required\n"
        "• Radiation therapy after surgery\n\n"
        "SURVIVAL & TRUTH:\n"
        "• High survival rate with early treatment\n"
        "• Delay can push cancer to stage 3\n\n"
        "PRECAUTIONS:\n"
        "• Follow full chemo cycle, do NOT stop midway\n"
        "• Maintain immunity and nutrition\n"
        "• Mental health support is important\n\n"
        "FOOD & LIFESTYLE:\n"
        "• High-protein diet (dal, eggs, nuts)\n"
        "• Fresh fruits, beetroot, pomegranate\n"
        "• Avoid outside food and infections\n\n"
        "BEST HOSPITALS:\n"
        "• Tata Memorial Mumbai\n"
        "• AIIMS Delhi\n"
        "• Regional Cancer Centres (RCC)"
    )

    if "stage 3" in raw or "stage3" in raw:
        return (
        "STAGE 3 BREAST CANCER (SERIOUS BUT TREATABLE)\n\n"
        "WHAT IS HAPPENING:\n"
        "• Cancer has spread to multiple lymph nodes\n"
        "• May involve chest wall or skin\n\n"
        "TREATMENT REALITY:\n"
        "• Chemotherapy FIRST (before surgery)\n"
        "• Major surgery after tumor shrinkage\n"
        "• Radiation therapy mandatory\n\n"
        "SURVIVAL & TRUTH:\n"
        "• Cure is possible, but treatment is long and exhausting\n"
        "• Discipline decides survival\n\n"
        "PRECAUTIONS:\n"
        "• Never skip chemotherapy sessions\n"
        "• Prevent infections aggressively\n"
        "• Emotional support is critical\n\n"
        "FOOD & LIFESTYLE:\n"
        "• Liquid foods during chemo if needed\n"
        "• Protein shakes, soups, fruits\n"
        "• Avoid raw food during low immunity\n\n"
        "BEST HOSPITALS:\n"
        "• Tata Memorial Mumbai\n"
        "• AIIMS Delhi\n"
        "• Apollo Cancer Centres"
    )
    if "stage 4" in raw or "stage4" in raw:
        return (
            "STAGE 4 BREAST CANCER (METASTATIC & LIFE-CHANGING)\n\n"
        "WHAT IS HAPPENING:\n"
        "• Cancer has spread to organs like bone, liver, lung or brain\n"
        "• Disease is systemic, not localized\n\n"
        "TREATMENT REALITY:\n"
        "• Cancer is NOT fully curable\n"
        "• Treatment focuses on control, not cure\n"
        "• Long-term medicines, chemo, targeted therapy\n\n"
        "SURVIVAL & TRUTH:\n"
        "• Patients can live years with proper treatment\n"
        "• Quality of life becomes the priority\n\n"
        "PRECAUTIONS:\n"
        "• Never stop treatment without doctor advice\n"
        "• Pain management is essential\n"
        "• Mental & family support is crucial\n\n"
        "FOOD & LIFESTYLE:\n"
        "• Easy-to-digest foods\n"
        "• Avoid sugar spikes and dehydration\n"
        "• Focus on comfort, nutrition, and peace\n\n"
        "BEST HOSPITALS:\n"
        "• Tata Memorial Mumbai\n"
        "• AIIMS Delhi\n"
        "• Advanced Oncology Centres"
    )
    # ---------- KNOWLEDGE SEARCH ----------
    for key in knowledge:
        if key in raw:
            return knowledge[key]

    return "Please ask a breast cancer related question."