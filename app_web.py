# 1.1. PYTHON LIBRARIES
#######################
import streamlit as st
import mediapipe as mp
import cv2
import pandas as pd
import numpy as np
import time
import base64
from random import randrange
import pickle
import streamlit_authenticator as stauth  # pip install streamlit-authenticator
import threading
from gtts import gTTS
from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1' #Oculta el texto --> Hello from the pygame community. https://www.pygame.org/contribute.html
import pygame
import io
import warnings
warnings.filterwarnings('ignore')

#1.2. OWN LIBRARIES
###################
import Libraries.database as db
import Libraries.home as home
import Libraries.UpcSystemCost as UpcSystemCost
import Libraries.UpcSystemAngles as UpcSystemAngles
import Libraries.dashboard as dashboard
import Libraries.utilitarios as util
import Libraries.reportes as reportes
import Libraries.ML_Tools as mlt


#1.3. SESSION STATE VARIABLES
st.session_state.count_pose_g   = 0
st.session_state.count_pose     = 0
st.session_state.count_rep      = 0
st.session_state.count_set      = 0

st.session_state.starting_secs = 5
st.session_state.landmark_visible = 0.2
st.session_state.prob_change_color = 33
st.session_state.prob_avance_pose = 40
st.session_state.prob_avance_pose = False

# 2. FUNCTIONS
##############
@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def load_exercise_metadata(id_exercise):
    df = pd.read_csv('02. trainers/exercises_metadata.csv', sep = '|', encoding='latin-1')

    st.session_state.short_name          = df.loc[df['id_exercise']==id_exercise, 'short_name'].values[0]
    st.session_state.vista               = df.loc[df['id_exercise']==id_exercise, 'vista'].values[0]
    st.session_state.articulaciones      = df.loc[df['id_exercise']==id_exercise, 'articulaciones'].values[0]
    st.session_state.posfijo             = df.loc[df['id_exercise']==id_exercise, 'posfijo'].values[0]
    st.session_state.n_poses             = df.loc[df['id_exercise']==id_exercise, 'n_poses'].values[0]
    st.session_state.n_sets_default      = int(df.loc[df['id_exercise']==id_exercise, 'n_sets_default'].values[0])
    st.session_state.n_reps_default      = int(df.loc[df['id_exercise']==id_exercise, 'n_reps_default'].values[0])
    st.session_state.n_rest_time_default = int(df.loc[df['id_exercise']==id_exercise, 'n_rest_time_default'].values[0])
    st.session_state.detail              = df.loc[df['id_exercise']==id_exercise, 'detail'].values[0]

#functions to use text to speech
def speak(text):
    substrs_to_compare = ['Felicitaciones']
    with io.BytesIO() as file:
        tts = gTTS(text=text, lang='es')
        tts.write_to_fp(file)
        file.seek(0)
        pygame.mixer.music.load(file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            continue
    if not any(text.startswith(substring) for substring in substrs_to_compare):
        time.sleep(3)
    elif text.startswith("Mantenga"):
        time.sleep(5)

def get_exercise_gif(id_exercise):
    gif_file = "02. trainers/" + id_exercise + "/images/" + id_exercise + ".gif"
    return gif_file

def update_trainer_image(id_exercise, pose):
    if pose != 0:
        placeholder_trainer.image("./02. trainers/{}/images/{}{}.png".format(id_exercise, id_exercise, pose))

def print_sidebar_main(id_exercise):
    load_exercise_metadata(id_exercise)
    #SIDEBAR START
    st.sidebar.markdown('---')
    st.sidebar.markdown(f'''**{st.session_state.short_name}**''', unsafe_allow_html=True)
    st.sidebar.image(get_exercise_gif(id_exercise))  
    vista_gif = '01. webapp_img/vista_' + st.session_state.vista + '.gif'
    with st.sidebar.expander("💡 Info"):
        st.info(st.session_state.detail)
    st.sidebar.markdown("<hr/>", unsafe_allow_html=True)
    st.session_state.n_sets = st.sidebar.number_input("Sets", min_value=1, max_value=10, value=st.session_state.n_sets_default)
    st.session_state.n_reps = st.sidebar.number_input("Reps", min_value=1, max_value=10, value=st.session_state.n_reps_default)
    st.session_state.seconds_rest_time = st.sidebar.number_input("Rest Time (seconds)", min_value=1, max_value=30, value=st.session_state.n_rest_time_default)
    position_image, position_text = st.sidebar.columns(2)
    with position_image:
        st.image(vista_gif, width=100)
    with position_text:
        st.markdown("<br/><br/>", unsafe_allow_html=True)
        st.text("Vista: {}".format(st.session_state.vista))
        st.text("N° poses: {}".format(st.session_state.n_poses))
    st.sidebar.markdown("<hr/>", unsafe_allow_html=True)
    
    #####EXTRA OPTIONS
    st.session_state.starting_secs = st.sidebar.slider('⏱️ SECONDS to start:', min_value=1, max_value=10,
                                                             value = 5)
    st.sidebar.markdown("<hr/>", unsafe_allow_html=True)
    st.session_state.landmark_visible = st.sidebar.slider('👁️‍🗨️ PERCENT min allowed for Landmarks Visibility:', min_value=0.0, max_value=1.0,
                                                                value = 0.2)
    st.sidebar.markdown("<hr/>", unsafe_allow_html=True)
    st.session_state.prob_change_color = st.sidebar.slider('🟣 PROBABILITY CLASS to change red-blue:', min_value=0, max_value=100,
                                                                value = 33)
    st.sidebar.markdown("<hr/>", unsafe_allow_html=True)
    st.session_state.prob_avance_pose = st.sidebar.slider('💯 PROBABILITY CLASS to advance next pose:', min_value=0, max_value=100,
                                                                value = 20)
    st.sidebar.markdown("<hr/>", unsafe_allow_html=True)
    st.session_state.developer_mode = st.sidebar.checkbox('Developer mode: Trainer costs [0-99]', value=False)
    st.sidebar.markdown("<hr/>", unsafe_allow_html=True)
    #####EXTRA OPTIONS

    placeholder_title.title('STARTER TRAINING - 🏋🏼‍♂️'+ st.session_state.short_name)
    st.markdown('---')
        
def get_trainer_coords(id_exercise, id_trainer):
    df = pd.read_csv("02. trainers/{}/costs/{}_puntos_trainer{}.csv".format(id_exercise, id_exercise, id_trainer))
    if id_exercise == "bird_dog":
        df = df.iloc[: , :-6]
    else:
        df = df.iloc[: , :-3]
    del df['pose']
    return df

def get_trainers_angles(id_exercise):
    df = pd.read_csv("02. trainers/{}/costs/angulos_{}_promedio.csv".format(id_exercise, id_exercise))
    return df

def LoadModel():
    model_weights = './04. model_weights/weights_body_language.pkl'
    with open(model_weights, "rb") as f:
        model = pickle.load(f)
    return model

def get_timestamp_log():
    now = time.time()
    mlsec = repr(now).split('.')[1][:3]
    timestamp1 = time.strftime("%Y-%m-%d %H:%M:%S.{}".format(mlsec))
    return timestamp1

def get_timestamp_txt(id_user,id_exer):
    timestamp2 = time.strftime("%Y%m%d_%H%M%S"+"_"+id_user+"_"+id_exer)
    return timestamp2

def update_counter_panel(season, count_pose_g, count_pose, count_rep, count_set):    
    if season == "training":
        placeholder_status.markdown(util.font_size_px("🏎️ TRAINING...", 26), unsafe_allow_html=True)
        placeholder_set.metric(  "🟡SET DONE",               str(count_set)    + " / " + str(st.session_state.n_sets),      "+1 rep")
        placeholder_rep.metric(  "🟡REPETITION DONE",        str(count_rep)    + " / " + str(st.session_state.n_reps),      "+1 rep")
        placeholder_pose.metric(       "🟡POSE DONE",        str(count_pose)   + " / " + str(st.session_state.n_poses),     "+1 pose")
        placeholder_pose_global.metric("🟡GLOBAL POSE DONE", str(count_pose_g) + " / " + str(st.session_state.total_poses), "+1 pose")
        
    elif season == "finished":
        placeholder_status.markdown(util.font_size_px("🥇 EXERCISE FINISHED!", 26), unsafe_allow_html=True)        
        placeholder_set.metric(        "🟢SET DONE",         str(count_set)    + " / " + str(st.session_state.n_sets), "COMPLETED" )
        placeholder_rep.metric(        "🟢REPETITION DONE",  str(count_rep)    + " / " + str(st.session_state.n_reps), "COMPLETED")
        placeholder_pose.metric(       "🟢POSE DONE",        str(count_pose)   + " / " + str(st.session_state.n_poses),"COMPLETED")
        placeholder_pose_global.metric("🟢GLOBAL POSE DONE", str(count_pose_g) + " / " + str(st.session_state.total_poses), "COMPLETED")
    else:
        placeholder_status.markdown(util.font_size_px("⚠️", 26), unsafe_allow_html=True)

# 3. HTML CODE
#############
st.set_page_config(
    page_title="STARTER TRAINING - UPC",
    page_icon ="01. webapp_img/upc_logo.png",
)

img_upc = get_base64_of_bin_file('01. webapp_img/upc_logo_50x50.png')
fontProgress = get_base64_of_bin_file('01. webapp_fonts/ProgressPersonalUse-EaJdz.ttf')

def web_app_background(color_1, color_2, color_3, deg):
    deg = str(deg)
    st.markdown(
        f"""
        <style>        
        .main {{
            background: linear-gradient({deg}deg,{color_1},{color_2},{color_3});
            background-size: 180% 180%;
            animation: gradient-animation 3s ease infinite;
            }}

            @keyframes gradient-animation {{
                0% {{ background-position: 0% 50%; }}
                50% {{ background-position: 100% 50%; }}
                100% {{ background-position: 0% 50%; }}
            }}        
        </style>
        """ ,
        unsafe_allow_html=True,
    )

st.markdown(
    f"""
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {{
        width: 336px;        
    }}
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {{
        width: 336px;
        margin-left: -336px;
        background-color: #00F;
    }}
    [data-testid="stVerticalBlock"] {{
        flex: 0;
        gap: 0;
    }}
    #starter-training{{
        padding: 0;
    }}
    #div-upc{{
        #border: 1px solid #DDDDDD;
        background-image: url("data:image/png;base64,{img_upc}");
        position: fixed !important;
        right: 14px;
        top: 60px;
        width: 50px;
        height: 50px;
        background-size: 50px 50px;
    }}
    @font-face {{
        font-family: ProgressFont;
        src: url("data:image/png;base64,{fontProgress}");
    }}
    .css-10trblm{{
        color: #FFF;
        font-size: 40px;
        font-family: ProgressFont;    
    }}
    .block-container{{
        max-width: 100%;
    }}
    .css-17qbjix {{
        font-size: 16px;
    }}
    .css-12oz5g7 {{
        padding-top: 3rem;
    }}
    .stButton{{
        text-align: center !important;
    }}
    footer{{
        visibility:visible;
        display:block;
        position:relative;
        color:white;
        padding:5px;
        top:3px;
        width:900px;
    }}
    footer:after{{
        content:' Copyright©️ 2022-2023   -   ♨️UPC-Maestría Data Science | 👨‍✈️Manuel Alcántara | 🤵‍♂️Renzo Bances | 👨‍🎓Cristian Cabrera | 🥷Leibnihtz Ayamamani';
        display:block;
        position:relative;
        color:white;
        padding:5px;
        top:3px;
        width:900px;
    }}
    </style>
    """ ,
    unsafe_allow_html=True,
)

# 4. PYTHON CODE
############# MANTENER EN PRIMERA FILA ⬇️ ############# 
web_app_background('#092de7', '#1a89de', '#a8e73d', 135)
placeholder_title = st.empty()
placeholder_title.title('STARTER TRAINING')
st.markdown("<div id='div-upc'></span>", unsafe_allow_html=True)
############# MANTENER EN PRIMERA FILA ⬆️ #############


pygame.mixer.init()
############# USER AUTHENTICATION ⬇️ #############
users = db.fetch_all_users()
usernames = [user["key"] for user in users]
names = [user["name"] for user in users]
hashed_passwords = [user["password"] for user in users]
authenticator = stauth.Authenticate(names, usernames, hashed_passwords,"app_training", "abcdef", cookie_expiry_days=30)
name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status == False:
    st.error("Username/password is incorrect")

############# USER AUTHENTICATION ⬆️ #############

if authentication_status:
    
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    if 'camera' not in st.session_state:
        st.session_state['camera'] = 0

    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh

    # ---- SIDEBAR ----
    authenticator.logout("Logout", "sidebar")
    st.sidebar.title('Welcome {}'.format(name))
    id_trainer = randrange(3) + 1

    app_exercise = st.sidebar.selectbox('Select your option:',
        ['🏠 HOME','🏋🏼‍♂️ WORKOUT ROUTINE', '📊 REPORTS', '⚙️ MACHINE LEARNING TOOLS']
    )

    if app_exercise == "🏋🏼‍♂️ WORKOUT ROUTINE":
        web_app_background('#201f22', '#343c4f', '#7ba6f5', 135)
        st.sidebar.markdown("<br>", unsafe_allow_html=True)
        app_mode = st.sidebar.selectbox('Select your workout routine:',
            ['Push Up', 'Curl Up', 'Front Plank', 'Forward Lunge', 'Bird Dog']
        )
    if app_exercise =='🏠 HOME':
        web_app_background('#9b21e9', '#c78db2', '#b19f29', 135)
        df = db.get_user(username)
        st.session_state.nombre = df['name']
        st.session_state.edad = df['edad']
        st.session_state.peso = df['peso']
        st.session_state.talla = df['talla']
        st.session_state.imc = df['imc']
        st.session_state.perabdominal = df['perabdominal']
        st.session_state.cexweek = df['perabdominal']
        st.session_state.genero = df['genero']

        home.load_home_sidebar(st.session_state.edad, st.session_state.peso, st.session_state.talla, st.session_state.imc, st.session_state.perabdominal, st.session_state.genero)
        home.load_home(st.session_state.edad, st.session_state.peso, st.session_state.talla, st.session_state.imc, st.session_state.perabdominal)

    elif app_exercise =='⚙️ MACHINE LEARNING TOOLS':
        web_app_background('#d32f5d', '#88539e', '#3c76df', 135)
        placeholder_title.title('STARTER TRAINING - ⚙️ MACHINE LEARNING TOOLS')
        mlt.load_ml_tools()
    elif app_exercise =='📊 REPORTS':
        web_app_background('#3c76df', '#88539e', '#d32f5d', 135)
        placeholder_title.title('STARTER TRAINING - 📊REPORTS')
        home.load_home_sidebar(st.session_state.edad, st.session_state.peso, st.session_state.talla, st.session_state.imc, st.session_state.perabdominal,st.session_state.genero)
        reportes.load_reportes(username)
    else:
        if app_mode =='Push Up':
            id_exercise = 'push_up'

        elif app_mode =='Curl Up':
            id_exercise = 'curl_up'

        elif app_mode =='Front Plank':
            id_exercise = 'front_plank'

        elif app_mode =='Forward Lunge':
            id_exercise = 'forward_lunge'

        elif app_mode =='Bird Dog':
            id_exercise = 'bird_dog'
        else:
            id_exercise = None

        print_sidebar_main(id_exercise)
        finishexercise = False
        # total_poses = Sets x Reps x N° Poses
        st.session_state.total_poses = st.session_state.n_sets * st.session_state.n_reps * st.session_state.n_poses
        exercise_control, exercise_number_set, exercise_number_rep, exercise_number_pose, exercise_number_pose_global, exercise_status  = st.columns(6)
            
        with exercise_control:
            placeholder_button_status = st.empty()
            placeholder_button_status.info('PRESS START', icon="📹")
            st.markdown("<br>", unsafe_allow_html=True)
            webcam = st.button("START / STOP")
            st.markdown("<spam style='color:cyan; font-size:12px;'>🌏Speech commands requires Internet connection</span>", unsafe_allow_html=True)
        with exercise_number_set:
            placeholder_set = st.empty()
            placeholder_set.metric(        "🟠SET",         "0 / " + str(st.session_state.n_sets),      "Not started", delta_color="inverse")
        with exercise_number_rep:
            placeholder_rep = st.empty()
            placeholder_rep.metric(        "🟠REPETITION",  "0 / " + str(st.session_state.n_reps),      "Not started", delta_color="inverse")
        with exercise_number_pose:
            placeholder_pose = st.empty()
            placeholder_pose.metric(       "🟠POSE",        "0 / " + str(st.session_state.n_poses),     "Not started", delta_color="inverse")
        with exercise_number_pose_global:
            placeholder_pose_global = st.empty()
            placeholder_pose_global.metric("🟠GLOBAL POSE", "0 / " + str(st.session_state.total_poses), "Not started", delta_color="inverse")
        with exercise_status:
            placeholder_status = st.empty()
            st.markdown("<br>", unsafe_allow_html=True)
            placeholder_status.markdown(util.font_size_px("⏱️ SECONDS to start", 26), unsafe_allow_html=True)        
        st.markdown('---')
        
        trainer, user = st.columns(2)
        with trainer:        
            st.markdown("**TRAINER**", unsafe_allow_html=True)
            placeholder_trainer = st.empty()
            placeholder_trainer.image("./01. webapp_img/trainer.png")
        with user:
            st.markdown("**USER**", unsafe_allow_html=True)
            stframe = st.empty()
            df_trainer_coords = get_trainer_coords(id_exercise, id_trainer)
            df_trainers_angles = get_trainers_angles(id_exercise)        
        st.markdown('---')
        placeholder_results_1 = st.empty()
        placeholder_results_2 = st.empty()
        
        with exercise_control:
            if(webcam):
                video_capture = cv2.VideoCapture(0,cv2.CAP_DSHOW)

                # Cámara apagada
                if st.session_state['camera'] % 2 != 0:
                    placeholder_button_status.warning('CAMERA OFF ⚫', icon="📹")
                    st.session_state['camera'] += 1
                    video_capture.release()
                    cv2.destroyAllWindows()
                    stframe.image("./01. webapp_img/user.png")

                # Cámara encendida
                else: 
                    placeholder_button_status.success('CAMERA ON  🚨', icon="📹")
                    st.session_state['camera'] += 1
                    
                    placeholder_trainer.image("./01. webapp_img/warm_up.gif")
                    stframe.image("./01. webapp_img/warm_up.gif")
                    mstart = "Asegúrate que los puntos del ejercicio sean visibles por tu cámara"
                    speak_start_msg = threading.Thread(target=speak, args=(mstart,))
                    speak_start_msg.start()
                    time.sleep(2)
                    for secs in range(st.session_state.starting_secs,0,-1):
                        ss = secs%60
                        placeholder_status.markdown(util.font_size_px(f"🏁 START IN {ss:02d}", 26), unsafe_allow_html=True)
                        time.sleep(1)
                    placeholder_status.markdown(util.font_size_px("🏎️ TRAINING...", 26), unsafe_allow_html=True)
                    placeholder_trainer.image("./02. trainers/{}/images/{}1.png".format(id_exercise, id_exercise))
                    selected_exercise = id_exercise

                    ############################################################
                    ##               📘 RESULT DATAFRAME (INICIO)             ##
                    ############################################################
                    df_results = util.create_df_results()
                    st.session_state.inicio_rutina = get_timestamp_log()                    
                    ############################################################
                    ##               📘 RESULT DATAFRAME (FIN)                ##
                    ############################################################

                    with user:
                        cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
                        up = False
                        down = False
                        mid = False
                        start = 0
                        last_set = 0
                        last_rep = 0
                        
                        while st.session_state.count_set < st.session_state.n_sets:
                            stage = ""
                            st.session_state.count_rep = 0
                            flagTime = False
                            # Setup mediapipe instance
                            with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                                cap.isOpened()
                                while st.session_state.count_rep < st.session_state.n_reps:
                                    
                                    ret, frame = cap.read()
                                    if ret == False:
                                        break
                                    frame = cv2.flip(frame,1)
                                    height, width, _ = frame.shape
                                    # Recolor image to RGB
                                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                    image.flags.writeable = False
                                
                                    # Make detection
                                    results = pose.process(image)

                                    # Recolor back to BGR
                                    image.flags.writeable = True
                                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                                    
                                    # Extract landmarks
                                    if results.pose_landmarks is None:
                                        cv2.putText(image, 
                                        "No se han detectado ninguno de los 33 puntos corporales",
                                        (100,250),
                                        cv2.FONT_HERSHEY_DUPLEX,
                                        0.5,
                                        (0, 0, 255),
                                        1, 
                                        cv2.LINE_AA)
                                        stframe.image(image,channels = 'BGR',use_column_width=True)   
                                    else:
                                        ############################################################
                                        ##         🏃‍♀️ SISTEMA PREDICCIÓN EJERCICIO (INICIO)       ##
                                        ############################################################

                                        landmarks = results.pose_landmarks.landmark
                                        pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in landmarks]).flatten())

                                        # Concate rows
                                        row = pose_row

                                        # Make Detections
                                        X = pd.DataFrame([row])
                                        # Load Model Clasification
                                        body_language_class = LoadModel().predict(X)[0]
                                        # Load Model Clasification                                        
                                        body_language_prob = max(LoadModel().predict_proba(X)[0])
                                        body_language_prob_p = round(body_language_prob*100,2)

                                        ############################################################
                                        ##         🏃‍♀️ SISTEMA PREDICCIÓN EJERCICIO (FIN)          ##
                                        ############################################################
                                        if (landmarks[11].visibility >= st.session_state.landmark_visible and \
                                            landmarks[13].visibility >= st.session_state.landmark_visible and \
                                            landmarks[15].visibility >= st.session_state.landmark_visible):
                                            right_arm_x1 = int(landmarks[11].x * width) #right_elbow_angle
                                            right_arm_x2 = int(landmarks[13].x * width)
                                            right_arm_x3 = int(landmarks[15].x * width)
                                            right_arm_y1 = int(landmarks[11].y * height)
                                            right_arm_y2 = int(landmarks[13].y * height)
                                            right_arm_y3 = int(landmarks[15].y * height)  

                                            right_arm_p1 = np.array([right_arm_x1, right_arm_y1])
                                            right_arm_p2 = np.array([right_arm_x2, right_arm_y2])
                                            right_arm_p3 = np.array([right_arm_x3, right_arm_y3])

                                            right_arm_l1 = np.linalg.norm(right_arm_p2 - right_arm_p3)
                                            right_arm_l2 = np.linalg.norm(right_arm_p1 - right_arm_p3)
                                            right_arm_l3 = np.linalg.norm(right_arm_p1 - right_arm_p2)

                                            # Calculate right_elbow_angle
                                            right_elbow_angle = UpcSystemAngles.calculate_angleacos(right_arm_l1, right_arm_l2, right_arm_l3)

                                        else:
                                            right_arm_x1 = 0
                                            right_arm_x2 = 0
                                            right_arm_x3 = 0
                                            right_arm_y1 = 0
                                            right_arm_y2 = 0
                                            right_arm_y3 = 0 
                                            right_elbow_angle = 0

                                        if (landmarks[12].visibility >= st.session_state.landmark_visible and \
                                            landmarks[14].visibility >= st.session_state.landmark_visible and \
                                            landmarks[16].visibility >= st.session_state.landmark_visible):
                                            left_arm_x1 = int(landmarks[12].x * width) #left_elbow_angle
                                            left_arm_x2 = int(landmarks[14].x * width)
                                            left_arm_x3 = int(landmarks[16].x * width)
                                            left_arm_y1 = int(landmarks[12].y * height)
                                            left_arm_y2 = int(landmarks[14].y * height)
                                            left_arm_y3 = int(landmarks[16].y * height)  

                                            left_arm_p1 = np.array([left_arm_x1, left_arm_y1])
                                            left_arm_p2 = np.array([left_arm_x2, left_arm_y2])
                                            left_arm_p3 = np.array([left_arm_x3, left_arm_y3])

                                            left_arm_l1 = np.linalg.norm(left_arm_p2 - left_arm_p3)
                                            left_arm_l2 = np.linalg.norm(left_arm_p1 - left_arm_p3)
                                            left_arm_l3 = np.linalg.norm(left_arm_p1 - left_arm_p2)

                                            # Calculate left_elbow_angle
                                            left_elbow_angle = UpcSystemAngles.calculate_angleacos(left_arm_l1, left_arm_l2, left_arm_l3)

                                        else:
                                            left_arm_x1 = 0
                                            left_arm_x2 = 0
                                            left_arm_x3 = 0
                                            left_arm_y1 = 0
                                            left_arm_y2 = 0
                                            left_arm_y3 = 0
                                            left_elbow_angle = 0

                                        if (landmarks[13].visibility >= st.session_state.landmark_visible and \
                                            landmarks[11].visibility >= st.session_state.landmark_visible and \
                                                landmarks[23].visibility >= st.session_state.landmark_visible):
                                            right_shoul_x1 = int(landmarks[13].x * width) #right_shoulder_angle
                                            right_shoul_x2 = int(landmarks[11].x * width)
                                            right_shoul_x3 = int(landmarks[23].x * width)
                                            right_shoul_y1 = int(landmarks[13].y * height)
                                            right_shoul_y2 = int(landmarks[11].y * height)
                                            right_shoul_y3 = int(landmarks[23].y * height)  

                                            right_shoul_p1 = np.array([right_shoul_x1, right_shoul_y1])
                                            right_shoul_p2 = np.array([right_shoul_x2, right_shoul_y2])
                                            right_shoul_p3 = np.array([right_shoul_x3, right_shoul_y3])

                                            right_shoul_l1 = np.linalg.norm(right_shoul_p2 - right_shoul_p3)
                                            right_shoul_l2 = np.linalg.norm(right_shoul_p1 - right_shoul_p3)
                                            right_shoul_l3 = np.linalg.norm(right_shoul_p1 - right_shoul_p2)

                                            # Calculate angle
                                            right_shoulder_angle = UpcSystemAngles.calculate_angleacos(right_shoul_l1, right_shoul_l2, right_shoul_l3)

                                        else:
                                            right_shoul_x1 = 0
                                            right_shoul_x2 = 0
                                            right_shoul_x3 = 0
                                            right_shoul_y1 = 0
                                            right_shoul_y2 = 0
                                            right_shoul_y3 = 0
                                            right_shoulder_angle = 0

                                        if (landmarks[25].visibility >= st.session_state.landmark_visible and \
                                            landmarks[27].visibility >= st.session_state.landmark_visible and \
                                                landmarks[31].visibility >= st.session_state.landmark_visible):
                                            right_ankle_x1 = int(landmarks[25].x * width) #right_ankle_angle
                                            right_ankle_x2 = int(landmarks[27].x * width)
                                            right_ankle_x3 = int(landmarks[31].x * width)
                                            right_ankle_y1 = int(landmarks[25].y * height)
                                            right_ankle_y2 = int(landmarks[27].y * height)
                                            right_ankle_y3 = int(landmarks[31].y * height)  

                                            right_ankle_p1 = np.array([right_ankle_x1, right_ankle_y1])
                                            right_ankle_p2 = np.array([right_ankle_x2, right_ankle_y2])
                                            right_ankle_p3 = np.array([right_ankle_x3, right_ankle_y3])

                                            right_ankle_l1 = np.linalg.norm(right_ankle_p2 - right_ankle_p3)
                                            right_ankle_l2 = np.linalg.norm(right_ankle_p1 - right_ankle_p3)
                                            right_ankle_l3 = np.linalg.norm(right_ankle_p1 - right_ankle_p2)

                                            # Calculate angle
                                            right_ankle_angle = UpcSystemAngles.calculate_angleacos(right_ankle_l1, right_ankle_l2, right_ankle_l3)

                                        else:
                                            right_ankle_x1 = 0
                                            right_ankle_x2 = 0
                                            right_ankle_x3 = 0
                                            right_ankle_y1 = 0
                                            right_ankle_y2 = 0
                                            right_ankle_y3 = 0  
                                            right_ankle_angle = 0

                                        if (landmarks[11].visibility >= st.session_state.landmark_visible and \
                                            landmarks[23].visibility >= st.session_state.landmark_visible and \
                                                landmarks[25].visibility >= st.session_state.landmark_visible):
                                            right_torso_x1 = int(landmarks[11].x * width) #right_hip_angle
                                            right_torso_x2 = int(landmarks[23].x * width)
                                            right_torso_x3 = int(landmarks[25].x * width) 
                                            right_torso_y1 = int(landmarks[11].y * height)
                                            right_torso_y2 = int(landmarks[23].y * height)
                                            right_torso_y3 = int(landmarks[25].y * height) 

                                            right_torso_p1 = np.array([right_torso_x1, right_torso_y1])
                                            right_torso_p2 = np.array([right_torso_x2, right_torso_y2])
                                            right_torso_p3 = np.array([right_torso_x3, right_torso_y3])

                                            right_torso_l1 = np.linalg.norm(right_torso_p2 - right_torso_p3)
                                            right_torso_l2 = np.linalg.norm(right_torso_p1 - right_torso_p3)
                                            right_torso_l3 = np.linalg.norm(right_torso_p1 - right_torso_p2)
                                            
                                            # Calculate right_hip_angle
                                            right_hip_angle = UpcSystemAngles.calculate_angleacos(right_torso_l1, right_torso_l2, right_torso_l3)
                                            
                                        else:
                                            right_torso_x1 = 0
                                            right_torso_x2 = 0
                                            right_torso_x3 = 0
                                            right_torso_y1 = 0
                                            right_torso_y2 = 0
                                            right_torso_y3 = 0
                                            right_hip_angle = 0

                                        if (landmarks[23].visibility >= st.session_state.landmark_visible and \
                                            landmarks[25].visibility >= st.session_state.landmark_visible and \
                                            landmarks[27].visibility >= st.session_state.landmark_visible):
                                            right_leg_x1 = int(landmarks[23].x * width) #right_knee_angle
                                            right_leg_x2 = int(landmarks[25].x * width)
                                            right_leg_x3 = int(landmarks[27].x * width) 
                                            right_leg_y1 = int(landmarks[23].y * height)
                                            right_leg_y2 = int(landmarks[25].y * height)
                                            right_leg_y3 = int(landmarks[27].y * height)

                                            right_leg_p1 = np.array([right_leg_x1, right_leg_y1])
                                            right_leg_p2 = np.array([right_leg_x2, right_leg_y2])
                                            right_leg_p3 = np.array([right_leg_x3, right_leg_y3])

                                            right_leg_l1 = np.linalg.norm(right_leg_p2 - right_leg_p3)
                                            right_leg_l2 = np.linalg.norm(right_leg_p1 - right_leg_p3)
                                            right_leg_l3 = np.linalg.norm(right_leg_p1 - right_leg_p2)

                                            # Calculate angle
                                            right_knee_angle = UpcSystemAngles.calculate_angleacos(right_leg_l1, right_leg_l2, right_leg_l3)
                                            
                                        else:
                                            right_leg_x1 = 0
                                            right_leg_x2 = 0
                                            right_leg_x3 = 0
                                            right_leg_y1 = 0
                                            right_leg_y2 = 0
                                            right_leg_y3 = 0
                                            right_knee_angle = 0
                                        
                                        if (landmarks[24].visibility >= st.session_state.landmark_visible and \
                                            landmarks[26].visibility >= st.session_state.landmark_visible and \
                                                landmarks[28].visibility >= st.session_state.landmark_visible):
                                            left_leg_x1 = int(landmarks[24].x * width) #left_knee_angle
                                            left_leg_x2 = int(landmarks[26].x * width)
                                            left_leg_x3 = int(landmarks[28].x * width) 
                                            left_leg_y1 = int(landmarks[24].y * height)
                                            left_leg_y2 = int(landmarks[26].y * height)
                                            left_leg_y3 = int(landmarks[28].y * height)

                                            left_leg_p1 = np.array([left_leg_x1, left_leg_y1])
                                            left_leg_p2 = np.array([left_leg_x2, left_leg_y2])
                                            left_leg_p3 = np.array([left_leg_x3, left_leg_y3])

                                            left_leg_l1 = np.linalg.norm(left_leg_p2 - left_leg_p3)
                                            left_leg_l2 = np.linalg.norm(left_leg_p1 - left_leg_p3)
                                            left_leg_l3 = np.linalg.norm(left_leg_p1 - left_leg_p2)

                                            # Calculate angle
                                            left_knee_angle = UpcSystemAngles.calculate_angleacos(left_leg_l1, left_leg_l2, left_leg_l3)

                                        else:
                                            left_leg_x1 = 0
                                            left_leg_x2 = 0
                                            left_leg_x3 = 0
                                            left_leg_y1 = 0
                                            left_leg_y2 = 0
                                            left_leg_y3 = 0
                                            left_knee_angle = 0

                                        ############################################################
                                        ##          💰 SISTEMA COSTOS - CÁLCULO (INICIO ⬇️)      ##
                                        ############################################################
                                        pose_trainer_cost_min, pose_trainer_cost_max = UpcSystemCost.get_cost_pose_trainer(id_exercise, st.session_state.count_pose + 1)
                                        pose_user_cost = UpcSystemCost.get_cost_pose_user(df_trainer_coords, results, st.session_state.count_pose+1)

                                        ############# DEVELOPER MODE ⬇️ #############
                                        if st.session_state.developer_mode == True:
                                            pose_trainer_cost_min = 0
                                            pose_trainer_cost_max = 99
                                        else:
                                            pose_trainer_cost_min = pose_trainer_cost_min
                                            pose_trainer_cost_max = pose_trainer_cost_max
                                        ############# DEVELOPER MODE ⬆️ #############

                                        ############################################################
                                        ##          💰 SISTEMA COSTOS - CÁLCULO (FIN ⬆️)         ##
                                        ############################################################
                                        
                                        ############################################################
                                        ##        📐 SISTEMA ÁNGULOS - CÁLCULO (INICIO ⬇️)       ##
                                        ############################################################
                                        if (pose_user_cost < pose_trainer_cost_min) or (pose_user_cost > pose_trainer_cost_max):
                                            cost_valid = "Asegúrate de imitar la pose del entrenador"
                                            try:
                                                if not speak_stage1.is_alive():
                                                    speak_cost.start()
                                            except:
                                                try:
                                                    if not speak_stage2.is_alive():
                                                        speak_cost.start()
                                                except:
                                                    try:
                                                        if not speak_stage3.is_alive():
                                                            speak_cost.start()
                                                    except:
                                                        try:
                                                            if not speak_stage4.is_alive():
                                                                speak_cost.start()
                                                        except:
                                                            try:
                                                                if not speak_cost.is_alive():
                                                                    speak_cost.start()
                                                            except:
                                                                speak_cost = threading.Thread(target=speak, args=(cost_valid,))
                                                                speak_cost.start() 
                                        else:
                                            #🟢 PUSH UP - 3 POSES (ejercicio 1 de 5)
                                            if id_exercise == "push_up" and body_language_prob_p > st.session_state.prob_avance_pose:

                                                right_elbow_angle_in        = UpcSystemAngles.get_valu_angle(df_trainers_angles, start, 'right_elbow_angles')
                                                right_hip_angle_in          = UpcSystemAngles.get_valu_angle(df_trainers_angles, start, 'right_hip_angles')
                                                right_knee_angle_in         = UpcSystemAngles.get_valu_angle(df_trainers_angles, start, 'right_knee_angles')
                                                desv_right_elbow_angle_in   = UpcSystemAngles.get_desv_angle(df_trainers_angles, start, 'right_elbow_angles')#10
                                                desv_right_hip_angle_in     = UpcSystemAngles.get_desv_angle(df_trainers_angles, start, 'right_hip_angles')#10
                                                desv_right_knee_angle_in    = UpcSystemAngles.get_desv_angle(df_trainers_angles, start, 'right_knee_angles')#10

                                                #🟢 PUSH UP - pose 1 de 3
                                                if  up == False and\
                                                    down == False and\
                                                    right_elbow_angle in range(int(right_elbow_angle_in-desv_right_elbow_angle_in), int(right_elbow_angle_in + desv_right_elbow_angle_in + 1)) and\
                                                    right_hip_angle in range(int(right_hip_angle_in - desv_right_hip_angle_in), int(right_hip_angle_in + desv_right_hip_angle_in + 1)) and\
                                                    right_knee_angle in range(int(right_knee_angle_in - desv_right_knee_angle_in),int(right_knee_angle_in + desv_right_knee_angle_in + 1)):                                                    

                                                    up = True
                                                    stage = "Arriba"
                                                    speak_stage1 = threading.Thread(target=speak, args=(stage,))
                                                    speak_stage1.start()
                                                    ############################################
                                                    st.session_state.count_pose_g += 1
                                                    st.session_state.count_pose = 1
                                                    update_counter_panel('training', st.session_state.count_pose_g, st.session_state.count_pose, st.session_state.count_rep + 1, st.session_state.count_set + 1)
                                                    update_trainer_image(id_exercise, 2)
                                                    ############################################
                                                    fin_rutina_timestamp = get_timestamp_log()
                                                    df_results = util.add_row_df_results(df_results,
                                                                                    id_exercise,                        #1 - str - id_exercise
                                                                                    st.session_state.inicio_rutina,     #2 - str - DateTime_Start
                                                                                    fin_rutina_timestamp,               #3 - str - DateTime_End
                                                                                    st.session_state.n_poses,           #4 - int - n_poses
                                                                                    st.session_state.n_sets,            #5 - int - n_sets
                                                                                    st.session_state.n_reps,            #6 - int - n_reps
                                                                                    st.session_state.total_poses,       #7 - int - total_poses
                                                                                    st.session_state.seconds_rest_time, #8 - int - seconds_rest_time
                                                                                    selected_exercise,                  #9 - str - Class
                                                                                    body_language_prob_p,               #10 - float - Prob
                                                                                    st.session_state.count_pose_g,      #11 - int - count_pose_g
                                                                                    st.session_state.count_pose,        #12 - int - count_pose
                                                                                    st.session_state.count_rep + 1,     #13 - int - count_rep
                                                                                    st.session_state.count_set + 1,     #14 - int - count_set
                                                                                    right_elbow_angle,                  #15 - float - right_elbow_angles_pu
                                                                                    right_hip_angle,                    #16 - float - right_hip_angles_pu
                                                                                    right_knee_angle,                   #17 - float - right_knee_angles_pu
                                                                                    None,                               #18 - float - right_shoulder_angles_cu
                                                                                    None,                               #19 - float - right_hip_angles_cu
                                                                                    None,                               #20 - float - right_knee_angles_cu
                                                                                    None,                               #21 - float - right_shoulder_angles_fp
                                                                                    None,                               #22 - float - right_hip_angles_fp
                                                                                    None,                               #23 - float - right_ankle_angles_fp
                                                                                    None,                               #24 - float - right_hip_angles_fl
                                                                                    None,                               #25 - float - right_knee_angles_fl
                                                                                    None,                               #26 - float - left_knee_angles_fl
                                                                                    None,                               #27 - float - right_shoulder_angles_bd
                                                                                    None,                               #28 - float - right_hip_angles_bd
                                                                                    None,                               #29 - float - right_knee_angles_bd
                                                                                    None,                               #30 - float - left_knee_angles_bd
                                                                                    None,                               #31 - float - right_elbow_angles_bd
                                                                                    None,                               #32 - float - left_elbow_angles_bd
                                                                                    pose_trainer_cost_min,              #33 - float - pose_trainer_cost_min
                                                                                    pose_trainer_cost_max,              #34 - float - pose_trainer_cost_max
                                                                                    pose_user_cost                      #35 - float - pose_user_cost
                                                    )
                                                    ############################################
                                                    last_set = st.session_state.count_set + 1
                                                    last_rep = st.session_state.count_rep + 1     
                                                    start +=1
                                                    st.session_state.inicio_rutina = fin_rutina_timestamp

                                                #🟢 PUSH UP - pose 2 de 3
                                                elif up == True and\
                                                    down == False and\
                                                    right_elbow_angle in range(int(right_elbow_angle_in - desv_right_elbow_angle_in) , int(right_elbow_angle_in + desv_right_elbow_angle_in + 1)) and\
                                                    right_hip_angle in range(int(right_hip_angle_in - desv_right_hip_angle_in), int(right_hip_angle_in + desv_right_hip_angle_in + 1)) and\
                                                    right_knee_angle in range(int(right_knee_angle_in - desv_right_knee_angle_in),int(right_knee_angle_in + desv_right_knee_angle_in + 1)):
                                                    
                                                    down = True
                                                    stage = "Abajo"
                                                    speak_stage2 = threading.Thread(target=speak, args=(stage,))
                                                    speak_stage2.start()
                                                    ############################################                                                    
                                                    st.session_state.count_pose_g += 1
                                                    st.session_state.count_pose = 2
                                                    update_counter_panel('training', st.session_state.count_pose_g, st.session_state.count_pose, st.session_state.count_rep + 1, st.session_state.count_set + 1)
                                                    update_trainer_image(id_exercise, 3)
                                                    ############################################
                                                    fin_rutina_timestamp = get_timestamp_log()
                                                    df_results = util.add_row_df_results(df_results,
                                                                                    id_exercise,                        #1 - str - id_exercise                                                                                    
                                                                                    st.session_state.inicio_rutina,     #2 - str - DateTime_Start
                                                                                    fin_rutina_timestamp,               #3 - str - DateTime_End
                                                                                    st.session_state.n_poses,           #4 - int - n_poses
                                                                                    st.session_state.n_sets,            #5 - int - n_sets
                                                                                    st.session_state.n_reps,            #6 - int - n_reps
                                                                                    st.session_state.total_poses,       #7 - int - total_poses
                                                                                    st.session_state.seconds_rest_time, #8 - int - seconds_rest_time
                                                                                    selected_exercise,                  #9 - str - Class
                                                                                    body_language_prob_p,               #10 - float - Prob
                                                                                    st.session_state.count_pose_g,      #11 - int - count_pose_g
                                                                                    st.session_state.count_pose,        #12 - int - count_pose
                                                                                    st.session_state.count_rep + 1,     #13 - int - count_rep
                                                                                    st.session_state.count_set + 1,     #14 - int - count_set
                                                                                    right_elbow_angle,                  #15 - float - right_elbow_angles_pu
                                                                                    right_hip_angle,                    #16 - float - right_hip_angles_pu
                                                                                    right_knee_angle,                   #17 - float - right_knee_angles_pu
                                                                                    None,    #18 - float - right_shoulder_angles_cu
                                                                                    None,    #19 - float - right_hip_angles_cu
                                                                                    None,    #20 - float - right_knee_angles_cu
                                                                                    None,    #21 - float - right_shoulder_angles_fp
                                                                                    None,    #22 - float - right_hip_angles_fp
                                                                                    None,    #23 - float - right_ankle_angles_fp
                                                                                    None,    #24 - float - right_hip_angles_fl
                                                                                    None,    #25 - float - right_knee_angles_fl
                                                                                    None,    #26 - float - left_knee_angles_fl
                                                                                    None,    #27 - float - right_shoulder_angles_bd
                                                                                    None,    #28 - float - right_hip_angles_bd
                                                                                    None,    #29 - float - right_knee_angles_bd
                                                                                    None,    #30 - float - left_knee_angles_bd
                                                                                    None,    #31 - float - right_elbow_angles_bd
                                                                                    None,    #32 - float - left_elbow_angles_bd
                                                                                    pose_trainer_cost_min,              #33 - float - pose_trainer_cost_min
                                                                                    pose_trainer_cost_max,              #34 - float - pose_trainer_cost_max
                                                                                    pose_user_cost                      #35 - float - pose_user_cost
                                                    )
                                                    ############################################
                                                    last_set = st.session_state.count_set + 1
                                                    last_rep = st.session_state.count_rep + 1
                                                    start +=1
                                                    st.session_state.inicio_rutina = fin_rutina_timestamp
                                                
                                                #🟢 PUSH UP - pose 3 de 3
                                                elif up == True and\
                                                    down == True and\
                                                    right_elbow_angle in range(int(right_elbow_angle_in - desv_right_elbow_angle_in) , int(right_elbow_angle_in + desv_right_elbow_angle_in + 1)) and\
                                                    right_hip_angle in range(int(right_hip_angle_in - desv_right_hip_angle_in), int(right_hip_angle_in + desv_right_hip_angle_in + 1)) and\
                                                    right_knee_angle in range(int(right_knee_angle_in - desv_right_knee_angle_in),int(right_knee_angle_in + desv_right_knee_angle_in + 1)):   
                                                    up = False
                                                    down = False
                                                    stage = "Arriba"
                                                    st.session_state.count_pose_g += 1
                                                    st.session_state.count_pose = 3
                                                    ############################################
                                                    update_counter_panel('training', st.session_state.count_pose_g, st.session_state.count_pose, st.session_state.count_rep + 1, st.session_state.count_set + 1)
                                                    # Cambio Renzo
                                                    # update_trainer_image(id_exercise, 1)
                                                    ############################################
                                                    speak_stage3 = threading.Thread(target=speak, args=(stage,))
                                                    speak_stage3.start()
                                                    fin_rutina_timestamp = get_timestamp_log()
                                                    df_results = util.add_row_df_results(df_results,
                                                                                    id_exercise,                        #1 - str - id_exercise
                                                                                    st.session_state.inicio_rutina,     #2 - str - DateTime_Start
                                                                                    fin_rutina_timestamp,               #3 - str - DateTime_End
                                                                                    st.session_state.n_poses,           #4 - int - n_poses
                                                                                    st.session_state.n_sets,            #5 - int - n_sets
                                                                                    st.session_state.n_reps,            #6 - int - n_reps
                                                                                    st.session_state.total_poses,       #7 - int - total_poses
                                                                                    st.session_state.seconds_rest_time, #8 - int - seconds_rest_time
                                                                                    selected_exercise,                  #9 - str - Class
                                                                                    body_language_prob_p,               #10 - float - Prob
                                                                                    st.session_state.count_pose_g,      #11 - int - count_pose_g
                                                                                    st.session_state.count_pose,        #12 - int - count_pose
                                                                                    st.session_state.count_rep + 1,     #13 - int - count_rep
                                                                                    st.session_state.count_set + 1,     #14 - int - count_set
                                                                                    right_elbow_angle,                  #15 - float - right_elbow_angles_pu
                                                                                    right_hip_angle,                    #16 - float - right_hip_angles_pu
                                                                                    right_knee_angle,                   #17 - float - right_knee_angles_pu
                                                                                    None,    #18 - float - right_shoulder_angles_cu
                                                                                    None,    #19 - float - right_hip_angles_cu
                                                                                    None,    #20 - float - right_knee_angles_cu
                                                                                    None,    #21 - float - right_shoulder_angles_fp
                                                                                    None,    #22 - float - right_hip_angles_fp
                                                                                    None,    #23 - float - right_ankle_angles_fp
                                                                                    None,    #24 - float - right_hip_angles_fl
                                                                                    None,    #25 - float - right_knee_angles_fl
                                                                                    None,    #26 - float - left_knee_angles_fl
                                                                                    None,    #27 - float - right_shoulder_angles_bd
                                                                                    None,    #28 - float - right_hip_angles_bd
                                                                                    None,    #29 - float - right_knee_angles_bd
                                                                                    None,    #30 - float - left_knee_angles_bd
                                                                                    None,    #31 - float - right_elbow_angles_bd
                                                                                    None,    #32 - float - left_elbow_angles_bd
                                                                                    pose_trainer_cost_min,              #33 - float - pose_trainer_cost_min
                                                                                    pose_trainer_cost_max,              #34 - float - pose_trainer_cost_max
                                                                                    pose_user_cost                      #35 - float - pose_user_cost
                                                    )
                                                    ######################################s######
                                                    last_set = st.session_state.count_set + 1
                                                    last_rep = st.session_state.count_rep + 1
                                                    start = 0 
                                                    st.session_state.count_rep += 1
                                                    st.session_state.count_pose = 0
                                                    st.session_state.inicio_rutina = fin_rutina_timestamp    

                                                #🟢 PUSH UP - ninguna pose
                                                else:
                                                    # ************************ INICIO SISTEMA DE RECOMENDACIONES - PUSH UP ************************ #                                                    
                                                    if st.session_state.count_pose+1 == 1 or st.session_state.count_pose+1 == 3:                                                    
                                                        if right_elbow_angle > right_elbow_angle_in+desv_right_elbow_angle_in:
                                                            rec_right_elbow_angle = "Flexiona más tu codo derecho"
                                                        elif right_elbow_angle < right_elbow_angle_in-desv_right_elbow_angle_in:
                                                            rec_right_elbow_angle = "Flexiona menos tu codo derecho"
                                                        else:
                                                            rec_right_elbow_angle = ""
                                                        if (right_hip_angle > right_hip_angle_in+desv_right_hip_angle_in) or\
                                                            (right_hip_angle < right_hip_angle_in-desv_right_hip_angle_in):
                                                            rec_right_hip_angle = ", mantén tu cadera recta"
                                                        else:
                                                            rec_right_hip_angle = ""
                                                        if (right_knee_angle > right_knee_angle_in+desv_right_knee_angle_in):
                                                            rec_right_knee_angle = ", flexiona menos tu rodilla derecha"
                                                        elif right_knee_angle < right_knee_angle_in-desv_right_knee_angle_in:
                                                            rec_right_knee_angle = ", flexiona más tu rodilla derecha"
                                                        else:
                                                            rec_right_knee_angle = ""
                                                        final_rec = rec_right_elbow_angle+rec_right_hip_angle+rec_right_knee_angle
                                                        
                                                    elif st.session_state.count_pose+1 == 2:
                                                        if right_elbow_angle > right_elbow_angle_in+desv_right_elbow_angle_in:
                                                            rec_right_elbow_angle = "Flexiona más tu codo derecho"
                                                        elif right_elbow_angle < right_elbow_angle_in-desv_right_elbow_angle_in:
                                                            rec_right_elbow_angle = "Flexiona menos tu codo derecho"
                                                        else:
                                                            rec_right_elbow_angle = ""
                                                        if (right_hip_angle > right_hip_angle_in+desv_right_hip_angle_in) or\
                                                            (right_hip_angle < right_hip_angle_in-desv_right_hip_angle_in):
                                                            rec_right_hip_angle = ", mantén tu cadera recta"
                                                        else:
                                                            rec_right_hip_angle = ""
                                                        if right_knee_angle > right_knee_angle_in+desv_right_knee_angle_in:
                                                            rec_right_knee_angle = ", flexiona más tu rodilla derecha"
                                                        elif right_knee_angle < right_knee_angle_in-desv_right_knee_angle_in:
                                                            rec_right_knee_angle = ", flexiona menos tu rodilla derecha"
                                                        else:
                                                            rec_right_knee_angle = ""
                                                        final_rec = rec_right_elbow_angle+rec_right_hip_angle+rec_right_knee_angle
                                                        
                                                    if final_rec != "":
                                                        if st.session_state.count_pose+1 == 1:
                                                            try:
                                                                if not speak_rec.is_alive():
                                                                    speak_rec.start()
                                                            except:
                                                                speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                speak_rec.start()
                                                        elif st.session_state.count_pose+1 == 2:
                                                            try:
                                                                if not speak_rec.is_alive():
                                                                    speak_rec.start()
                                                            except:
                                                                try:
                                                                    if not speak_stage1.is_alive():
                                                                        speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                        speak_rec.start()
                                                                except:
                                                                    speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                    speak_rec.start()
                                                        elif st.session_state.count_pose+1 == 3:
                                                            try:
                                                                if not speak_rec.is_alive():
                                                                    speak_rec.start()
                                                            except:
                                                                try:
                                                                    if not speak_stage2.is_alive():
                                                                        speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                        speak_rec.start()
                                                                except:
                                                                    speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                    speak_rec.start()
                                                # ************************ FIN SISTEMA DE RECOMENDACIONES - PUSH UP ************************ #                                   

                                            #🟢 CURL UP - 3 POSES (ejercicio 2 de 5)
                                            elif selected_exercise == "curl_up" and body_language_prob_p > st.session_state.prob_avance_pose:
                                                right_shoulder_angle_in     = UpcSystemAngles.get_valu_angle(df_trainers_angles, start, 'right_shoulder_angles')
                                                right_hip_angle_in          = UpcSystemAngles.get_valu_angle(df_trainers_angles, start, 'right_hip_angles')
                                                right_knee_angle_in         = UpcSystemAngles.get_valu_angle(df_trainers_angles, start, 'right_knee_angles')
                                                desv_right_shoulder_angle_in= UpcSystemAngles.get_desv_angle(df_trainers_angles, start, 'right_shoulder_angles')#15
                                                desv_right_hip_angle_in     = UpcSystemAngles.get_desv_angle(df_trainers_angles, start, 'right_hip_angles')#15
                                                desv_right_knee_angle_in    = UpcSystemAngles.get_desv_angle(df_trainers_angles, start, 'right_knee_angles')#15

                                                #🟢 CURL UP - pose 1 de 3
                                                if  up == False and\
                                                    down == False and\
                                                    right_shoulder_angle in range(int(right_shoulder_angle_in-desv_right_shoulder_angle_in), int(right_shoulder_angle_in + desv_right_shoulder_angle_in + 1)) and\
                                                    right_hip_angle in range(int(right_hip_angle_in - desv_right_hip_angle_in), int(right_hip_angle_in + desv_right_hip_angle_in + 1)) and\
                                                    right_knee_angle in range(int(right_knee_angle_in - desv_right_knee_angle_in),int(right_knee_angle_in + desv_right_knee_angle_in + 1)):                                                    
                                                    up = True
                                                    stage = "Abajo"
                                                    speak_stage1 = threading.Thread(target=speak, args=(stage,))
                                                    speak_stage1.start()
                                                    ############################################
                                                    st.session_state.count_pose_g += 1
                                                    st.session_state.count_pose = 1
                                                    update_counter_panel('training', st.session_state.count_pose_g, st.session_state.count_pose, st.session_state.count_rep + 1, st.session_state.count_set + 1)
                                                    update_trainer_image(id_exercise, 2)
                                                    ############################################
                                                    fin_rutina_timestamp = get_timestamp_log()
                                                    df_results = util.add_row_df_results(df_results,
                                                                                    id_exercise,                        #1 - str - id_exercise
                                                                                    st.session_state.inicio_rutina,     #2 - str - DateTime_Start
                                                                                    fin_rutina_timestamp,               #3 - str - DateTime_End
                                                                                    st.session_state.n_poses,           #4 - int - n_poses
                                                                                    st.session_state.n_sets,            #5 - int - n_sets
                                                                                    st.session_state.n_reps,            #6 - int - n_reps
                                                                                    st.session_state.total_poses,       #7 - int - total_poses
                                                                                    st.session_state.seconds_rest_time, #8 - int - seconds_rest_time
                                                                                    selected_exercise,                  #9 - str - Class
                                                                                    body_language_prob_p,               #10 - float - Prob
                                                                                    st.session_state.count_pose_g,      #11 - int - count_pose_g
                                                                                    st.session_state.count_pose,        #12 - int - count_pose
                                                                                    st.session_state.count_rep + 1,     #13 - int - count_rep
                                                                                    st.session_state.count_set + 1,     #14 - int - count_set
                                                                                    None,   #15 - float - right_elbow_angles_pu
                                                                                    None,   #16 - float - right_hip_angles_pu
                                                                                    None,   #17 - float - right_knee_angles_pu
                                                                                    right_shoulder_angle,               #18 - float - right_shoulder_angles_cu
                                                                                    right_hip_angle,                    #19 - float - right_hip_angles_cu
                                                                                    right_knee_angle,                   #20 - float - right_knee_angles_cu
                                                                                    None,    #21 - float - right_shoulder_angles_fp
                                                                                    None,    #22 - float - right_hip_angles_fp
                                                                                    None,    #23 - float - right_ankle_angles_fp
                                                                                    None,    #24 - float - right_hip_angles_fl
                                                                                    None,    #25 - float - right_knee_angles_fl
                                                                                    None,    #26 - float - left_knee_angles_fl
                                                                                    None,    #27 - float - right_shoulder_angles_bd
                                                                                    None,    #28 - float - right_hip_angles_bd
                                                                                    None,    #29 - float - right_knee_angles_bd
                                                                                    None,    #30 - float - left_knee_angles_bd
                                                                                    None,    #31 - float - right_elbow_angles_bd
                                                                                    None,    #32 - float - left_elbow_angles_bd
                                                                                    pose_trainer_cost_min,              #33 - float - pose_trainer_cost_min
                                                                                    pose_trainer_cost_max,              #34 - float - pose_trainer_cost_max
                                                                                    pose_user_cost                      #35 - float - pose_user_cost
                                                    )
                                                    ############################################ 
                                                    last_set = st.session_state.count_set + 1
                                                    last_rep = st.session_state.count_rep + 1
                                                    start +=1
                                                    st.session_state.inicio_rutina = fin_rutina_timestamp
                                                
                                                #🟢 CURL UP - pose 2 de 3
                                                elif up == True and\
                                                    down == False and\
                                                    right_shoulder_angle in range(int(right_shoulder_angle_in - desv_right_shoulder_angle_in) , int(right_shoulder_angle_in + desv_right_shoulder_angle_in + 1)) and\
                                                    right_hip_angle in range(int(right_hip_angle_in - desv_right_hip_angle_in), int(right_hip_angle_in + desv_right_hip_angle_in + 1)) and\
                                                    right_knee_angle in range(int(right_knee_angle_in - desv_right_knee_angle_in),int(right_knee_angle_in + desv_right_knee_angle_in + 1)):                                                    
                                                    down = True
                                                    stage = "Arriba"                                                    
                                                    speak_stage2 = threading.Thread(target=speak, args=(stage,))  
                                                    speak_stage2.start()                                                                                                      
                                                    ############################################
                                                    st.session_state.count_pose_g += 1
                                                    st.session_state.count_pose = 2
                                                    update_counter_panel('training', st.session_state.count_pose_g, st.session_state.count_pose, st.session_state.count_rep + 1, st.session_state.count_set + 1)
                                                    update_trainer_image(id_exercise, 3)
                                                    ############################################
                                                    fin_rutina_timestamp = get_timestamp_log()
                                                    df_results = util.add_row_df_results(df_results,
                                                                                    id_exercise,                        #1 - str - id_exercise
                                                                                    st.session_state.inicio_rutina,     #2 - str - DateTime_Start
                                                                                    fin_rutina_timestamp,               #3 - str - DateTime_End
                                                                                    st.session_state.n_poses,           #4 - int - n_poses
                                                                                    st.session_state.n_sets,            #5 - int - n_sets
                                                                                    st.session_state.n_reps,            #6 - int - n_reps
                                                                                    st.session_state.total_poses,       #7 - int - total_poses
                                                                                    st.session_state.seconds_rest_time, #8 - int - seconds_rest_time
                                                                                    selected_exercise,                  #9 - str - Class
                                                                                    body_language_prob_p,               #10 - float - Prob
                                                                                    st.session_state.count_pose_g,      #11 - int - count_pose_g
                                                                                    st.session_state.count_pose,        #12 - int - count_pose
                                                                                    st.session_state.count_rep + 1,     #13 - int - count_rep
                                                                                    st.session_state.count_set + 1,     #14 - int - count_set
                                                                                    None,   #15 - float - right_elbow_angles_pu
                                                                                    None,   #16 - float - right_hip_angles_pu
                                                                                    None,   #17 - float - right_knee_angles_pu
                                                                                    right_shoulder_angle,               #18 - float - right_shoulder_angles_cu
                                                                                    right_hip_angle,                    #19 - float - right_hip_angles_cu
                                                                                    right_knee_angle,                   #20 - float - right_knee_angles_cu
                                                                                    None,    #21 - float - right_shoulder_angles_fp
                                                                                    None,    #22 - float - right_hip_angles_fp
                                                                                    None,    #23 - float - right_ankle_angles_fp
                                                                                    None,    #24 - float - right_hip_angles_fl
                                                                                    None,    #25 - float - right_knee_angles_fl
                                                                                    None,    #26 - float - left_knee_angles_fl
                                                                                    None,    #27 - float - right_shoulder_angles_bd
                                                                                    None,    #28 - float - right_hip_angles_bd
                                                                                    None,    #29 - float - right_knee_angles_bd
                                                                                    None,    #30 - float - left_knee_angles_bd
                                                                                    None,    #31 - float - right_elbow_angles_bd
                                                                                    None,    #32 - float - left_elbow_angles_bd
                                                                                    pose_trainer_cost_min,              #33 - float - pose_trainer_cost_min
                                                                                    pose_trainer_cost_max,              #34 - float - pose_trainer_cost_max
                                                                                    pose_user_cost                      #35 - float - pose_user_cost
                                                    )
                                                    ############################################
                                                    last_set = st.session_state.count_set + 1
                                                    last_rep = st.session_state.count_rep + 1
                                                    start +=1
                                                    st.session_state.inicio_rutina = fin_rutina_timestamp
                                                
                                                #🟢 CURL UP - pose 3 de 3
                                                elif up == True and\
                                                    down == True and\
                                                    right_shoulder_angle in range(int(right_shoulder_angle_in - desv_right_shoulder_angle_in) , int(right_shoulder_angle_in + desv_right_shoulder_angle_in + 1)) and\
                                                    right_hip_angle in range(int(right_hip_angle_in - desv_right_hip_angle_in), int(right_hip_angle_in + desv_right_hip_angle_in + 1)) and\
                                                    right_knee_angle in range(int(right_knee_angle_in - desv_right_knee_angle_in),int(right_knee_angle_in + desv_right_knee_angle_in + 1)):
                                                    up = False
                                                    down = False
                                                    stage = "Abajo"                                                    
                                                    speak_stage3 = threading.Thread(target=speak, args=(stage,))
                                                    speak_stage3.start()                                                    
                                                    ############################################
                                                    st.session_state.count_pose_g += 1
                                                    st.session_state.count_pose = 3
                                                    update_counter_panel('training', st.session_state.count_pose_g, st.session_state.count_pose, st.session_state.count_rep + 1, st.session_state.count_set + 1)
                                                    ############################################
                                                    fin_rutina_timestamp = get_timestamp_log()
                                                    df_results = util.add_row_df_results(df_results,
                                                                                    id_exercise,                        #1 - str - id_exercise
                                                                                    st.session_state.inicio_rutina,     #2 - str - DateTime_Start
                                                                                    fin_rutina_timestamp,               #3 - str - DateTime_End
                                                                                    st.session_state.n_poses,           #4 - int - n_poses
                                                                                    st.session_state.n_sets,            #5 - int - n_sets
                                                                                    st.session_state.n_reps,            #6 - int - n_reps
                                                                                    st.session_state.total_poses,       #7 - int - total_poses
                                                                                    st.session_state.seconds_rest_time, #8 - int - seconds_rest_time
                                                                                    selected_exercise,                  #9 - str - Class
                                                                                    body_language_prob_p,               #10 - float - Prob
                                                                                    st.session_state.count_pose_g,      #11 - int - count_pose_g
                                                                                    st.session_state.count_pose,        #12 - int - count_pose
                                                                                    st.session_state.count_rep + 1,     #13 - int - count_rep
                                                                                    st.session_state.count_set + 1,     #14 - int - count_set
                                                                                    None,   #15 - float - right_elbow_angles_pu
                                                                                    None,   #16 - float - right_hip_angles_pu
                                                                                    None,   #17 - float - right_knee_angles_pu
                                                                                    right_shoulder_angle,               #18 - float - right_shoulder_angles_cu
                                                                                    right_hip_angle,                    #19 - float - right_hip_angles_cu
                                                                                    right_knee_angle,                   #20 - float - right_knee_angles_cu
                                                                                    None,    #21 - float - right_shoulder_angles_fp
                                                                                    None,    #22 - float - right_hip_angles_fp
                                                                                    None,    #23 - float - right_ankle_angles_fp
                                                                                    None,    #24 - float - right_hip_angles_fl
                                                                                    None,    #25 - float - right_knee_angles_fl
                                                                                    None,    #26 - float - left_knee_angles_fl
                                                                                    None,    #27 - float - right_shoulder_angles_bd
                                                                                    None,    #28 - float - right_hip_angles_bd
                                                                                    None,    #29 - float - right_knee_angles_bd
                                                                                    None,    #30 - float - left_knee_angles_bd
                                                                                    None,    #31 - float - right_elbow_angles_bd
                                                                                    None,    #32 - float - left_elbow_angles_bd
                                                                                    pose_trainer_cost_min,              #33 - float - pose_trainer_cost_min
                                                                                    pose_trainer_cost_max,              #34 - float - pose_trainer_cost_max
                                                                                    pose_user_cost                      #35 - float - pose_user_cost
                                                    )
                                                    #####################################s######
                                                    last_set = st.session_state.count_set + 1
                                                    last_rep = st.session_state.count_rep + 1
                                                    start = 0

                                                    st.session_state.count_rep += 1
                                                    st.session_state.count_pose = 0
                                                    st.session_state.inicio_rutina = fin_rutina_timestamp

                                                #🟢 CURL UP - ninguna pose
                                                else:
                                                    # ************************ INICIO SISTEMA DE RECOMENDACIONES - CURL UP ************************ #                                                
                                                    if start+1 == 1 or start+1 == 3:                                                    
                                                        if right_shoulder_angle > right_shoulder_angle_in+desv_right_shoulder_angle_in:
                                                            rec_right_shoulder_angle = "Flexiona menos tu hombro derecho"
                                                        elif right_shoulder_angle < right_shoulder_angle_in-desv_right_shoulder_angle_in:
                                                            rec_right_shoulder_angle = "Flexiona más tu hombro derecho"
                                                        else:
                                                            rec_right_shoulder_angle = ""
                                                        if right_hip_angle > right_hip_angle_in+desv_right_hip_angle_in:
                                                            rec_right_hip_angle = ", flexiona más tu cadera"
                                                        elif right_hip_angle < right_hip_angle_in-desv_right_hip_angle_in:
                                                            rec_right_hip_angle = ", flexiona menos tu cadera"
                                                        else:
                                                            rec_right_hip_angle = ""
                                                        if right_knee_angle > right_knee_angle_in+desv_right_knee_angle_in:
                                                            rec_right_knee_angle = ", flexiona más tus rodillas"
                                                        elif right_knee_angle < right_knee_angle_in-desv_right_knee_angle_in:
                                                            rec_right_knee_angle = ", flexiona menos tus rodillas"
                                                        else:
                                                            rec_right_knee_angle = ""
                                                        final_rec = rec_right_shoulder_angle+rec_right_hip_angle+rec_right_knee_angle
                                                        
                                                    elif start+1 == 2:
                                                        if right_shoulder_angle > right_shoulder_angle_in+desv_right_shoulder_angle_in:
                                                            rec_right_shoulder_angle = "Flexiona más tu hombro derecho"
                                                        elif right_shoulder_angle < right_shoulder_angle_in-desv_right_shoulder_angle_in:
                                                            rec_right_shoulder_angle = "Flexiona menos tu hombro derecho"
                                                        else:
                                                            rec_right_shoulder_angle = ""
                                                        if right_hip_angle > right_hip_angle_in+desv_right_hip_angle_in:
                                                            rec_right_hip_angle = ", flexiona más tu cadera"
                                                        elif right_hip_angle < right_hip_angle_in-desv_right_hip_angle_in:
                                                            rec_right_hip_angle = ", flexiona menos tu cadera"
                                                        else:
                                                            rec_right_hip_angle = ""
                                                        if right_knee_angle > right_knee_angle_in+desv_right_knee_angle_in:
                                                            rec_right_knee_angle = ", flexiona más tus rodillas"
                                                        elif right_knee_angle < right_knee_angle_in-desv_right_knee_angle_in:
                                                            rec_right_knee_angle = ", flexiona menos tus rodillas"
                                                        else:
                                                            rec_right_knee_angle = ""
                                                        final_rec = rec_right_shoulder_angle+rec_right_hip_angle+rec_right_knee_angle
                                                        
                                                    if final_rec != "":
                                                        if start+1 == 1:
                                                            try:
                                                                if not speak_rec.is_alive():
                                                                    speak_rec.start()
                                                            except:
                                                                speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                speak_rec.start()
                                                        elif start+1 == 2:
                                                            try:
                                                                if not speak_rec.is_alive():
                                                                    speak_rec.start()
                                                            except:
                                                                try:
                                                                    if not speak_stage1.is_alive():
                                                                        speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                        speak_rec.start()
                                                                except:
                                                                    speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                    speak_rec.start()
                                                        elif start+1 == 3:
                                                            try:
                                                                if not speak_rec.is_alive():
                                                                    speak_rec.start()
                                                            except:
                                                                try:
                                                                    if not speak_stage2.is_alive():
                                                                        speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                        speak_rec.start()
                                                                except:
                                                                    speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                    speak_rec.start()
                                                    # ************************ FIN SISTEMA DE RECOMENDACIONES - CURL UP ************************ #

                                            #🟢 FRONT PLANK - 3 POSES (ejercicio 3 de 5)
                                            elif selected_exercise == "front_plank" and body_language_prob_p > st.session_state.prob_avance_pose:
                                                right_shoulder_angle_in     = UpcSystemAngles.get_valu_angle(df_trainers_angles, start, 'right_shoulder_angles')
                                                right_hip_angle_in          = UpcSystemAngles.get_valu_angle(df_trainers_angles, start, 'right_hip_angles')
                                                right_ankle_angle_in        = UpcSystemAngles.get_valu_angle(df_trainers_angles, start, 'right_ankle_angles')
                                                desv_right_shoulder_angle_in= UpcSystemAngles.get_desv_angle(df_trainers_angles, start, 'right_shoulder_angles')#15
                                                desv_right_hip_angle_in     = UpcSystemAngles.get_desv_angle(df_trainers_angles, start, 'right_hip_angles')#15
                                                desv_right_ankle_angle_in   = UpcSystemAngles.get_desv_angle(df_trainers_angles, start, 'right_ankle_angles')#15

                                                #🟢 FRONT PLANK - pose 1 de 3
                                                if  up == False and\
                                                    down == False and\
                                                    right_shoulder_angle in range(int(right_shoulder_angle_in-desv_right_shoulder_angle_in), int(right_shoulder_angle_in + desv_right_shoulder_angle_in + 1)) and\
                                                    right_hip_angle in range(int(right_hip_angle_in - desv_right_hip_angle_in), int(right_hip_angle_in + desv_right_hip_angle_in + 1)) and\
                                                    right_ankle_angle in range(int(right_ankle_angle_in - desv_right_ankle_angle_in),int(right_ankle_angle_in + desv_right_ankle_angle_in + 1)):
                                                    up = True
                                                    stage = "Abajo"                                                    
                                                    speak_stage1 = threading.Thread(target=speak, args=(stage,))
                                                    speak_stage1.start()                                                    
                                                    ############################################
                                                    st.session_state.count_pose_g += 1
                                                    st.session_state.count_pose = 1
                                                    update_counter_panel('training', st.session_state.count_pose_g, st.session_state.count_pose, st.session_state.count_rep + 1, st.session_state.count_set + 1)
                                                    update_trainer_image(id_exercise, 2)
                                                    ############################################

                                                    fin_rutina_timestamp = get_timestamp_log()
                                                    df_results = util.add_row_df_results(df_results,
                                                                                    id_exercise,                        #1 - str - id_exercise
                                                                                    st.session_state.inicio_rutina,     #2 - str - DateTime_Start
                                                                                    fin_rutina_timestamp,               #3 - str - DateTime_End
                                                                                    st.session_state.n_poses,           #4 - int - n_poses
                                                                                    st.session_state.n_sets,            #5 - int - n_sets
                                                                                    st.session_state.n_reps,            #6 - int - n_reps
                                                                                    st.session_state.total_poses,       #7 - int - total_poses
                                                                                    st.session_state.seconds_rest_time, #8 - int - seconds_rest_time
                                                                                    selected_exercise,                  #9 - str - Class
                                                                                    body_language_prob_p,               #10 - float - Prob
                                                                                    st.session_state.count_pose_g,      #11 - int - count_pose_g
                                                                                    st.session_state.count_pose,        #12 - int - count_pose
                                                                                    st.session_state.count_rep + 1,     #13 - int - count_rep
                                                                                    st.session_state.count_set + 1,     #14 - int - count_set
                                                                                    None,   #15 - float - right_elbow_angles_pu
                                                                                    None,   #16 - float - right_hip_angles_pu
                                                                                    None,   #17 - float - right_knee_angles_pu
                                                                                    None,   #18 - float - right_shoulder_angles_cu
                                                                                    None,   #19 - float - right_hip_angles_cu
                                                                                    None,   #20 - float - right_knee_angles_cu
                                                                                    right_shoulder_angle,               #21 - float - right_shoulder_angles_fp
                                                                                    right_hip_angle,                    #22 - float - right_hip_angles_fp
                                                                                    right_ankle_angle,                  #23 - float - right_ankle_angles_fp
                                                                                    None,    #24 - float - right_hip_angles_fl
                                                                                    None,    #25 - float - right_knee_angles_fl
                                                                                    None,    #26 - float - left_knee_angles_fl
                                                                                    None,    #27 - float - right_shoulder_angles_bd
                                                                                    None,    #28 - float - right_hip_angles_bd
                                                                                    None,    #29 - float - right_knee_angles_bd
                                                                                    None,    #30 - float - left_knee_angles_bd
                                                                                    None,    #31 - float - right_elbow_angles_bd
                                                                                    None,    #32 - float - left_elbow_angles_bd
                                                                                    pose_trainer_cost_min,              #33 - float - pose_trainer_cost_min
                                                                                    pose_trainer_cost_max,              #34 - float - pose_trainer_cost_max
                                                                                    pose_user_cost                      #35 - float - pose_user_cost
                                                    )
                                                    ############################################
                                                    last_set = st.session_state.count_set + 1
                                                    last_rep = st.session_state.count_rep + 1
                                                    start +=1
                                                    st.session_state.inicio_rutina = fin_rutina_timestamp
                                                
                                                #🟢 FRONT PLANK - pose 2 de 3 ⏰ Esta pose requiere mantener la posición unos segundos
                                                elif up == True and\
                                                    down == False and\
                                                    right_shoulder_angle in range(int(right_shoulder_angle_in - desv_right_shoulder_angle_in) , int(right_shoulder_angle_in + desv_right_shoulder_angle_in + 1)) and\
                                                    right_hip_angle in range(int(right_hip_angle_in - desv_right_hip_angle_in), int(right_hip_angle_in + desv_right_hip_angle_in + 1)) and\
                                                    right_ankle_angle in range(int(right_ankle_angle_in - desv_right_ankle_angle_in),int(right_ankle_angle_in + desv_right_ankle_angle_in + 1)):
                                                    
                                                    down = True
                                                    stage = "Arriba"
                                                    speak_stage2 = threading.Thread(target=speak, args=(stage,))
                                                    speak_stage2.start()  
                                                    start +=1
                                                    flagTime = True
                                                    
                                                #🟢 FRONT PLANK - pose 3 de 3
                                                elif up == True and\
                                                    down == True and\
                                                    right_shoulder_angle in range(int(right_shoulder_angle_in - desv_right_shoulder_angle_in) , int(right_shoulder_angle_in + desv_right_shoulder_angle_in + 1)) and\
                                                    right_hip_angle in range(int(right_hip_angle_in - desv_right_hip_angle_in), int(right_hip_angle_in + desv_right_hip_angle_in + 1)) and\
                                                    right_ankle_angle in range(int(right_ankle_angle_in - desv_right_ankle_angle_in),int(right_ankle_angle_in + desv_right_ankle_angle_in + 1)):
                                                    up = False
                                                    down = False
                                                    stage = "Abajo"
                                                    speak_stage3 = threading.Thread(target=speak, args=(stage,))
                                                    speak_stage3.start()                                                    
                                                    ############################################
                                                    st.session_state.count_pose_g += 1
                                                    st.session_state.count_pose = 3
                                                    update_counter_panel('training', st.session_state.count_pose_g, st.session_state.count_pose, st.session_state.count_rep + 1, st.session_state.count_set + 1)
                                                    ############################################

                                                    fin_rutina_timestamp = get_timestamp_log()
                                                    df_results = util.add_row_df_results(df_results,
                                                                                    id_exercise,                        #1 - str - id_exercise
                                                                                    st.session_state.inicio_rutina,     #2 - str - DateTime_Start
                                                                                    fin_rutina_timestamp,               #3 - str - DateTime_End
                                                                                    st.session_state.n_poses,           #4 - int - n_poses
                                                                                    st.session_state.n_sets,            #5 - int - n_sets
                                                                                    st.session_state.n_reps,            #6 - int - n_reps
                                                                                    st.session_state.total_poses,       #7 - int - total_poses
                                                                                    st.session_state.seconds_rest_time, #8 - int - seconds_rest_time
                                                                                    selected_exercise,                  #9 - str - Class
                                                                                    body_language_prob_p,               #10 - float - Prob
                                                                                    st.session_state.count_pose_g,      #11 - int - count_pose_g
                                                                                    st.session_state.count_pose,        #12 - int - count_pose
                                                                                    st.session_state.count_rep + 1,     #13 - int - count_rep
                                                                                    st.session_state.count_set + 1,     #14 - int - count_set
                                                                                    None,   #15 - float - right_elbow_angles_pu
                                                                                    None,   #16 - float - right_hip_angles_pu
                                                                                    None,   #17 - float - right_knee_angles_pu
                                                                                    None,   #18 - float - right_shoulder_angles_cu
                                                                                    None,   #19 - float - right_hip_angles_cu
                                                                                    None,   #20 - float - right_knee_angles_cu
                                                                                    right_shoulder_angle,               #21 - float - right_shoulder_angles_fp
                                                                                    right_hip_angle,                    #22 - float - right_hip_angles_fp
                                                                                    right_ankle_angle,                  #23 - float - right_ankle_angles_fp
                                                                                    None,    #24 - float - right_hip_angles_fl
                                                                                    None,    #25 - float - right_knee_angles_fl
                                                                                    None,    #26 - float - left_knee_angles_fl
                                                                                    None,    #27 - float - right_shoulder_angles_bd
                                                                                    None,    #28 - float - right_hip_angles_bd
                                                                                    None,    #29 - float - right_knee_angles_bd
                                                                                    None,    #30 - float - left_knee_angles_bd
                                                                                    None,    #31 - float - right_elbow_angles_bd
                                                                                    None,    #32 - float - left_elbow_angles_bd
                                                                                    pose_trainer_cost_min,              #33 - float - pose_trainer_cost_min
                                                                                    pose_trainer_cost_max,              #34 - float - pose_trainer_cost_max
                                                                                    pose_user_cost                      #35 - float - pose_user_cost
                                                    )
                                                    ######################################s######
                                                    last_set = st.session_state.count_set + 1
                                                    last_rep = st.session_state.count_rep + 1
                                                    start = 0
                                                    st.session_state.count_rep += 1                                                
                                                    st.session_state.count_pose = 0
                                                    st.session_state.inicio_rutina = fin_rutina_timestamp

                                                #🟢 FRONT PLANK - ninguna pose
                                                else:
                                                    # ************************ INICIO SISTEMA DE RECOMENDACIONES - FRONT PLANK ************************ #
                                                    if start+1 == 1 or start+1 == 3:                                                    
                                                        if right_shoulder_angle > right_shoulder_angle_in+desv_right_shoulder_angle_in:
                                                            rec_right_shoulder_angle = "Flexiona más tu hombro derecho"
                                                        elif right_shoulder_angle < right_shoulder_angle_in-desv_right_shoulder_angle_in:
                                                            rec_right_shoulder_angle = "Flexiona menos tu hombro derecho"
                                                        else:
                                                            rec_right_shoulder_angle = ""
                                                        if right_hip_angle > right_hip_angle_in+desv_right_hip_angle_in:
                                                            rec_right_hip_angle = ", flexiona más tu cadera"
                                                        elif right_hip_angle < right_hip_angle_in-desv_right_hip_angle_in:
                                                            rec_right_hip_angle = ", flexiona menos tu cadera"
                                                        else:
                                                            rec_right_hip_angle = ""
                                                        if right_ankle_angle > right_ankle_angle_in+desv_right_ankle_angle_in:
                                                            rec_right_ankle_angle = ", flexiona más tu tobillo derecho"
                                                        elif right_ankle_angle < right_ankle_angle_in-desv_right_ankle_angle_in:
                                                            rec_right_ankle_angle = ", flexiona menos tu tobillo derecho"
                                                        else:
                                                            rec_right_ankle_angle = ""
                                                        final_rec = rec_right_shoulder_angle+rec_right_hip_angle+rec_right_ankle_angle
                                                        
                                                    elif start+1 == 2:
                                                        if right_shoulder_angle > right_shoulder_angle_in+desv_right_shoulder_angle_in:
                                                            rec_right_shoulder_angle = "Flexiona más tu hombro derecho"
                                                        elif right_shoulder_angle < right_shoulder_angle_in-desv_right_shoulder_angle_in:
                                                            rec_right_shoulder_angle = "Flexiona menos tu hombro derecho"
                                                        else:
                                                            rec_right_shoulder_angle = ""
                                                        if right_hip_angle > right_hip_angle_in+desv_right_hip_angle_in:
                                                            rec_right_hip_angle = ", flexiona menos tu cadera"
                                                        elif right_hip_angle < right_hip_angle_in-desv_right_hip_angle_in:
                                                            rec_right_hip_angle = ", flexiona más tu cadera"
                                                        else:
                                                            rec_right_hip_angle = ""
                                                        if right_ankle_angle > right_ankle_angle_in+desv_right_ankle_angle_in:
                                                            rec_right_ankle_angle = ", flexiona más tu tobillo derecho"
                                                        elif right_ankle_angle < right_ankle_angle_in-desv_right_ankle_angle_in:
                                                            rec_right_ankle_angle = ", flexiona menos tu tobillo derecho"
                                                        else:
                                                            rec_right_ankle_angle = ""
                                                        final_rec = rec_right_shoulder_angle+rec_right_hip_angle+rec_right_ankle_angle
                                                        
                                                    if final_rec != "":
                                                        if start+1 == 1:
                                                            try:
                                                                if not speak_rec.is_alive():
                                                                    speak_rec.start()
                                                            except:
                                                                speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                speak_rec.start()
                                                        elif start+1 == 2:
                                                            try:
                                                                if not speak_rec.is_alive():
                                                                    speak_rec.start()
                                                            except:
                                                                try:
                                                                    if not speak_stage1.is_alive():
                                                                        speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                        speak_rec.start()
                                                                except:
                                                                    speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                    speak_rec.start()
                                                        elif start+1 == 3:
                                                            try:
                                                                if not speak_rec.is_alive():
                                                                    speak_rec.start()
                                                            except:
                                                                try:
                                                                    if not speak_stage2.is_alive():
                                                                        speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                        speak_rec.start()
                                                                except:
                                                                    speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                    speak_rec.start()
                                                    # ************************ FIN SISTEMA DE RECOMENDACIONES - FRONT PLANK ************************ #                                             

                                            #🟢 FORWARD LUNGE - 5 POSES (ejercicio 4 de 5)
                                            elif selected_exercise == "forward_lunge" and body_language_prob_p > st.session_state.prob_avance_pose:
                                                right_hip_angle_in      = UpcSystemAngles.get_valu_angle(df_trainers_angles, start, 'right_hip_angles')
                                                right_knee_angle_in     = UpcSystemAngles.get_valu_angle(df_trainers_angles, start, 'right_knee_angles')
                                                left_knee_angle_in      = UpcSystemAngles.get_valu_angle(df_trainers_angles, start, 'left_knee_angles')
                                                desv_right_hip_angle_in = UpcSystemAngles.get_desv_angle(df_trainers_angles, start, 'right_hip_angles')#25
                                                desv_right_knee_angle_in= UpcSystemAngles.get_desv_angle(df_trainers_angles, start, 'right_knee_angles')#25
                                                desv_left_knee_angle_in = UpcSystemAngles.get_desv_angle(df_trainers_angles, start, 'left_knee_angles')#25

                                                #🟢 FORWARD LUNGE  - pose 1 de 5 
                                                if  up == False and\
                                                    down == False and\
                                                    right_hip_angle in range(int(right_hip_angle_in-desv_right_hip_angle_in), int(right_hip_angle_in + desv_right_hip_angle_in + 1)) and\
                                                    right_knee_angle in range(int(right_knee_angle_in - desv_right_knee_angle_in), int(right_knee_angle_in + desv_right_knee_angle_in + 1)):
                                                    stage = "Arriba"                                                    
                                                    speak_stage1 = threading.Thread(target=speak, args=(stage,))
                                                    speak_stage1.start()                                                    
                                                    ############################################
                                                    st.session_state.count_pose_g += 1
                                                    st.session_state.count_pose = 1
                                                    update_counter_panel('training', st.session_state.count_pose_g, st.session_state.count_pose, st.session_state.count_rep + 1, st.session_state.count_set + 1)
                                                    update_trainer_image(id_exercise, 2)
                                                    ############################################
                                                    fin_rutina_timestamp = get_timestamp_log()
                                                    df_results = util.add_row_df_results(df_results,
                                                                                    id_exercise,                        #1 - str - id_exercise
                                                                                    st.session_state.inicio_rutina,     #2 - str - DateTime_Start
                                                                                    fin_rutina_timestamp,               #3 - str - DateTime_End
                                                                                    st.session_state.n_poses,           #4 - int - n_poses
                                                                                    st.session_state.n_sets,            #5 - int - n_sets
                                                                                    st.session_state.n_reps,            #6 - int - n_reps
                                                                                    st.session_state.total_poses,       #7 - int - total_poses
                                                                                    st.session_state.seconds_rest_time, #8 - int - seconds_rest_time
                                                                                    selected_exercise,                  #9 - str - Class
                                                                                    body_language_prob_p,               #10 - float - Prob
                                                                                    st.session_state.count_pose_g,      #11 - int - count_pose_g
                                                                                    st.session_state.count_pose,        #12 - int - count_pose
                                                                                    st.session_state.count_rep + 1,     #13 - int - count_rep
                                                                                    st.session_state.count_set + 1,     #14 - int - count_set
                                                                                    None,   #15 - float - right_elbow_angles_pu
                                                                                    None,   #16 - float - right_hip_angles_pu
                                                                                    None,   #17 - float - right_knee_angles_pu
                                                                                    None,   #18 - float - right_shoulder_angles_cu
                                                                                    None,   #19 - float - right_hip_angles_cu
                                                                                    None,   #20 - float - right_knee_angles_cu
                                                                                    None,   #21 - float - right_shoulder_angles_fp
                                                                                    None,   #22 - float - right_hip_angles_fp
                                                                                    None,   #23 - float - right_ankle_angles_fp
                                                                                    right_hip_angle,                    #24 - float - right_hip_angles_fl
                                                                                    right_knee_angle,                   #25 - float - right_knee_angles_fl
                                                                                    left_knee_angle,                    #26 - float - left_knee_angles_fl
                                                                                    None,    #27 - float - right_shoulder_angles_bd
                                                                                    None,    #28 - float - right_hip_angles_bd
                                                                                    None,    #29 - float - right_knee_angles_bd
                                                                                    None,    #30 - float - left_knee_angles_bd
                                                                                    None,    #31 - float - right_elbow_angles_bd
                                                                                    None,    #32 - float - left_elbow_angles_bd
                                                                                    pose_trainer_cost_min,              #33 - float - pose_trainer_cost_min
                                                                                    pose_trainer_cost_max,              #34 - float - pose_trainer_cost_max
                                                                                    pose_user_cost                      #35 - float - pose_user_cost
                                                    )
                                                    ########################################### 
                                                    last_set = st.session_state.count_set + 1
                                                    last_rep = st.session_state.count_rep + 1
                                                    up = True
                                                    start +=1
                                                    st.session_state.inicio_rutina = fin_rutina_timestamp
                                                
                                                #🟢 FORWARD LUNGE  - pose 2 de 5 
                                                elif up == True and\
                                                    down == False and\
                                                    right_hip_angle in range(int(right_hip_angle_in-desv_right_hip_angle_in), int(right_hip_angle_in + desv_right_hip_angle_in + 1)) and\
                                                    right_knee_angle in range(int(right_knee_angle_in - desv_right_knee_angle_in), int(right_knee_angle_in + desv_right_knee_angle_in + 1)) and\
                                                    left_knee_angle in range(int(left_knee_angle_in - desv_left_knee_angle_in), int(left_knee_angle_in + desv_left_knee_angle_in + 1)):
                                                    stage = "Abajo"                                                   
                                                    speak_stage2 = threading.Thread(target=speak, args=(stage,))
                                                    speak_stage2.start()
                                                    ############################################
                                                    st.session_state.count_pose_g += 1
                                                    st.session_state.count_pose = 2
                                                    update_counter_panel('training', st.session_state.count_pose_g, st.session_state.count_pose, st.session_state.count_rep + 1, st.session_state.count_set + 1)
                                                    update_trainer_image(id_exercise, 3)
                                                    ############################################
                                                    fin_rutina_timestamp = get_timestamp_log()
                                                    df_results = util.add_row_df_results(df_results,
                                                                                    id_exercise,                        #1 - str - id_exercise
                                                                                    st.session_state.inicio_rutina,     #2 - str - DateTime_Start
                                                                                    fin_rutina_timestamp,               #3 - str - DateTime_End
                                                                                    st.session_state.n_poses,           #4 - int - n_poses
                                                                                    st.session_state.n_sets,            #5 - int - n_sets
                                                                                    st.session_state.n_reps,            #6 - int - n_reps
                                                                                    st.session_state.total_poses,       #7 - int - total_poses
                                                                                    st.session_state.seconds_rest_time, #8 - int - seconds_rest_time
                                                                                    selected_exercise,                  #9 - str - Class
                                                                                    body_language_prob_p,               #10 - float - Prob
                                                                                    st.session_state.count_pose_g,      #11 - int - count_pose_g
                                                                                    st.session_state.count_pose,        #12 - int - count_pose
                                                                                    st.session_state.count_rep + 1,     #13 - int - count_rep
                                                                                    st.session_state.count_set + 1,     #14 - int - count_set
                                                                                    None,   #15 - float - right_elbow_angles_pu
                                                                                    None,   #16 - float - right_hip_angles_pu
                                                                                    None,   #17 - float - right_knee_angles_pu
                                                                                    None,   #18 - float - right_shoulder_angles_cu
                                                                                    None,   #19 - float - right_hip_angles_cu
                                                                                    None,   #20 - float - right_knee_angles_cu
                                                                                    None,   #21 - float - right_shoulder_angles_fp
                                                                                    None,   #22 - float - right_hip_angles_fp
                                                                                    None,   #23 - float - right_ankle_angles_fp
                                                                                    right_hip_angle,                    #24 - float - right_hip_angles_fl
                                                                                    right_knee_angle,                   #25 - float - right_knee_angles_fl
                                                                                    left_knee_angle,                    #26 - float - left_knee_angles_fl
                                                                                    None,    #27 - float - right_shoulder_angles_bd
                                                                                    None,    #28 - float - right_hip_angles_bd
                                                                                    None,    #29 - float - right_knee_angles_bd
                                                                                    None,    #30 - float - left_knee_angles_bd
                                                                                    None,    #31 - float - right_elbow_angles_bd
                                                                                    None,    #32 - float - left_elbow_angles_bd
                                                                                    pose_trainer_cost_min,              #33 - float - pose_trainer_cost_min
                                                                                    pose_trainer_cost_max,              #34 - float - pose_trainer_cost_max
                                                                                    pose_user_cost                      #35 - float - pose_user_cost
                                                    )
                                                    ########################################### 
                                                    last_set = st.session_state.count_set + 1
                                                    last_rep = st.session_state.count_rep + 1
                                                    down = True
                                                    start +=1
                                                    st.session_state.inicio_rutina = fin_rutina_timestamp
                                                
                                                #🟢 FORWARD LUNGE  - pose 3 de 5 
                                                elif up == True and\
                                                    down == True and\
                                                    right_hip_angle in range(int(right_hip_angle_in-desv_right_hip_angle_in), int(right_hip_angle_in + desv_right_hip_angle_in + 1)) and\
                                                    right_knee_angle in range(int(right_knee_angle_in - desv_right_knee_angle_in), int(right_knee_angle_in + desv_right_knee_angle_in + 1)):
                                                    stage = "Arriba"
                                                    speak_stage3 = threading.Thread(target=speak, args=(stage,))
                                                    speak_stage3.start()
                                                    ############################################
                                                    st.session_state.count_pose_g += 1
                                                    st.session_state.count_pose = 3
                                                    update_counter_panel('training', st.session_state.count_pose_g, st.session_state.count_pose, st.session_state.count_rep + 1, st.session_state.count_set + 1)
                                                    update_trainer_image(id_exercise, 4)
                                                    ############################################

                                                    fin_rutina_timestamp = get_timestamp_log()
                                                    df_results = util.add_row_df_results(df_results,
                                                                                    id_exercise,                        #1 - str - id_exercise
                                                                                    st.session_state.inicio_rutina,     #2 - str - DateTime_Start
                                                                                    fin_rutina_timestamp,               #3 - str - DateTime_End
                                                                                    st.session_state.n_poses,           #4 - int - n_poses
                                                                                    st.session_state.n_sets,            #5 - int - n_sets
                                                                                    st.session_state.n_reps,            #6 - int - n_reps
                                                                                    st.session_state.total_poses,       #7 - int - total_poses
                                                                                    st.session_state.seconds_rest_time, #8 - int - seconds_rest_time
                                                                                    selected_exercise,                  #9 - str - Class
                                                                                    body_language_prob_p,               #10 - float - Prob
                                                                                    st.session_state.count_pose_g,      #11 - int - count_pose_g
                                                                                    st.session_state.count_pose,        #12 - int - count_pose
                                                                                    st.session_state.count_rep + 1,     #13 - int - count_rep
                                                                                    st.session_state.count_set + 1,     #14 - int - count_set
                                                                                    None,   #15 - float - right_elbow_angles_pu
                                                                                    None,   #16 - float - right_hip_angles_pu
                                                                                    None,   #17 - float - right_knee_angles_pu
                                                                                    None,   #18 - float - right_shoulder_angles_cu
                                                                                    None,   #19 - float - right_hip_angles_cu
                                                                                    None,   #20 - float - right_knee_angles_cu
                                                                                    None,   #21 - float - right_shoulder_angles_fp
                                                                                    None,   #22 - float - right_hip_angles_fp
                                                                                    None,   #23 - float - right_ankle_angles_fp
                                                                                    right_hip_angle,                    #24 - float - right_hip_angles_fl
                                                                                    right_knee_angle,                   #25 - float - right_knee_angles_fl
                                                                                    left_knee_angle,                    #26 - float - left_knee_angles_fl
                                                                                    None,    #27 - float - right_shoulder_angles_bd
                                                                                    None,    #28 - float - right_hip_angles_bd
                                                                                    None,    #29 - float - right_knee_angles_bd
                                                                                    None,    #30 - float - left_knee_angles_bd
                                                                                    None,    #31 - float - right_elbow_angles_bd
                                                                                    None,    #32 - float - left_elbow_angles_bd
                                                                                    pose_trainer_cost_min,              #33 - float - pose_trainer_cost_min
                                                                                    pose_trainer_cost_max,              #34 - float - pose_trainer_cost_max
                                                                                    pose_user_cost                      #35 - float - pose_user_cost
                                                    )
                                                    #####################################s######
                                                    last_set = st.session_state.count_set + 1
                                                    last_rep = st.session_state.count_rep + 1
                                                    up = False
                                                    down = True
                                                    start +=1
                                                    st.session_state.inicio_rutina = fin_rutina_timestamp
                                                    #####################################s######
                                                
                                                #🟢 FORWARD LUNGE  - pose 4 de 5 
                                                elif up == False and\
                                                    down == True and\
                                                    mid == False and\
                                                    right_hip_angle in range(int(right_hip_angle_in-desv_right_hip_angle_in), int(right_hip_angle_in + desv_right_hip_angle_in + 1)) and\
                                                    right_knee_angle in range(int(right_knee_angle_in - desv_right_knee_angle_in), int(right_knee_angle_in + desv_right_knee_angle_in + 1)) and\
                                                    left_knee_angle in range(int(left_knee_angle_in - desv_left_knee_angle_in), int(left_knee_angle_in + desv_left_knee_angle_in + 1)):
                                                    stage = "Abajo"
                                                    speak_stage4 = threading.Thread(target=speak, args=(stage,))
                                                    speak_stage4.start()
                                                    ############################################
                                                    st.session_state.count_pose_g += 1
                                                    st.session_state.count_pose = 4
                                                    update_counter_panel('training', st.session_state.count_pose_g, st.session_state.count_pose, st.session_state.count_rep + 1, st.session_state.count_set + 1)
                                                    update_trainer_image(id_exercise, 5)
                                                    ############################################
                                                    fin_rutina_timestamp = get_timestamp_log()
                                                    df_results = util.add_row_df_results(df_results,
                                                                                    id_exercise,                        #1 - str - id_exercise
                                                                                    st.session_state.inicio_rutina,     #2 - str - DateTime_Start
                                                                                    fin_rutina_timestamp,               #3 - str - DateTime_End
                                                                                    st.session_state.n_poses,           #4 - int - n_poses
                                                                                    st.session_state.n_sets,            #5 - int - n_sets
                                                                                    st.session_state.n_reps,            #6 - int - n_reps
                                                                                    st.session_state.total_poses,       #7 - int - total_poses
                                                                                    st.session_state.seconds_rest_time, #8 - int - seconds_rest_time
                                                                                    selected_exercise,                  #9 - str - Class
                                                                                    body_language_prob_p,               #10 - float - Prob
                                                                                    st.session_state.count_pose_g,      #11 - int - count_pose_g
                                                                                    st.session_state.count_pose,        #12 - int - count_pose
                                                                                    st.session_state.count_rep + 1,     #13 - int - count_rep
                                                                                    st.session_state.count_set + 1,     #14 - int - count_set
                                                                                    None,   #15 - float - right_elbow_angles_pu
                                                                                    None,   #16 - float - right_hip_angles_pu
                                                                                    None,   #17 - float - right_knee_angles_pu
                                                                                    None,   #18 - float - right_shoulder_angles_cu
                                                                                    None,   #19 - float - right_hip_angles_cu
                                                                                    None,   #20 - float - right_knee_angles_cu
                                                                                    None,   #21 - float - right_shoulder_angles_fp
                                                                                    None,   #22 - float - right_hip_angles_fp
                                                                                    None,   #23 - float - right_ankle_angles_fp
                                                                                    right_hip_angle,                    #24 - float - right_hip_angles_fl
                                                                                    right_knee_angle,                   #25 - float - right_knee_angles_fl
                                                                                    left_knee_angle,                    #26 - float - left_knee_angles_fl
                                                                                    None,    #27 - float - right_shoulder_angles_bd
                                                                                    None,    #28 - float - right_hip_angles_bd
                                                                                    None,    #29 - float - right_knee_angles_bd
                                                                                    None,    #30 - float - left_knee_angles_bd
                                                                                    None,    #31 - float - right_elbow_angles_bd
                                                                                    None,    #32 - float - left_elbow_angles_bd
                                                                                    pose_trainer_cost_min,              #33 - float - pose_trainer_cost_min
                                                                                    pose_trainer_cost_max,              #34 - float - pose_trainer_cost_max
                                                                                    pose_user_cost                      #35 - float - pose_user_cost
                                                    )
                                                    ########################################### 
                                                    last_set = st.session_state.count_set + 1
                                                    last_rep = st.session_state.count_rep + 1
                                                    mid = True
                                                    start +=1
                                                    st.session_state.inicio_rutina = fin_rutina_timestamp
                                                #🟢 FORWARD LUNGE  - pose 5 de 5 
                                                elif up == False and\
                                                    down == True and\
                                                    mid == True and\
                                                    right_hip_angle in range(int(right_hip_angle_in-desv_right_hip_angle_in), int(right_hip_angle_in + desv_right_hip_angle_in + 1)) and\
                                                    right_knee_angle in range(int(right_knee_angle_in - desv_right_knee_angle_in), int(right_knee_angle_in + desv_right_knee_angle_in + 1)):
                                                    stage = "Arriba"
                                                    speak_stage5 = threading.Thread(target=speak, args=(stage,))
                                                    speak_stage5.start()
                                                    ############################################
                                                    st.session_state.count_pose_g += 1
                                                    st.session_state.count_pose = 5
                                                    update_counter_panel('training', st.session_state.count_pose_g, st.session_state.count_pose, st.session_state.count_rep + 1, st.session_state.count_set + 1)
                                                    ############################################
                                                    fin_rutina_timestamp = get_timestamp_log()
                                                    df_results = util.add_row_df_results(df_results,
                                                                                    id_exercise,                        #1 - str - id_exercise
                                                                                    st.session_state.inicio_rutina,     #2 - str - DateTime_Start
                                                                                    fin_rutina_timestamp,               #3 - str - DateTime_End
                                                                                    st.session_state.n_poses,           #4 - int - n_poses
                                                                                    st.session_state.n_sets,            #5 - int - n_sets
                                                                                    st.session_state.n_reps,            #6 - int - n_reps
                                                                                    st.session_state.total_poses,       #7 - int - total_poses
                                                                                    st.session_state.seconds_rest_time, #8 - int - seconds_rest_time
                                                                                    selected_exercise,                  #9 - str - Class
                                                                                    body_language_prob_p,               #10 - float - Prob
                                                                                    st.session_state.count_pose_g,      #11 - int - count_pose_g
                                                                                    st.session_state.count_pose,        #12 - int - count_pose
                                                                                    st.session_state.count_rep + 1,     #13 - int - count_rep
                                                                                    st.session_state.count_set + 1,     #14 - int - count_set
                                                                                    None,   #15 - float - right_elbow_angles_pu
                                                                                    None,   #16 - float - right_hip_angles_pu
                                                                                    None,   #17 - float - right_knee_angles_pu
                                                                                    None,   #18 - float - right_shoulder_angles_cu
                                                                                    None,   #19 - float - right_hip_angles_cu
                                                                                    None,   #20 - float - right_knee_angles_cu
                                                                                    None,   #21 - float - right_shoulder_angles_fp
                                                                                    None,   #22 - float - right_hip_angles_fp
                                                                                    None,   #23 - float - right_ankle_angles_fp
                                                                                    right_hip_angle,                    #24 - float - right_hip_angles_fl
                                                                                    right_knee_angle,                   #25 - float - right_knee_angles_fl
                                                                                    left_knee_angle,                    #26 - float - left_knee_angles_fl
                                                                                    None,    #27 - float - right_shoulder_angles_bd
                                                                                    None,    #28 - float - right_hip_angles_bd
                                                                                    None,    #29 - float - right_knee_angles_bd
                                                                                    None,    #30 - float - left_knee_angles_bd
                                                                                    None,    #31 - float - right_elbow_angles_bd
                                                                                    None,    #32 - float - left_elbow_angles_bd
                                                                                    pose_trainer_cost_min,              #33 - float - pose_trainer_cost_min
                                                                                    pose_trainer_cost_max,              #34 - float - pose_trainer_cost_max
                                                                                    pose_user_cost                      #35 - float - pose_user_cost
                                                    )
                                                    #####################################s######
                                                    last_set = st.session_state.count_set + 1
                                                    last_rep = st.session_state.count_rep + 1
                                                    up = False
                                                    down = False
                                                    mid = False
                                                    start = 0
                                                    st.session_state.count_rep += 1
                                                    st.session_state.count_pose = 0
                                                    st.session_state.inicio_rutina = fin_rutina_timestamp                                                

                                                #🟢 FORWARD LUNGE - ninguna pose
                                                else:
                                                    # ************************ INICIO SISTEMA DE RECOMENDACIONES - FORWARD LUNGE ************************ #
                                                    if st.session_state.count_pose+1 == 1 or st.session_state.count_pose+1 == 3 or st.session_state.count_pose+1 == 5:
                                                        if (right_hip_angle > right_hip_angle_in+desv_right_hip_angle_in) or\
                                                            (right_hip_angle < right_hip_angle_in-desv_right_hip_angle_in):
                                                            rec_right_hip_angle = "Mantén tu cadera recta"
                                                        else:
                                                            rec_right_hip_angle = ""
                                                        if (right_knee_angle > right_knee_angle_in+desv_right_knee_angle_in) or\
                                                            (right_knee_angle < right_knee_angle_in-desv_right_knee_angle_in):
                                                            if rec_right_hip_angle != "":
                                                                rec_right_hip_angle = "Mantén tu cadera"
                                                                rec_right_knee_angle = " y tu rodilla derecha recta"
                                                            else:
                                                                rec_right_knee_angle = "Mantén tu rodilla derecha recta"
                                                        else:
                                                            rec_right_knee_angle = ""
                                                        final_rec = rec_right_hip_angle+rec_right_knee_angle
                                                        
                                                    elif st.session_state.count_pose+1 == 2:
                                                        if (right_hip_angle > right_hip_angle_in+desv_right_hip_angle_in) or\
                                                            (right_hip_angle < right_hip_angle_in-desv_right_hip_angle_in):
                                                            rec_right_hip_angle = "Mantén tu cadera recta"
                                                        else:
                                                            rec_right_hip_angle = ""
                                                        if right_knee_angle > right_knee_angle_in+desv_right_knee_angle_in:                                                                
                                                            rec_right_knee_angle = ", flexiona más tu rodilla derecha manteniendola hacia Abajo"
                                                        elif right_knee_angle < right_knee_angle_in-desv_right_knee_angle_in:            
                                                            rec_right_knee_angle = ", flexiona menos tu rodilla derecha manteniendola hacia Abajo"
                                                        else:
                                                            rec_right_knee_angle = ""
                                                        if left_knee_angle > left_knee_angle_in+desv_left_knee_angle_in:
                                                            rec_left_knee_angle = ", flexiona más tu rodilla izquierda manteniendola frente a tu cadera"
                                                        elif left_knee_angle < left_knee_angle_in-desv_left_knee_angle_in:
                                                            rec_left_knee_angle = ", flexiona menos tu rodilla izquierda manteniendola frente a tu cadera"
                                                        else:
                                                            rec_left_knee_angle = ""                                                        
                                                        final_rec = rec_right_hip_angle+rec_right_knee_angle+rec_left_knee_angle
                                                        
                                                    elif st.session_state.count_pose+1 == 4:
                                                        if (right_hip_angle > right_hip_angle_in+desv_right_hip_angle_in):                                                            
                                                            rec_right_hip_angle = "Flexiona más tu cadera"
                                                        elif (right_hip_angle < right_hip_angle_in-desv_right_hip_angle_in):
                                                            rec_right_hip_angle = "Flexiona menos tu cadera"
                                                        else:
                                                            rec_right_hip_angle = ""
                                                        if right_knee_angle > right_knee_angle_in+desv_right_knee_angle_in:
                                                            rec_right_knee_angle = ", flexiona más tu rodilla derecha manteniendola frente a tu cadera"
                                                        elif right_knee_angle < right_knee_angle_in-desv_right_knee_angle_in:
                                                            rec_right_knee_angle = ", flexiona menos tu rodilla derecha manteniendola frente a tu cadera"                                                        
                                                        else:
                                                            rec_right_knee_angle = ""
                                                        if left_knee_angle > left_knee_angle_in+desv_left_knee_angle_in:
                                                            rec_left_knee_angle = ", flexiona más tu rodilla izquierda manteniendola hacia Abajo"
                                                        elif left_knee_angle < left_knee_angle_in-desv_left_knee_angle_in:
                                                            rec_left_knee_angle = ", flexiona menos tu rodilla izquierda manteniendola hacia Abajo"
                                                        else:
                                                            rec_left_knee_angle = ""
                                                        final_rec = rec_right_hip_angle+rec_right_knee_angle+rec_left_knee_angle
                                                        
                                                    if final_rec != "":
                                                        if st.session_state.count_pose+1 == 1:
                                                            try:
                                                                if not speak_rec.is_alive():
                                                                    speak_rec.start()
                                                            except:
                                                                speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                speak_rec.start()
                                                        elif st.session_state.count_pose+1 == 2:
                                                            try:
                                                                if not speak_rec.is_alive():
                                                                    speak_rec.start()
                                                            except:
                                                                try:
                                                                    if not speak_stage1.is_alive():
                                                                        speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                        speak_rec.start()
                                                                except:
                                                                    speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                    speak_rec.start()
                                                        elif st.session_state.count_pose+1 == 3:
                                                            try:
                                                                if not speak_rec.is_alive():
                                                                    speak_rec.start()
                                                            except:
                                                                try:
                                                                    if not speak_stage2.is_alive():
                                                                        speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                        speak_rec.start()
                                                                except:
                                                                    speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                    speak_rec.start()
                                                        elif st.session_state.count_pose+1 == 4:
                                                            try:
                                                                if not speak_rec.is_alive():
                                                                    speak_rec.start()
                                                            except:
                                                                try:
                                                                    if not speak_stage3.is_alive():
                                                                        speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                        speak_rec.start()
                                                                except:
                                                                    speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                    speak_rec.start()
                                                        elif st.session_state.count_pose+1 == 5:
                                                            try:
                                                                if not speak_rec.is_alive():
                                                                    speak_rec.start()
                                                            except:
                                                                try:
                                                                    if not speak_stage4.is_alive():
                                                                        speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                        speak_rec.start()
                                                                except:
                                                                    speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                    speak_rec.start()
                                                    # ************************ FIN SISTEMA DE RECOMENDACIONES - FORWARD LUNGE ************************ #

                                            #🟢 BIRD DOG - 5 POSES (ejercicio 5 de 5)
                                            elif selected_exercise == "bird_dog" and body_language_prob_p > st.session_state.prob_avance_pose:
                                                right_shoulder_angle_in = UpcSystemAngles.get_valu_angle(df_trainers_angles, start, 'right_shoulder_angles')
                                                right_hip_angle_in      = UpcSystemAngles.get_valu_angle(df_trainers_angles, start, 'right_hip_angles')
                                                right_knee_angle_in     = UpcSystemAngles.get_valu_angle(df_trainers_angles, start, 'right_knee_angles')
                                                left_knee_angle_in      = UpcSystemAngles.get_valu_angle(df_trainers_angles, start, 'left_knee_angles')
                                                right_elbow_angle_in    = UpcSystemAngles.get_valu_angle(df_trainers_angles, start, 'right_elbow_angles')
                                                left_elbow_angle_in     = UpcSystemAngles.get_valu_angle(df_trainers_angles, start, 'left_elbow_angles')
                                                
                                                desv_right_shoulder_angle_in= UpcSystemAngles.get_desv_angle(df_trainers_angles, start, 'right_shoulder_angles')#25                                                
                                                desv_right_hip_angle_in     = UpcSystemAngles.get_desv_angle(df_trainers_angles, start, 'right_hip_angles')#25                                                
                                                desv_right_knee_angle_in    = UpcSystemAngles.get_desv_angle(df_trainers_angles, start, 'right_knee_angles')#25                                                
                                                desv_left_knee_angle_in     = UpcSystemAngles.get_desv_angle(df_trainers_angles, start, 'left_knee_angles')#25                                                
                                                desv_right_elbow_angle_in   = UpcSystemAngles.get_desv_angle(df_trainers_angles, start, 'right_elbow_angles')#25                                                
                                                desv_left_elbow_angle_in    = UpcSystemAngles.get_desv_angle(df_trainers_angles, start, 'left_elbow_angles')#25
                                                
                                                #🟢  BIRD DOG - pose 1 de 5
                                                if  up == False and\
                                                    down == False and\
                                                    right_hip_angle in range(int(right_hip_angle_in-desv_right_hip_angle_in), int(right_hip_angle_in + desv_right_hip_angle_in + 1)) and\
                                                    right_knee_angle in range(int(right_knee_angle_in - desv_right_knee_angle_in), int(right_knee_angle_in + desv_right_knee_angle_in + 1)) and\
                                                    right_shoulder_angle in range(int(right_shoulder_angle_in - desv_right_shoulder_angle_in), int(right_shoulder_angle_in + desv_right_shoulder_angle_in + 1)):
                                                    
                                                    up = True
                                                    stage = "Abajo"
                                                    speak_stage1 = threading.Thread(target=speak, args=(stage,))
                                                    speak_stage1.start()
                                                    ############################################
                                                    st.session_state.count_pose_g += 1
                                                    st.session_state.count_pose = 1
                                                    update_counter_panel('training', st.session_state.count_pose_g, st.session_state.count_pose, st.session_state.count_rep + 1, st.session_state.count_set + 1)
                                                    update_trainer_image(id_exercise, 2)
                                                    ############################################
                                                    fin_rutina_timestamp = get_timestamp_log()
                                                    df_results = util.add_row_df_results(df_results,
                                                                                    id_exercise,                        #1 - str - id_exercise
                                                                                    st.session_state.inicio_rutina,     #2 - str - DateTime_Start
                                                                                    fin_rutina_timestamp,               #3 - str - DateTime_End
                                                                                    st.session_state.n_poses,           #4 - int - n_poses
                                                                                    st.session_state.n_sets,            #5 - int - n_sets
                                                                                    st.session_state.n_reps,            #6 - int - n_reps
                                                                                    st.session_state.total_poses,       #7 - int - total_poses
                                                                                    st.session_state.seconds_rest_time, #8 - int - seconds_rest_time
                                                                                    selected_exercise,                  #9 - str - Class
                                                                                    body_language_prob_p,               #10 - float - Prob
                                                                                    st.session_state.count_pose_g,      #11 - int - count_pose_g
                                                                                    st.session_state.count_pose,        #12 - int - count_pose
                                                                                    st.session_state.count_rep + 1,     #13 - int - count_rep
                                                                                    st.session_state.count_set + 1,     #14 - int - count_set
                                                                                    None,   #15 - float - right_elbow_angles_pu
                                                                                    None,   #16 - float - right_hip_angles_pu
                                                                                    None,   #17 - float - right_knee_angles_pu
                                                                                    None,   #18 - float - right_shoulder_angles_cu
                                                                                    None,   #19 - float - right_hip_angles_cu
                                                                                    None,   #20 - float - right_knee_angles_cu
                                                                                    None,   #21 - float - right_shoulder_angles_fp
                                                                                    None,   #22 - float - right_hip_angles_fp
                                                                                    None,   #23 - float - right_ankle_angles_fp
                                                                                    None,   #24 - float - right_hip_angles_fl
                                                                                    None,   #25 - float - right_knee_angles_fl
                                                                                    None,   #26 - float - left_knee_angles_fl
                                                                                    right_shoulder_angle,               #27 - float - right_shoulder_angles_bd
                                                                                    right_hip_angle,                    #28 - float - right_hip_angles_bd
                                                                                    right_knee_angle,                   #29 - float - right_knee_angles_bd
                                                                                    left_knee_angle,                    #30 - float - left_knee_angles_bd
                                                                                    right_elbow_angle,                  #31 - float - right_elbow_angles_bd
                                                                                    left_elbow_angle,                   #32 - float - left_elbow_angles_bd
                                                                                    pose_trainer_cost_min,              #33 - float - pose_trainer_cost_min
                                                                                    pose_trainer_cost_max,              #34 - float - pose_trainer_cost_max
                                                                                    pose_user_cost                      #35 - float - pose_user_cost
                                                    )
                                                    ############################################ 
                                                    last_set = st.session_state.count_set + 1
                                                    last_rep = st.session_state.count_rep + 1
                                                    start +=1                                                    
                                                    st.session_state.inicio_rutina = fin_rutina_timestamp

                                                #🟢  BIRD DOG - pose 2 de 5
                                                elif up == True and\
                                                    down == False and\
                                                    left_knee_angle in range(int(left_knee_angle_in-desv_left_knee_angle_in), int(left_knee_angle_in + desv_left_knee_angle_in + 1)) and\
                                                    right_elbow_angle in range(int(right_elbow_angle_in - desv_right_elbow_angle_in), int(right_elbow_angle_in + desv_right_knee_angle_in + 1)):
                                                    down = True
                                                    stage = "Arriba"
                                                    speak_stage2 = threading.Thread(target=speak, args=(stage,))
                                                    speak_stage2.start()
                                                    ############################################
                                                    st.session_state.count_pose_g += 1
                                                    st.session_state.count_pose = 2
                                                    update_counter_panel('training', st.session_state.count_pose_g, st.session_state.count_pose, st.session_state.count_rep + 1, st.session_state.count_set + 1)
                                                    update_trainer_image(id_exercise, 3)
                                                    ############################################
                                                    fin_rutina_timestamp = get_timestamp_log()
                                                    df_results = util.add_row_df_results(df_results,
                                                                                    id_exercise,                        #1 - str - id_exercise
                                                                                    st.session_state.inicio_rutina,     #2 - str - DateTime_Start
                                                                                    fin_rutina_timestamp,               #3 - str - DateTime_End
                                                                                    st.session_state.n_poses,           #4 - int - n_poses
                                                                                    st.session_state.n_sets,            #5 - int - n_sets
                                                                                    st.session_state.n_reps,            #6 - int - n_reps
                                                                                    st.session_state.total_poses,       #7 - int - total_poses
                                                                                    st.session_state.seconds_rest_time, #8 - int - seconds_rest_time
                                                                                    selected_exercise,                  #9 - str - Class
                                                                                    body_language_prob_p,               #10 - float - Prob
                                                                                    st.session_state.count_pose_g,      #11 - int - count_pose_g
                                                                                    st.session_state.count_pose,        #12 - int - count_pose
                                                                                    st.session_state.count_rep + 1,     #13 - int - count_rep
                                                                                    st.session_state.count_set + 1,     #14 - int - count_set
                                                                                    None,   #15 - float - right_elbow_angles_pu
                                                                                    None,   #16 - float - right_hip_angles_pu
                                                                                    None,   #17 - float - right_knee_angles_pu
                                                                                    None,   #18 - float - right_shoulder_angles_cu
                                                                                    None,   #19 - float - right_hip_angles_cu
                                                                                    None,   #20 - float - right_knee_angles_cu
                                                                                    None,   #21 - float - right_shoulder_angles_fp
                                                                                    None,   #22 - float - right_hip_angles_fp
                                                                                    None,   #23 - float - right_ankle_angles_fp
                                                                                    None,   #24 - float - right_hip_angles_fl
                                                                                    None,   #25 - float - right_knee_angles_fl
                                                                                    None,   #26 - float - left_knee_angles_fl
                                                                                    right_shoulder_angle,               #27 - float - right_shoulder_angles_bd
                                                                                    right_hip_angle,                    #28 - float - right_hip_angles_bd
                                                                                    right_knee_angle,                   #29 - float - right_knee_angles_bd
                                                                                    left_knee_angle,                    #30 - float - left_knee_angles_bd
                                                                                    right_elbow_angle,                  #31 - float - right_elbow_angles_bd
                                                                                    left_elbow_angle,                   #32 - float - left_elbow_angles_bd
                                                                                    pose_trainer_cost_min,              #33 - float - pose_trainer_cost_min
                                                                                    pose_trainer_cost_max,              #34 - float - pose_trainer_cost_max
                                                                                    pose_user_cost                      #35 - float - pose_user_cost
                                                    )
                                                    ############################################ 
                                                    last_set = st.session_state.count_set + 1
                                                    last_rep = st.session_state.count_rep + 1
                                                    start +=1
                                                    st.session_state.inicio_rutina = fin_rutina_timestamp                                            
                                                
                                                #🟢  BIRD DOG - pose 3 de 5
                                                elif up == True and\
                                                    down == True and\
                                                    right_hip_angle in range(int(right_hip_angle_in-desv_right_hip_angle_in), int(right_hip_angle_in + desv_right_hip_angle_in + 1)) and\
                                                    right_knee_angle in range(int(right_knee_angle_in - desv_right_knee_angle_in), int(right_knee_angle_in + desv_right_knee_angle_in + 1)) and\
                                                    right_shoulder_angle in range(int(right_shoulder_angle_in - desv_right_shoulder_angle_in), int(right_shoulder_angle_in + desv_right_shoulder_angle_in + 1)):
                                                    up = False
                                                    down = True
                                                    stage = "Abajo"
                                                    speak_stage3 = threading.Thread(target=speak, args=(stage,))
                                                    speak_stage3.start()
                                                    ############################################
                                                    st.session_state.count_pose_g += 1
                                                    st.session_state.count_pose = 3
                                                    update_counter_panel('training', st.session_state.count_pose_g, st.session_state.count_pose, st.session_state.count_rep + 1, st.session_state.count_set + 1)
                                                    update_trainer_image(id_exercise, 4)
                                                    ############################################
                                                    fin_rutina_timestamp = get_timestamp_log()
                                                    df_results = util.add_row_df_results(df_results,
                                                                                    id_exercise,                        #1 - str - id_exercise
                                                                                    st.session_state.inicio_rutina,     #2 - str - DateTime_Start
                                                                                    fin_rutina_timestamp,               #3 - str - DateTime_End
                                                                                    st.session_state.n_poses,           #4 - int - n_poses
                                                                                    st.session_state.n_sets,            #5 - int - n_sets
                                                                                    st.session_state.n_reps,            #6 - int - n_reps
                                                                                    st.session_state.total_poses,       #7 - int - total_poses
                                                                                    st.session_state.seconds_rest_time, #8 - int - seconds_rest_time
                                                                                    selected_exercise,                  #9 - str - Class
                                                                                    body_language_prob_p,               #10 - float - Prob
                                                                                    st.session_state.count_pose_g,      #11 - int - count_pose_g
                                                                                    st.session_state.count_pose,        #12 - int - count_pose
                                                                                    st.session_state.count_rep + 1,     #13 - int - count_rep
                                                                                    st.session_state.count_set + 1,     #14 - int - count_set
                                                                                    None,   #15 - float - right_elbow_angles_pu
                                                                                    None,   #16 - float - right_hip_angles_pu
                                                                                    None,   #17 - float - right_knee_angles_pu
                                                                                    None,   #18 - float - right_shoulder_angles_cu
                                                                                    None,   #19 - float - right_hip_angles_cu
                                                                                    None,   #20 - float - right_knee_angles_cu
                                                                                    None,   #21 - float - right_shoulder_angles_fp
                                                                                    None,   #22 - float - right_hip_angles_fp
                                                                                    None,   #23 - float - right_ankle_angles_fp
                                                                                    None,   #24 - float - right_hip_angles_fl
                                                                                    None,   #25 - float - right_knee_angles_fl
                                                                                    None,   #26 - float - left_knee_angles_fl
                                                                                    right_shoulder_angle,               #27 - float - right_shoulder_angles_bd
                                                                                    right_hip_angle,                    #28 - float - right_hip_angles_bd
                                                                                    right_knee_angle,                   #29 - float - right_knee_angles_bd
                                                                                    left_knee_angle,                    #30 - float - left_knee_angles_bd
                                                                                    right_elbow_angle,                  #31 - float - right_elbow_angles_bd
                                                                                    left_elbow_angle,                   #32 - float - left_elbow_angles_bd
                                                                                    pose_trainer_cost_min,              #33 - float - pose_trainer_cost_min
                                                                                    pose_trainer_cost_max,              #34 - float - pose_trainer_cost_max
                                                                                    pose_user_cost                      #35 - float - pose_user_cost
                                                    )
                                                    ######################################s######
                                                    last_set = st.session_state.count_set + 1
                                                    last_rep = st.session_state.count_rep + 1
                                                    start +=1
                                                    st.session_state.inicio_rutina = fin_rutina_timestamp 
                                                #🟢  BIRD DOG - pose 4 de 5
                                                elif up == False and\
                                                    down == True and\
                                                    mid == False and\
                                                    right_knee_angle in range(int(right_knee_angle_in-desv_right_knee_angle_in), int(right_knee_angle_in + desv_right_knee_angle_in + 1)) and\
                                                    left_elbow_angle in range(int(left_elbow_angle_in - desv_left_elbow_angle_in), int(left_elbow_angle_in + desv_left_elbow_angle_in + 1)):
                                                    
                                                    mid = True
                                                    stage = "Arriba"
                                                    speak_stage4 = threading.Thread(target=speak, args=(stage,))
                                                    speak_stage4.start()
                                                    ############################################
                                                    st.session_state.count_pose_g += 1
                                                    st.session_state.count_pose = 4
                                                    update_counter_panel('training', st.session_state.count_pose_g, st.session_state.count_pose, st.session_state.count_rep + 1, st.session_state.count_set + 1)
                                                    update_trainer_image(id_exercise, 5)
                                                    ############################################ 
                                                    fin_rutina_timestamp = get_timestamp_log()
                                                    df_results = util.add_row_df_results(df_results,
                                                                                    id_exercise,                        #1 - str - id_exercise
                                                                                    st.session_state.inicio_rutina,     #2 - str - DateTime_Start
                                                                                    fin_rutina_timestamp,               #3 - str - DateTime_End
                                                                                    st.session_state.n_poses,           #4 - int - n_poses
                                                                                    st.session_state.n_sets,            #5 - int - n_sets
                                                                                    st.session_state.n_reps,            #6 - int - n_reps
                                                                                    st.session_state.total_poses,       #7 - int - total_poses
                                                                                    st.session_state.seconds_rest_time, #8 - int - seconds_rest_time
                                                                                    selected_exercise,                  #9 - str - Class
                                                                                    body_language_prob_p,               #10 - float - Prob
                                                                                    st.session_state.count_pose_g,      #11 - int - count_pose_g
                                                                                    st.session_state.count_pose,        #12 - int - count_pose
                                                                                    st.session_state.count_rep + 1,     #13 - int - count_rep
                                                                                    st.session_state.count_set + 1,     #14 - int - count_set
                                                                                    None,   #15 - float - right_elbow_angles_pu
                                                                                    None,   #16- float - right_hip_angles_pu
                                                                                    None,   #17 - float - right_knee_angles_pu
                                                                                    None,   #18 - float - right_shoulder_angles_cu
                                                                                    None,   #19 - float - right_hip_angles_cu
                                                                                    None,   #20 - float - right_knee_angles_cu
                                                                                    None,   #21 - float - right_shoulder_angles_fp
                                                                                    None,   #22 - float - right_hip_angles_fp
                                                                                    None,   #23 - float - right_ankle_angles_fp
                                                                                    None,   #24 - float - right_hip_angles_fl
                                                                                    None,   #25 - float - right_knee_angles_fl
                                                                                    None,   #26 - float - left_knee_angles_fl
                                                                                    right_shoulder_angle,               #27 - float - right_shoulder_angles_bd
                                                                                    right_hip_angle,                    #28 - float - right_hip_angles_bd
                                                                                    right_knee_angle,                   #29 - float - right_knee_angles_bd
                                                                                    left_knee_angle,                    #30 - float - left_knee_angles_bd
                                                                                    right_elbow_angle,                  #31 - float - right_elbow_angles_bd
                                                                                    left_elbow_angle,                   #32 - float - left_elbow_angles_bd
                                                                                    pose_trainer_cost_min,              #33 - float - pose_trainer_cost_min
                                                                                    pose_trainer_cost_max,              #34 - float - pose_trainer_cost_max
                                                                                    pose_user_cost                      #35 - float - pose_user_cost
                                                    )
                                                    ############################################ 
                                                    last_set = st.session_state.count_set + 1
                                                    last_rep = st.session_state.count_rep + 1
                                                    start +=1
                                                    st.session_state.inicio_rutina = fin_rutina_timestamp
                                                
                                                #🟢  BIRD DOG - pose 5 de 5
                                                elif up == False and\
                                                    down == True and\
                                                    mid == True and\
                                                    right_hip_angle in range(int(right_hip_angle_in-desv_right_hip_angle_in), int(right_hip_angle_in + desv_right_hip_angle_in + 1)) and\
                                                    right_knee_angle in range(int(right_knee_angle_in - desv_right_knee_angle_in), int(right_knee_angle_in + desv_right_knee_angle_in + 1)) and\
                                                    right_shoulder_angle in range(int(right_shoulder_angle_in - desv_right_shoulder_angle_in), int(right_shoulder_angle_in + desv_right_shoulder_angle_in + 1)):
                                                    up = False
                                                    down = False
                                                    mid = False
                                                    stage = "Abajo"
                                                    speak_stage5 = threading.Thread(target=speak, args=(stage,))
                                                    speak_stage5.start()
                                                    ############################################
                                                    st.session_state.count_pose_g += 1
                                                    st.session_state.count_pose = 5
                                                    update_counter_panel('training', st.session_state.count_pose_g, st.session_state.count_pose, st.session_state.count_rep + 1, st.session_state.count_set + 1)
                                                    # Cambio Renzo
                                                    # update_trainer_image(id_exercise, 1)
                                                    ############################################
                                                    fin_rutina_timestamp = get_timestamp_log()
                                                    df_results = util.add_row_df_results(df_results,
                                                                                    id_exercise,                        #1 - str - id_exercise
                                                                                    st.session_state.inicio_rutina,     #2 - str - DateTime_Start
                                                                                    fin_rutina_timestamp,               #3 - str - DateTime_End
                                                                                    st.session_state.n_poses,           #4 - int - n_poses
                                                                                    st.session_state.n_sets,            #5 - int - n_sets
                                                                                    st.session_state.n_reps,            #6 - int - n_reps
                                                                                    st.session_state.total_poses,       #7 - int - total_poses
                                                                                    st.session_state.seconds_rest_time, #8 - int - seconds_rest_time
                                                                                    selected_exercise,                  #9 - str - Class
                                                                                    body_language_prob_p,               #10 - float - Prob
                                                                                    st.session_state.count_pose_g,      #11 - int - count_pose_g
                                                                                    st.session_state.count_pose,        #12 - int - count_pose
                                                                                    st.session_state.count_rep + 1,     #13 - int - count_rep
                                                                                    st.session_state.count_set + 1,     #14 - int - count_set
                                                                                    None,   #15 - float - right_elbow_angles_pu
                                                                                    None,   #16 - float - right_hip_angles_pu
                                                                                    None,   #17 - float - right_knee_angles_pu
                                                                                    None,   #18 - float - right_shoulder_angles_cu
                                                                                    None,   #19 - float - right_hip_angles_cu
                                                                                    None,   #20 - float - right_knee_angles_cu
                                                                                    None,   #21 - float - right_shoulder_angles_fp
                                                                                    None,   #22 - float - right_hip_angles_fp
                                                                                    None,   #23 - float - right_ankle_angles_fp
                                                                                    None,   #24 - float - right_hip_angles_fl
                                                                                    None,   #25 - float - right_knee_angles_fl
                                                                                    None,   #26 - float - left_knee_angles_fl
                                                                                    right_shoulder_angle,               #27 - float - right_shoulder_angles_bd
                                                                                    right_hip_angle,                    #28 - float - right_hip_angles_bd
                                                                                    right_knee_angle,                   #29 - float - right_knee_angles_bd
                                                                                    left_knee_angle,                    #30 - float - left_knee_angles_bd
                                                                                    right_elbow_angle,                  #31 - float - right_elbow_angles_bd
                                                                                    left_elbow_angle,                   #32 - float - left_elbow_angles_bd
                                                                                    pose_trainer_cost_min,              #33 - float - pose_trainer_cost_min
                                                                                    pose_trainer_cost_max,              #34 - float - pose_trainer_cost_max
                                                                                    pose_user_cost                      #35 - float - pose_user_cost
                                                    )
                                                    ######################################s######
                                                    last_set = st.session_state.count_set + 1
                                                    last_rep = st.session_state.count_rep + 1
                                                    start = 0
                                                    st.session_state.count_rep += 1
                                                    st.session_state.count_pose = 0
                                                    st.session_state.inicio_rutina = fin_rutina_timestamp

                                                #🟢  BIRD DOG - ninguna pose
                                                else:
                                                    # ************************ INICIO SISTEMA DE RECOMENDACIONES - BIRD DOG ************************ #
                                                    if start+1 == 1 or start+1 == 3 or start+1 == 5 : 
                                                        if right_shoulder_angle > right_shoulder_angle_in+desv_right_shoulder_angle_in:
                                                            rec_right_shoulder_angle = "Flexiona más tu hombro derecho"
                                                        elif right_shoulder_angle < right_shoulder_angle_in-desv_right_shoulder_angle_in:
                                                            rec_right_shoulder_angle = "Flexiona menos tu hombro derecho"
                                                        else:
                                                            rec_right_shoulder_angle = ""
                                                        if right_hip_angle > right_hip_angle_in+desv_right_hip_angle_in:
                                                            rec_right_hip_angle = ", flexiona más tu cadera"
                                                        elif right_hip_angle < right_hip_angle_in-desv_right_hip_angle_in:
                                                            rec_right_hip_angle = ", flexiona menos tu cadera"
                                                        else:
                                                            rec_right_hip_angle = ""
                                                        if right_knee_angle > right_knee_angle_in+desv_right_knee_angle_in:
                                                            rec_right_knee_angle = ", flexiona más tu rodilla derecha"                                                        
                                                        elif right_knee_angle < right_knee_angle_in-desv_right_knee_angle_in:
                                                            rec_right_knee_angle = ", flexiona menos tu rodilla derecha"
                                                        else:
                                                            rec_right_knee_angle = ""
                                                        final_rec = rec_right_shoulder_angle+rec_right_hip_angle+rec_right_knee_angle
                                                        
                                                    elif start+1 == 2:
                                                        if right_elbow_angle > right_elbow_angle_in+desv_right_elbow_angle_in:
                                                            rec_right_elbow_angle = "Flexiona más tu codo derecho"
                                                        elif right_elbow_angle < right_elbow_angle_in-desv_right_elbow_angle_in:
                                                            rec_right_elbow_angle = "Flexiona menos tu codo derecho"
                                                        else:
                                                            rec_right_elbow_angle = ""
                                                        if left_knee_angle > left_knee_angle_in+desv_left_knee_angle_in:
                                                            rec_left_knee_angle = ", flexiona más tu rodilla izquierda"
                                                        elif left_knee_angle < left_knee_angle_in-desv_left_knee_angle_in:
                                                            rec_left_knee_angle = ", flexiona menos tu rodilla izquierda"
                                                        else:
                                                            rec_left_knee_angle = ""
                                                        final_rec = rec_right_elbow_angle+rec_left_knee_angle
                                                        
                                                    elif start+1 == 4:
                                                        if left_elbow_angle > left_elbow_angle_in+desv_left_elbow_angle_in:
                                                            rec_left_elbow_angle = "Flexiona más tu codo izquierdo"
                                                        elif left_elbow_angle < left_elbow_angle_in-desv_left_elbow_angle_in:
                                                            rec_left_elbow_angle = "Flexiona menos tu codo izquierdo"
                                                        else:
                                                            rec_left_elbow_angle = ""
                                                        if right_knee_angle > right_knee_angle_in+desv_right_knee_angle_in:
                                                            rec_right_knee_angle = ", flexiona más tu rodilla derecha"
                                                        elif right_knee_angle < right_knee_angle_in-desv_right_knee_angle_in:
                                                            rec_right_knee_angle = ", flexiona menos tu rodilla derecha"
                                                        else:
                                                            rec_right_knee_angle = ""
                                                        final_rec = rec_left_elbow_angle+rec_right_knee_angle
                                                        
                                                    if final_rec != "":
                                                        if start+1 == 1:
                                                            try:
                                                                if not speak_rec.is_alive():
                                                                    speak_rec.start()
                                                            except:
                                                                speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                speak_rec.start()
                                                        elif start+1 == 2:
                                                            try:
                                                                if not speak_rec.is_alive():
                                                                    speak_rec.start()
                                                            except:
                                                                try:
                                                                    if not speak_stage1.is_alive():
                                                                        speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                        speak_rec.start()
                                                                except:
                                                                    speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                    speak_rec.start()
                                                        elif start+1 == 3:
                                                            try:
                                                                if not speak_rec.is_alive():
                                                                    speak_rec.start()
                                                            except:
                                                                try:
                                                                    if not speak_stage2.is_alive():
                                                                        speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                        speak_rec.start()
                                                                except:
                                                                    speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                    speak_rec.start()
                                                        elif start+1 == 4:
                                                            try:
                                                                if not speak_rec.is_alive():
                                                                    speak_rec.start()
                                                            except:
                                                                try:
                                                                    if not speak_stage3.is_alive():
                                                                        speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                        speak_rec.start()
                                                                except:
                                                                    speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                    speak_rec.start()
                                                        elif start+1 == 5:
                                                            try:
                                                                if not speak_rec.is_alive():
                                                                    speak_rec.start()
                                                            except:
                                                                try:
                                                                    if not speak_stage4.is_alive():
                                                                        speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                        speak_rec.start()
                                                                except:
                                                                    speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                    speak_rec.start()
                                                    # ************************ FIN SISTEMA DE RECOMENDACIONES - BIRD DOG ************************ #
                                            
                                            #🟢 0. NINGÚN EJERCICIO
                                            else:
                                                print(f'NINGÚN EJERCICIO')
                                        ############################################################
                                        ##        📐 SISTEMA ÁNGULOS - CÁLCULO (FIN ⬆️)          ##
                                        ############################################################

                                        ############################################################
                                        ##         💰🖥️ PANTALLA SISTEMA COSTOS (INICIO ⬇️)      ##
                                        ############################################################
                                        color_validation_cost = (255, 0, 0) #Azul - dentro del rango
                                        if (pose_user_cost >= pose_trainer_cost_min) and (pose_user_cost <= pose_trainer_cost_max):
                                            color_validation_cost = (255, 0, 0) #Azul - dentro del rango
                                        else:
                                            color_validation_cost = (0, 0, 255) #Rojo - fuera del rango

                                        # #4A. Rectángulo 4A: Pose esperada
                                        r4A_x = 305             # x inicial
                                        r4A_y = 3             # y inicial
                                        r4A_w = 112             # width
                                        r4A_h = 24              # height
                                        r4A_rgb = (178,255,1)  # color BGR
                                        r4A_thickness = -1      #Sin ancho de borde
                                        cv2.rectangle(image, (r4A_x, r4A_y), (r4A_x + r4A_w, r4A_y + r4A_h), r4A_rgb, r4A_thickness)

                                        # #4B. Rectángulo 4B: Valor de Pose esperada
                                        r4B_x = 305             # x inicial
                                        r4B_y = 30              # y inicial
                                        r4B_w = 112             # width
                                        r4B_h = 24              # height
                                        r4B_rgb = (255,255,255) # color BGR
                                        r4B_thickness = -1      #Sin ancho de borde
                                        cv2.rectangle(image, (r4B_x, r4B_y), (r4B_x + r4B_w, r4B_y + r4B_h), r4B_rgb, r4B_thickness)

                                        cv2.putText(image, 
                                                    "EXPECTED POSE",
                                                    (310,20),
                                                    cv2.FONT_HERSHEY_SIMPLEX,
                                                    0.4,
                                                    (70,70,70),
                                                    1, 
                                                    cv2.LINE_AA)
                                        cv2.putText(image, 
                                                    "Pose {}".format(st.session_state.count_pose + 1),
                                                    (320,46),
                                                    cv2.FONT_HERSHEY_SIMPLEX,
                                                    0.4,
                                                    (70,70,70),
                                                    1, 
                                                    cv2.LINE_AA)

                                        # #2A. Rectángulo 2A: Evaluación de costos trainer vs user
                                        r2A_x = 420             # x inicial
                                        r2A_y = 3             # y inicial
                                        r2A_w = 215             # width
                                        r2A_h = 24              # height
                                        r2A_rgb = (227,138,64)  # color BGR
                                        r2A_thickness = -1      #Sin ancho de borde
                                        cv2.rectangle(image, (r2A_x, r2A_y), (r2A_x + r2A_w, r2A_y + r2A_h), r2A_rgb, r2A_thickness)

                                        # #2B. Rectángulo 2B: Costos resultantes
                                        r2B_x = 420             # x inicial
                                        r2B_y = 30              # y inicial
                                        r2B_w = 215             # width
                                        r2B_h = 24              # height
                                        r2B_rgb = (255,255,255) # color BGR
                                        r2B_thickness = -1      #Sin ancho de borde
                                        cv2.rectangle(image, (r2B_x, r2B_y), (r2B_x + r2B_w, r2B_y + r2B_h), r2B_rgb, r2B_thickness)

                                        cv2.putText(image, 
                                                    "TRAINER COST",
                                                    (432,20),
                                                    cv2.FONT_HERSHEY_SIMPLEX,
                                                    0.4,
                                                    (255,255,255),
                                                    1, 
                                                    cv2.LINE_AA)
                                        cv2.putText(image, 
                                                    "[{:.2f} to {:.2f}]".format(pose_trainer_cost_min , pose_trainer_cost_max), #Rango costos
                                                    (432,48),
                                                    cv2.FONT_HERSHEY_SIMPLEX,
                                                    0.4,
                                                    (70,70,70),
                                                    1, 
                                                    cv2.LINE_AA)
                                        cv2.putText(image, 
                                                    "USER COST",
                                                    (552,20),
                                                    cv2.FONT_HERSHEY_SIMPLEX,
                                                    0.4,
                                                    (255,255,255),
                                                    1, 
                                                    cv2.LINE_AA)
                                        cv2.putText(image, 
                                                     "{:.4f}".format(pose_user_cost), #Costo resultante 
                                                     (552,48),
                                                     cv2.FONT_HERSHEY_SIMPLEX, 
                                                     0.4,
                                                     color_validation_cost,
                                                     1, 
                                                     cv2.LINE_AA)

                                        ############################################################
                                        ##          💰🖥️ PANTALLA SISTEMA COSTOS (FIN ⬆️)        ##
                                        ############################################################


                                        ############################################################
                                        ##     🚩🖥️ PANTALLA SISTEMA AVANCE POSES (INICIO ⬇️)   ##
                                        ############################################################
                                        # #1A. Rectángulo 1A: Sets, repeticiones y stages
                                        r1A_x = 3               # x inicial
                                        r1A_y = 3               # y inicial
                                        r1A_w = 140             # width
                                        r1A_h = 24              # height
                                        r1A_rgb = (81,33,235)  # color BGR
                                        r1A_thickness = -1      # Sin ancho de borde
                                        cv2.rectangle(image, (r1A_x, r1A_y), (r1A_x + r1A_w, r1A_y + r1A_h), r1A_rgb, r1A_thickness)

                                        # #1B. Rectángulo 1B: Resultados de sets, repeticiones y stages
                                        r1B_x = 3               # x inicial
                                        r1B_y = 30              # y inicial
                                        r1B_w = 140             # width
                                        r1B_h = 24              # height
                                        r1B_rgb = (255,255,255) # color BGR
                                        r1B_thickness = -1      # Sin ancho de borde
                                        cv2.rectangle(image, (r1B_x, r1B_y), (r1B_x + r1B_w, r1B_y + r1B_h), r1B_rgb, r1B_thickness)
                                        
                                        # Set data
                                        cv2.putText(image, 'SET', (15,20), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
                                        cv2.putText(image, str(last_set), 
                                                    (15,48), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (70,70,70), 1, cv2.LINE_AA)

                                        # Rep data
                                        cv2.putText(image, 'REPS', (50,20), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
                                        cv2.putText(image, str(last_rep), 
                                                    (50,48), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (70,70,70), 1, cv2.LINE_AA)
                                        
                                        # Stage data
                                        cv2.putText(image, 'STAGE', (94,20), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
                                        cv2.putText(image, stage, 
                                                    (94,48), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (70,70,70), 1, cv2.LINE_AA)

                                        ############################################################
                                        ##     🚩🖥️ PANTALLA SISTEMA AVANCE POSES (FIN ⬆️)      ##
                                        ############################################################

                                        ############################################################
                                        ## 🏃‍♀️🖥️ PANTALLA SISTEMA PREDICCIÓN EJERCICIO (INICIO⬇️) ##
                                        ############################################################

                                        color_validation_class = (255, 0, 0) #Azul - dentro del rango

                                        if (body_language_prob_p > st.session_state.prob_change_color):
                                            color_validation_class = (255, 0, 0) #Azul - dentro del rango
                                        else:
                                            color_validation_class = (0, 0, 255) #Rojo - fuera del rango

                                        # #3A. Rectángulo 3A: Clase & Prob de pose
                                        r3A_x = 146               # x inicial
                                        r3A_y = 3             # y inicial
                                        r3A_w = 156             # width
                                        r3A_h = 24              # height
                                        r3A_rgb = (2,254,255) # color BGR
                                        r3A_thickness = -1      #Sin ancho de borde
                                        cv2.rectangle(image, (r3A_x, r3A_y), (r3A_x + r3A_w, r3A_y + r3A_h), r3A_rgb, r3A_thickness)

                                        # #3B. Rectángulo 3B: Resultados Clase & Prob de pose
                                        r3B_x = 146               # x inicial
                                        r3B_y = 30             # y inicial
                                        r3B_w = 156             # width
                                        r3B_h = 24              # height
                                        r3B_rgb = (255,255,255) # color BGR
                                        r3B_thickness = -1      #Sin ancho de borde
                                        cv2.rectangle(image, (r3B_x, r3B_y), (r3B_x + r3B_w, r3B_y + r3B_h), r3B_rgb, r3B_thickness)

                                        # Class data
                                        cv2.putText(image, 'CLASS', (158,20), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (70,70,70), 1, cv2.LINE_AA)
                                        cv2.putText(image, str(selected_exercise), 
                                                    (158,46), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (70,70,70), 1, cv2.LINE_AA)

                                        # Prob data
                                        cv2.putText(image, 'PROB', (250,20), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (70,70,70), 1, cv2.LINE_AA)
                                        cv2.putText(image, ("{:.2f}%".format(body_language_prob_p)),
                                                    (250,48), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_validation_class, 1, cv2.LINE_AA)
                                    
                                        ############################################################
                                        ## 🏃‍♀️🖥️ PANTALLA SISTEMA PREDICCIÓN EJERCICIO (FIN ⬆️)   ##
                                        ############################################################

                                        ############################################################
                                        ##     🦾🖥️ PANTALLA SISTEMA ARTICULACIONES (INICIO⬇️)   ##
                                        ############################################################

                                        if selected_exercise == "push_up": 
                                            cv2.line(image, (right_arm_x1, right_arm_y1), (right_arm_x2, right_arm_y2), (242, 14, 14), 3)
                                            cv2.line(image, (right_arm_x2, right_arm_y2), (right_arm_x3, right_arm_y3), (242, 14, 14), 3)
                                            cv2.circle(image, (right_arm_x1, right_arm_y1), 6, (128, 0, 255),-1)
                                            cv2.circle(image, (right_arm_x2, right_arm_y2), 6, (128, 0, 255),-1)
                                            cv2.circle(image, (right_arm_x3, right_arm_y3), 6, (128, 0, 255),-1)
                                            cv2.putText(image, str(int(right_elbow_angle)), (right_arm_x2 + 30, right_arm_y2), 1, 1.5, (128, 0, 250), 2)

                                            cv2.line(image, (right_torso_x1, right_torso_y1), (right_torso_x2, right_torso_y2), (14, 242, 59), 3)
                                            cv2.line(image, (right_torso_x2, right_torso_y2), (right_torso_x3, right_torso_y3), (14, 242, 59), 3)
                                            cv2.circle(image, (right_torso_x1, right_torso_y1), 6, (128, 0, 255),-1)
                                            cv2.circle(image, (right_torso_x2, right_torso_y2), 6, (128, 0, 255),-1)
                                            cv2.circle(image, (right_torso_x3, right_torso_y3), 6, (128, 0, 255),-1)
                                            cv2.putText(image, str(int(right_hip_angle)), (right_torso_x2 + 30, right_torso_y2), 1, 1.5, (128, 0, 250), 2)

                                            cv2.line(image, (right_leg_x1, right_leg_y1), (right_leg_x2, right_leg_y2), (14, 83, 242), 3)
                                            cv2.line(image, (right_leg_x2, right_leg_y2), (right_leg_x3, right_leg_y3), (14, 83, 242), 3)
                                            cv2.circle(image, (right_leg_x1, right_leg_y1), 6, (128, 0, 255),-1)
                                            cv2.circle(image, (right_leg_x2, right_leg_y2), 6, (128, 0, 255),-1)
                                            cv2.circle(image, (right_leg_x3, right_leg_y3), 6, (128, 0, 255),-1)
                                            cv2.putText(image, str(int(right_knee_angle)), (right_leg_x2 + 30, right_leg_y2), 1, 1.5, (128, 0, 250), 2)
                                            stframe.image(image,channels = 'BGR',use_column_width=True)
                                        
                                        if selected_exercise == "curl_up": 
                                            cv2.line(image, (right_shoul_x1, right_shoul_y1), (right_shoul_x2, right_shoul_y2), (242, 14, 14), 3)
                                            cv2.line(image, (right_shoul_x2, right_shoul_y2), (right_shoul_x3, right_shoul_y3), (242, 14, 14), 3)
                                            cv2.circle(image, (right_shoul_x1, right_shoul_y1), 6, (128, 0, 255),-1)
                                            cv2.circle(image, (right_shoul_x2, right_shoul_y2), 6, (128, 0, 255),-1)
                                            cv2.circle(image, (right_shoul_x3, right_shoul_y3), 6, (128, 0, 255),-1)
                                            cv2.putText(image, str(int(right_shoulder_angle)), (right_shoul_x2 + 30, right_shoul_y2), 1, 1.5, (128, 0, 250), 2)

                                            cv2.line(image, (right_torso_x1, right_torso_y1), (right_torso_x2, right_torso_y2), (14, 242, 59), 3)
                                            cv2.line(image, (right_torso_x2, right_torso_y2), (right_torso_x3, right_torso_y3), (14, 242, 59), 3)
                                            cv2.circle(image, (right_torso_x1, right_torso_y1), 6, (128, 0, 255),-1)
                                            cv2.circle(image, (right_torso_x2, right_torso_y2), 6, (128, 0, 255),-1)
                                            cv2.circle(image, (right_torso_x3, right_torso_y3), 6, (128, 0, 255),-1)
                                            cv2.putText(image, str(int(right_hip_angle)), (right_torso_x2 + 30, right_torso_y2), 1, 1.5, (128, 0, 250), 2)

                                            cv2.line(image, (right_leg_x1, right_leg_y1), (right_leg_x2, right_leg_y2), (14, 83, 242), 3)
                                            cv2.line(image, (right_leg_x2, right_leg_y2), (right_leg_x3, right_leg_y3), (14, 83, 242), 3)
                                            cv2.circle(image, (right_leg_x1, right_leg_y1), 6, (128, 0, 255),-1)
                                            cv2.circle(image, (right_leg_x2, right_leg_y2), 6, (128, 0, 255),-1)
                                            cv2.circle(image, (right_leg_x3, right_leg_y3), 6, (128, 0, 255),-1)
                                            cv2.putText(image, str(int(right_knee_angle)), (right_leg_x2 + 30, right_leg_y2), 1, 1.5, (128, 0, 250), 2)
                                            stframe.image(image,channels = 'BGR',use_column_width=True)
                                        
                                        if selected_exercise == "front_plank": 
                                            cv2.line(image, (right_shoul_x1, right_shoul_y1), (right_shoul_x2, right_shoul_y2), (242, 14, 14), 3)
                                            cv2.line(image, (right_shoul_x2, right_shoul_y2), (right_shoul_x3, right_shoul_y3), (242, 14, 14), 3)
                                            cv2.circle(image, (right_shoul_x1, right_shoul_y1), 6, (128, 0, 255),-1)
                                            cv2.circle(image, (right_shoul_x2, right_shoul_y2), 6, (128, 0, 255),-1)
                                            cv2.circle(image, (right_shoul_x3, right_shoul_y3), 6, (128, 0, 255),-1)
                                            cv2.putText(image, str(int(right_shoulder_angle)), (right_shoul_x2 + 30, right_shoul_y2), 1, 1.5, (128, 0, 250), 2)

                                            cv2.line(image, (right_torso_x1, right_torso_y1), (right_torso_x2, right_torso_y2), (14, 242, 59), 3)
                                            cv2.line(image, (right_torso_x2, right_torso_y2), (right_torso_x3, right_torso_y3), (14, 242, 59), 3)
                                            cv2.circle(image, (right_torso_x1, right_torso_y1), 6, (128, 0, 255),-1)
                                            cv2.circle(image, (right_torso_x2, right_torso_y2), 6, (128, 0, 255),-1)
                                            cv2.circle(image, (right_torso_x3, right_torso_y3), 6, (128, 0, 255),-1)
                                            cv2.putText(image, str(int(right_hip_angle)), (right_torso_x2 + 30, right_torso_y2), 1, 1.5, (128, 0, 250), 2)

                                            cv2.line(image, (right_ankle_x1, right_ankle_y1), (right_ankle_x2, right_ankle_y2), (14, 83, 242), 3)
                                            cv2.line(image, (right_ankle_x2, right_ankle_y2), (right_ankle_x3, right_ankle_y3), (14, 83, 242), 3)
                                            cv2.circle(image, (right_ankle_x1, right_ankle_y1), 6, (128, 0, 255),-1)
                                            cv2.circle(image, (right_ankle_x2, right_ankle_y2), 6, (128, 0, 255),-1)
                                            cv2.circle(image, (right_ankle_x3, right_ankle_y3), 6, (128, 0, 255),-1)
                                            cv2.putText(image, str(int(right_ankle_angle)), (right_ankle_x2 + 30, right_ankle_y2), 1, 1.5, (128, 0, 250), 2)
                                            
                                            #🟢 FRONT PLANK - pose 2 de 3 ⏰ Esta pose requiere mantener la posición unos segundos
                                            if start == 2 and flagTime == True:
                                                keep_pose_sec = 5 
                                                cv2.putText(image, 'WAIT FOR ' + str(keep_pose_sec) + ' s' , (155,350), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 3, cv2.LINE_AA)
                                                stframe.image(image,channels = 'BGR',use_column_width=True)
                                                time.sleep(1)
                                                mifrontplank = "Mantenga la posicion " + str(keep_pose_sec) + " segundos"
                                                speak(mifrontplank)
                                                time.sleep(keep_pose_sec)
                                                mffrontplank = "Baje!"
                                                speak_stage2 = threading.Thread(target=speak, args=(mffrontplank,))
                                                speak_stage2.start()
                                                ############################################
                                                st.session_state.count_pose_g += 1
                                                st.session_state.count_pose = 2
                                                update_counter_panel('training', st.session_state.count_pose_g, st.session_state.count_pose, st.session_state.count_rep + 1, st.session_state.count_set + 1)
                                                update_trainer_image(id_exercise, 3)
                                                ############################################
                                                
                                                fin_rutina_timestamp = get_timestamp_log()
                                                df_results = util.add_row_df_results(df_results,
                                                                                id_exercise,                        #1 - str - id_exercise
                                                                                st.session_state.inicio_rutina,     #2 - str - DateTime_Start
                                                                                fin_rutina_timestamp,               #3 - str - DateTime_End
                                                                                st.session_state.n_poses,           #4 - int - n_poses
                                                                                st.session_state.n_sets,            #5 - int - n_sets
                                                                                st.session_state.n_reps,            #6 - int - n_reps
                                                                                st.session_state.total_poses,       #7 - int - total_poses
                                                                                st.session_state.seconds_rest_time, #8 - int - seconds_rest_time
                                                                                selected_exercise,                  #9 - str - Class
                                                                                body_language_prob_p,               #10 - float - Prob
                                                                                st.session_state.count_pose_g,      #11 - int - count_pose_g
                                                                                st.session_state.count_pose,        #12 - int - count_pose
                                                                                st.session_state.count_rep + 1,     #13 - int - count_rep
                                                                                st.session_state.count_set + 1,     #14 - int - count_set
                                                                                None,   #15 - float - right_elbow_angles_pu
                                                                                None,   #16 - float - right_hip_angles_pu
                                                                                None,   #17 - float - right_knee_angles_pu
                                                                                None,   #18 - float - right_shoulder_angles_cu
                                                                                None,   #19 - float - right_hip_angles_cu
                                                                                None,   #20 - float - right_knee_angles_cu
                                                                                right_shoulder_angle,               #21 - float - right_shoulder_angles_fp
                                                                                right_hip_angle,                    #22 - float - right_hip_angles_fp
                                                                                right_ankle_angle,                  #23 - float - right_ankle_angles_fp
                                                                                None,    #24 - float - right_hip_angles_fl
                                                                                None,    #25 - float - right_knee_angles_fl
                                                                                None,    #26 - float - left_knee_angles_fl
                                                                                None,    #27 - float - right_shoulder_angles_bd
                                                                                None,    #28 - float - right_hip_angles_bd
                                                                                None,    #29 - float - right_knee_angles_bd
                                                                                None,    #30 - float - left_knee_angles_bd
                                                                                None,    #31 - float - right_elbow_angles_bd
                                                                                None,    #32 - float - left_elbow_angles_bd
                                                                                pose_trainer_cost_min,              #33 - float - pose_trainer_cost_min
                                                                                pose_trainer_cost_max,              #34 - float - pose_trainer_cost_max
                                                                                pose_user_cost                      #35 - float - pose_user_cost
                                                )
                                                ######################################s######
                                                last_set = st.session_state.count_set + 1
                                                last_rep = st.session_state.count_rep + 1
                                                st.session_state.inicio_rutina = fin_rutina_timestamp
                                                cv2.putText(image, '' , (155,350), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 3, cv2.LINE_AA)
                                                stframe.image(image,channels = 'BGR',use_column_width=True)
                                                flagTime = False
                                            else:
                                                stframe.image(image,channels = 'BGR',use_column_width=True)

                                        if selected_exercise == "forward_lunge": 
                                            cv2.line(image, (left_leg_x1, left_leg_y1), (left_leg_x2, left_leg_y2), (242, 14, 14), 3)
                                            cv2.line(image, (left_leg_x2, left_leg_y2), (left_leg_x3, left_leg_y3), (242, 14, 14), 3)
                                            cv2.circle(image, (left_leg_x1, left_leg_y1), 6, (128, 0, 255),-1)
                                            cv2.circle(image, (left_leg_x2, left_leg_y2), 6, (128, 0, 255),-1)
                                            cv2.circle(image, (left_leg_x3, left_leg_y3), 6, (128, 0, 255),-1)
                                            cv2.putText(image, str(int(left_knee_angle)), (left_leg_x2 + 30, left_leg_y2), 1, 1.5, (128, 0, 250), 2)

                                            cv2.line(image, (right_torso_x1, right_torso_y1), (right_torso_x2, right_torso_y2), (14, 242, 59), 3)
                                            cv2.line(image, (right_torso_x2, right_torso_y2), (right_torso_x3, right_torso_y3), (14, 242, 59), 3)
                                            cv2.circle(image, (right_torso_x1, right_torso_y1), 6, (128, 0, 255),-1)
                                            cv2.circle(image, (right_torso_x2, right_torso_y2), 6, (128, 0, 255),-1)
                                            cv2.circle(image, (right_torso_x3, right_torso_y3), 6, (128, 0, 255),-1)
                                            cv2.putText(image, str(int(right_hip_angle)), (right_torso_x2 + 30, right_torso_y2), 1, 1.5, (128, 0, 250), 2)

                                            cv2.line(image, (right_leg_x1, right_leg_y1), (right_leg_x2, right_leg_y2), (14, 83, 242), 3)
                                            cv2.line(image, (right_leg_x2, right_leg_y2), (right_leg_x3, right_leg_y3), (14, 83, 242), 3)
                                            cv2.circle(image, (right_leg_x1, right_leg_y1), 6, (128, 0, 255),-1)
                                            cv2.circle(image, (right_leg_x2, right_leg_y2), 6, (128, 0, 255),-1)
                                            cv2.circle(image, (right_leg_x3, right_leg_y3), 6, (128, 0, 255),-1)
                                            cv2.putText(image, str(int(right_knee_angle)), (right_leg_x2 + 30, right_leg_y2), 1, 1.5, (128, 0, 250), 2)
                                            stframe.image(image,channels = 'BGR',use_column_width=True)
                                        
                                        if selected_exercise == "bird_dog": 
                                            cv2.line(image, (left_leg_x1, left_leg_y1), (left_leg_x2, left_leg_y2), (242, 14, 14), 3)
                                            cv2.line(image, (left_leg_x2, left_leg_y2), (left_leg_x3, left_leg_y3), (242, 14, 14), 3)
                                            cv2.circle(image, (left_leg_x1, left_leg_y1), 6, (128, 0, 255),-1)
                                            cv2.circle(image, (left_leg_x2, left_leg_y2), 6, (128, 0, 255),-1)
                                            cv2.circle(image, (left_leg_x3, left_leg_y3), 6, (128, 0, 255),-1)
                                            cv2.putText(image, str(int(left_knee_angle)), (left_leg_x2 + 30, left_leg_y2), 1, 1.5, (128, 0, 250), 2)

                                            cv2.line(image, (right_torso_x1, right_torso_y1), (right_torso_x2, right_torso_y2), (14, 242, 59), 3)
                                            cv2.line(image, (right_torso_x2, right_torso_y2), (right_torso_x3, right_torso_y3), (14, 242, 59), 3)
                                            cv2.circle(image, (right_torso_x1, right_torso_y1), 6, (128, 0, 255),-1)
                                            cv2.circle(image, (right_torso_x2, right_torso_y2), 6, (128, 0, 255),-1)
                                            cv2.circle(image, (right_torso_x3, right_torso_y3), 6, (128, 0, 255),-1)
                                            cv2.putText(image, str(int(right_hip_angle)), (right_torso_x2 + 30, right_torso_y2), 1, 1.5, (128, 0, 250), 2)

                                            cv2.line(image, (right_leg_x1, right_leg_y1), (right_leg_x2, right_leg_y2), (14, 83, 242), 3)
                                            cv2.line(image, (right_leg_x2, right_leg_y2), (right_leg_x3, right_leg_y3), (14, 83, 242), 3)
                                            cv2.circle(image, (right_leg_x1, right_leg_y1), 6, (128, 0, 255),-1)
                                            cv2.circle(image, (right_leg_x2, right_leg_y2), 6, (128, 0, 255),-1)
                                            cv2.circle(image, (right_leg_x3, right_leg_y3), 6, (128, 0, 255),-1)
                                            cv2.putText(image, str(int(right_knee_angle)), (right_leg_x2 + 30, right_leg_y2), 1, 1.5, (128, 0, 250), 2)
                                            stframe.image(image,channels = 'BGR',use_column_width=True)
                                        
                                        ############################################################
                                        ##     🦾🖥️ PANTALLA SISTEMA ARTICULACIONES (FIN ⬆️)    ##
                                        ############################################################

                                st.session_state.count_set += 1

                                if (st.session_state.count_set!=st.session_state.n_sets):
                                    try:
                                        placeholder_status.markdown(util.font_size_px("🧘 RESTING TIME", 26), unsafe_allow_html=True)
                                        cv2.putText(image, 'FINISHED SET', (100,250), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 3, cv2.LINE_AA)
                                        cv2.putText(image, 'REST FOR ' + str(st.session_state.seconds_rest_time) + ' s' , (155,350), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 3, cv2.LINE_AA)
                                        stframe.image(image,channels = 'BGR',use_column_width=True)
                                        msucessset = "Felicitaciones, vas por buen camino"
                                        time.sleep(1)
                                        speak(msucessset)
                                        time.sleep(int(st.session_state.seconds_rest_time))
                                        placeholder_status.markdown(util.font_size_px("🏎️ TRAINING...", 26), unsafe_allow_html=True)
                                        update_trainer_image(id_exercise,1)
                                    except:
                                        stframe.image(image,channels = 'BGR',use_column_width=True)
                                        pass 
                        update_counter_panel('finished', st.session_state.count_pose_g, st.session_state.n_poses, st.session_state.count_rep, st.session_state.count_set)
                        cv2.rectangle(image, (r4B_x, r4B_y), (r4B_x + r4B_w, r4B_y + r4B_h), r4B_rgb, r4B_thickness)
                        cv2.putText(image, "EXPECTED POSE",(310,20),cv2.FONT_HERSHEY_SIMPLEX,0.4,(70,70,70),1, cv2.LINE_AA)
                        cv2.rectangle(image, (50,180), (600,300), (0,255,0), -1)
                        cv2.putText(image, 'EXERCISE FINISHED!', (100,250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3, cv2.LINE_AA)
                        stframe.image(image,channels = 'BGR',use_column_width=True)
                        msucess = "Felicitaciones, bien hecho"
                        time.sleep(1)
                        speak(msucess)
                        finishexercise = True
                        #Finalización de hilos
                        speak_start_msg.join() # Mensaje de inicio ("Por favor asegurese que su dispositivo pueda ...")
                        speak_stage1.join() # Stage 1
                        speak_stage2.join() # Stage 2
                        speak_stage3.join() # Stage 3

                        if id_exercise == "forward_lunge" or id_exercise == "bird_dog": #Se terminan los hilos de ejercicios de más de 3 poses
                            speak_stage4.join() # Stage 4
                            speak_stage5.join() # Stage 5
                        try:
                            speak_rec.join() # Recomendaciones según pose y articulaciones
                        except:
                            print("No hubieron recomendaciones")
                        time.sleep(5)          
                        cap.release()
                        cv2.destroyAllWindows()

                    placeholder_button_status.warning('CAMERA OFF ⚫', icon="📹")
                    st.session_state['camera'] += 1
                    video_capture.release()
                    cv2.destroyAllWindows()

                    st.balloons()                    
                    placeholder_results_1.markdown(util.font_size_px("RESULTADOS", 26), unsafe_allow_html=True)

                    #Cargar dataset de resultados
                    timestamp_show_results = get_timestamp_txt(username,id_exercise)
                    df_results.to_csv('03. users/{}.csv'.format(timestamp_show_results), index=False, sep='|')
                    placeholder_results_2.dataframe(dashboard.get_result_filter_dataframe(df_results, st.session_state.articulaciones, st.session_state.posfijo))

        # Recent training
        if finishexercise == True:
            st.markdown("<br />", unsafe_allow_html=True)
            st.markdown(util.font_size_px("Métricas del entrenamiento", 26), unsafe_allow_html=True)

            dateTime_start_str, dateTime_end_str, minutes, seconds = dashboard.get_training_time(df_results)
            burned_kcal_min, burned_kcal_factor, kcal_min_calc_info, kcal_table, col_value = dashboard.\
                get_burned_calories(minutes, id_exercise, st.session_state.peso)

            st.text("🏁 Inicio entrenamiento :{}".format(dateTime_start_str))
            st.text("🥇 Fin entrenamiento    :{}".format(dateTime_end_str))
            st.text("🕒 Tiempo entrenamiento :{:.2f} minutos y {:.2f} segundos".format(minutes, seconds))
            st.text("⚖️ Peso                 :{:.2f} Kg".format(st.session_state.peso))            
            st.text("🔥 Calorías quemadas    :{:.2f} calorías".format(burned_kcal_min))
            st.text("🧨 Factor de Calorías   :{:.2f} KCal (Fuente: {}) [Ver cuadro 1]".format(burned_kcal_factor, kcal_min_calc_info))
            st.markdown("<br />", unsafe_allow_html=True)
            st.text("Cuadro 1")
            st.dataframe(kcal_table.style.highlight_max(col_value, axis=0, color = 'blueviolet'))
            st.markdown("---------", unsafe_allow_html=True)

            system_pred_exercise, system_angles, system_cost = st.tabs(["🏃‍♀️ SISTEMA PREDICCIÓN EJERCICIO", "📐 SISTEMA ÁNGULOS", "💰 SISTEMA COSTOS"])
            
            # 1.🏃‍♀️ SISTEMA CLASIFICACIÓN EJERCICIO                 
            with system_pred_exercise:
                st.markdown("**🏃‍♀️ SISTEMA CLASIFICACIÓN EJERCICIO**", unsafe_allow_html=True)
                st.markdown("Evalúa cada pose y a través de algoritmos de Machine Learning identifica a cual tiene una mayor aproximación.")
                st.markdown("<br><br>", unsafe_allow_html=True)

                st.markdown("🔵 __APROXIMACIÓN MEDIA TOTAL:__")
                st.markdown("<br>", unsafe_allow_html=True)
                st.plotly_chart(dashboard.plot_aprox_gauge_chart(df_results.loc[:, 'Prob'].mean(), 
                                                                        (("Aprox(%) Media Total {}").format(st.session_state.short_name)), 0, 16, 32, 48), 
                                                                        use_container_width=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("🔵 __APROXIMACIÓN MEDIA POR POSE:__")
                st.markdown("<br>", unsafe_allow_html=True)

                if (st.session_state.n_poses == 3):
                    #POSE 1
                    pose_1_spe, pose_2_spe, pose_3_spe = st.columns(3)
                    with pose_1_spe:
                        st.markdown("**Pose 1**", unsafe_allow_html=True)                        
                        st.plotly_chart(dashboard.plot_aprox_gauge_chart(df_results.loc[df_results['count_pose']==1].\
                                                                        loc[:, 'Prob'].mean(), "Aprox(%) Media Pose 1", 0, 16, 32, 48, st.session_state.n_poses))
                    #POSE 2
                    with pose_2_spe:
                        st.markdown("**Pose 2**", unsafe_allow_html=True)
                        st.plotly_chart(dashboard.plot_aprox_gauge_chart(df_results.loc[df_results['count_pose']==2].\
                                                                        loc[:, 'Prob'].mean(), "Aprox(%) Media Pose 2", 0, 16, 32, 48, st.session_state.n_poses))
                    #POSE 3    
                    with pose_3_spe:
                        st.markdown("**Pose 3**", unsafe_allow_html=True)
                        st.plotly_chart(dashboard.plot_aprox_gauge_chart(df_results.loc[df_results['count_pose']==3].\
                                                                    loc[:, 'Prob'].mean(), "Aprox(%) Media Pose 3", 0, 16, 32, 48, st.session_state.n_poses))
                elif (st.session_state.n_poses == 5):
                    #POSE 1
                    pose_1_spe, pose_2_spe, pose_3_spe, pose_4_spe, pose_5_spe = st.columns(5)
                    with pose_1_spe:
                        st.markdown("**Pose 1**", unsafe_allow_html=True)                        
                        st.plotly_chart(dashboard.plot_aprox_gauge_chart(df_results.loc[df_results['count_pose']==1].\
                                                                        loc[:, 'Prob'].mean(), "Aprox(%) Media Pose 1", 0, 16, 32, 48, st.session_state.n_poses))
                    #POSE 2
                    with pose_2_spe:
                        st.markdown("**Pose 2**", unsafe_allow_html=True)
                        st.plotly_chart(dashboard.plot_aprox_gauge_chart(df_results.loc[df_results['count_pose']==2].\
                                                                        loc[:, 'Prob'].mean(), "Aprox(%) Media Pose 2", 0, 16, 32, 48, st.session_state.n_poses))
                    #POSE 3    
                    with pose_3_spe:
                        st.markdown("**Pose 3**", unsafe_allow_html=True)
                        st.plotly_chart(dashboard.plot_aprox_gauge_chart(df_results.loc[df_results['count_pose']==3].\
                                                                    loc[:, 'Prob'].mean(), "Aprox(%) Media Pose 3", 0, 16, 32, 48, st.session_state.n_poses))
                    #POSE 4    
                    with pose_4_spe:
                        st.markdown("**Pose 4**", unsafe_allow_html=True)
                        st.plotly_chart(dashboard.plot_aprox_gauge_chart(df_results.loc[df_results['count_pose']==4].\
                                                                    loc[:, 'Prob'].mean(), "Aprox(%) Media Pose 4", 0, 16, 32, 48, st.session_state.n_poses))
                    #POSE 5    
                    with pose_5_spe:
                        st.markdown("**Pose 5**", unsafe_allow_html=True)
                        st.plotly_chart(dashboard.plot_aprox_gauge_chart(df_results.loc[df_results['count_pose']==5].\
                                                                    loc[:, 'Prob'].mean(), "Aprox(%) Media Pose 5", 0, 16, 32, 48, st.session_state.n_poses))
                else:
                    st.markdown("**Ejercicio con N° Poses ≠ [2,5]**", unsafe_allow_html=True)
                
                st.markdown("🔵 __ANÁLISIS APROXIMACIÓN POR CADA POSE:__")
                gauges_img = dashboard.plot_all_aprox_gauge_chart(df_results)
                for gauge in gauges_img:
                    st.plotly_chart(gauge, use_container_width=True)
                    
            # 2.📐 SISTEMA ÁNGULOS 
            with system_angles:
                st.markdown("**📐 SISTEMA ÁNGULOS**", unsafe_allow_html=True)
                articulaciones = dashboard.get_articulaciones_list(st.session_state.articulaciones)

                st.markdown("Articulaciones evaluadas angularmente en este ejercicio:")
                st.markdown("<br><br>", unsafe_allow_html=True)

                st.markdown("🟡 __ARTICULACIÓN VS TODAS LAS POSES:__")
                st.markdown("<br>", unsafe_allow_html=True)

                for articulacion in articulaciones:
                    st.markdown("🧬 __" + (dashboard.get_articulacion_name(articulacion)[0]).upper() + "__")
                    img_algles, img_acorr, correlation_trainer_user = dashboard.plot_articulacion_performance_by_exerc(articulacion, id_exercise, df_results, st.session_state.posfijo)
                    
                    st.plotly_chart(img_algles, use_container_width=True)
                    st.text("Correlación Trainer vs User : {:.4f}".format(correlation_trainer_user))
                    st.pyplot(img_acorr)

                    st.markdown("<br ><br >", unsafe_allow_html=True)
                
                st.markdown("🟡 __ANÁLISIS POR CADA POSE:__")
                st.markdown("<br>", unsafe_allow_html=True)

                radars_img = dashboard.plot_arts_performance_radar(df_results, id_exercise, st.session_state.posfijo)
                for radar in radars_img:
                    st.plotly_chart(radar, use_container_width=True)

            # 3.📐 💰 SISTEMA COSTOS                
            with system_cost:
                st.markdown("**💰 SISTEMA COSTOS**", unsafe_allow_html=True)
                st.markdown("Cálculo de los costos del usuario vs costo del trainer en este ejercicio:")
                st.markdown("<br><br>", unsafe_allow_html=True)

                st.markdown("🟣 __COMPARACIÓN MEDIA COSTO USUARIO VS COSTO TRAINER__")
                st.markdown("<br>", unsafe_allow_html=True)

                img_costs = dashboard.plot_cost(
                    df_results.loc[:, 'pose_trainer_cost_min'].mean(), 
                    df_results.loc[:, 'pose_trainer_cost_max'].mean(), 
                    df_results.loc[:, 'pose_user_cost'].mean(), 
                    "COSTO MEDIA TOTAL TODO EL EJERCICIO %")
                st.plotly_chart(img_costs, use_container_width=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("🟣 __ANÁLISIS COSTOS POR CADA POSE:__")

                costs_img = dashboard.plot_all_cost(df_results)
                for cost_im in costs_img:
                    st.plotly_chart(cost_im, use_container_width=True)

            st.markdown("<hr/>", unsafe_allow_html=True)