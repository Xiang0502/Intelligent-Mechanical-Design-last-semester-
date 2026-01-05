import streamlit as st
import os

st.set_page_config(page_title="國立虎尾科技大學機械設計工程系", layout="wide")

# --- 自訂 CSS 樣式 ---
st.markdown("""
<style>
    /* 設定全站背景為深色 */
    .stApp {
        background-color: #0E1117;
    }

    /* 強制所有主要文字元件為白色，並增加行高提升閱讀體驗 */
    h1, h2, h3, h4, h5, h6, p, div, span, label, li, .stMarkdown {
        color: #FFFFFF !important;
        line-height: 1.6 !important;
    }

    /* 修正程式碼區塊的文字顏色 */
    code {
        color: #ff4b4b !important;
    }

    /* 側邊欄文字顏色 */
    .css-17lntkn {
        color: #FFFFFF !important;
    }

    /* --- GitHub 風格 Expander 樣式 --- */
    .streamlit-expanderHeader {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 6px !important;
        color: #c9d1d9 !important;
        font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif,"Apple Color Emoji","Segoe UI Emoji" !important;
        font-size: 14px !important;
        font-weight: 600 !important;
    }
    .streamlit-expanderHeader:hover {
        background-color: #20252c !important;
        color: #58a6ff !important;
    }
    .streamlit-expanderContent {
        background-color: #0d1117 !important;
        border: 1px solid #30363d !important;
        border-top: none !important;
        border-bottom-left-radius: 6px !important;
        border-bottom-right-radius: 6px !important;
        color: #c9d1d9 !important;
    }
    .streamlit-expanderContent code {
        background-color: transparent !important;
    }

    /* --- 側邊欄選單樣式 (Gemini 風格) --- */
    section[data-testid="stSidebar"] .stRadio > div {
        gap: 12px;
        padding-top: 10px;
    }
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label > div:first-child {
        display: none;
    }
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label {
        display: flex;
        align-items: center;
        justify-content: center !important; /* 強制 Flex 容器內容置中 */
        width: 100%;
        padding: 12px 16px;       
        border-radius: 12px;
        transition: all 0.3s ease;
        border: 1px solid transparent; 
        color: #FFFFFF !important;
        cursor: pointer;
        background-color: transparent;
        text-align: center !important; /* 強制文字置中 */
        margin: 0 auto; /* 容器置中 */
    }
    /* 確保 label 內部的文字容器也置中 */
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label > div[data-testid="stMarkdownContainer"] {
        text-align: center !important;
        width: 100%;
        display: flex;
        justify-content: center;
    }
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label p {
        text-align: center !important;
        width: 100%;
        margin: 0;
    }

    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:hover {
        background-color: rgba(255, 255, 255, 0.1);
    }
    /* 藍色毛玻璃醒目框風格 */
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:has(input:checked) {
        background-color: rgba(16, 83, 210, 0.5) !important;
        border: 1px solid rgba(16, 83, 210, 0);
        backdrop-filter: blur(10px);
        color: #FFFFFF !important;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    /* --- 首頁置中專用樣式 --- */
    .home-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        width: 100%;
        margin-top: 20px;
    }
    .home-container h1, .home-container h2, .home-container h3, .home-container p {
        text-align: center !important;
        width: 100%;
    }

</style>
""", unsafe_allow_html=True)

# --- 側邊欄選單 ---
menu = st.sidebar.radio(
    "目錄",
    [
        "首頁",
        "深度學習分類問題",
        "YOLO影像辨識",
        "TurtleBot Burger平台",
        "Streamlit UI設計與資料可視化",
        "RL建模與訓練"
    ],
    label_visibility="collapsed",
    key="main_menu"
)


# --- Helper Function: 安全顯示媒體 ---
def show_media(path, media_type='image', caption="", width=None):
    if os.path.exists(path):
        if media_type == 'image':
            st.image(path, caption=caption, width=width, use_container_width=(width is None))
        elif media_type == 'video':
            st.video(path)
    else:
        st.warning(f"⚠️找不到檔案: {path} (請確認檔案是否已放入資料夾)")


# --- 頁面內容 ---

if menu == "首頁":
    st.markdown("""
        <div class="home-container">
            <h1>114(上)學年度『智慧機械設計』課程期末報告</h1>
            <h2>ROS自主移動平台與AI整合之研究</h2>
            <br>
            <h3>指導老師：周榮源</h3>
            <h3>班級：碩設計一甲</h3>
            <h3>組別：第一組</h3>
            <h3>組員：11473132 陳威誌
            <h3>     11473107 紀閔翔
            <h3>     11473143 朱王黃</h3>           
            <br>
            <p style='font-size: 1.1em;'>歡迎來到智慧機械設計課程期末報告。請從左側選單選擇要查看的實驗項目。</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 300, 1])
    with col2:
        show_media("img/Turtlebot/first.jpg", caption="Turtlebot3")


elif menu == "深度學習分類問題":
    st.title("深度學習分類問題")
    st.markdown("---")

    st.header("一、實作過程：")

    st.subheader('步驟 1：PC 端啟動 ROS Master')
    st.caption("用途：作為整個系統的 ROS 中樞，負責管理所有 ROS 節點與 topic")
    show_media("img/rviz/1.jpg")
    code = """
export ROS_MASTER_URI = http://192.168.1.203:11311
export ROS_IP = 192.168.1.203
roscore"""
    st.code(code, language="bash")

    st.markdown("---")

    st.subheader('步驟 2：連線至 TurtleBot3 Burger（Raspberry Pi）')
    st.caption("用途：遠端控制 TurtleBot3，啟動機器人端節點")
    show_media("img/rviz/2.jpg")
    st.code("ssh pi@192.168.1.199", language="bash")

    st.markdown("---")

    st.subheader('步驟 3：TurtleBot3 端：Bringup（底層系統啟動）')
    st.caption("用途：啟動馬達、雷射感測器、TF 架構，使機器人可接收 /cmd_vel")
    show_media("img/rviz/3.jpg")
    code = """
export TURTLEBOT3_MODEL = burger
export ROS_MASTER_URI = http://192.168.1.203:11311
export ROS_IP = 192.168.1.199 
roslaunch turtlebot3_bringup turtlebot3_robot.launch"""
    st.code(code, language="bash")

    st.markdown("---")

    st.subheader('步驟 4：TurtleBot3 端：啟動 USB Camera')
    st.caption("用途：取得即時影像並發布為 ROS topic")
    show_media("img/rviz/usb.jpg")
    code = """
export ROS_MASTER_URI = http://192.168.1.203:11311
export ROS_IP=192.168.1.199
roslaunch usb_cam usb_cam-test.launch """
    st.code(code, language="bash")

    st.markdown('**成功後影像會發布至**')
    st.code("/usb_cam/image_raw", language="bash")

    st.markdown("---")

    st.subheader('步驟 5：PC 端鍵盤控制測試')
    st.caption("用途：確認 /cmd_vel 控制通道正常，避免後續誤判為模型錯誤")
    show_media("img/rviz/5.jpg")
    st.header("成功畫面")
    show_media("img/rviz/6.jpg")
    code = """
export TURTLEBOT3_MODEL=burger
export ROS_MASTER_URI=http://192.168.1.203:11311
export ROS_IP=192.168.1.203
roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch"""
    st.code(code, language="bash")

    st.markdown("---")

    st.title("二、影像辨識與控制核心程式碼")
    st.markdown("""
    以下為完整可執行版本，此程式負責：
    1. 訂閱 TurtleBot3 相機影像
    2. 進行前進 / 後退影像辨識
    3. 發布 `/cmd_vel` 控制 Burger 移動
    """)
    show_media("img/rviz/7.jpg")
    code = """
source /opt/ros/noetic/setup.bash
source ~/mde_ws/devel_isolated/setup.bash
export ROS_MASTER_URI=http://192.168.1.203:11311
export ROS_IP=192.168.1.203
rosrun gesture_control gesture_cmd_vel.py"""
    st.code(code, language="bash")
    st.header("成功畫面")
    show_media("img/rviz/8.jpg")



    code_gesture_cmd_vel = """
#!/usr/bin/env python3
import rospy
import rospkg
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
from keras.models import load_model

# =========================
# ROS node initialization
# =========================
rospy.init_node("gesture_control_node")

# =========================
# Load model & labels
# =========================
rospack = rospkg.RosPack()
pkg_path = rospack.get_path("gesture_control")

model_path = pkg_path + "/model/keras_model.h5"
label_path = pkg_path + "/model/labels.txt"

model = load_model(model_path, compile=False)
class_names = open(label_path, "r").readlines()

# =========================
# ROS publisher / subscriber
# =========================
bridge = CvBridge()
cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

CONF_TH = 0.8   # 信心值門檻

# =========================
# Image callback function
# =========================
def image_cb(msg):
    frame = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    # Resize & normalize
    img = cv2.resize(frame, (224, 224))
    img = np.asarray(img, dtype=np.float32).reshape(1,224,224,3)
    img = (img / 127.5) - 1

    # Predict
    pred = model.predict(img, verbose=0)
    idx = np.argmax(pred)
    conf = pred[0][idx]
    label = class_names[idx].strip()

    twist = Twist()

    if conf > CONF_TH:
        if "forward" in label:
            twist.linear.x = 0.2      # 前進
        elif "back" in label:
            twist.linear.x = -0.2     # 後退

        cmd_pub.publish(twist)

    # Debug display
    cv2.putText(frame, f"{label} {conf:.2f}",
                (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0,255,0), 2)
    cv2.imshow("Gesture Control", frame)
    cv2.waitKey(1)

# =========================
# Subscriber
# =========================
rospy.Subscriber("/usb_cam/image_raw", Image, image_cb)
rospy.spin()
"""
    with st.expander("點擊複製相關程式碼 (gesture_cmd_vel.py)"):
        st.code(code_gesture_cmd_vel, language="python")

    st.markdown("---")

    st.header('影像辨識情形')
    st.caption("用途：Techable Machine 資料集程式辨識情況")
    show_media("img/rviz/T.jpg")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown('#### (a) 左轉')
        show_media("img/rviz/left.jpg")
    with col2:
        st.markdown('#### (b) 右轉')
        show_media("img/rviz/right.jpg")
    with col3:
        st.markdown('#### (c) 前進')
        show_media("img/rviz/up.jpg")
    with col4:
        st.markdown('#### (d) 後退')
        show_media("img/rviz/down.jpg")
    with col5:
        st.markdown('#### (e) STOP')
        show_media("img/rviz/stop.jpg")

    st.markdown("---")

    st.header("二、結果展示")
    st.markdown('### 成功使 Turtlebot3 依箭頭方向做出相應動作')
    st.subheader('(A) 左轉')
    show_media("img/rviz/left.mp4", "video")
    st.markdown("---")
    st.subheader('(B) 右轉')
    show_media("img/rviz/right.mp4", "video")
    st.markdown("---")
    st.subheader('(C) 前進')
    show_media("img/rviz/up.mp4", "video")
    st.markdown("---")
    st.subheader('(D) 後退')
    show_media("img/rviz/down.mp4", "video")
    st.markdown("---")
    st.subheader('(E) STOP')
    show_media("img/rviz/stop.mp4", "video")

elif menu == "YOLO影像辨識":
    st.title("YOLO 影像辨識實作步驟")
    st.markdown("---")

    st.subheader("步驟 1：影像數據收集 (Data images collection)")
    show_media("img/yolo/yolo1.png", caption="影像收集示意圖")

    st.markdown("---")

    st.subheader("步驟 2：標註類別 (Annotating Labels Classes - Roboflow)")
    show_media("img/yolo/yolo2.png", caption="Roboflow 標註畫面")

    st.markdown("---")

    st.subheader("步驟 3：建立 Mushroom 資料集 (Roboflow)")
    st.markdown("總數: 124 張 (訓練集: 111, 驗證集: 9, 測試集: 4)")
    st.markdown("[🔗 點擊前往 Roboflow 專案連結](https://app.roboflow.com/mushrooms-object-detection/king_mushroom/3)")
    show_media("img/yolo/yolo3.png", caption="資料集分佈圖")

    st.markdown("---")

    st.subheader("步驟 4：訓練資料集 (Google Colab 高效能 GPU T4)")
    show_media("img/yolo/yolo4.png", caption="Colab 訓練過程")

    st.markdown("---")

    st.subheader("步驟 5：在 Ubuntu Linux 上進行 Mushroom 偵測")

    st.markdown("**圖一：開啟攝影機並檢查裝置**")
    show_media("img/yolo/yolo5.png", caption="開啟 webcam")

    st.markdown("**圖二：執行 Python 程式**")
    show_media("img/yolo/yolo6.png", caption="Step 2: Run python file")

    code_mushroom = """
import cv2
from ultralytics import solutions

# 1. model
VIDEO_SOURCE = 0            # 0: Camera laptop
MODEL_PATH = "best.pt"      # Model .pt
CONF_THRESHOLD = 0.8        # threshold
LINE_POSITION = 0.66        # line (right side)

# 2. CAMERA 
cap = cv2.VideoCapture(VIDEO_SOURCE)
assert cap.isOpened(), "Can not open Camera"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Line zone (right side)
line_x = int(w * LINE_POSITION)
line_points = [(line_x, 0), (line_x, h)]

# (OBJECT COUNTER)
counter = solutions.ObjectCounter(
    show=False,              # show video output
    region=line_points,     # line
    model=MODEL_PATH,       # Link model
    conf=CONF_THRESHOLD,    # threshold
    line_width=2,
    # classes=[0]           
)

print("Running... Press 'q' to exit.")

while cap.isOpened():
    success, im0 = cap.read()
    if not success: break
    result = counter.process(im0)
    im_output = result.plot_im

    # show
    cv2.imshow("Mushroom Counter Optimized", im_output)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
"""
    with st.expander("點擊複製相關程式碼 (run_mushroom.py)"):
        st.code(code_mushroom, language="python")

    st.markdown("**圖三：偵測結果畫面**")
    show_media("img/yolo/yolo7.png", caption="偵測執行中")

    code_video_counter = """
import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("test_mushroom.mp4")
assert cap.isOpened(), "Error reading video file"

# region_points = [(20, 400), (1080, 400)]                                      # line counting
#region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]  # rectangular region
region_points = [[691, 113], [959, 115], [957, 535], [700, 528]]   # polygon region

# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("object_counting.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Initialize object counter object
counter = solutions.ObjectCounter(
    show=True,  # display the output
    region=region_points,  # pass region points
    model="best.pt",  # model="yolo11n-obb.pt" for object counting with OBB model.
    # classes=[0, 2],  # count specific classes, e.g., person and car with the COCO pretrained model.
    # tracker="botsort.yaml",  # choose trackers, e.g., "bytetrack.yaml"
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()

    if not success:
        print("Video frame is empty or processing is complete.")
        break

    results = counter(im0)

    # print(results)  # access the output

    video_writer.write(results.plot_im)  # write the processed frame.

cap.release()
video_writer.release()
cv2.destroyAllWindows()  # destroy all opened windows
"""
    with st.expander("點擊複製相關程式碼 (counting_mushroom.py)"):
        st.code(code_video_counter, language="python")

    st.markdown("---")

    st.subheader("步驟 6：使用 Webcam 進行偵測與計數")
    show_media("img/yolo/yolo1.mp4", "video")

    st.markdown("---")

    st.subheader("步驟 7：使用圖片/影片進行偵測與計數")
    show_media("img/yolo/yolo2.mp4", "video")


elif menu == "RL建模與訓練":
    st.title("RL 建模與訓練")
    # 新增子選單分支
    rl_nav = st.sidebar.radio(
        "RL 選單",
        ["系統架構與實作", "獎勵函數詳細解說"],
        label_visibility="collapsed",
        key="rl_menu"
    )

    if rl_nav == "系統架構與實作":
        st.header("一、研究動機與專案目標")
        st.subheader('概述')
        st.markdown("""
        Three_link_rl 專案旨在建立一套基於深度強化學習（Deep Reinforcement Learning, DRL）之三連桿平面機械手臂控制系統。
        系統透過學習型控制策略，使三連桿機械臂能在未知目標位置條件下，自主追蹤並穩定停留於目標區域，同時具備良好的動作平順性與控制穩定度。
        本專案不依賴傳統解析式逆運動學（Inverse Kinematics, IK），而是採用模型自由（model-free）強化學習方法，使系統具備良好的泛化能力與延展性。
        """)
        st.markdown("---")

        st.header("二、系統架構概述")
        st.markdown("本專案系統由三個核心模組組成，形成標準 Agent–Environment 互動閉環：")
        code = """
three_link_rl/
├── env.py   # 環境建模（MDP + 控制穩定化）
├── rl.py    # DDPG Actor–Critic 學習器
└── main.py  # 訓練與測試流程控制 """
        st.code(code, language="bash")

        st.markdown("""
        <div style="font-size: 16px;">
            <ul>
                <li><strong>env.py</strong>：負責機械結構建模、狀態定義、獎勵設計與控制後處理</li>
                <li><strong>rl.py</strong>：實作 DDPG 強化學習演算法</li>
                <li><strong>main.py</strong>：負責訓練流程與推論展示</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        st.header("三、強化學習環境設計")
        st.subheader("3.1 三連桿機械手臂建模")
        st.markdown("系統模擬一個平面三連桿機械手臂，每個關節皆為旋轉關節（Revolute Joint），透過前向運動學計算末端位置：")
        code = """
def _get_joint_positions(self):
    p0 = self.base
    p1 = p0 + [cos(θ1), sin(θ1)] * l1
    p2 = p1 + [cos(θ1+θ2), sin(θ1+θ2)] * l2
    p3 = p2 + [cos(θ1+θ2+θ3), sin(θ1+θ2+θ3)] * l3
    return p0, p1, p2, p3 """
        st.code(code, language="python")

        st.markdown("---")

        st.subheader("3.2 狀態空間（State Space）")
        st.markdown(
            "態向量共 15 維，包含：末端與目標之相對向量與距離末端速度（平滑性評估）是否進入目標區域的狀態記憶關節角度之 sin / cos 表示前一時間步的控制動作")
        code = """
state = [
    dist_vec(2), dist(1),
    ee_vel(2), on_goal(1),
    cos(theta)(3), sin(theta)(3),
    prev_action(3)
]"""
        st.code(code, language="python")
        st.caption("此設計同時兼顧幾何關係、動態特性與控制連續性。")

        st.markdown("---")

        st.subheader("3.3 動作空間（Action Space）")
        code = """
action = np.clip(action, -0.5, 0.5)
self.arm_info[:, 1] += action * dt"""
        st.code(code, language="python")
        st.caption("每一維動作對應一個關節角速度增量，適用於連續控制型演算法。")

        st.markdown("---")

        st.subheader("3.4 獎勵函數設計（Reward Function）")
        st.markdown("本專案採用距離差分式獎勵函數：")
        st.latex(r"r = (d_{t-1} - d_t) \times 20.0")
        st.markdown("""
        <div style="font-size: 16px;">
            <strong>並額外加入：</strong>
            <ul>
                <li>末端速度懲罰（避免高頻抖動）</li>
                <li>角速度懲罰（避免多餘動作）</li>
                <li>目標命中持續獎勵（鼓勵穩定停留）</li>
            </ul>
            <p>此設計能有效避免 sparse reward 問題，並提升學習穩定度。</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        st.subheader("3.5 控制後處理與抖動抑制")
        st.markdown("為解決強化學習控制中常見的高頻抖動問題，本研究於環境層加入控制後處理：")
        code = """
# 動作平滑（低通濾波）
action = alpha * prev_action + (1 - alpha) * action

# 微小角速度死區
action[abs(action) < deadband] = 0

# 目標區 soft stop
if dist < goal_radius:
    action *= 0.2"""
        st.code(code, language="python")
        st.caption("此作法使最終行為呈現接近工業級機械手臂之平順控制效果")
        code = """
# env.py
import numpy as np
import pyglet


class ArmEnv(object):
    viewer = None
    dt = 0.1
    action_bound = [-1, 1]
    action_dim = 3
    state_dim = 15  # dist_vec(2) + dist(1) + ee_vel(2) + on_goal(1) + cos(3)+sin(3) + prev_action(3)

    def __init__(self, allow_mouse_goal=False, random_goal_on_reset=True):
        self.W, self.H = 400, 400
        self.base = np.array([200., 200.], dtype=np.float32)

        self.goal = {'x': 100., 'y': 100., 'l': 50.0}

        self.allow_mouse_goal = allow_mouse_goal
        self.random_goal_on_reset = random_goal_on_reset

        self.arm_info = np.zeros((3, 2), dtype=np.float32)
        self.arm_info[:, 0] = [100.0, 100.0, 50.0]
        self.arm_info[:, 1] = np.random.uniform(0, 2*np.pi, size=3).astype(np.float32)

        self.on_goal = 0
        self.prev_action = np.zeros(3, dtype=np.float32)
        self.prev_ee_pos = self._get_ee_pos()
        self.prev_dist = self._dist_to_goal(self.prev_ee_pos)

    # ===================== Kinematics =====================
    def _get_joint_positions(self):
        tr = self.arm_info[:, 1]
        l = self.arm_info[:, 0]

        p0 = self.base.copy()
        p1 = p0 + np.array([np.cos(tr[0]), np.sin(tr[0])]) * l[0]
        p2 = p1 + np.array([np.cos(tr[0]+tr[1]), np.sin(tr[0]+tr[1])]) * l[1]
        p3 = p2 + np.array([np.cos(tr[0]+tr[1]+tr[2]), np.sin(tr[0]+tr[1]+tr[2])]) * l[2]
        return p0, p1, p2, p3

    def _get_ee_pos(self):
        return self._get_joint_positions()[-1]

    def _dist_to_goal(self, ee_pos):
        g = np.array([self.goal['x'], self.goal['y']], dtype=np.float32)
        return float(np.linalg.norm(ee_pos - g))

    def _get_state(self):
        tr = self.arm_info[:, 1]
        ee = self._get_ee_pos()
        g = np.array([self.goal['x'], self.goal['y']], dtype=np.float32)

        dist_vec = (g - ee) / 200.0
        dist = np.linalg.norm(g - ee) / 200.0
        ee_vel = (ee - self.prev_ee_pos) / 20.0
        touch = 1.0 if self.on_goal > 0 else 0.0
        c = np.cos(tr)
        s = np.sin(tr)

        return np.concatenate([
            dist_vec, [dist], ee_vel, [touch], c, s, self.prev_action
        ]).astype(np.float32)

    # ===================== RL Interface =====================
    def step(self, action):
        done = False

        # --- clip ---
        action = np.clip(action, -0.5, 0.5).astype(np.float32)

        ee = self._get_ee_pos()
        dist = self._dist_to_goal(ee)

        # === 1️⃣ 距離自適應平滑（越近越穩）===
        if dist < self.goal['l']:
            alpha = 0.9
        else:
            alpha = 0.75
        action = alpha * self.prev_action + (1 - alpha) * action

        # === 2️⃣ 微小角速度死區（抖動殺手）===
        rate_deadband = 0.01
        action[np.abs(action) < rate_deadband] = 0.0

        # === 3️⃣ 目標區 soft stop ===
        if dist < self.goal['l'] * 0.5:
            action *= 0.2

        # --- update joint ---
        self.arm_info[:, 1] += action * self.dt
        self.arm_info[:, 1] %= (2 * np.pi)

        ee_new = self._get_ee_pos()
        dist_new = self._dist_to_goal(ee_new)

        # --- reward ---
        r = (self.prev_dist - dist_new) * 20.0
        self.prev_dist = dist_new

        ee_vel = np.linalg.norm(ee_new - self.prev_ee_pos)
        r -= 0.01 * ee_vel
        r -= 0.001 * np.sum(np.abs(action))

        if dist_new < self.goal['l']:
            r += 5.0
            self.on_goal += 1
            if self.on_goal >= 50:
                done = True
                r += 50.0
        else:
            self.on_goal = 0

        self.prev_action = action.copy()
        self.prev_ee_pos = ee_new.copy()

        return self._get_state(), float(r), done

    def reset(self):
        self.on_goal = 0
        self.prev_action = np.zeros(3, dtype=np.float32)
        self.arm_info[:, 1] = np.random.uniform(0, 2*np.pi, size=3).astype(np.float32)

        if self.random_goal_on_reset and not self.allow_mouse_goal:
            margin = 70
            self.goal['x'] = float(np.random.uniform(margin, self.W - margin))
            self.goal['y'] = float(np.random.uniform(margin, self.H - margin))

        self.prev_ee_pos = self._get_ee_pos()
        self.prev_dist = self._dist_to_goal(self.prev_ee_pos)

        return self._get_state()

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(self.arm_info, self.goal, self.base, self.allow_mouse_goal, self.W, self.H)
        self.viewer.render()

    def sample_action(self):
        return np.random.uniform(-0.5, 0.5, size=3).astype(np.float32)


# ===================== Viewer =====================
class Viewer(pyglet.window.Window):
    bar_thc = 5

    def __init__(self, arm_info, goal, base, allow_mouse_goal, W, H):
        super().__init__(width=W, height=H, caption='Arm')
        pyglet.gl.glClearColor(1, 1, 1, 1)

        self.arm_info = arm_info
        self.goal_info = goal
        self.base = base
        self.allow_mouse_goal = allow_mouse_goal
        self.batch = pyglet.graphics.Batch()

        self.point = self.batch.add(4, pyglet.gl.GL_QUADS, None,
            ('v2f', [0]*8), ('c3B', (86, 109, 249)*4))

        self.arm1 = self.batch.add(4, pyglet.gl.GL_QUADS, None, ('v2f', [0]*8), ('c3B', (249, 86, 86)*4))
        self.arm2 = self.batch.add(4, pyglet.gl.GL_QUADS, None, ('v2f', [0]*8), ('c3B', (249, 86, 86)*4))
        self.arm3 = self.batch.add(4, pyglet.gl.GL_QUADS, None, ('v2f', [0]*8), ('c3B', (249, 86, 86)*4))

    def render(self):
        self._update()
        self.switch_to()
        self.dispatch_events()
        self.clear()
        self.batch.draw()
        self.flip()

    def _update(self):
        x, y, l = self.goal_info['x'], self.goal_info['y'], self.goal_info['l']
        self.point.vertices = [
            x-l/2, y-l/2, x-l/2, y+l/2, x+l/2, y+l/2, x+l/2, y-l/2
        ]

        a1l, a2l, a3l = self.arm_info[:, 0]
        a1r, a2r, a3r = self.arm_info[:, 1]
        p0 = self.base
        p1 = p0 + np.array([np.cos(a1r), np.sin(a1r)]) * a1l
        p2 = p1 + np.array([np.cos(a1r+a2r), np.sin(a1r+a2r)]) * a2l
        p3 = p2 + np.array([np.cos(a1r+a2r+a3r), np.sin(a1r+a2r+a3r)]) * a3l

        def quad(pA, pB):
            v = pB - pA
            v = v / (np.linalg.norm(v) + 1e-6)
            n = np.array([-v[1], v[0]]) * self.bar_thc
            return np.concatenate([pA-n, pA+n, pB+n, pB-n]).astype(int).tolist()

        self.arm1.vertices = quad(p0, p1)
        self.arm2.vertices = quad(p1, p2)
        self.arm3.vertices = quad(p2, p3)

    def on_mouse_motion(self, x, y, dx, dy):
        if self.allow_mouse_goal:
            self.goal_info['x'] = float(x)
            self.goal_info['y'] = float(y)"""

        with st.expander("點擊複製完整程式碼 (env.py)"):
            st.code(code, language="python")

        st.markdown("---")

        st.header("四、學習器設計")
        st.subheader("4.1 演算法選擇：DDPG")
        st.markdown("""
        本專案採用 **Deep Deterministic Policy Gradient (DDPG)**，適用於：
        * 連續動作空間
        * 高維非線性控制問題

        **DDPG 架構包含：**
        * Actor Network：輸出控制動作
        * Critic Network：估計 Q-value
        * Target Network 與 Soft Update
        * Replay Buffer 打破資料相關性
        """)
        code = """
a_loss = -tf.reduce_mean(q)
td_error = mse(r + γ * Q_target, Q_eval)"""
        st.code(code, language="python")

        st.markdown("---")

        st.subheader("4.2 參考來源（莫煩 Python）")
        st.markdown("""
        本專案 DDPG 架構與訓練流程主要參考：
        * **莫煩（Mofan）Python 強化學習教學系列** – DDPG 實作架構

        參考重點包括：
        * Actor–Critic 分離式網路架構
        * Replay Buffer 設計方式
        * Target Network 與 Soft Update 機制

        **本研究進一步針對機械手臂控制問題進行以下改良：**
        * 更高維且連續的狀態設計
        * 距離差分型 reward
        * 控制後處理抑制抖動（莫煩原始範例未包含）     
        """)

        code = """
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

#####################  hyper parameters  ####################

LR_A = 0.001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 30000
BATCH_SIZE = 32


class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.memory_full = False
        self.sess = tf.Session()
        self.a_replace_counter, self.c_replace_counter = 0, 0

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound[1]
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [[tf.assign(ta, (1 - TAU) * ta + TAU * ea), tf.assign(tc, (1 - TAU) * tc + TAU * ec)]
                             for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]

        q_target = self.R + GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)    # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[None, :]})[0]

    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1
        if self.pointer > MEMORY_CAPACITY:      # indicator for learning
            self.memory_full = True

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.keras.layers.Dense(
                300,
                activation='relu',
                trainable=trainable,
                name='l1'
            )(s)

            a = tf.keras.layers.Dense(
                self.a_dim,
                activation='tanh',
                trainable=trainable,
                name='a'
            )(net)

            return tf.multiply(a, self.a_bound, name='scaled_a')


    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 300
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)

            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)

            q = tf.keras.layers.Dense(
                1,
                trainable=trainable,
                name='q'
            )(net)

            return q


    def save(self):
        saver = tf.train.Saver()
        saver.save(self.sess, './params', write_meta_graph=False)

    def restore(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, './params')"""

        with st.expander("點擊複製完整程式碼 (rl.py)"):
            st.code(code, language="python")

        st.markdown("---")

        st.header('五、訓練與測試流程')
        st.subheader("5.1 訓練模式")
        show_media("img/RL/rl_train.mp4","video")
        code = """
for episode:
    s = env.reset()
    for step:
        a = actor(s) + noise
        s_, r = env.step(a)
        buffer.store(s, a, r, s_)
        agent.learn()"""
        st.code(code, language="python")
        st.markdown("""
        * 每個 episode 隨機生成目標位置
        * 加入探索噪音促進探索
        * 透過 replay buffer 持續更新策略
        """)

        st.markdown("---")

        st.subheader("5.2 測試與展示模式")
        code = """
a = rl.choose_action(s)  # 無 noise
env.render()"""
        st.code(code, language="python")
        st.caption("測試時以滑鼠即時控制目標位置，直觀展示學習後之追蹤能力與穩定性。")

        st.markdown("---")

        st.header("六、專案特色與貢獻")
        st.markdown("""
        <div style="font-size: 20px; line-height: 1.8;">
            <strong>不依賴解析式逆運動學</strong>
            <ol style="margin-top: 10px;">
                <li>成功實現三連桿末端之即時追蹤控制</li>
                <li>結合強化學習與控制後處理，顯著降低抖動</li>
                <li>架構模組化，易於擴展至：
                    <ul style="list-style-type: circle; margin-left: 20px; margin-top: 5px;">
                        <li>多連桿機械手臂</li>
                        <li>3D 空間控制</li>
                        <li>實體機械手臂平台</li>
                    </ul>
                </li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        st.header("七、結果展示")
        show_media("img/RL/RL final.mp4", "video")
        code_main_py = """
# main.py
from env import ArmEnv
from rl import DDPG
import numpy as np

# ===============================
# 訓練相關參數設定（★重點）
# ===============================
MAX_EPISODES = 1200        # 三連桿至少要 1000+
MAX_EP_STEPS = 400
ON_TRAIN = False            # ★ 一定要 True 才會學

# ===============================
# Training Function
# ===============================
def train():
    # 🔹 訓練時：目標固定在一個 episode 內
    # 🔹 每個 episode 換一個目標（學泛化）
    env = ArmEnv(
        allow_mouse_goal=False,
        random_goal_on_reset=True
    )

    s_dim = env.state_dim
    a_dim = env.action_dim
    a_bound = env.action_bound

    rl = DDPG(a_dim, s_dim, a_bound)

    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_r = 0.0

        for j in range(MAX_EP_STEPS):

            # -----------------------------
            # Actor + exploration noise
            # -----------------------------
            a = rl.choose_action(s)

            # ★ noise 不要太小，不然學不到
            a = np.clip(
                np.random.normal(a, 0.15),
                -1, 1
            )

            # -----------------------------
            # Env step
            # -----------------------------
            s_, r, done = env.step(a)

            rl.store_transition(s, a, r, s_)
            ep_r += r
            s = s_

            if rl.memory_full:
                rl.learn()

            # ★ 成功就結束 episode
            if done:
                print(f'Ep {i:04d} | DONE | ep_r={ep_r:.2f} | step={j}')
                break

            if j == MAX_EP_STEPS - 1:
                print(f'Ep {i:04d} | ---- | ep_r={ep_r:.2f}')

    rl.save()
    print('[INFO] Training finished & model saved.')

# ===============================
# Evaluation Function
# ===============================
def eval():
    # 🔹 測試時：目標跟滑鼠
    env = ArmEnv(
        allow_mouse_goal=True,
        random_goal_on_reset=False
    )

    s_dim = env.state_dim
    a_dim = env.action_dim
    a_bound = env.action_bound

    rl = DDPG(a_dim, s_dim, a_bound)
    rl.restore()

    s = env.reset()
    while True:
        env.render()
        a = rl.choose_action(s)   # ★ 不加 noise
        s, r, done = env.step(a)

# ===============================
# 主程式入口
# ===============================
if ON_TRAIN:
    train()
else:
    eval()
"""
        with st.expander("點擊複製完整程式碼 (main.py)"):
          st.code(code_main_py, language="python")

        code_env_py = """
# env.py
import numpy as np
import pyglet

class ArmEnv(object):
    viewer = None
    dt = 0.1
    action_bound = [-1, 1]
    action_dim = 3
    state_dim = 15  # dist_vec(2) + dist(1) + ee_vel(2) + on_goal(1) + cos(3)+sin(3) + prev_action(3)

    def __init__(self, allow_mouse_goal=False, random_goal_on_reset=True):
        self.W, self.H = 400, 400
        self.base = np.array([200., 200.], dtype=np.float32)

        self.goal = {'x': 100., 'y': 100., 'l': 50.0}

        self.allow_mouse_goal = allow_mouse_goal
        self.random_goal_on_reset = random_goal_on_reset

        self.arm_info = np.zeros((3, 2), dtype=np.float32)
        self.arm_info[:, 0] = [100.0, 100.0, 50.0]
        self.arm_info[:, 1] = np.random.uniform(0, 2*np.pi, size=3).astype(np.float32)

        self.on_goal = 0
        self.prev_action = np.zeros(3, dtype=np.float32)
        self.prev_ee_pos = self._get_ee_pos()
        self.prev_dist = self._dist_to_goal(self.prev_ee_pos)

    # ===================== Kinematics =====================
    def _get_joint_positions(self):
        tr = self.arm_info[:, 1]
        l = self.arm_info[:, 0]

        p0 = self.base.copy()
        p1 = p0 + np.array([np.cos(tr[0]), np.sin(tr[0])]) * l[0]
        p2 = p1 + np.array([np.cos(tr[0]+tr[1]), np.sin(tr[0]+tr[1])]) * l[1]
        p3 = p2 + np.array([np.cos(tr[0]+tr[1]+tr[2]), np.sin(tr[0]+tr[1]+tr[2])]) * l[2]
        return p0, p1, p2, p3

    def _get_ee_pos(self):
        return self._get_joint_positions()[-1]

    def _dist_to_goal(self, ee_pos):
        g = np.array([self.goal['x'], self.goal['y']], dtype=np.float32)
        return float(np.linalg.norm(ee_pos - g))

    def _get_state(self):
        tr = self.arm_info[:, 1]
        ee = self._get_ee_pos()
        g = np.array([self.goal['x'], self.goal['y']], dtype=np.float32)

        dist_vec = (g - ee) / 200.0
        dist = np.linalg.norm(g - ee) / 200.0
        ee_vel = (ee - self.prev_ee_pos) / 20.0
        touch = 1.0 if self.on_goal > 0 else 0.0
        c = np.cos(tr)
        s = np.sin(tr)

        return np.concatenate([
            dist_vec, [dist], ee_vel, [touch], c, s, self.prev_action
        ]).astype(np.float32)

    # ===================== RL Interface =====================
    def step(self, action):
        done = False

        # --- clip ---
        action = np.clip(action, -0.5, 0.5).astype(np.float32)

        ee = self._get_ee_pos()
        dist = self._dist_to_goal(ee)

        # === 1️⃣ 距離自適應平滑（越近越穩）===
        if dist < self.goal['l']:
            alpha = 0.9
        else:
            alpha = 0.75
        action = alpha * self.prev_action + (1 - alpha) * action

        # === 2️⃣ 微小角速度死區（抖動殺手）===
        rate_deadband = 0.01
        action[np.abs(action) < rate_deadband] = 0.0

        # === 3️⃣ 目標區 soft stop ===
        if dist < self.goal['l'] * 0.5:
            action *= 0.2

        # --- update joint ---
        self.arm_info[:, 1] += action * self.dt
        self.arm_info[:, 1] %= (2 * np.pi)

        ee_new = self._get_ee_pos()
        dist_new = self._dist_to_goal(ee_new)

        # --- reward ---
        r = (self.prev_dist - dist_new) * 20.0
        self.prev_dist = dist_new

        ee_vel = np.linalg.norm(ee_new - self.prev_ee_pos)
        r -= 0.01 * ee_vel
        r -= 0.001 * np.sum(np.abs(action))

        if dist_new < self.goal['l']:
            r += 5.0
            self.on_goal += 1
            if self.on_goal >= 50:
                done = True
                r += 50.0
        else:
            self.on_goal = 0

        self.prev_action = action.copy()
        self.prev_ee_pos = ee_new.copy()

        return self._get_state(), float(r), done

    def reset(self):
        self.on_goal = 0
        self.prev_action = np.zeros(3, dtype=np.float32)
        self.arm_info[:, 1] = np.random.uniform(0, 2*np.pi, size=3).astype(np.float32)

        if self.random_goal_on_reset and not self.allow_mouse_goal:
            margin = 70
            self.goal['x'] = float(np.random.uniform(margin, self.W - margin))
            self.goal['y'] = float(np.random.uniform(margin, self.H - margin))

        self.prev_ee_pos = self._get_ee_pos()
        self.prev_dist = self._dist_to_goal(self.prev_ee_pos)

        return self._get_state()

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(self.arm_info, self.goal, self.base, self.allow_mouse_goal, self.W, self.H)
        self.viewer.render()

    def sample_action(self):
        return np.random.uniform(-0.5, 0.5, size=3).astype(np.float32)


# ===================== Viewer =====================
class Viewer(pyglet.window.Window):
    bar_thc = 5

    def __init__(self, arm_info, goal, base, allow_mouse_goal, W, H):
        super().__init__(width=W, height=H, caption='Arm')
        pyglet.gl.glClearColor(1, 1, 1, 1)

        self.arm_info = arm_info
        self.goal_info = goal
        self.base = base
        self.allow_mouse_goal = allow_mouse_goal
        self.batch = pyglet.graphics.Batch()

        self.point = self.batch.add(4, pyglet.gl.GL_QUADS, None,
            ('v2f', [0]*8), ('c3B', (86, 109, 249)*4))

        self.arm1 = self.batch.add(4, pyglet.gl.GL_QUADS, None, ('v2f', [0]*8), ('c3B', (249, 86, 86)*4))
        self.arm2 = self.batch.add(4, pyglet.gl.GL_QUADS, None, ('v2f', [0]*8), ('c3B', (249, 86, 86)*4))
        self.arm3 = self.batch.add(4, pyglet.gl.GL_QUADS, None, ('v2f', [0]*8), ('c3B', (249, 86, 86)*4))

    def render(self):
        self._update()
        self.switch_to()
        self.dispatch_events()
        self.clear()
        self.batch.draw()
        self.flip()

    def _update(self):
        x, y, l = self.goal_info['x'], self.goal_info['y'], self.goal_info['l']
        self.point.vertices = [
            x-l/2, y-l/2, x-l/2, y+l/2, x+l/2, y+l/2, x+l/2, y-l/2
        ]

        a1l, a2l, a3l = self.arm_info[:, 0]
        a1r, a2r, a3r = self.arm_info[:, 1]
        p0 = self.base
        p1 = p0 + np.array([np.cos(a1r), np.sin(a1r)]) * a1l
        p2 = p1 + np.array([np.cos(a1r+a2r), np.sin(a1r+a2r)]) * a2l
        p3 = p2 + np.array([np.cos(a1r+a2r+a3r), np.sin(a1r+a2r+a3r)]) * a3l

        def quad(pA, pB):
            v = pB - pA
            v = v / (np.linalg.norm(v) + 1e-6)
            n = np.array([-v[1], v[0]]) * self.bar_thc
            return np.concatenate([pA-n, pA+n, pB+n, pB-n]).astype(int).tolist()

        self.arm1.vertices = quad(p0, p1)
        self.arm2.vertices = quad(p1, p2)
        self.arm3.vertices = quad(p2, p3)

    def on_mouse_motion(self, x, y, dx, dy):
        if self.allow_mouse_goal:
            self.goal_info['x'] = float(x)
            self.goal_info['y'] = float(y)"""

        with st.expander("點擊複製完整程式碼 (env.py)"):
            st.code(code_env_py, language="python")

        code_rl_py = """
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

#####################  hyper parameters  ####################

LR_A = 0.001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 30000
BATCH_SIZE = 32


class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.memory_full = False
        self.sess = tf.Session()
        self.a_replace_counter, self.c_replace_counter = 0, 0

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound[1]
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [[tf.assign(ta, (1 - TAU) * ta + TAU * ea), tf.assign(tc, (1 - TAU) * tc + TAU * ec)]
                             for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]

        q_target = self.R + GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)    # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[None, :]})[0]

    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1
        if self.pointer > MEMORY_CAPACITY:      # indicator for learning
            self.memory_full = True

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.keras.layers.Dense(
                300,
                activation='relu',
                trainable=trainable,
                name='l1'
            )(s)

            a = tf.keras.layers.Dense(
                self.a_dim,
                activation='tanh',
                trainable=trainable,
                name='a'
            )(net)

            return tf.multiply(a, self.a_bound, name='scaled_a')


    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 300
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)

            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)

            q = tf.keras.layers.Dense(
                1,
                trainable=trainable,
                name='q'
            )(net)

            return q


    def save(self):
        saver = tf.train.Saver()
        saver.save(self.sess, './params', write_meta_graph=False)

    def restore(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, './params')"""

        with st.expander("點擊複製完整程式碼 (rl.py)"):
            st.code(code_rl_py, language="python")


    elif rl_nav == "獎勵函數詳細解說":
        st.title("獎勵函數設計（Reward Function Design）")

        st.header("一、設計目標與原則")
        st.markdown("""
        在三連桿機械手臂的連續控制問題中，獎勵函數的設計對於學習效率與策略穩定性具有決定性影響。本研究在設計獎勵函數時，遵循以下原則：

        1. **避免 Sparse Reward 問題**
           * 若僅在成功到達目標時給予獎勵，將導致學習初期回饋極度稀疏，策略難以收斂。

        2. **提供連續且具方向性的回饋**
           * Agent 需要即時知道「目前動作是否朝正確方向前進」，而非僅得知結果好壞。

        3. **兼顧穩定性與平滑性**
           * 除了到達目標，亦需避免高頻震盪與多餘關節運動，使行為更符合實際機械手臂之控制需求。

        基於上述考量，本研究採用距離差分式（Distance Difference-based）獎勵設計，並搭配多項懲罰與成功獎勵項。
        """)

        st.markdown("---")

        st.header("二、核心獎勵項：距離差分獎勵")
        st.subheader("2.1 定義方式")
        st.markdown("主要獎勵項定義為末端執行器到目標距離的變化量：")
        st.latex(r"R_{dist} = (d_{t-1} - d_t) \times k")
        st.markdown("""
        其中：
        * $d_{t-1}$：前一時間步末端與目標之距離 (`prev_dist`)
        * $d_t$：目前時間步末端與目標之距離 (`current_dist`)
        * $k$：距離縮放係數（本研究設定為 20）
        """)

        st.subheader("2.2 設計動機")
        st.markdown("""
        此設計具備以下特性：
        * 若末端靠近目標 ($d_t < d_{t-1}$)，則獲得 **正獎勵**
        * 若末端遠離目標，則獲得 **負獎勵**
        * 當末端停滯不動時，獎勵趨近於 0

        相較於直接使用 `-distance` 作為 reward，此方式能：
        * 提供更明確的「方向性梯度」
        * 避免在距離很遠時 reward 變化過小而導致學習停滯
        * 提升學習初期的探索效率

        因此，此距離差分式獎勵能有效引導 Agent 逐步學習朝向目標移動的策略。
        """)

        st.markdown("---")

        st.header("三、穩定性相關懲罰項（Stability-related Penalties）")

        st.subheader("3.1 末端速度懲罰（End-effector Velocity Penalty）")
        st.markdown("為避免末端在接近目標時產生高頻震盪，本研究引入末端速度懲罰項：")
        st.latex(r"R_{vel} = - \lambda_v \cdot \| \mathbf{v}_{ee} \|")
        st.markdown("""
        其中：
        * $\mathbf{v}_{ee}$ 代表末端執行器的速度向量 (或 $\| ee_t - ee_{t-1} \|$)
        * $\lambda_v$ 為權重係數（本研究設定為 0.01）

        **設計意義**
        * 抑制末端在目標附近來回擺動
        * 鼓勵平滑且連續的運動軌跡
        * 避免策略透過「劇烈修正」來換取微小距離改善

        此懲罰項使 Agent 在學習過程中自然偏好低速度、穩定收斂的控制行為。
        """)

        st.subheader("3.2 關節角速度懲罰（Action Magnitude Penalty）")
        st.markdown("為進一步減少不必要的關節動作，本研究對動作大小加入懲罰：")
        st.latex(r"R_{action} = - \lambda_a \cdot \sum_i |a_i|")
        st.markdown("""
        其中：
        * $a_i$ 為第 i 個關節之角速度控制量
        * $\lambda_a$ 為懲罰權重（本研究設定為 0.001）

        **設計意義**
        * 降低關節在目標附近的高頻小幅震盪
        * 引導 Actor 學習「能不動就不動」的策略
        * 提升整體控制的能源效率與平順性

        此項在不影響追蹤能力的前提下，有效改善視覺上的抖動問題。
        """)

        st.markdown("---")

        st.header("四、成功與終止獎勵（Success and Termination Reward）")
        st.subheader("4.1 進入目標區域獎勵")
        st.markdown("當末端進入目標半徑範圍內時，給予即時獎勵：")
        st.latex(r"R_{in\_goal} = +5.0")
        st.markdown("""
        此設計能：
        * 明確告知 Agent「已達到目標」
        * 加速策略收斂至目標附近
        """)

        st.subheader("4.2 穩定停留獎勵與 Episode 終止")
        st.markdown("為避免策略僅短暫觸碰目標後離開，本研究進一步設計持續命中機制：")
        st.markdown("""
        若連續 **N** 步停留在目標區：
        """)
        st.latex(r"R_{terminal} = +r_{success}")
        st.markdown("""
        其中：
        * $N$ 為連續停留步數（本研究設定為 50）
        * $r_{success}$ 為成功終止獎勵（本研究設定為 50）

        **設計意義**
        * 鼓勵末端「穩定停留」而非短暫碰觸
        * 強化長期穩定控制行為
        * 提供明確 episode 結束條件，有助於策略收斂
        """)

        st.markdown("---")

        st.header("五、整體獎勵函數總結")
        st.markdown("綜合上述設計，本研究之獎勵函數可表示為：")
        st.latex(r"R_{total} = R_{dist} + R_{vel} + R_{action} + R_{in\_goal} + R_{terminal}")
        st.markdown("""
        此獎勵函數同時兼顧：
        1. **引導性（Guidance）**：距離差分項
        2. **穩定性（Stability）**：速度與動作懲罰
        3. **成功性（Success）**：目標命中與終止獎勵

        使得 Agent 能在連續控制問題中，學習到兼具準確性與平順性的控制策略。
        """)

        st.markdown("---")

        st.header("六、設計成效說明（實驗觀察）")
        st.markdown("""
        實驗結果顯示，透過上述獎勵設計：
        * 學習初期能快速學會朝目標移動
        * 中後期能穩定停留於目標區域
        * 配合控制後處理後，可顯著降低高頻抖動現象

        顯示本獎勵函數設計適合應用於連續型機械手臂控制問題。
        """)

elif menu == "TurtleBot Burger平台":
    st.title("TurtleBot Burger平台")
    st.markdown("---")

    st.header("一、實作過程：避障與導航")
    st.subheader('步驟 1：VirtualBox 設定')
    show_media("img/Turtlebot/1.jpg")

    st.markdown("---")

    st.subheader('步驟 2：網路設定與建立工作空間')
    col_tb2_1, col_tb2_2 = st.columns(2)
    with col_tb2_1:
        show_media("img/Turtlebot/2.jpg")
    with col_tb2_2:
        show_media("img/Turtlebot/2-2.jpg")

    st.markdown("---")

    st.subheader('步驟 3：安裝 Turtlebot3 套件')
    show_media("img/Turtlebot/3.jpg")
    st.code("git clone https://github.com/ROBOTIS-GIT/turtlebot3\ncd ..\ncatkin_make", language="bash")

    st.markdown("---")

    st.subheader('步驟 4：啟動 ROS Core 與連接 Turtlebot3')
    show_media("img/Turtlebot/ros core.jpg")
    st.code("source /opt/ros/noetic/setup.bash\nsource ~/mde_ws/devel_isolated/setup.bash\nroscore", language="bash")

    st.markdown("---")

    st.subheader('步驟 5：啟動 SLAM 建圖')
    show_media("img/Turtlebot/slam.jpg")
    st.code("export TURTLEBOT3_MODEL=burger\nroslaunch turtlebot3_slam turtlebot3_slam.launch", language="bash")

    st.markdown("---")

    st.subheader('步驟 6：掃描地形')
    show_media("img/Turtlebot/map.jpg")
    show_media("img/Turtlebot/real.jpg")


    st.markdown("---")

    st.subheader('步驟 7：儲存與開啟地圖')
    col_tb11_1, col_tb11_2 = st.columns(2)
    with col_tb11_1:
        show_media("img/Turtlebot/save map.jpg")
    with col_tb11_2:
        show_media("img/Turtlebot/save map 2.jpg")
    st.code("rosrun map_server map_saver -f ~/mde_ws/map00", language="bash")

    st.markdown("---")

    st.header("二、路徑規劃與結果展示")
    st.markdown('### 成功導航使 Turtlebot3 到目的地，並且避開障礙物')
    show_media("img/Turtlebot/road.jpg")

    st.markdown("---")

    show_media("img/Turtlebot/final.mp4", "video")

elif menu == "Streamlit UI設計與資料可視化":
    st.title("Streamlit UI 設計與資料可視化")
    st.subheader("步驟 1：安裝 streamlit 套件")
    st.code("pip install streamlit", language="bash")

    st.subheader("步驟 2：啟動 streamlit，開啟網頁")
    st.code("streamlit run app.py", language="bash")

    st.subheader("步驟 3：開啟官網程式庫")
    st.markdown("開啟 [🔗Streamlit Cheat Sheet](https://cheat-sheet.streamlit.app/)，從中可獲的各種程式以供網頁書寫")