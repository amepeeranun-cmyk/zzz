"""
Ehrlich-Z Web App
==================
ระบบตรวจหาเชื้อ Ehrlichia canis จากภาพสเมียร์เลือดสุนัข

วิธีรันในเครื่อง:
    pip install streamlit requests opencv-python-headless pillow
    streamlit run app.py

วิธี deploy ให้คนอื่นใช้ได้ฟรี:
    1. อัปโหลดไฟล์นี้ + requirements.txt ขึ้น GitHub (repo แยกต่างหาก)
    2. ไปที่ share.streamlit.io -> เชื่อม GitHub -> เลือก repo -> Deploy
    3. ได้ลิงก์เว็บสาธารณะทันที (ฟรี ไม่ต้องมีเซิร์ฟเวอร์ของตัวเอง)
"""

import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image
import io

# ============================================================
# ตั้งค่า Roboflow (แก้ให้ตรงกับของจริง)
# ============================================================
ROBOFLOW_API_KEY = "2G5Lbz1TQC0doTcK4YiO"          # <-- แก้เป็นของจริง
MODEL_ID = "my-first-project-cp4zt/1"                # <-- แก้เป็นของจริง (project/version)
TARGET_CLASS = "Ehrilchia canis"                     # <-- ต้องตรงกับที่ label ไว้ใน Roboflow เป๊ะๆ
DEFAULT_CONFIDENCE = 0.5

st.set_page_config(page_title="Ehrlich-Z", page_icon="🩸", layout="wide")


# ============================================================
# ฟังก์ชันหลัก (ย้ายมาจาก Colab)
# ============================================================
def infer_image_bytes(image_bytes, confidence=0.5):
    """ส่งภาพ (bytes) เข้าโมเดล Roboflow ผ่าน REST API"""
    url = f"https://detect.roboflow.com/{MODEL_ID}"
    params = {
        "api_key": ROBOFLOW_API_KEY,
        "confidence": int(confidence * 100),
    }
    response = requests.post(
        url,
        params=params,
        files={"file": image_bytes}
    )
    response.raise_for_status()
    return response.json()


def split_image_into_9(pil_image):
    """แบ่งภาพ PIL เป็น 9 ส่วน (3x3 grid) คืนค่าเป็น list ของภาพ PIL"""
    img = np.array(pil_image.convert("RGB"))
    h, w = img.shape[:2]
    tile_w = w // 3
    tile_h = h // 3

    tiles = []
    for row in range(3):
        for col in range(3):
            x_start = col * tile_w
            y_start = row * tile_h
            x_end = w if col == 2 else x_start + tile_w
            y_end = h if row == 2 else y_start + tile_h
            tile = img[y_start:y_end, x_start:x_end]
            tiles.append(Image.fromarray(tile))
    return tiles


def pil_to_bytes(pil_image):
    """แปลงภาพ PIL เป็น bytes สำหรับส่งเข้า API"""
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG")
    buf.seek(0)
    return buf


def draw_boxes(pil_image, predictions):
    """วาดกรอบ bounding box ลงบนภาพ สีแดง=ติดเชื้อ สีเขียว=ปกติ"""
    img = np.array(pil_image.convert("RGB"))
    for pred in predictions:
        x, y = int(pred["x"]), int(pred["y"])
        w, h = int(pred["width"]), int(pred["height"])
        x1, y1 = x - w // 2, y - h // 2
        x2, y2 = x + w // 2, y + h // 2
        is_infected = pred["class"].lower() == TARGET_CLASS.lower()
        color = (255, 0, 0) if is_infected else (0, 200, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    return Image.fromarray(img)


def calculate_infection_rate(total_cells, infected_cells):
    """คำนวณ % infected และระดับความรุนแรง"""
    if total_cells == 0:
        return 0.0, "ไม่พบเซลล์เม็ดเลือดขาวในภาพ — ลองถ่ายภาพใหม่"
    percent = (infected_cells / total_cells) * 100
    if percent == 0:
        severity = "ไม่พบการติดเชื้อ (Negative)"
    elif percent <= 2:
        severity = "ระดับต่ำ (Mild)"
    elif percent <= 5:
        severity = "ระดับปานกลาง (Moderate)"
    else:
        severity = "ระดับสูง (Severe)"
    return round(percent, 2), severity


# ============================================================
# หน้าเว็บ (UI)
# ============================================================
st.title("🩸 Ehrlich-Z")
st.caption("ระบบ AI คัดกรองเชื้อ Ehrlichia canis จากภาพสเมียร์เลือดสุนัข")

with st.sidebar:
    st.header("⚙️ ตั้งค่า")
    confidence = st.slider("Confidence Threshold", 0.0, 1.0, DEFAULT_CONFIDENCE, 0.05)
    st.markdown("---")
    st.markdown(
        "**วิธีใช้**\n"
        "1. อัปโหลดภาพสเมียร์เลือดที่ถ่ายจากกล้องจุลทรรศน์\n"
        "2. รอระบบวิเคราะห์ (ใช้เวลาไม่กี่วินาที)\n"
        "3. ดูผล % เซลล์ติดเชื้อและระดับความรุนแรง"
    )
    st.markdown("---")
    st.caption("⚠️ ผลลัพธ์เป็นการคัดกรองเบื้องต้นเท่านั้น ไม่ใช่การวินิจฉัยขั้นสุดท้าย ควรให้สัตวแพทย์ตรวจสอบซ้ำและยืนยันด้วย ELISA/PCR")

uploaded_file = st.file_uploader(
    "อัปโหลดภาพสเมียร์เลือด (JPG, PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("ภาพต้นฉบับ")
        st.image(pil_image, use_container_width=True)

    with st.spinner("กำลังวิเคราะห์ภาพ... (แบ่งเป็น 9 ส่วน และตรวจจับด้วย AI)"):
        tiles = split_image_into_9(pil_image)

        total_cells = 0
        infected_cells = 0
        annotated_tiles = []

        progress = st.progress(0)
        for i, tile in enumerate(tiles):
            try:
                img_bytes = pil_to_bytes(tile)
                result = infer_image_bytes(img_bytes, confidence=confidence)
                predictions = result.get("predictions", [])

                total_cells += len(predictions)
                n_infected = sum(
                    1 for p in predictions
                    if p["class"].lower() == TARGET_CLASS.lower()
                )
                infected_cells += n_infected

                annotated_tiles.append(draw_boxes(tile, predictions))
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดที่ tile {i+1}: {e}")
                annotated_tiles.append(tile)

            progress.progress((i + 1) / 9)

    percent, severity = calculate_infection_rate(total_cells, infected_cells)

    with col2:
        st.subheader("ผลการวิเคราะห์")
        m1, m2, m3 = st.columns(3)
        m1.metric("เซลล์ทั้งหมด", total_cells)
        m2.metric("เซลล์ติดเชื้อ", infected_cells)
        m3.metric("% Infected", f"{percent}%")

        if percent == 0:
            st.success(f"**{severity}**")
        elif percent <= 2:
            st.info(f"**{severity}**")
        elif percent <= 5:
            st.warning(f"**{severity}**")
        else:
            st.error(f"**{severity}**")

    st.markdown("---")
    st.subheader("ภาพที่ตรวจจับแล้ว (แดง = สงสัยติดเชื้อ, เขียว = ปกติ)")
    grid_cols = st.columns(3)
    for i, tile_img in enumerate(annotated_tiles):
        with grid_cols[i % 3]:
            st.image(tile_img, caption=f"ส่วนที่ {i+1}", use_container_width=True)

else:
    st.info("⬆️ อัปโหลดภาพสเมียร์เลือดเพื่อเริ่มวิเคราะห์")
