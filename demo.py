import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image

# Load mô hình đã train
model = load_model("model.h5")

# Từ điển nhãn
class_labels = {
    0: "Melanocytic nevi",
    1: "Basal cell carcinoma",
    2: "Benign keratosis-like lesions",
    3: "Dermatofibroma",
    4: "Melanoma",
    5: "Actinic keratoses",
    6: "Vascular lesions"
}

# Mô tả bệnh
disease_info = {
    "Melanocytic nevi": "U sắc tố lành tính của da, thường xuất hiện dưới dạng nốt ruồi.",
    "Melanoma": "Ung thư da nguy hiểm nhất, phát triển từ tế bào sắc tố.",
    "Benign keratosis-like lesions": "Tổn thương da lành tính thường gặp ở người lớn tuổi.",
    "Basal cell carcinoma": "Loại ung thư da phổ biến nhất, thường do tiếp xúc nhiều với ánh nắng.",
    "Actinic keratoses": "Tổn thương tiền ung thư do ánh nắng mặt trời gây ra.",
    "Vascular lesions": "Tổn thương mạch máu trên da như u máu hoặc xuất huyết dưới da.",
    "Dermatofibroma": "Tổn thương da lành tính, thường xuất hiện dưới dạng nốt cứng nhỏ."
}

def preprocess_image(img):
    img = img.resize((224, 224)) 
    img_array = np.array(img)  
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = preprocess_input(img_array) 
    return img_array

# Giao diện Streamlit
st.title("PHÂN LOẠI TỔN THƯƠNG DA")
st.write("Upload ảnh để dự đoán.")

# Upload ảnh từ người dùng
uploaded_file = st.file_uploader("Chọn ảnh", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Ảnh đã tải lên", use_column_width=True)

    img_array = preprocess_image(img)
    
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    predicted_label = class_labels[predicted_class]

    st.subheader(f"Kết quả dự đoán là {predicted_label}:  {disease_info[predicted_label]}")

