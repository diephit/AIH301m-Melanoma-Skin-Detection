import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
##############################################################
model = load_model('MobileNetV2_melanoma_detection.h5')  # Tải mô hình đã huấn luyện
##############################################################
st.title("Hệ thống phát hiện ung thư da Melanoma chạy bằng cơm")

uploaded = st.file_uploader("Tải lên ảnh da của bạn", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded)
    st.image(image, caption="Ảnh đã tải lên", use_column_width=True)
    
    def preprocessimg(image):# Tiền xử lý ảnh
        img_array = img_to_array(image.resize((224, 224)))  # Resize ảnh về kích thước phù hợp với mô hình
        img_array = np.expand_dims(img_array, axis=0)  # Thêm chiều batch
        return img_array / 255.0
    
    def predict(img_array):
    # Dự đoán
        prediction = model.predict(img_array)
        return prediction
    
    input = preprocessimg(image)
    answer = predict(input)
    classes = ['Not Melanoma', 'Melanoma']
    
    pred_class = np.argmax(answer)
    confidence, pred_class_name = answer[0][pred_class], classes[pred_class]
    
    confidence_percentage = round(confidence * 100, 2)

    data = {
        'Predicted Class': [pred_class_name],
        'Confidence (%)': [confidence_percentage]
    }
    df = pd.DataFrame(data)
    st.write(df)