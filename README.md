# 🧠 PersonalityTypes API

## 🚀 Giới thiệu
API PersonalityTypes sử dụng tiêu chí Keirsey để dự đoán 4 nhóm tính cách chính dựa trên các đặc điểm tâm lý.

### 📊 Hiệu suất mô hình:
- **✅ Accuracy tổng thể:** 91%
- **📈 Recall trung bình:** 84,5%
- **🎯 Precision trung bình:** 79,25%
- **🛡 Hiệu suất dự đoán nhóm Guardians:** tạm thời ở mức ổn

## 🛠 Công nghệ sử dụng
- **🤖 Mô hình:** LightGBM
- **🌐 API Framework:** Flask
- **🚀 Triển khai:** Render

## 🔥 Cách sử dụng API
### 📍 Endpoint:
**GET**: `https://personalitytypes.onrender.com/predict`

### 📝 Body JSON mẫu:
```json
{
    "extraversion_score": 9.47,
    "thinking_score": 6.03,
    "age": 19,
    "sensing_score": 7.14,
    "judging_score": 4.3
}
```

### 🔍 Các biến đầu vào:
- `🗣 extraversion_score`: Điểm thể hiện mức độ hướng ngoại
- `🧠 thinking_score`: Điểm thể hiện mức độ tư duy logic
- `🎂 age`: Độ tuổi của người được đánh giá
- `👀 sensing_score`: Điểm thể hiện mức độ cảm nhận chi tiết
- `⚖ judging_score`: Điểm thể hiện mức độ quyết đoán

## 📚 Tham khảo
Xem nội dung trình bày chung về dự án và LightGBM tại đây: [Google Drive](https://drive.google.com/file/d/1GtmmvBQWHLMUmyfvwpXf_GFxNkTXjjoI/view?usp=sharing)

