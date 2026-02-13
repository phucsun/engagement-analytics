# AI Engagement Analytics System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![MediaPipe](https://img.shields.io/badge/MediaPipe-FaceMesh-orange?style=for-the-badge&logo=google)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-green?style=for-the-badge&logo=opencv)
![Status](https://img.shields.io/badge/Status-Research_Grade-red?style=for-the-badge)

> **Hệ thống phân tích mức độ tập trung dựa trên hành vi (Behavioral Analytics) sử dụng Computer Vision và Xử lý tín hiệu số (DSP).**

Hệ thống này được thiết kế để giải quyết bài toán giám sát học tập trực tuyến (E-learning) hoặc thi cử (Proctoring) thông qua việc phân tích video offline. Khác với các mô hình Deep Learning "hộp đen" (Black-box), hệ thống sử dụng tiếp cận **Geometric Rule-based** kết hợp với **Multi-pass Signal Processing** để đảm bảo tính giải thích được (Explainability), độ chính xác cao và khả năng tự thích nghi với từng người dùng.

---

## Tính Năng Nổi Bật (Research Highlights)

Hệ thống V14 vượt trội so với các giải pháp Real-time đơn giản nhờ kiến trúc xử lý 4 pha (4-Pass Architecture):

### 1. Robust Signal Processing (Xử lý tín hiệu bền vững)
- **Savitzky-Golay Filter:** Thay vì dùng Moving Average gây trễ (lag), hệ thống sử dụng bộ lọc Savitzky-Golay để làm mượt tín hiệu nhiễu nhưng vẫn giữ nguyên biên độ và độ sắc nét của các hành vi nhanh (như chớp mắt, nói chuyện).
- **Dual-Window Smoothing:** Áp dụng cửa sổ lọc ngắn (~0.3s) cho mắt và cửa sổ dài (~1.0s) cho đầu để tối ưu hóa độ nhạy.

### 2. Advanced Calibration (Cân chỉnh nâng cao)
- **Top-K Median:** Tự động tìm ngưỡng mắt mở (EAR Base) dựa trên top 20% khung hình tốt nhất, giúp hệ thống hoạt động đúng ngay cả khi người dùng mắt nhỏ hoặc buồn ngủ phần lớn thời gian.
- **Cross-Modal Filtering:** Chỉ học dữ liệu mống mắt (Gaze) khi đầu đang giữ thẳng, loại bỏ nhiễu do góc quay đầu gây ra.
- **Histogram Mode Estimation:** Sử dụng đỉnh mật độ phân phối (Mode) để tìm tư thế ngồi chuẩn thay vì trung vị (Median) dễ bị sai lệch bởi dữ liệu nhiễu.

### 3. Logic Hành Vi Thông Minh (Behavioral Logic)
- **Hysteresis (Schmitt Trigger):** Cơ chế ngưỡng đôi (Ngưỡng vào > Ngưỡng ra) giúp loại bỏ hoàn toàn hiện tượng nhấp nháy trạng thái (State Flickering) ở vùng ranh giới.
- **Non-linear Scoring:** Hàm phạt mũ (Exponential Penalty) trừng phạt nặng các hành vi mất tập trung kéo dài.
- **Strict Counters:** Bộ đếm frame nghiêm ngặt giúp phân biệt rõ ràng giữa *Chớp mắt* (Blink) và *Ngủ gật* (Microsleep).

---

## Cài Đặt

### Yêu cầu hệ thống
- Python 3.8 trở lên
- Webcam (nếu muốn test luồng video) hoặc File Video MP4.

### Các bước cài đặt

**1. Clone repository:**
```bash
git clone [https://github.com/USERNAME_CUA_BAN/engagement-analytics.git](https://github.com/USERNAME_CUA_BAN/engagement-analytics.git)
cd engagement-analytics
