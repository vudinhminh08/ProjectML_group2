# ProjectML_group2

Ở đây sẽ có 2 phần cần chú ý là build model và phần dùng camera nhận diện mặt

- Build model: File age-prediction-from-images-cnn kia là notebook build model nhận diện tuổi, đã có kết quả trong đó gồm rsme và r^2. Sau đó sẽ save ra 1 file model riêng modelV1.h5 để áp dụng vào phần camera
- Phần nhận diện mặt: Camera.py kia là setup ra GUI để nhận diện mặt, khi đó sẽ capture hình ảnh có khuôn mặt để sử dụng để gọi hàm detect trong face_detection.py sẽ predict ra tuổi và hiện lên màn hình luôn
- Các file XML kia là trọng số của thư viện Haarcascade để nhận diện được mặt người thôi
