# Marker-based Visual Pose Estimation Demo (ArUco / ChArUco)


Минимально работающее демо системы визуального позиционирования камеры относительно маркера на базе **OpenCV + ArUco / ChArUco**.

Проект реализует:

- калибровку камеры по **ChArUco board**,
- оценку 6D-позы (**x, y, z, roll, pitch, yaw**) камеры относительно ArUco-маркера,
- визуализацию позы в видео,
- вывод позы в консоль с заданной частотой и логирование в файл

## Зависимости
- Python ≥ 3.9  
- OpenCV (contrib)  
- NumPy
## Пайплайн использования
1. Генерация ChArUco-доски
```bash
python generate_charuco_board.py
```


2. Сбор изображений для калибровки
```bash
python capture_charuco.py
```


3. Калибровка камеры
```
python calibrate_charuco.py captures/charuco_YYYYMMDD_HHMMSS
```
Результат:
camera_calibration.json в той же папке

4. Оценка позы ArUco-маркера
```bash
python detect_pose_aruco.py captures/charuco_YYYYMMDD_HHMMSS/camera_calibration.json
```

OpenCV camera frame:

X — вправо
Y — вниз
Z — вперёд от камеры
Поза описывает маркер относительно камеры
Углы: roll, pitch, yaw в градусах (ZYX)