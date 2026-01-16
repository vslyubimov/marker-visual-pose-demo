# Marker-based Visual Pose Estimation Demo (ArUco / ChArUco)


Минимально работающее демо системы визуального позиционирования камеры относительно маркера на базе **OpenCV + ArUco / ChArUco**.

Проект реализует:

- калибровку камеры по **ChArUco board**,
- оценку 6D-позы (**x, y, z, roll, pitch, yaw**) камеры относительно ArUco-маркера,
- визуализацию позы в видео,
- вывод позы в консоль с заданной частотой и логирование в файл

## Зависимости
- Python ≥ 3.9  (3.13.9 used)
- OpenCV (contrib) >= 4.7 (4.12 used)
- NumPy (2.2.6 used)
## Пайплайн использования
config.json содержит настраиваемые параметры ChArUco-доски, камеры, ArUco-маркера.

1. Генерация ChArUco-доски. 
```bash
python generate_charuco_board.py
```

2. Сбор изображений для калибровки
```bash
python capture_charuco.py
```
Изображения сохраняются на клавишу "space" в папку captures/charuco_YYYYMMDD_HHMMSS  

3. Калибровка камеры
```
python calibrate_charuco.py captures/charuco_YYYYMMDD_HHMMSS
```
Результаты калибровки записывается в captures/charuco_YYYYMMDD_HHMMSS/camera_calibration.json

4. Оценка позы ArUco-маркера. 

Указать путь до json с калибровкой камеры.
```bash
python detect_pose_aruco.py captures/charuco_YYYYMMDD_HHMMSS/camera_calibration.json
```
В консоли, в HUD и в логах (captures/logs) отображатся: x, y, z, roll, pitch, yaw. 


OpenCV camera frame:  
  - X — вправо
  - Y — вниз
  - Z — вперёд от камеры

Поза ArUco-маркера относительно камеры в углах:

Углы: roll, pitch, yaw в градусах (ZYX)


Modern OpenCV ArUco API используется (нет deprecated ArUco функций).
