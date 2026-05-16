@echo off
echo ================================
echo  YOLO Export + Send to Pi
echo ================================

REM ---- Config ----
set PI_USER=ai
set PI_IP=192.168.0.102
set MODEL=..\T1\test_model_detection\weights\best.pt
set ONNX=..\T1\test_model_detection\weights\best.onnx

echo.
echo [1/2] Exporting best.pt to ONNX...
python -c "from ultralytics import YOLO; YOLO('%MODEL%').export(format='onnx', imgsz=640)"
if errorlevel 1 (
    echo ERROR: Export failed. Make sure ultralytics is installed.
    echo Run: pip install ultralytics
    pause
    exit /b 1
)

echo.
echo [2/2] Transferring best.onnx to Raspberry Pi...
scp %ONNX% %PI_USER%@%PI_IP%:/home/ai/ai/piTEST/best.onnx
if errorlevel 1 (
    echo ERROR: Transfer failed. Make sure the Pi is on and SSH is enabled.
    pause
    exit /b 1
)

echo.
echo Done! Now on the Pi run:
echo   source /opt/ai_venv/bin/activate
echo   cd /home/ai/ai
echo   python3 piTEST/yolo_stream.py
echo.
pause
