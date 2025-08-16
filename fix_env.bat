@echo off
echo 🛠 Fixing Python environment...

:: Uninstall broken versions
pip uninstall -y numpy scipy supervision ultralytics opencv-python opencv-contrib-python

:: Install stable versions for your project
pip install numpy==2.3.1
pip install scipy==1.16.1
pip install supervision==0.26.0
pip install ultralytics==8.3.161
pip install opencv-python==4.11.0.86 opencv-contrib-python==4.11.0.86

:: ✅ Environment check
echo.
echo 🔍 Checking installed versions...
python -c "import numpy, scipy, supervision, cv2, ultralytics; \
print('numpy:', numpy.__version__); \
print('scipy:', scipy.__version__); \
print('supervision:', supervision.__version__); \
print('opencv:', cv2.__version__); \
import ultralytics; print('ultralytics:', ultralytics.__version__)"
