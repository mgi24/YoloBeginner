@echo off
:: Membuat folder "yoloproject"
mkdir yoloproject

:: Membuat folder "yoloenv"
mkdir yoloenv

:: Membuat virtual environment di dalam folder "yoloenv"
python -m venv yoloenv

:: Mengaktifkan virtual environment
call yoloenv\Scripts\activate

:: Menginstal ultralytics
pip install ultralytics

:: Memberi tahu bahwa semua perintah telah selesai dijalankan
echo Semua perintah telah selesai dijalankan.
pause
