@echo off
:: Mengaktifkan virtual environment di folder yoloenv
call yoloenv\Scripts\activate

:: Menginstal PyTorch dengan versi tertentu dan CUDA 11.8
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118

:: Menginstal opencv_contrib_python dari file .whl
pip install opencv_contrib_python-4.10.0.84-cp37-abi3-win_amd64.whl

:: Memberi tahu bahwa semua perintah telah selesai dijalankan
echo Semua perintah telah selesai dijalankan.
pause
