# เสียงกระซิบในคลิปลับ
## Audio Steganography in Video with AES-256-GCM Dual-Layer Security

แอปเดสก์ท็อป Windows สำหรับซ่อนไฟล์เสียงไว้ในวิดีโอด้วย **frequency-domain steganography** เท่านั้น โดยมี 3 โหมด: DCT, DWT และ Hybrid DWT-DCT

> **คำแนะนำด้านความปลอดภัย:** โปรเจกต์นี้เป็น reference implementation สำหรับงานวิจัย/การเรียนรู้ ควรใช้วิดีโอที่เข้ารหัสด้วย PNG frames + FFV1 (`.mkv`) เพื่อรักษาสัมประสิทธิ์ให้เสถียรที่สุด หากนำวิดีโอไปบีบอัดใหม่ด้วย H.264/H.265 หรือส่งผ่านแพลตฟอร์มที่ re-encode อาจทำให้การกู้คืนล้มเหลว

## สิ่งที่มีให้

- PyQt6 dark-theme GUI: Hide Audio, Extract Audio, Quality Metrics
- FFmpeg → Opus pipeline ปรับ bitrate 16–32 kbps
- Dual-layer AES-256-GCM: PBKDF2-HMAC-SHA256, 100,000 iterations, salt 16 bytes, nonce 12 bytes ต่อชั้น
- Authenticated encryption: password ผิดหรือข้อมูลถูกแก้ไขจะถูกปฏิเสธ
- DCT 8×8, DWT Haar และ Hybrid DWT-DCT
- Magic bytes + mode + payload length + SHA-256 integrity header
- MP4 (H.264 Lossless RGB) หรือ Matroska (FFV1) พร้อมคงแทร็กเสียงเดิมของวิดีโอ
- PSNR/SSIM เบื้องต้นแบบ frame-by-frame
- type hints, worker thread, progress, log และ error handling

## ติดตั้งบน Windows

1. ติดตั้ง Python 3.10+ และเพิ่มเข้า PATH
2. ติดตั้ง FFmpeg และให้คำสั่ง `ffmpeg`/`ffprobe` ใช้งานได้จาก Command Prompt
3. เปิดโฟลเดอร์นี้ แล้วรัน:

```powershell
py -3.10 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

หรือดับเบิลคลิก `run_windows.bat`

## วิธีใช้งาน

### ซ่อนเสียง

1. เลือกวิดีโอต้นฉบับ และไฟล์เสียง
2. ตั้ง password ที่มีความยาวอย่างน้อย 8 ตัวอักษร
3. เลือกโหมดและ bitrate Opus
4. เลือกไฟล์ผลลัพธ์ รองรับทั้ง `.mp4` (สำหรับเปิดเล่นทั่วไป) และ `.mkv`
5. ตรวจสอบค่า capacity แล้วกด **Start Embedding**

### กู้คืนเสียง

1. เลือก stego video และ mode เดียวกับที่ใช้ฝัง
2. กรอก password เดิม
3. กด **Extract & Play**
4. เมื่อสำเร็จสามารถฟังหรือ Save เป็น `.opus` ได้

### วัดคุณภาพ

เลือก original video กับ stego video แล้วกด **Calculate PSNR / SSIM** ค่าเป็นค่าเฉลี่ยของเฟรมที่จับคู่กันได้

## รูปแบบข้อมูลเข้ารหัส

แพ็กเกจไบนารีมี magic `ASV1`, version, iterations, salt, nonce ของแต่ละชั้น, outer tag และ ciphertext โดยชั้นในเข้ารหัส Opus ก่อน แล้วชั้นนอกเข้ารหัสผลลัพธ์ของชั้นในอีกครั้ง AAD ผูก header กับ ciphertext ทำให้แก้ไขข้อมูลหรือสลับพารามิเตอร์แล้วตรวจพบ

## หมายเหตุเรื่อง capacity

capacity ที่แสดงเป็นค่าประมาณจากจำนวนเฟรมและจำนวนบล็อกที่ฝังได้จริง แต่ payload มี header และ digest เพิ่มเติมด้วย หาก payload ใหญ่เกิน capacity แอปจะแจ้งข้อผิดพลาดก่อนเขียนไฟล์

## ทดสอบ core แบบไม่เปิด GUI

```powershell
python -m unittest discover -s tests -v
```
