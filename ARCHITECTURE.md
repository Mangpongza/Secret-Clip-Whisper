# สถาปัตยกรรมและ Security Notes

## Data flow

```text
Audio input
  → FFmpeg/libopus (16–32 kbps)
  → AES-256-GCM ชั้นใน
  → AES-256-GCM ชั้นนอก
  → Magic/Version/Mode/Length/SHA-256 header
  → DCT | DWT | DWT-DCT embedder
  → PNG frame sequence
  → FFmpeg FFV1 Matroska + original audio stream
```

Extraction ทำย้อนลำดับ โดยตรวจ Magic/Mode/Length ก่อน จำกัด length ไม่ให้เกิน capacity ของวิดีโอ ตรวจ SHA-256 แล้วจึงทำ authenticated decryption สองชั้น

## Package boundaries

- `audio.py` — FFmpeg Opus encode/decode และ dependency checks
- `crypto.py` — PBKDF2-HMAC-SHA256 และ AES-256-GCM สองชั้น
- `payload.py` — steganography header, length และ digest
- `steg.py` — QIM บน DCT/DWT/Hybrid frequency coefficients
- `pipeline.py` — orchestration, frame allocation, FFV1 reassembly
- `metrics.py` — frame PSNR และ local-window SSIM
- `gui.py` — PyQt6 UI และ worker threads

## Security model

- แต่ละชั้นสุ่ม salt 16 bytes และ nonce 12 bytes แยกกัน
- PBKDF2 100,000 รอบตาม requirement; ใน production ใหม่ควร benchmark แล้วเพิ่มรอบ หรือพิจารณา Argon2id
- AES-GCM tag 16 bytes; tag ผิดจะไม่คืน plaintext
- Outer header ถูก bind เป็น AAD ป้องกันการแก้ salt/nonce/length แบบไม่ถูกตรวจพบ
- SHA-256 ใน steganography header ใช้ตรวจ corruption อย่างรวดเร็ว ไม่ได้แทน GCM authentication
- password ไม่ถูกเขียนลง disk/log
- temporary data ถูกสร้างใน OS temp directory และลบเมื่อ pipeline จบ อย่างไรก็ตาม secure deletion บน SSD ไม่สามารถรับประกันได้

## Robustness boundaries

Frequency-domain embedding ไม่ได้แปลว่าทนต่อ re-encoding ทุกชนิด ตัว implementation ใช้ QIM และ coefficient ระดับกลาง แต่รับประกันการกู้คืนดีที่สุดเมื่อเก็บ output เป็น FFV1 lossless หาก upload ไป social media, resize, crop, change FPS หรือ transcode payload อาจเสียหาย ควรเพิ่ม ECC/interleaving หากต้องการ channel ที่มี loss สูง
