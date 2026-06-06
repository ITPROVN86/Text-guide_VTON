# ST-VTON Experimental Source Code

Mã nguồn này dùng để làm **mô hình thực nghiệm cho Text-guided / Structure-aware Virtual Try-On** ở mức nghiên cứu, phù hợp để chạy baseline nội bộ và sinh các chỉ số:

- SSIM ↑
- PSNR ↑
- LPIPS ↓
- FID ↓
- Text Alignment / CLIPScore ↑

Ý tưởng mô hình:

```text
Person image + Garment image + Text caption
        │
        ├── Garment encoder
        ├── Person encoder
        ├── Text encoder / CLIP text feature, optional
        │
        ├── Structure-aware garment deformation / flow warping
        │
        ├── Attention U-Net mask refinement
        │
        └── Image synthesis network
                ↓
          Try-on result
```

## 1. Cấu trúc dataset gợi ý

Đặt dataset theo dạng:

```text
DataSet/VITON-HD/
├── train_pairs.txt
├── test_pairs.txt
├── train/
│   ├── image/
│   ├── cloth/
│   ├── image-parse-v3/        optional
│   └── openpose_img/          optional
└── test/
    ├── image/
    ├── cloth/
    ├── image-parse-v3/        optional
    └── openpose_img/          optional
```

Mỗi dòng trong `train_pairs.txt` hoặc `test_pairs.txt`:

```text
person_image.jpg cloth_image.jpg a photo of person wearing the target garment
```

Nếu chưa có caption, có thể chỉ để:

```text
person_image.jpg cloth_image.jpg
```

## 2. Cài đặt

```bash
cd st_vton_experiment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Các thư viện nâng cao như `lpips`, `torchmetrics`, `open_clip_torch` là optional. Nếu chưa cài, code vẫn chạy PSNR/SSIM.

## 3. Train

Sửa đường dẫn trong `configs/default.yaml`, sau đó chạy:

```bash
python -m src.train --config configs/default.yaml
```

Checkpoint sẽ nằm ở:

```text
outputs/checkpoints/st_vton_latest.pth
outputs/checkpoints/st_vton_best.pth
```

## 4. Inference sinh ảnh

```bash
python -m src.infer   --config configs/default.yaml   --checkpoint outputs/checkpoints/st_vton_best.pth   --split test   --save_dir outputs/inference
```

## 5. Evaluate metric

```bash
python -m src.eval   --config configs/default.yaml   --pred_dir outputs/inference   --split test   --save_csv outputs/metrics.csv
```

