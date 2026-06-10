import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.detector import TrashDetector
from src.models.classifier import TrashClassifier
from src.utils.cropper import ImageCropper

def run_ablation_comparison(image_path, yolo_weights='yolov8n.pt', classifier_model='resnet50', classifier_weights=None):
    print(f"BẮT ĐẦU THỰC NGHIỆM ABLATION STUDY TRÊN ẢNH: {image_path}")
    print(f"Mô hình sử dụng: {classifier_model.upper()} và {yolo_weights}")
    
    # Nạp mô hình
    classifier = TrashClassifier(model_name=classifier_model, num_classes=7, model_weights_path=classifier_weights)
    detector = TrashDetector(model_path=yolo_weights, model_type='yolo' if 'yolo' in yolo_weights else 'rtdetr')
    cropper = ImageCropper(padding=10)
    
    # Đọc ảnh gốc
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Lỗi: Không thể đọc được ảnh {image_path}")
        return
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # -------------------------------------------------------------------------
    # PHƯƠNG PHÁP 1: TRUYỀN THẲNG ẢNH GỐC VÀO CLASSIFIER (BASELINE)
    # -------------------------------------------------------------------------
    print("\n--- PHƯƠNG PHÁP 1: CLASSIFIER THUẦN (KHÔNG DETECT) ---")
    # Mạng ResNet50 sẽ bị nhiễu bởi hậu cảnh, bầu trời, mặt đất...
    baseline_res = classifier.predict([img_rgb], return_prob=True)[0]
    baseline_pred = baseline_res[0]
    baseline_prob = baseline_res[1]
    print(f"Kết quả dự đoán toàn cảnh: {baseline_pred} ({baseline_prob*100:.1f}%)")
    
    # -------------------------------------------------------------------------
    # PHƯƠNG PHÁP 2: DETECT -> CROP -> CLASSIFY (PROPOSED)
    # -------------------------------------------------------------------------
    print("\n--- PHƯƠNG PHÁP 2: PIPELINE ĐỀ XUẤT (DETECT + CROP + CLASSIFY) ---")
    # YOLO loại bỏ hậu cảnh, chỉ giữ lại cục rác
    boxes = detector.predict(image_path, conf_threshold=0.1)
    
    if len(boxes) > 0:
        cropped_images = cropper.crop_objects(image_path, boxes)
        proposed_preds = classifier.predict(cropped_images, return_prob=True)
    else:
        boxes = []
        proposed_preds = []
        print("YOLO không tìm thấy cục rác nào trong ảnh này.")

    # -------------------------------------------------------------------------
    # VẼ BIỂU ĐỒ SO SÁNH TRỰC QUAN
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Khung 1: Baseline
    axes[0].imshow(img_rgb)
    axes[0].set_title(f"[Baseline] Truyền thẳng ảnh gốc vào {classifier_model.upper()}\nDự đoán: {baseline_pred} ({baseline_prob*100:.1f}%)", fontsize=14, color='red')
    axes[0].axis('off')
    
    # Khung 2: Proposed
    img_boxes = img_rgb.copy()
    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = map(int, box[:4])
        # Vẽ khung xanh lá cây cho YOLO
        cv2.rectangle(img_boxes, (x_min, y_min), (x_max, y_max), (0, 255, 0), 4)
        
        # Vẽ nhãn phân loại và xác suất từ ResNet50
        label_name, prob = proposed_preds[i]
        label = f"{label_name} ({prob*100:.1f}%)"
        
        # Tạo nền đen cho chữ dễ đọc
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(img_boxes, (x_min, y_min - 30), (x_min + w, y_min), (0, 255, 0), -1)
        cv2.putText(img_boxes, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        
    axes[1].imshow(img_boxes)
    if len(boxes) > 0:
        axes[1].set_title(f"[Đề xuất] YOLO Cắt rác -> Đưa vào {classifier_model.upper()}\nTìm thấy {len(boxes)} đối tượng rác", fontsize=14, color='green')
    else:
        axes[1].set_title(f"[Đề xuất] YOLO Cắt rác -> Đưa vào {classifier_model.upper()}\nKhông thấy rác", fontsize=14, color='green')
    axes[1].axis('off')
    
    plt.tight_layout()
    output_filename = "ablation_comparison_result.png"
    plt.savefig(output_filename, dpi=150)
    print(f"\n[HOÀN TẤT] Đã lưu ảnh so sánh cực kỳ trực quan tại: {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help="Đường dẫn tới bức ảnh test")
    parser.add_argument('--detector', type=str, default='yolov8s.pt', help="File weights của YOLO")
    parser.add_argument('--classifier', type=str, default='resnet50', help="Mô hình phân loại (resnet50/efficientnet_b0/mobilenet_v3)")
    parser.add_argument('--classifier_weights', type=str, default='weights/best_resnet50.pth', help="File weights của Classifier")
    
    args = parser.parse_args()
    run_ablation_comparison(args.image, args.detector, args.classifier, args.classifier_weights)
