import urllib.request
import cv2
import numpy as np
import os
import sys

# Thêm đường dẫn src để import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.infer import DetectAndClassifyPipeline
from src.utils.visualizer import Visualizer
from src.utils.grad_cam_explainer import ErrorAnalyzer
from torchvision import transforms
from src.config.app_config import AppConfigs

def download_image(url, save_path):
    print(f"Đang tải ảnh thử nghiệm từ Internet...")
    # Thêm header giả lập trình duyệt để không bị chặn
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response, open(save_path, 'wb') as out_file:
        out_file.write(response.read())
    print("Tải xong ảnh!")

def run_test(image_path, detector_model, classifier_model):
    if not os.path.exists(image_path):
        print(f"Không tìm thấy ảnh {image_path}. Sẽ thử tải một ảnh test mặc định...")
        image_path = 'test_trash.jpg'
        img_url = "https://images.unsplash.com/photo-1595278069441-2cf29f8005a4?q=80&w=800&auto=format&fit=crop"
        if not os.path.exists(image_path):
            download_image(img_url, image_path)

    print("\n--- BƯỚC 1 & 2: CHẠY NHẬN DIỆN VÀ PHÂN LOẠI ---")
    pipeline = DetectAndClassifyPipeline(yolo_weights=detector_model, classifier_model_name=classifier_model)
    
    try:
        bounding_boxes, labels = pipeline.run(image_path)
    except Exception as e:
        print(f"Lỗi khi chạy Model: {e}")
        print("Có thể bạn chưa cài thư viện. Hãy gõ: pip install ultralytics torch torchvision opencv-python matplotlib pytorch-grad-cam")
        return

    print("\n--- BƯỚC 3: PHÂN TÍCH LỖI VÀ VẼ BIỂU ĐỒ ---")
    if bounding_boxes:
        # Giả lập hộp thực tế (Ground Truth) để so sánh lỗi khoanh vùng
        # (Ở đây mình bịa ra 1 cái khung lệch đi 1 tí để hàm Visualizer có cái để so sánh)
        fake_ground_truth_box = [
            [bounding_boxes[0][0] - 20, bounding_boxes[0][1] - 10, bounding_boxes[0][2] + 20, bounding_boxes[0][3] + 10]
        ]
        
        # 1. Vẽ lỗi Khoanh vùng (Detection)
        print("Đang vẽ biểu đồ so sánh Bounding Box...")
        Visualizer.plot_bounding_boxes(
            image_path=image_path,
            true_boxes=fake_ground_truth_box,
            pred_boxes=bounding_boxes,
            save_path="test_error_analysis_detection.png"
        )
        
        # 2. Vẽ Bản đồ nhiệt Grad-CAM (Phân loại)
        print("Đang vẽ Bản đồ nhiệt Grad-CAM cho cục rác vừa cắt...")
        try:
            # Lấy model ResNet từ pipeline
            resnet_model = pipeline.classifier.model
            # Chọn layer cuối cùng của ResNet50 để soi Grad-CAM
            target_layer = resnet_model.layer4[-1]
            
            explainer = ErrorAnalyzer(model=resnet_model, target_layer=target_layer)
            
            # Cắt cái ảnh thực tế ra để cho vào explainer
            original_img = cv2.imread(image_path)
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            x1, y1, x2, y2 = map(int, bounding_boxes[0])
            cropped_rgb = original_img[y1:y2, x1:x2]
            
            # Tiền xử lý như ResNet
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            input_tensor = transform(cropped_rgb).unsqueeze(0)
            
            # Đưa ảnh đã resize về kích thước 224 để hiển thị heatmap cho khớp
            cropped_resized_rgb = cv2.resize(cropped_rgb, (224, 224))
            
            explainer.explain_prediction(
                input_tensor=input_tensor,
                original_image_rgb=cropped_resized_rgb,
                save_path="test_heatmap_gradcam.png"
            )
            print("Toàn bộ bài Test đã thành công! Hãy mở các file ảnh .png vừa tạo ra để xem kết quả.")
        except Exception as e:
            print(f"Lỗi chạy Grad-CAM: {e}")
    else:
        print("Không tìm thấy rác nên không vẽ biểu đồ được!")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Khối XAI: Vẽ Heatmap Grad-CAM và Error Analysis")
    parser.add_argument('--image', type=str, default='test_trash.jpg', help='Đường dẫn ảnh test')
    parser.add_argument('--detector', type=str, default='runs/detect/yolov8m_trashnet/weights/best.pt', help='Đường dẫn mô hình YOLO')
    parser.add_argument('--classifier', type=str, default='efficientnet_b0', choices=['resnet50', 'efficientnet_b0', 'mobilenet_v3'], help='Kiến trúc mạng phân loại')
    
    args = parser.parse_args()
    run_test(args.image, args.detector, args.classifier)
