import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.infer import DetectAndClassifyPipeline

def main():
    parser = argparse.ArgumentParser(description="Hệ thống Nhận diện và Phân loại Rác thải")
    parser.add_argument('--pipeline', type=str, required=True, 
                        choices=['classify_only', 'detect_and_classify', 'segment_and_classify'],
                        help="Chọn luồng chạy: classify_only (Người 1), detect_and_classify (Người 2), segment_and_classify (Người 3)")
    
    # Cho phép linh hoạt chọn model (Ablation Study)
    parser.add_argument('--detector', type=str, default='yolov8s.pt', help="Tên trọng số mô hình khoanh vùng (VD: yolov8s.pt, rtdetr-l.pt)")
    parser.add_argument('--classifier', type=str, default='efficientnet_b0', choices=['resnet50', 'efficientnet_b0', 'mobilenet_v3'], help="Tên mạng phân loại")
    parser.add_argument('--classifier_weights', type=str, default='weights/best_resnet50.pth', help="Đường dẫn file trọng số của mạng phân loại")
    parser.add_argument('--image', type=str, default='test_image.jpg', help="Đường dẫn ảnh test")
    parser.add_argument('--conf', type=float, default=0.25, help="Ngưỡng độ tin cậy của YOLO (ví dụ: 0.25, 0.5)")
    
    args = parser.parse_args()

    if args.pipeline == 'classify_only':
        print(f">>> ĐANG CHẠY PHÂN LOẠI THUẦN (MÔ HÌNH: {args.classifier.upper()}) <<<")
        # pipeline_p1 = ClassificationOnlyPipeline(classifier_model=args.classifier)
        print("Chưa có code cho luồng này!")

    elif args.pipeline == 'detect_and_classify':
        print(f">>> ĐANG CHẠY DETECT ({args.detector}) + CROP + CLASSIFY ({args.classifier}) <<<")
        pipeline_p2 = DetectAndClassifyPipeline(yolo_weights=args.detector, resnet_weights=args.classifier_weights, classifier_model_name=args.classifier)
        try:
            pipeline_p2.run(args.image, conf_threshold=args.conf)
        except Exception as e:
            print(f"Lỗi: {e}. Nhớ tải ảnh test_image.jpg vào thư mục nhé!")

    elif args.pipeline == 'segment_and_classify':
        print(f">>> ĐANG CHẠY SEGMENTATION + MASK + CLASSIFY ({args.classifier}) <<<")
        # pipeline_p3 = SegmentAndClassifyPipeline(...)
        print("Chưa có code cho luồng này!")

if __name__ == "__main__":
    main()
