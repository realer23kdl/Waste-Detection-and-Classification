import logging
from src.config.app_config import AppConfig
from src.core.logger import LoggerSetup
from src.pipeline.taco_pipeline import WastePreprocessingPipeline

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Chạy Pipeline Tiền xử lý dữ liệu")
    parser.add_argument('--raw_annotations', type=str, default=None, help="Đường dẫn tới file raw annotations.json trên Kaggle")
    parser.add_argument('--mapping_label', type=str, default=None, help="Đường dẫn tới file mapping_label.json trên Kaggle")
    args = parser.parse_args()

    config = AppConfig()
    
    # Ghi đè đường dẫn nếu người dùng truyền từ dòng lệnh
    if args.raw_annotations:
        config.data.RAW_ANNOTATIONS_PATH = args.raw_annotations
    if args.mapping_label:
        config.data.TACO_TO_7_CLASSES_MAP = args.mapping_label

    LoggerSetup.initialize(config.system.log_file_path, clear_old_logs=True)
    
    try:
        logging.info("BẮT ĐẦU CHẠY PIPELINE CHUẨN BỊ DỮ LIỆU...")
        data_pipeline = WastePreprocessingPipeline(config=config.data)
        data_pipeline.execute()
        logging.info("✅ Đã hoàn thành toàn bộ Pipeline!")
    except Exception as e:
        logging.exception(f"Pipeline sập do lỗi: {e}")
