import logging
from src.config.app_config import AppConfig
from src.core.logger import LoggerSetup
from src.pipeline.taco_pipeline import WastePreprocessingPipeline

if __name__ == '__main__':
    config = AppConfig()
    LoggerSetup.initialize(config.system.log_file_path, clear_old_logs=True)
    
    try:
        logging.info("BẮT ĐẦU CHẠY PIPELINE CHUẨN BỊ DỮ LIỆU...")
        data_pipeline = WastePreprocessingPipeline(config=config.data)
        data_pipeline.execute()
        logging.info("✅ Đã hoàn thành toàn bộ Pipeline!")
    except Exception as e:
        logging.exception(f"Pipeline sập do lỗi: {e}")
