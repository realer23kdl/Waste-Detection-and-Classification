import os
import logging
import sys
import atexit

class LoggerSetup:
    @staticmethod
    def initialize(log_file: str, clear_old_logs: bool = True):
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        root = logging.getLogger()
        root.handlers = []

        formatter = logging.Formatter('%(asctime)s - [%(name)s] - %(levelname)s - %(message)s')
        file_mode = 'w' if clear_old_logs else 'a'

        fh = logging.FileHandler(log_file, mode=file_mode, encoding='utf-8', delay=False)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        root.addHandler(fh)

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        root.addHandler(ch)

        root.setLevel(logging.DEBUG)

        def shutdown_logging():
            fh.flush()
            fh.close()
            root.removeHandler(fh)
            root.removeHandler(ch)

        atexit.register(shutdown_logging)
        logging.info(f"Hệ thống Logger đã sẵn sàng. File log: {log_file}")
