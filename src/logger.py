import logging
import os
from datetime import datetime

# Create log file name
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Create logs folder path (NOT including file)
logs_dir = os.path.join(os.getcwd(), "logs")

# Create folder if not exists
os.makedirs(logs_dir, exist_ok=True)

# Full file path
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Test log
logging.info("Logging setup is working!")