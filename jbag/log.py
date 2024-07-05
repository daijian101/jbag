import logging
import sys

logger = logging.getLogger('jbag')
logger.setLevel(logging.INFO)
log_format = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
console_handler = logging.StreamHandler(stream=sys.stdout)
console_handler.setFormatter(log_format)
logger.addHandler(console_handler)
