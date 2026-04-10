import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 替换print语句
logger.info(f'vocab_size:{len(word_to_num)}')
if size % 500 == 0:
    logger.info(f'prepare:{size}')