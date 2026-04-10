import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='ChnSentiCorp_htl_all.csv')
parser.add_argument('--vocab_path', type=str, default='./data.json')
parser.add_argument('--freq_path', type=str, default='./f.json')
args = parser.parse_args()

# 在类中使用args.data_path等