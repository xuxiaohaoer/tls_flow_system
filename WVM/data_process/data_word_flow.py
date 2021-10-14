import os
import tqdm
from flow import FlowWord
import dpkt
import numpy as np
def pre_flow(data_path, save_path, type):
    dataset = []
    for filename in os.listdir(data_path):
        if '.pcap' in filename:
            try:
                with open(data_path + filename, 'rb') as f:
                    capture = dpkt.pcap.Reader(f)
                    flow_sample = FlowWord(capture, type)
                    flow_sample.name = filename.replace('.pcap', '')
                    flow_sample.analyse()
                    dataset.append(flow_sample.toPac())
                f.close()
            except IOError:
                print('could not parse {0}'.format(filename))
    dataset_np = np.array(dataset)
    np.save(save_path, dataset_np)




if __name__ == "__main__":
    print("begin")
    base_path = "data_feature"
    save_path = "f_data_word"
    path = base_path + "/" + save_path
    if not os.path.exists(path):
        os.makedirs(path)
    pre_flow("./data_raw/train/black/", '{}/train_black.npy'.format(path), 'black')
    pre_flow("./data_raw/train/white/", '{}/train_white.npy'.format(path), 'white')
    pre_flow("./data_raw/test/black/", '{}/test_black.npy'.format(path), 'black')
    pre_flow("./data_raw/test/white/", '{}/test_white.npy'.format(path), 'white')
    print("end")