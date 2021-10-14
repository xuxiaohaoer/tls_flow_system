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
    print(dataset_np)
    np.save(save_path, dataset_np)




if __name__ == "__main__":
    print("begin")
    base_path = "./WVM/data_feature"
    save_path = "f_data_word"
    save_path = base_path + "/" + save_path
    # 保存路径
    data_path = "./WVM/data_cut"
    # 原始数据路径
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    pre_flow("{}/train/black/".format(data_path), '{}/train_black.npy'.format(save_path), 'black')
    pre_flow("{}/train/white/".format(data_path), '{}/train_white.npy'.format(save_path), 'white')
    pre_flow("{}/test/black/".format(data_path), '{}/test_black.npy'.format(save_path), 'black')
    pre_flow("{}/test/white/".format(data_path), '{}/test_white.npy'.format(save_path), 'white')
    print("end")