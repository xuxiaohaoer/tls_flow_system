import  numpy as np
import math
from cal import *
import dpkt
import OpenSSL
import sys
import socket
import time
from datetime import datetime
from constants import PRETTY_NAMES



class FlowRecord:
    def __init__(self, num, flow_size, flow_starttime, flow_endtime):
        self.num = num
        self.flow_size = flow_size
        self.flow_duration = 0
        self.flow_starttime = flow_starttime
        self.flow_endtime = flow_endtime


class FlowWord(object):
    def __init__(self, capture, label):

        self.capture = capture   #待处理数据     
        self.contact = {}  # 链接
        self.flow = {}

        self.ip_src = ''  # 目的ip地址
        self.ip_dst = ''  # 源ip地址
        self.dport = 0  # 源端口号
        self.sport = 0  # 目的端口号
        
        self.pack_num = 0  # 包数量
        
        self.num_src = 0  # 源包数目
        self.num_dst = 0    # 目的包数目
        self.num_ratio = 0  # 上下行流量比
        self.size_src = 0   # 源总包大小
        self.size_dst = 0   # 目的总包大小
        self.size_ratio = 0  # 上下行包大小比
        self.by_s = 0   # 每秒字节传输速度
        self.pk_s = 0   # 每秒包传输速度

        self.time = 0  # 整体持续时间
        self.time_seq = []  # 时间序列
        self.max_time = 0  # 最大间隔时间
        self.min_time = 0  # 最小间隔时间
        self.mean_time = 0  # 平均间隔时间
        self.std_time = 0  # 均差间隔时间

        self.time_src_seq = []  # 源时间间隔序列
        self.max_time_src = 0  # 最大源时间间隔
        self.min_time_src = 0  # 最小源时间间隔
        self.mean_time_src = 0  # 平均源时间间隔
        self.std_time_src = 0  # 均差源时间间隔

        self.time_dst_seq = []  # 目的时间间隔序列
        self.max_time_dst = 0  # 最大目的时间间隔
        self.min_time_dst = 0  # 最小目的时间间隔
        self.mean_time_dst = 0  # 平均目的时间间隔
        self.std_time_dst = 0  # 均差目的时间间隔

        self.packetsize_src_seq = []  # 源包大小序列
        self.max_packetsize_src = 0  # 最大源包大小
        self.min_packetsize_src = 0  # 最小源包大小
        self.mean_packetsize_src = 0  # 平均源包大小
        self.std_packetsize_src = 0  # 均差源包大小

        self.packetsize_dst_seq = []  # 目的包大小序列
        self.max_packetsize_dst = 0  # 最大目的包大小
        self.min_packetsize_dst = 0  # 最小目的包大小
        self.mean_packetsize_dst = 0  # 均值目的包大小
        self.std_packetsize_dst = 0  # 均差目的包大小

       
        self.time_flow_seq = []  # 流时间序列
        self.max_time_flow = 0  # 最大流时间
        self.min_time_flow = 0  # 最小流时间
        self.mean_time_flow = 0  # 平均流时间
        self.std_time_flow = 0  # 均差流时间
        self.packetsize_all = 0  # # 平均包大小

        self.packetsize_packet_seq = []  # 包大小序列
        self.max_packetsize_packet = 0  # 最大包大小
        self.min_packetsize_packet = 0  # 最小包大小
        self.mean_packetsize_packet = 0  # 平均包大小
        self.std_packetsize_packet = 0  # 均差包大小
    
        self.seq = []  # 自TLS开始的有向序列
        self.payload_seq = []
        self.tls_seq = []
        self.dir_seq = []
        self.num = 0  # 数据流数量

        self.bitFre = np.zeros(256) # 所有负载内各字节出现次数
        self.entropy = 0
        self.entropy_seq = []
        self.max_entropy = 0
        self.min_entropy = 0
        self.mean_entropy = 0
        self.std_entropy = 0


        self.cipher_num = 0  # 加密组件长度
        self.cipher_support = []  # 加密支持组件序列
        self.cipher_support_num = 0  # 加密支持组件编码
        self.cipher = 0  # 加密组件
        self.cipher_app_content = bytes(0)
        self.cipher_bitFre = np.zeros(256)  # 加密内容里各字节出现次数

        self.cipher_content_ratio = 0  # 加密内容位中0出现次数
        
        self.need_more_certificate = True
        self.certificate = []
        self.cipher_self_signature = []  # 是否自签名，是1，否为0
        self.cipher_certifcate_time = []  # 证书有效时间
        self.cipher_subject = []  # 证书中subject
        self.cipher_issue = []  # 证书中issue
        self.cipher_extension_count = []
        self.cipher_sigature_alo = []
        self.cipher_version = []
        self.cipher_pubkey = []
        self.cipher_serial_number = []

        self.cipher_entropy = 0 # 总熵值
        self.cipher_app_num = 0 # 加密应用数据数目
        self.cipher_app_entropy = [] # 加密内容熵序列

        self.max_cipher_app_entropy = 0
        self.min_cipher_app_entropy = 0
        self.mean_cipher_app_entropy = 0
        self.std_cipher_app_entropy = 0

        self.flag = False  # 只取第一个certificate

        self.fin = 0  # 标志位Fin的数量
        self.syn = 0  # 标志位Syn的数量
        self.rst = 0  # 标志位RST的数量
        self.ack = 0  # 标志位ACK的数量
        self.urg = 0  # 标志位URG的数量
        self.psh = 0  # 标志位PSH的数量
        self.ece = 0  # 标志位ECE的数量
        self.cwe = 0  # 标志位CWE的数量

        self.client_hello_num = 0
        self.server_hello_num = 0
        self.certificate_num = 0

        self.client_hello_content = bytes(0)
        self.server_hello_content = bytes(0)
        self.certificate_content = bytes(0)
        self.packet = {"client_hello":[], "server_hello":[], "certificate":[]}


        self.transition_matrix = np.zeros((15, 15), dtype=int)  # 马尔可夫转移矩阵
        self.label = label  # 若有，则为具体攻击类型
        self.name = ''  # pacp包名称

        self.content = [] # 包负载内容
        self.content_payload = []

        self.need_flow_all = False
        self.flow_num = 0  # 流数目
        self.packetsize_flow_seq = []  # 流大小序列
        self.max_packetsize_flow = 0  # 最大流大小
        self.min_packetsize_flow = 0  # 最小流大小
        self.mean_packetsize_flow = 0  # 平均流大小
        self.std_packetsize_flow = 0  # 均差流大小

    def tolist(self):
        """change to list that is the model input"""
        # print(self.cipher_application_data)
        # 存在application data

        time = round(self.time)
        ip_src = int(self.ip_src.replace('.', ''))
        self.packetsize_all = round(self.packetsize_all / self.pack_num)
        self.time = cal_seq(self.time)
        self.max_time, self.min_time, self.mean_time, self.std_time = cal(self.time_seq)
        if self.need_flow_all:
            self.max_packetsize_flow, self.min_packetsize_flow, self.mean_packetsize_flow, self.std_packetsize_flow = cal(
                self.packetsize_flow_seq)
        self.max_time_flow, self.min_time_flow, self.mean_time_flow, self.std_time_flow = cal(self.time_flow_seq)
        self.time_src_seq = cal_seq(self.time_src_seq)
        self.time_dst_seq = cal_seq(self.time_dst_seq)
        self.max_time_src, self.min_time_src, self.mean_time_src, self.std_time_src = cal(self.time_src_seq)
        self.max_time_dst, self.min_time_dst, self.mean_time_dst, self.std_time_dst = cal(self.time_dst_seq)
        self.max_packetsize_src, self.min_packetsize_src, self.mean_packetsize_src, self.std_packetsize_src = cal(
            self.packetsize_src_seq)
        self.max_packetsize_dst, self.min_packetsize_dst, self.mean_packetsize_dst, self.std_packetsize_dst = cal(
            self.packetsize_dst_seq)
        self.max_packetsize_packet, self.min_packetsize_packet, self.mean_packetsize_packet, self.std_packetsize_packet = cal(
            self.packetsize_packet_seq)
        self.cipher_support_num = cal_hex(self.cipher_support)
        self.cipher_content_ratio = round(cal_ratio(self.bitFre), 4)

        self.transition_matrix = cal_matrix(self.packetsize_packet_seq)

        self.num_ratio = cal_div(self.num_src, self.num_dst)
        self.size_ratio = cal_div(self.size_src, self.num_dst)
        self.by_s = cal_div(self.packetsize_all, self.time)
        self.pk_s = cal_div(self.pack_num, self.time)

        self.max_entropy, self.min_entropy, self.mean_entropy, self.std_entropy = cal(self.bitFre)
        self.max_cipher_app_entropy, self.min_cipher_app_entropy, self.mean_cipher_app_entropy, self.std_cipher_app_entropy = cal(self.cipher_bitFre)
        # if self.cipher_bitFre.sum() != 0:
        #     self.cipher_bitFre /= self.cipher_bitFre.sum()
        #     self.cipher_entropy = self.cal_entropy(self.cipher_bitFre)
        # if self.bitFre.sum() != 0:
        #     self.bitFre /= self.bitFre.sum()
        #     self.entropy = self.cal_entropy(self.bitFre)

        return [self.pack_num, time, ip_src, self.packetsize_all, self.dport,
                # 5
                self.max_time, self.min_time, self.mean_time, self.std_time,
                self.max_time_src, self.min_time_src, self.mean_time_src, self.std_time_src,
                self.max_time_dst, self.min_time_dst, self.mean_time_dst, self.std_time_dst,
                self.max_time_flow, self.min_time_flow, self.mean_time_flow, self.std_time_flow,
                # 21
                self.max_packetsize_packet, self.mean_packetsize_packet, self.std_packetsize_packet,
                self.max_packetsize_src, self.mean_packetsize_src, self.std_packetsize_src,
                self.max_packetsize_dst, self.mean_packetsize_dst, self.std_packetsize_dst,
                self.fin, self.syn, self.rst, self.ack, self.urg, self.psh, self.ece, self.cwe,
                self.num_src, self.num_dst, self.num_ratio,
                self.size_src, self.size_dst, self.size_ratio,
                self.by_s, self.pk_s,
                self.cipher_self_signature, self.cipher_certifcate_time, self.cipher_subject,
                self.cipher_issue, self.cipher_extension_count, self.cipher_sigature_alo, self.cipher_version,
                self.cipher_num, self.cipher_support, self.cipher_support_num, self.cipher,
                self.cipher_content_ratio,
                self.cipher_app_num,
                self.transition_matrix,
                self.tls_seq, self.payload_seq, self.dir_seq,
                self.client_hello_num, self.server_hello_num, self.certificate_num,
                self.label, self.name
                ]
        # self.entropy, self.bitFre, self.max_entropy, self.min_entropy, self.mean_entropy, self.std_entropy,
        # # 69
        # self.cipher_entropy, self.cipher_bitFre, self.max_cipher_app_entropy, self.min_cipher_app_entropy, self.mean_cipher_app_entropy, self.std_cipher_app_entropy

    def toSeq(self):
        contents = []
        for key in self.content:
            content = []
            for value in key:
                content.append(value/255)
            content += [0]* 144
            contents.append(content[:144])
        while len(contents)<3:
            contents.append([0]*144)
        while len(self.content_payload) <3:
            self.content_payload.append(0)
        # print(contents)
        return [contents, self.content_payload, self.client_hello_num, self.label, self.name]

    def toImag(self):
        while len(self.content)<100:
            self.content.append(0)
        tem = np.array(self.content[:100])
        # tem = tem/255
        # tem = tem.reshape(28, 28)    
        return [tem, self.client_hello_num, self.label, self.name]


    def toCut(self):
        contents = []
        contents.append(self.client_hello_content )
        contents.append(self.server_hello_content )
        contents.append(self.certificate_content)

        return [contents, self.label, self.name]

    def cal_entropy(self, content):
        result = 0
        for key in content:
            if key != 0 :
                result += (-key) * math.log(key, 2)
        return result

    def toWord(self):
        res = []
     
        len_base = 4
        data = {}
        len_type = {"client_hello":30, "server_hello":10, "certificate":46}
        for key in ["client_hello", "server_hello", "certificate"]:
            # 30
            res = []
            i = 0
            for content in self.packet[key]:
                tem = []
                for value in content:
                    tem.append(value/255)
                while len(tem) < len_base:
                    tem.insert(0, 0.0)
                res.append(tem[: len_base])
            while len(res)< len_type[key]:
                res.append([0] * len_base)
            data[key] = res[:len_type[key]]
            # data[key] = res
            # data[key] = res
        data['label'] = self.label
        data['name'] = self.name
        return data
    

    def analyse(self):
        nth = 1
        
        for timestamp, packet in self.capture:
            if(self.client_hello_content == bytes(0) or self.server_hello_content == bytes(0) or self.certificate_content == bytes(0) ):
                self.parse_packet(packet, timestamp, nth)
            if nth == 1:
                time_begin = timestamp
            time = timestamp - time_begin
            nth += 1
            self.time_seq.append(time)


        if self.need_flow_all:
            for key in self.contact:
                self.contact[key].duration = self.contact[key].flow_endtime - self.contact[key].flow_starttime
                self.packetsize_flow_seq.append(self.contact[key].flow_size)
                self.time_flow_seq.append(self.contact[key].duration)
        
   
        self.pack_num = nth
        self.time = time
        while len(self.seq) < 20:
            self.seq.append(0)

    def parse_packet(self, packet, timestamp, nth):
        """
        Main analysis loop for pcap.
        """
        eth = dpkt.ethernet.Ethernet(packet)
        self.tls_seq.append(0)
        if isinstance(eth.data, dpkt.ip.IP):
            self.parse_ip_packet(eth, nth, timestamp)

    
    def parse_ip_packet(self, eth, nth, timestamp):
        """
        Parses IP packet.
        """

        ip = eth.data

        sys.stdout.flush()
        size = len(eth)  # 包大小
        self.packetsize_packet_seq.append(size)
        self.packetsize_all += size
        payload = len(ip.data.data)  # 有效负载大小
        self.payload_seq.append(payload)
        rest_load = None
        if isinstance(ip.data, dpkt.tcp.TCP):
            if (len(ip.data.data)!=0):
                self.parse_tcp_packet(ip, nth, timestamp)
        # 提取 ip地址、端口号
        if nth == 1:
            self.ip_src = socket.inet_ntoa(ip.src)
            self.ip_dst = socket.inet_ntoa(ip.dst)
            self.sport = int(ip.data.sport)
            self.dport = int(ip.data.dport)
        if socket.inet_ntoa(ip.src) == self.ip_src:
            self.time_src_seq.append(timestamp)
            self.packetsize_src_seq.append(size)
            self.num_src += 1
            self.size_src += size
            self.dir_seq.append(1)
        else:
            self.time_dst_seq.append(timestamp)
            self.packetsize_dst_seq.append(size)
            self.num_dst += 1
            self.size_dst += size
            self.dir_seq.append(-1)

        if self.need_flow_all:
            flag = socket.inet_ntoa(ip.src) + ' ' + socket.inet_ntoa(ip.dst) + ' ' + str(ip.data.dport) + ' ' + str(
                ip.data.sport)
            flag_1 = socket.inet_ntoa(ip.dst) + ' ' + socket.inet_ntoa(ip.src) + ' ' + str(ip.data.sport) + ' ' + str(
                ip.data.dport)
            if self.contact.__contains__(flag):
                self.contact[flag].num += 1
                self.contact[flag].flow_endtime = timestamp
                self.contact[flag].flow_size += size
            # elif contact.__contains__(flag_1):
            #     contact[flag_1].num += 1
            #     contact[flag_1].flow_endtime = timestamp
            #     contact[flag_1].flow_size += size
            else:
                tem = FlowRecord(0, size, timestamp, timestamp)
                self.contact[flag] = tem



        if isinstance(ip.data, dpkt.tcp.TCP) and payload:
            if socket.inet_ntoa(ip.src) == self.ip_dst:
                direction = 1
            else:
                direction = -1
            dirpath = direction * payload
            if len(self.seq) < 20:
                self.seq.append(dirpath)

        if self.need_more_certificate:
            class FlowFlag:
                def __init__(self, seq, data):
                    self.seq = seq

                    self.seq_exp = seq + len(data)
                    self.data = data
                    self.sequence = []
                    self.nth_seq = []
                    self.ip = dpkt.ip.IP()
                    self.timestamp = 0

            # 设置flow记录流的各条记录，以解决tcp resseambeld segment
            flow_flag = socket.inet_ntoa(ip.src) + '->' + socket.inet_ntoa(ip.dst)
            flow_flag1 = socket.inet_ntoa(ip.dst) + '->' + socket.inet_ntoa(ip.src)
            # 存在udp 没有seq和ack
            try:
                seq = ip.data.seq
                ack = ip.data.ack
            except AttributeError as exception:
                seq = 0
                ack = 0
            data = ip.data.data
            data_flag = data
            try:
                if data[0] in {20, 21, 22, 23}:
                    # 直接可以解压一部分，且返回剩余负载部分
                    data_tem, flag = self.parse_tls_records(ip, data, nth, [nth], timestamp)
                    if flag:
                        if len(data_tem) == 0:
                            data_tem = bytes(0)
                        data = data_tem
            except:
                pass

            # 接收到反向的包
            if flow_flag1 in self.flow.keys():
                if ack >= self.flow[flow_flag1].seq:
                    if len(self.flow[flow_flag1].data) != 0:
                        tem = self.flow[flow_flag1].data
                        nth_flag = self.flow[flow_flag1].nth_seq
                        ip_tem = self.flow[flow_flag1].ip
                        timestamp_tem = self.flow[flow_flag1].timestamp
                        if tem[0] in {20, 21, 22, 23}:
                            rest_load, flag = self.parse_tls_records(ip_tem, tem, nth_flag[-1], nth_flag, timestamp_tem)

                    try:
                        if rest_load != None and not len(data_flag):
                            if rest_load == bytes(0):
                                self.flow[flow_flag1].sequence.clear()
                                self.flow[flow_flag1].nth_seq.clear()
                                self.flow[flow_flag1].ip = dpkt.ip.IP()
                                self.flow[flow_flag1].timestamp = 0
                            if rest_load[0] in {20, 21, 22, 23}:
                                self.flow[flow_flag1].data = rest_load
                                # 中间插入一条ack较大值
                                self.flow[flow_flag1].sequence = [rest_load]
                                self.flow[flow_flag1].ip = ip
                                self.flow[flow_flag1].timestamp = timestamp
                        else:
                            self.flow.pop(flow_flag1)
                            # flow[flow_flag1].data = bytes(0)
                            # flow[flow_flag1].seq.clear()
                            # flow[flow_flag1].nth_seq.clear()
                    except:
                        self.flow.pop(flow_flag1)
                        # flow[flow_flag1].data = bytes(0)
                        # flow[flow_flag1].seq.clear()
                        # flow[flow_flag1].nth_seq.clear()

            # if data == None:
            #     print(nth)

            if len(data):
                if flow_flag not in self.flow.keys():
                    if data != bytes(0):
                        if data[0] in {20, 21, 22, 23}:
                            self.flow[flow_flag] = FlowFlag(seq, data)
                            self.flow[flow_flag].sequence.append(data)
                            self.flow[flow_flag].nth_seq.append(nth)
                            self.flow[flow_flag].seq_exp = seq + len(data_flag)
                            self.flow[flow_flag].ip = ip
                            self.flow[flow_flag].timestamp = timestamp
                else:
                    # if flow[flow_flag].seq < seq:
                    if self.flow[flow_flag].seq_exp == seq:
                        # print(nth, "###")
                        self.flow[flow_flag].seq = seq
                        self.flow[flow_flag].seq_exp += len(data_flag)
                        if data not in self.flow[flow_flag].sequence:
                            if data not in self.flow[flow_flag].data:
                                self.flow[flow_flag].data += data
                                self.flow[flow_flag].sequence.append(data)
                                self.flow[flow_flag].nth_seq.append(nth)
                                self.flow[flow_flag].ip = ip
                                self.flow[flow_flag].timestamp = timestamp
                    else:
                        pass


    def parse_tcp_packet(self, ip, nth, timestamp):
        """
        Parses TCP packet.
        """
        rest_load = None
        tcp_data = ip.data
        stream = ip.data.data

        # ssl flow
        #  提取标志位

        # if cal_psh(tcp_data.flags):
        #     self.psh += 1
        # if cal_urg(tcp_data.data.flags):
        #     feature.urg += 1

        if (stream[0]) in {20, 21, 22, 23, 128, 25}:
            if (stream[0]) in {20, 21, 22}:
                # print("---")
                pass
                # rest_load = parse_tls_records(ip, stream, nth)
            if (stream[0]) == 128:  # sslv2 client hello

                # feature.flag = True
                try:
                    cipher_length = stream[6] + stream[5] * 256
                except:
                    cipher_length = 0
                if len(stream) > 6:
                    if stream[2] == 1:  # sslv2 client hello

                        self.client_hello_num += 1

                        packet = []
                        pos_flag  = False
                    
                        packet.append(nth.to_bytes(length=4, byteorder='big', signed=False))
                        # packet.append(math.floor(timestamp * 1000).to_bytes(length=4, byteorder='big', signed=False))
                        packet.append((len(ip) + 14).to_bytes(length=4, byteorder='big', signed=False))
                        packet.append(ip.src)
                        packet.append(ip.dst)
                        packet.append(ip.data.sport.to_bytes(length=4, byteorder='big', signed=False))
                        packet.append(ip.data.dport.to_bytes(length=4, byteorder='big', signed=False))
                        packet.append(bytes(3) + stream[2:3])  # 类型
                        packet.append(bytes(3) + stream[1:2])  # 长度
                        packet.append(bytes(2) +int(2).to_bytes(length=4, byteorder='big', signed=False)) # 版本
                        packet.append(bytes(0) + stream[5:7])  # 长度    
                        head = bytes(0)
                        head += nth.to_bytes(length=4, byteorder='big', signed=False)
                        # head += math.floor(timestamp * 1000).to_bytes(length=4, byteorder='big', signed=False)
                        head += (len(ip) + 14).to_bytes(length=4, byteorder='big', signed=False)
                        head += ip.src
                        head += ip.dst
                        head += ip.data.sport.to_bytes(length=4, byteorder='big', signed=False)
                        head += ip.data.dport.to_bytes(length=4, byteorder='big', signed=False)
                        if (self.client_hello_content == bytes(0)):
                            self.client_hello_content = head + stream[2:]
                        # length = stream[1]
                        # dataClientHello = stream[:length+2]

                        self.tls_seq[nth - 1] = stream[2]
                        self.cipher_num = max(cipher_length, self.cipher_num)
                        tem = stream[7] * 256 + stream[8] + 11  # 加密组件开始的stream的index
                        i = 0
                        while i < cipher_length:
                            cipher = 0
                            if tem + i + 2 < len(stream):
                                cipher = stream[tem + i + 2] + stream[tem + i + 1] * 256 + stream[tem + i] * 256 * 256
                                packet.append(bytes(1) + stream[tem+i: tem+i+3])
                            if cipher not in self.cipher_support:
                                self.cipher_support.append(cipher)
                            i += 3
                        if not self.packet["client_hello"]:
                            self.packet["client_hello"] = packet


                    # print(nth, stream[6])
            # if (stream[0]) == 25:
            #     rest_load = parse_tls_records(ip, stream, nth)
        return rest_load


    def multiple_handshake(self, nth, buf, ip):
        i, n = 0, len(buf)
        msgs = []
    
        while i + 5 <= n:
            tot = 0
            v = buf[i + 1:i + 3]
            if v in dpkt.ssl.SSL3_VERSION_BYTES:
                head = buf[i:i + 5]
                tot_len = int.from_bytes(buf[i + 3:i + 5], byteorder='big')
                j = i + 5
                while j <= tot_len + 5:
                    try:
                        Record_len = int.from_bytes(buf[j + 1:j + 4], byteorder='big', signed=False)
                        len_tem_b = (Record_len + 4).to_bytes(length=2, byteorder='big', signed=False)
                        head_tem = head[0:3] + len_tem_b
                        tem = head_tem + buf[j:j + Record_len + 4]
                    
                    except:
                        # Record_len = 0
                        pass
                    try:
                        msg = dpkt.ssl.TLSRecord(tem)
                        msgs.append(msg)
                        record_type = self.pretty_name('tls_record', msg.type)

                        # print(nth, "***{}***".format(msg))

                    except dpkt.NeedData:
                        i = n
                        break
                    try:
                        j += Record_len + 4
                        i += Record_len + 4
                    except:
                        pass
                    # if Record_len != 0:
                    #     j += Record_len + 4
                    #     i += j
                    # else:
                    #     j += 4
                    #     i += j
                # 防止无限循环
                if j == i + 5:
                    i = n


            else:
                raise dpkt.ssl.SSL3Exception('Bad TLS version in buf: %r' % buf[i:i + 5])
            # i += tot
        return msgs, i +5


    


    def parse_tls_records(self, ip, stream, nth, nth_seq, timestamp):
      
        packet = []
    
        flag = False
        try:
            record_len = int.from_bytes(stream[3: 5], byteorder='big', signed=False)
            if record_len + 5 > len(stream):
                return stream, False
        except:
            return stream, False
        try:
            records, bytes_used = dpkt.ssl.tls_multi_factory(stream)
        except dpkt.ssl.SSL3Exception as exception:
            return stream, False
        # mutliple

        try:
            record = records[0]
                # length = record.length
            if record.type == 22 and record.data[0] in {1, 2, 11, 16}:
                length = int.from_bytes(record.data[1:4], byteorder='big', signed=False) + 4
            else:
                length = record.length
        except:
            length = record.length
            pass
        
        if bytes_used == 0 or length != record_len:
            try:
                records, bytes_used = self.multiple_handshake(nth, stream, ip)
            except:
                return stream, False
            if bytes_used > len(stream):
                return stream, False

      
        flag = True
        n = 0
        type = []
        handshake_scope = [1, 2, 11, 12, 14, 16]
        for record in records:
            record_type = self.pretty_name('tls_record', record.type)
            if record_type == 'application_data':
                i = (len(stream) - bytes_used) // 1460
                # print(nth, nth_seq[-1-i],record_type)
                # 存在多余bytes_used，说明组合了多余的包，应该回退
                try:
                    nth = nth_seq[-1 - i]
                except:
                    pass
                    # print(len(records))
                    # print(len(stream), bytes_used)
                    # print(self.ip_dst, self.ip_src, nth, i)
                content = np.zeros(256)
                entropy = 0
                for key in record.data:
                    content[key] += 1
                self.cipher_bitFre += content

                if content.sum() != 0:
                    content /= content.sum()
                for key in content:
                    if key != 0:
                        entropy -= (key) * math.log(key, 2)
                self.cipher_app_entropy.append(entropy)
                # if len(self.cipher_app_content) < 1600:
                #     self.cipher_app_content += record.data
                #     self.cipher_app_content = self.cipher_app_content[:1600]

                self.cipher_app_num += 1

            if record_type == 'handshake':
                handshake_type = ord(record.data[:1])
                if handshake_type in handshake_scope:
                    type.append(handshake_type)
                if length +4 != len(record.data) or self.last_record == 'change_cipher': # encrypted handshake
                    handshake_type == -1
                # print(nth, "handshake_type", handshake_type)
                if handshake_type == 2:  # server hello

                    # buf_cont = record.type.to_bytes(length=1, byteorder='big', signed=False)
                    # buf_ver = record.version.to_bytes(length=2, byteorder = 'big', signed=False)
                    # buf_len = record.length.to_bytes(length = 2, byteorder= 'big', signed=False)
                    # dataServerHello = buf_cont + buf_ver + buf_len + record.data
                    self.server_hello_num += 1
                    self.cipher = (record.data[-2] + record.data[-3] * 256)
                    if (not self.packet["server_hello"]):
                        packet = []
                        packet.append(nth.to_bytes(length=4, byteorder='big', signed=False))
                        # packet.append(math.floor(timestamp * 1000).to_bytes(length=4, byteorder='big', signed=False))
                        packet.append((len(ip) + 14).to_bytes(length=4, byteorder='big', signed=False))
                        packet.append(ip.src)
                        packet.append(ip.dst)
                        packet.append(ip.data.sport.to_bytes(length=4, byteorder='big', signed=False))
                        packet.append(ip.data.dport.to_bytes(length=4, byteorder='big', signed=False))
                        packet.append(bytes(3) + record.data[0:1])  # 类型
                        packet.append(bytes(1) + record.data[1:4])  # 长度
                        packet.append(bytes(2) + record.data[4:6])  # 版本
                        packet.append(bytes(2) + record.data[record.data[38] + 39: record.data[38] + 41]) # 组件
                        self.packet["server_hello"] = packet

                    head = bytes(0)
                    head += nth.to_bytes(length = 2, byteorder='big', signed=False)
                    # head += math.floor(timestamp *1000).to_bytes(length =4, byteorder= 'big', signed= False)
                    head += (len(ip) +14).to_bytes(length = 2, byteorder = 'big', signed = False)
                    head += ip.src
                    head += ip.dst
                    head += ip.data.sport.to_bytes(length = 2, byteorder ='big', signed = False)
                    head += ip.data.dport.to_bytes(length =2, byteorder = 'big', signed = False)

                    if self.server_hello_content == bytes(0):
                        self.server_hello_content = head + record.data[:6] + record.data[38:]


                if handshake_type == 11:  # certificate

                    # buf_cont = record.type.to_bytes(length=1, byteorder='big', signed=False)
                    # buf_ver = record.version.to_bytes(length=2, byteorder='big', signed=False)
                    # buf_len = record.length.to_bytes(length=2, byteorder='big', signed=False)
                    # dataCertificate = buf_cont + buf_ver + buf_len + record.data
                    packet = []
                    packet.append(nth.to_bytes(length = 4, byteorder = 'big', signed = False))
                    # packet.append(math.floor(timestamp * 1000).to_bytes(length=4, byteorder='big', signed=False))
                    packet.append((len(ip) + 14).to_bytes(length = 4, byteorder='big', signed=False))
                    packet.append(ip.src)
                    packet.append(ip.dst)
                    packet.append(ip.data.sport.to_bytes(length = 4, byteorder= 'big', signed = False))
                    packet.append(ip.data.dport.to_bytes(length = 4, byteorder= 'big', signed = False))
                    packet.append(bytes(3) + record.data[0:1])  # 类型
                    packet.append(bytes(1) + record.data[1:4])  # 长度

                    self.certificate_num += 1
                    head = bytes(0)
                    head += nth.to_bytes(length = 2, byteorder='big', signed=False)
                    # head += math.floor(timestamp *1000).to_bytes(length =4, byteorder= 'big', signed= False)
                    head += (len(ip) +14).to_bytes(length = 2, byteorder = 'big', signed=False)
                    head += ip.src
                    head += ip.dst
                    head += ip.data.sport.to_bytes(length = 2, byteorder = 'big', signed= False)
                    head += ip.data.dport.to_bytes(length = 2, byteorder = 'big', signed= False)
                    a = []
                    a = self.parse_tls_certs(nth, record.data, record.length, packet)
                    if self.certificate_content == bytes(0):
                        self.certificate_content = head + record.data
                    

                if handshake_type == 1:
                    self.client_hello_num += 1

                if handshake_type == 1:  # sslv3 tlsv1 client hello
                    # self.flag = True
                    try:
                        cipher_len = int(record.data[40 + record.data[38]])
                    except IndexError as exception:
                        cipher_len = 0

                    head = bytes(0)
                    head += nth.to_bytes(length=2, byteorder='big', signed=False)

                    # head += math.floor(timestamp * 1000).to_bytes(length=4, byteorder='big', signed=False)

                    # head += math.floor(timestamp * 1000).to_bytes(length=2, byteorder='big', signed=False)
                    head += (len(ip) + 14).to_bytes(length=2, byteorder='big', signed=False)
                    head += ip.src
                    head += ip.dst
                    head += ip.data.sport.to_bytes(length=2, byteorder='big', signed=False)
                    head += ip.data.dport.to_bytes(length=2, byteorder='big', signed=False)
                    if self.client_hello_content == bytes(0):
                        self.client_hello_content = head + record.data[:6] + record.data[38:]

                    # buf_cont = record.type.to_bytes(length=1, byteorder='big', signed=False)
                    # buf_ver = record.version.to_bytes(length=2, byteorder='big', signed=False)
                    # buf_len = record.length.to_bytes(length=2, byteorder='big', signed=False)
                    # dataClientHello= buf_cont + buf_ver + buf_len + record.data
                    packet = []
                    packet.append(nth.to_bytes(length=4, byteorder='big', signed=False))
                    # packet.append(math.floor(timestamp * 1000).to_bytes(length=4, byteorder='big', signed=False))
                    packet.append((len(ip) + 14).to_bytes(length=4, byteorder='big', signed=False))
                    packet.append(ip.src)
                    packet.append(ip.dst)
                    packet.append(ip.data.sport.to_bytes(length=4, byteorder='big', signed=False))
                    packet.append(ip.data.dport.to_bytes(length=4, byteorder='big', signed=False))
                    packet.append(bytes(3) + record.data[0:1])  # 类型
                    packet.append(bytes(1) + record.data[1:4])  # 长度
                    packet.append(bytes(2) + record.data[4:6])  # 版本
                    try:
                        packet.append(bytes(2) + record.data[record.data[38]+39:record.data[38]+41]) # 组件长度
                    except:
                        pass
                        # print(self.ip_dst)
                        # print(self.ip_src)
                        # print(nth)

                    self.cipher_num = max(cipher_len, self.cipher_num)
                    tem = 40 + record.data[38] + 1
                    i = 0
                    try:
                        while i < cipher_len:
                            cipher = record.data[tem + i] * 256 + record.data[tem + i + 1]

                            packet.append(bytes(2) + record.data[tem+i :tem+i +2])
                            if cipher not in self.cipher_support:
                                self.cipher_support.append(cipher)
                            i += 2
                        if not self.packet["client_hello"]:
                            self.packet["client_hello"] = packet
                    except IndexError:
                        # encypted hanshake message或者application报文中头字节为1，不是client_hello,匹配错误
                        pass


                        # print(nth, record.data[40])
                    


            else:
                type.append(record.type)
            n += 1
            sys.stdout.flush()
        try:
            self.tls_seq[nth - 1] = type
        except:
            print(nth, len(self.tls_seq))
            print(self.ip_src, self.ip_dst)
        # ressembled tcp segments
        load = stream[bytes_used:]
        if load == None:
            load = bytes(0)
        return load, flag


    def parse_tls_certs(self, nth, data, record_length, packet):
        """
        Parses TLS Handshake message contained in data according to their type.
        """
        ans = []
        handshake_type = ord(data[:1])  # 握手类型
        if handshake_type == 4:
            print('[#] New Session Ticket is not implemented yet')
            return ans

        buffers = data[0:]
        try:
            handshake = dpkt.ssl.TLSHandshake(buffers)
        except dpkt.ssl.SSL3Exception as exception:
            pass
            # print('exception while parsing TLS handshake record: {0}'.format(exception))
        except dpkt.dpkt.NeedData as exception:
            pass
            # print('exception while parsing TLS handshake record: {0}'.format(exception))
        try:
            ch = handshake.data
        except UnboundLocalError as exception:
            pass
        else:
            if handshake.type == 11:  # TLS Certificate
                # ssl_servers_with_handshake.add(client)
                hd_data = handshake.data
                assert isinstance(hd_data, dpkt.ssl.TLSCertificate)
                certs = []
                # print(dir(hd))
                if len(hd_data.certificates) != 0:
                    cert_1 = hd_data.certificates[0]
                    try:
                        cert_1 = OpenSSL.crypto.load_certificate(OpenSSL.crypto.FILETYPE_ASN1, cert_1)
                        if cert_1 not in self.certificate:
                            self.certificate.append(cert_1)
                            self.cipher_subject.append(cert_1.get_subject().CN)
                            self.cipher_issue.append(cert_1.get_issuer().CN)
                            # self.cipher_certifcate_time.append(cert_1.get_notAfter()-cert_1.get_notBefore())
                            packet.append(cert_1.get_version().to_bytes(length=4, byteorder='big', signed=False))
                            signature = cert_1.get_signature_algorithm()
                            # cut_bytes(signature, packet)
                            try:
                                issue = bytes(cert_1.get_issuer().CN.replace(" ", ""), encoding='utf-8')
                            except:
                                issue = bytes(0)
                            self.cut_bytes(issue, packet)
                            
                            time_before = cert_1.get_notBefore()[:8]
                            time_after = cert_1.get_notAfter()[:8]
                            packet.append(time_before[:4])
                            packet.append(time_before[4:])
                            packet.append(time_after[:4])
                            packet.append(time_after[4:])                
                            try:
                                subject = bytes(cert_1.get_subject().CN.replace(" ", ""), encoding='utf-8')
                            except:
                                subject = bytes(0)
                            self.cut_bytes(subject, packet)
                        
                            packet.append(cert_1.get_extension_count().to_bytes(length=4, byteorder='big', signed=False))
                            

                            before = datetime.strptime(cert_1.get_notBefore().decode()[:-7], '%Y%m%d')
                            after = datetime.strptime(cert_1.get_notAfter().decode()[:-7], '%Y%m%d')
                            self.cipher_certifcate_time.append((after - before).days)
                            self.cipher_extension_count.append(cert_1.get_extension_count())
                            self.cipher_sigature_alo.append(cert_1.get_signature_algorithm())
                            self.cipher_version.append(cert_1.get_version())
                            self.cipher_pubkey.append(cert_1.get_pubkey())
                            self.cipher_serial_number.append(cert_1.get_serial_number())
                            if cert_1.get_subject() == cert_1.get_issuer():
                                # 自签名
                                self.cipher_self_signature.append(1)
                            else:
                                # 非自签名
                                self.cipher_self_signature.append(0)
                        if not self.packet["certificate"]:
                            self.packet["certificate"] = packet
                        ans += certs
                    except:
                        print("certificate证书解析错误")
                        pass
                
        return ans


    def cut_bytes(self, tem, packet):
        i = 0
        while (i<len(tem)):
            if (i+4)<=len(tem):
                packet.append(tem[i:i+4])
            else:
                packet.append(bytes(4-len(tem) +i) + tem[i:])
            i += 4 

    def pretty_name(self, name_type, name_value):
        """Returns the pretty name for type name_type."""
        if name_type in PRETTY_NAMES:
            if name_value in PRETTY_NAMES[name_type]:
                name_value = PRETTY_NAMES[name_type][name_value]
            else:
                name_value = '{0}: unknown value {1}'.format(name_value, name_type)
        else:
            name_value = 'unknown type: {0}'.format(name_type)
        return name_value

        
    

    
    


            