# GPU

import torch
from torch.nn.utils import weight_norm
from torch import nn
from random import randrange


class Node():
    def __init__(self, node_num, x, sp):
        self.x = x
        self.node_num = node_num
        # special padding with mean std and median
        self.sp = sp


class Graph():
    def __init__(self, num_levels, layer_nodes, batch_size, input_dim, device):
        self.num_levels = num_levels
        self.node_per_level = layer_nodes
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.total_node_num = num_levels * self.node_per_level
        self.device = device
        # info: current node info
        self.node_info = { i:{'info': None} for i in range(self.total_node_num) }


    def build_graph(self, x, cur_num, cur_level, index, mean, std, median):
        """
        Generate the graph structure

        :param x: input data
        :param cur_num: current number
        :param cur_level: current level
        :param index: index
        :param mean: mean of an array
        :param std: std of an array
        :param median: median of an array
        :return: graph
        """
        sp = False
        x = x[:, :, index::self.node_per_level]
        if x.shape[2] < 10:
            sp = True
            _ = [mean, std, median]
            x_cat = torch.tensor([[[_[j] for j in range(3) for i in range(3)]]]).cuda()
            x_cat = x_cat.expand(self.batch_size, self.input_dim, x_cat.shape[2])
            x = torch.cat((x, x_cat), 2).cuda()

        next_level_next = cur_num + self.node_per_level # 当前位置的下一层指向节点
        self.node_info[cur_num]['info'] = Node(node_num=cur_num, x=x, sp=sp)

        if cur_level < self.num_levels-1:
            self.build_graph(x=x, cur_num=next_level_next, cur_level=cur_level+1,
                             index=index, mean=mean, std=std, median=median)


class Block(nn.Module):
    def __init__(self, configs):
        super(Block, self).__init__()
        self.device = configs.gpu
        self.input_dim = configs.enc_in
        self.batch_size = configs.batch_size
        self.output_dim = configs.enc_in
        self.seq_len = configs.seq_len
        self.kernel_size = configs.kernel_size
        self.stride = 2
        self.dropout = configs.dropout
        self.dilation = 3
        self.num_levels = configs.num_levels
        self.layer_nodes = configs.layer_nodes
        self.xgraph = Graph(num_levels=self.num_levels, layer_nodes=self.layer_nodes,
                            batch_size=self.batch_size, input_dim=self.input_dim,
                            device=self.device)
        self.skip_p = 1 # The number of times the first node in the list was skipped
        self.skip_s = 1 # The number of times a single node was skipped
        self.skip_d = 2 # even node turns
        self.skip_n = 3 # The number of times a normal node was skipped
        self.skip_t = False

        # TCN
        modules_conv = [
            weight_norm(nn.Conv1d(in_channels=self.input_dim, out_channels=self.output_dim,
                                  kernel_size=self.kernel_size, stride=self.stride,
                                  dilation=self.dilation)),
            nn.MaxPool1d(kernel_size=self.kernel_size, stride=self.stride, padding=1),
            nn.ELU(),
            nn.Dropout(self.dropout),
        ]

        self.conv = nn.Sequential(*modules_conv)

        # others
        self.padding = nn.Conv1d(in_channels=self.input_dim, out_channels=self.input_dim,
                                 kernel_size=1, padding=0, stride=1)
        self.norm = nn.BatchNorm1d(self.input_dim)
        self.elu = nn.ELU()
        self.dpt = nn.Dropout(0.1)
        self.pooling = nn.AvgPool1d(kernel_size=self.kernel_size, stride=self.stride,
                                    padding=1)

        # test
        modules_conv_e = [
            weight_norm(nn.Conv1d(in_channels=self.input_dim, out_channels=self.output_dim,
                                  kernel_size=3, stride=2,
                                  dilation=3)),
            nn.MaxPool1d(kernel_size=self.kernel_size, stride=self.stride, padding=1),
            nn.ELU(),
            nn.Dropout(self.dropout),
        ]

        self.conv_e = nn.Sequential(*modules_conv_e)


    def joint(self, out_cur, out_prev, node_idx=-1):
        a = out_cur.permute(2, 0, 1)
        a_len = a.shape[0]
        b = out_prev.permute(2, 0, 1)
        b_len = b.shape[0]
        _ = []
        cnt = 0
        skip = 2 ** (node_idx - 1)

        for i in range(a_len):
            # several values need to be skipped before interpolation
            for j in range(skip):
                if cnt >= b_len:
                    break
                _.append(b[cnt].unsqueeze(0))
                cnt += 1
            _.append(a[i].unsqueeze(0))

        return torch.cat(_, 0).permute(1, 2, 0)


    def joint_explict(self, out_a, out_b, out_c):
        a = out_a.permute(2, 0, 1)
        a_len = a.shape[0]
        b = out_b.permute(2, 0, 1)
        b_len = b.shape[0]
        c = out_c.permute(2, 0, 1)
        c_len = c.shape[0]
        _ = []
        cnt = 0
        min_len = min(a_len, b_len, c_len)

        while cnt < min_len:
            _.append(a[cnt].unsqueeze(0))
            _.append(b[cnt].unsqueeze(0))
            _.append(c[cnt].unsqueeze(0))
            cnt += 1

        if cnt < a_len:
            _.append(a[cnt:, :, :])

        if cnt < b_len:
            _.append(b[cnt:, :, :])

        if cnt < c_len:
            _.append(c[cnt:, :, :])

        return torch.cat(_, 0).permute(1, 2, 0)


    def sew_up(self, cur_layer_out, prev_layer_out, cur_layer, sp):
        if cur_layer == 0:
            return cur_layer_out

        a = cur_layer_out.permute(2, 0, 1)
        a_len = a.shape[0]
        b = prev_layer_out.permute(2, 0, 1)
        b_len = b.shape[0]
        _ = []
        cnt_a = 0
        cnt_b = 0

        while cnt_a < a_len:
            cur_val = a[cnt_a].unsqueeze(0)
            prev_val = b[cnt_b].unsqueeze(0)

            if sp:
                break

            # whether you need to skip the first node
            if cnt_a == 0:
                for j in range(self.skip_p):
                    _.append(prev_val)
                    cnt_b += 1

                # odd layers
                if cur_layer % 2 != 0:
                    _.append(cur_val)
                    cnt_a += 1

                # if even layers, skip
                if cnt_a == 0:
                    cnt_a = 1
                continue

            for j in range(self.skip_n):
                if cnt_b >= b_len:
                    break
                _.append(b[cnt_b].unsqueeze(0))
                cnt_b += 1

            # If skip_trans is false, the next node to join is not a contiguous node
            if not self.skip_t:
                for j in range(self.skip_s):
                    if cnt_b >= b_len:
                        break
                    _.append(b[cnt_b].unsqueeze(0))
                    cnt_b += 1
                    self.skip_t = True
            else:
                for j in range(self.skip_d):
                    for i in range(self.skip_s):
                        if cnt_b >= b_len:
                            break
                        _.append(b[cnt_b].unsqueeze(0))
                        cnt_b += 1

                    if cnt_a >= a_len:
                        break
                    _.append(a[cnt_a].unsqueeze(0))
                    cnt_a += 1

                    self.skip_t = False
                continue

            _.append(a[cnt_a].unsqueeze(0))
            cnt_a += 1

        _.append(b[cnt_b:, :, :])
        if sp:
            _.append(a[cnt_a:, :, :])

        # Update skipped data
        self.skip_s += 1
        self.skip_n = 3 * self.skip_n + self.skip_s + self.skip_d * self.skip_s
        if cur_layer % 2 != 0:
            self.skip_p += 1

        return torch.cat(_, 0).permute(1, 2, 0)


    def dpadding(self, x, num_padding):
        """
        The value of x is supplemented as many times as needed

        :param x: input data
        :param num_padding: times
        :return: data
        """
        if num_padding == 0:
            return x

        a = x.permute(2, 0, 1)
        a_len = a.shape[0]
        target_len = a_len + num_padding
        skip_cnt = a_len // num_padding
        mult = None
        _ = []

        # If the skip_cnt is 0, the current length is less than the number that needs to be inserted
        if skip_cnt == 0:
            mult = num_padding // a_len # The number of supplements is a multiple of num_padding
            skip_cnt = 1

        for i in range(a_len):
            cur_val = a[i].unsqueeze(0)
            if num_padding == 0:
                _.append(a[i:, :, :])
                break

            if i % skip_cnt == 0:
                _.append(cur_val)
                num_padding -= 1

            if mult is not None:
                for k in range(mult):
                    _.append(cur_val)

            _.append(cur_val)

        # If the padding is excessive, the element is randomly deleted
        if len(_) > target_len:
            for k in range(len(_)-target_len):
                _.pop(randrange(len(_)))

        return torch.cat(_, 0).permute(1, 2, 0)


    def communicate(self, x, prev_out, node_idx, mean, std):
        out_cur = x

        if node_idx != 0:
            while prev_out.shape[2] - out_cur.shape[2]:
                prev_out = torch.diff(prev_out)

            out_mean = 0.8 * out_cur/mean + 0.2 * prev_out
            out_std  = 0.8 * out_cur/std + 0.2 * prev_out

            out_cur = self.joint_explict(out_mean, prev_out, out_std)
            out_cur = torch.exp(self.conv(out_cur)**2/-2)

            num_padding = x.shape[2] - out_cur.shape[2]
            out_cur = self.dpadding(x=out_cur, num_padding=num_padding)
            out_cur = self.norm(out_cur)
            out_cur = self.dpt(self.elu(out_cur))
            out_cur = x + out_cur

        return out_cur


    def forward(self, x):
        x = x.permute(0, 2, 1)
        mean = torch.mean(x)
        std = torch.std(x)
        median = torch.median(x)

        # Initialize the diagram structure
        for i in range(self.layer_nodes):
            self.xgraph.build_graph(x=x, cur_num=i, cur_level=0, index=i,
                                    mean=mean, std=std, median=median)

        prev_layer_out = None
        for i in range(self.num_levels):
            cur_layer_out = None
            prev_out = None
            for j in range(self.layer_nodes + 1):
                # If it is the last node, go back to the original starting point, otherwise to the next node
                if j == self.layer_nodes:
                    cur_node_num = i * self.layer_nodes
                else:
                    cur_node_num = j + i * self.layer_nodes
                node_idx = j % self.layer_nodes
                cur_node_x = self.xgraph.node_info[cur_node_num]['info'].x
                # Whether it was specially populated during the downsampling phase
                cur_node_sp = self.xgraph.node_info[cur_node_num]['info'].sp

                out = self.communicate(x=cur_node_x, prev_out=prev_out, node_idx=node_idx, mean=mean, std=std)

                prev_out = out
                if cur_layer_out is None:
                    cur_layer_out = out
                if node_idx != 0: # Merges the value of the current node with the value of the current layer
                    cur_layer_out = self.joint(out_cur=prev_out, out_prev=cur_layer_out, node_idx=node_idx)

                # In this state, it means that it needs to merge with the previous layer and enter the next layer
                if j == self.layer_nodes:
                    out = self.sew_up(cur_layer_out=cur_layer_out, prev_layer_out=prev_layer_out,
                                      cur_layer=i, sp=cur_node_sp)
                    # out = torch.diff(out)
                    prev_layer_out = out

        # Initialize the individual metrics
        self.skip_p = 1
        self.skip_s = 1
        self.skip_d = 2
        self.skip_n = 3
        self.skip_t = False

        return prev_layer_out.permute(0, 2, 1)


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.device = configs.gpu
        self.network = Block(configs)
        # in_features adjust with settings
        self.linear = nn.Linear(in_features=303, out_features=self.configs.pred_len)

        print(">>>>>>>>> network details printing >>>>>>>>>>")
        print(self.network)

    def forward(self, x):
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        x = x.cuda()
        x = self.network(x).cuda()
        x = self.linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + seq_last

        return x