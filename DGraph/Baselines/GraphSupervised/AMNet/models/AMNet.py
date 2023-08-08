import torch
import torch.nn.functional as F
from itertools import chain
import torch.nn as nn
from .layers.bernconv import BernConv


class AMNet_model(nn.Module):
    '''
    AMNet model: https://arxiv.org/abs/1905.06214

    :param in_channels: int
        number of input features
    :param hid_channels: int
        number of hidden features
    :param num_class: int
        number of classes
    :param K: int
        number of Bernoulli trials
    :param filter_num: int
        number of filters
    :param dropout: float, optional
        dropout rate
    '''

    def __init__(self, in_channels, hid_channels, num_class, K, filter_num, dropout=0.3):

        super(AMNet_model, self).__init__()
        self.act_fn = nn.ReLU()
        self.attn_fn = nn.Tanh()
        self.linear_transform_in = nn.Sequential(nn.Linear(in_channels, hid_channels),
                                                 self.act_fn,
                                                 nn.Linear(hid_channels, hid_channels),
                                                 )
        self.K = K
        self.filters = nn.ModuleList([BernConv(hid_channels, K, normalization=True, bias=True) for _ in range(filter_num)])
        self.filter_num = filter_num
        self.W_f = nn.Sequential(nn.Linear(hid_channels, hid_channels),
                                 self.attn_fn,
                                 )
        self.W_x = nn.Sequential(nn.Linear(hid_channels, hid_channels),
                                 self.attn_fn,
                                 )
        self.linear_cls_out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hid_channels, num_class))


        self.attn = list(self.W_x.parameters())
        self.attn.extend(list(self.W_f.parameters()))
        self.lin = list(self.linear_transform_in.parameters())
        self.lin.extend(list(self.linear_cls_out.parameters()))
        self.reset_parameters()


    def reset_parameters(self):
        pass


    def forward(self, x, edge_index, label=None):
        '''
        forward function

        :param x: torch_geometric.data.Data
            input features
        :param edge_index: torch.tensor
            edge index
        :param label: torch.tensor, optional
            label for training
        '''
    
        x = self.linear_transform_in(x)
        h_list = []
        for i, filter_ in enumerate(self.filters):
            h = filter_(x, edge_index)
            h_list.append(h)

        h_filters = torch.stack(h_list, dim=1)
        h_filters_proj = self.W_f(h_filters)
        x_proj = self.W_x(x).unsqueeze(-1)

        score_logit = torch.bmm(h_filters_proj, x_proj)
        soft_score = F.softmax(score_logit, dim=1)
        score = soft_score

        res = h_filters[:, 0, :] * score[:, 0]
        for i in range(1, self.filter_num):
            res += h_filters[:, i, :] * score[:, i]


        y_hat = self.linear_cls_out(res)
        marginal_loss = 0.

        if self.training:
            anomaly_train, normal_train = label
            normal_bias = score[normal_train][:, 1] - score[normal_train][:, 0]
            anomaly_bias = score[anomaly_train][:, 0] - score[anomaly_train][:, 1]
            normal_bias = torch.clamp(normal_bias, -0.)
            anomaly_bias = torch.clamp(anomaly_bias, -0.)
            normal_bias = torch.mean(normal_bias)
            anomaly_bias = torch.mean(anomaly_bias)
            bias = anomaly_bias + normal_bias
            marginal_loss = bias

        if self.training:
            return y_hat,  marginal_loss
        else:
            return y_hat



    @torch.no_grad()
    def get_attn(self, label, train_index, test_index):
        '''
        get attention score

        :param label: torch.tensor
            label
        :param train_index: torch.tensor
            train index
        :param test_index: torch.tensor
            test index
        '''
        anomaly, normal = label
        test_attn_anomaly = list(chain(*torch.mean(self.attn_score[test_index & anomaly], dim=0).tolist()))
        test_attn_normal = list(chain(*torch.mean(self.attn_score[test_index & normal], dim=0).tolist()))
        train_attn_anomaly = list(chain(*torch.mean(self.attn_score[train_index & anomaly], dim=0).tolist()))
        train_attn_normal = list(chain(*torch.mean(self.attn_score[train_index & normal], dim=0).tolist()))

        return (train_attn_anomaly, train_attn_normal), \
               (test_attn_anomaly, test_attn_normal)







