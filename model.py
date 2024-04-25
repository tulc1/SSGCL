import torch
import torch.nn as nn
from utils import sparse_dropout, spmm
import torch.nn.functional as F

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


class LightGCL(nn.Module):
    def __init__(self, args, n_u, n_i, d, h_n, train_csr, adj, adj_norm, l, dropout, batch_user, device):
        super(LightGCL,self).__init__()
        self.args = args
        self.dim = d
        self.user = n_u
        self.item = n_i
        self.layer = args.gnn_layer
        self.gnn_layers = args.gnn_layer
        self.hyper_layers = args.hyper_layer
        self.h_num = h_n

        # layer
        self.gcnLayer = GCNLayer()
        self.h_gnn_layer = HGNNLayer()
        self.hyper_encoder = HyperEncoder()

        # node embedding
        self.u_embeds = nn.Parameter(init(torch.empty(self.user, self.dim)))
        self.i_embeds = nn.Parameter(init(torch.empty(self.item, self.dim)))

        # hyper embedding
        # self.u_hyper_embeds = nn.Parameter(init(torch.empty(self.h_num, self.dim)))
        # self.i_hyper_embeds = nn.Parameter(init(torch.empty(self.h_num, self.dim)))

        # edge to hyper graph
        self.u_hyper_graph = nn.Parameter(init(torch.empty(self.user, self.h_num)))
        self.i_hyper_graph = nn.Parameter(init(torch.empty(self.item, self.h_num)))

        # embedding trans to edge
        # self.u_hyper = nn.Parameter(init(torch.empty(self.dim, self.h_num)))
        # self.i_hyper = nn.Parameter(init(torch.empty(self.dim, self.h_num)))

        # batch norm
        self.bn_layers_1 = nn.ModuleList([nn.BatchNorm1d(self.dim) for _ in range(self.layer)])
        # self.bn_layers_2 = nn.ModuleList([nn.BatchNorm1d(self.dim) for _ in range(self.layer)])

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU(0.01)
        self.edgeDropper = SpAdjDropEdge()

        self.E_u, self.E_i = None, None
        self.train_csr = train_csr
        self.adj_norm = adj_norm
        self.adj = adj

        self.alpha = args.alpha
        self.temp_1 = args.temp1
        self.temp_2 = args.temp2
        self.temp_3 = args.temp3
        self.lambda_1 = args.lambda1
        self.lambda_2 = args.lambda2
        self.lambda_3 = args.lambda3
        self.reg = args.reg
        self.dropout = dropout
        self.eps = args.eps
        self.perturbed = False
        self.act = nn.LeakyReLU(0.5)
        self.batch_user = batch_user

        self.device = device

    def contrastLoss(self, embeds1, embeds2, nodes, temp, normal=True):
        pckEmbeds1 = embeds1[nodes]
        pckEmbeds2 = embeds2[nodes]
        if normal:
            pckEmbeds1, pckEmbeds2 = F.normalize(pckEmbeds1, dim=1), F.normalize(pckEmbeds2, dim=1)
        pos_score = (pckEmbeds1 @ pckEmbeds2.T) / temp
        score = torch.diag(F.log_softmax(pos_score, dim=1))
        return -score.mean()

    def bpr_loss(self, u_embeds, i_embeds, uids, pos, neg):
        u_emb = u_embeds[uids]
        pos_emb = i_embeds[pos]
        neg_emb = i_embeds[neg]
        pos_scores = (u_emb * pos_emb).sum(-1)
        neg_scores = (u_emb * neg_emb).sum(-1)
        p = ((pos_scores - neg_scores) / u_embeds.shape[-1]).sigmoid()
        # p = (pos_scores - neg_scores).sigmoid()
        loss_r = -(p + 1e-15).log().mean()
        return loss_r

    def forward(self, uids, iids, pos, neg, test=False):
        if test==True:  # testing phase
            if uids is not None:
                preds = self.E_u[uids] @ self.E_i.T
                mask = self.train_csr[uids.cpu().numpy()].toarray()
                mask = torch.Tensor(mask).cuda(torch.device(self.device))
                preds = preds * (1-mask) - 1e8 * mask
                predictions = preds.argsort(descending=True)
                return predictions
            else:
                return self.E_u, self.E_i
        else:  # training phase
            embeds = torch.concat([self.u_embeds, self.i_embeds], dim=0)
            # hyper_embeds = torch.concat([self.u_hyper_embeds, self.i_hyper_embeds], dim=0)

            # res_lats_1, res_lats_2, hyper_res_lat = [embeds], [embeds], []
            bn_lats, hbn_lats, bn_lats_2 = [], [], []
            lats, lats_1, lats_2, lats_3 = [embeds], [], [], []

            # uu = torch.mm(self.u_embeds, self.u_hyper)
            # ii = torch.mm(self.i_embeds, self.i_hyper)

            # uu = torch.spmm(self.adj_norm, self.i_hyper_graph)
            # ii = torch.spmm(self.adj_norm.transpose(0, 1), self.u_hyper_graph)

            uu = torch.spmm(self.adj, self.i_hyper_graph)
            ii = torch.spmm(self.adj.transpose(0, 1), self.u_hyper_graph)

            # uu = F.gumbel_softmax(uu, self.lambda_3, dim=1, hard=False)
            # ii = F.gumbel_softmax(ii, self.lambda_3, dim=1, hard=False)

            # dropout
            # self.adj_norm
            # adj = self.edgeDropper(self.adj_norm, 1 - self.dropout)
            # uu = F.dropout(uu, p=self.dropout)
            # ii = F.dropout(ii, p=self.dropout)

            for i in range(self.layer):
                # embedding batch norm
                bn_lats.append(self.bn_layers_1[i](lats[-1]))
                # hbn_lats.append(self.bn_layers_2[i](hyper_lats[-1]))

                # GCN
                user_embeds_1 = self.gcnLayer(self.adj_norm, bn_lats[-1][self.user:])
                item_embeds_1 = self.gcnLayer(self.adj_norm.transpose(0, 1), bn_lats[-1][:self.user])

                # HGCN
                user_embeds_2 = self.h_gnn_layer(uu, bn_lats[-1][:self.user])
                item_embeds_2 = self.h_gnn_layer(ii, bn_lats[-1][self.user:])
                # user_embeds_2, user_hyper_embeds_1 = self.hyper_encoder(uu, [bn_lats[-1][:self.user], hbn_lats[-1][:self.h_num]])
                # item_embeds_2, item_hyper_embeds_1 = self.hyper_encoder(ii, [bn_lats[-1][self.user:], hbn_lats[-1][self.h_num:]])

                # user item connect
                embeds_1 = torch.cat([user_embeds_1, item_embeds_1], dim=0)
                # if self.perturbed:
                #     random_noise = torch.rand_like(embeds_1).cuda()
                #     embeds_1 += torch.sign(embeds_1) * F.normalize(random_noise, dim=-1) * self.eps
                embeds_2 = torch.cat([user_embeds_2, item_embeds_2], dim=0)
                # embeds_3 = torch.cat([user_hyper_embeds_1, item_hyper_embeds_1], dim=0)

                # fusion
                lats_1.append(embeds_1), lats_2.append(embeds_2)
                embeds_3 = embeds_1 + lats[-1]
                lats_3.append(embeds_3)
                lats.append(embeds_3 + self.alpha * F.normalize(embeds_2))

            # lats.append(sum(lats))
            self.E_u = lats[-1][:self.user]
            self.E_i = lats[-1][self.user:]

            # bpr loss
            loss_r = self.bpr_loss(self.E_u, self.E_i, uids, pos, neg)

            loss_s = 0
            loss_s_2 = 0
            for i in range(self.layer):
                embeds1 = lats_3[i]
                embeds2 = lats_1[i]
                loss_s += self.contrastLoss(embeds1[:self.user], embeds2[:self.user], torch.unique(uids), self.temp_1) + \
                          self.contrastLoss(embeds1[self.user:], embeds2[self.user:], torch.unique(pos), self.temp_1)

                # embeds4 = lats[i+1]
                embeds3 = lats_2[i]
                loss_s_2 += self.contrastLoss(embeds2[:self.user].detach(), embeds3[:self.user], torch.unique(uids), self.temp_2) + \
                          self.contrastLoss(embeds2[self.user:].detach(), embeds3[self.user:], torch.unique(pos), self.temp_2)
            loss_s = (loss_s * self.lambda_1 + loss_s_2 * self.lambda_2) / self.layer

            # reg loss
            loss_reg = 0
            for param in self.parameters():
                loss_reg += param.norm(2).square()
            loss_reg *= self.reg


            # total loss
            loss = loss_r + loss_s + loss_reg
            # print('loss',loss.item(),'loss_r',loss_r.item(),'loss_s',loss_s.item())
            return loss, loss_r, loss_s


class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()

    def forward(self, adj, embeds):
        return torch.spmm(adj, embeds)


class HGNNLayer(nn.Module):
    def __init__(self):
        super(HGNNLayer, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, adj, embeddings):
        tmp = torch.mm(adj.T, embeddings)
        lat = torch.mm(adj, tmp)
        return lat


class HyperEncoder(nn.Module):
    def __init__(self):
        super(HyperEncoder, self).__init__()

    def forward(self, adj, embeddings):
        embedding, hyper_embedding = embeddings
        new_embedding = torch.mm(adj, hyper_embedding)
        new_hyper_embedding = torch.mm(adj.T, embedding)
        return new_embedding, new_hyper_embedding


class FNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FNN, self).__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)

    def forward(self, embeddings):
        return self.linear2(self.linear1(embeddings))


class SpAdjDropEdge(nn.Module):
    def __init__(self):
        super(SpAdjDropEdge, self).__init__()

    def forward(self, adj, keepRate):
        if keepRate == 1.0:
            return adj
        vals = adj._values()
        idxs = adj._indices()
        edgeNum = vals.size()
        mask = ((torch.rand(edgeNum) + keepRate).floor()).type(torch.bool)
        newVals = vals[mask] / keepRate
        newIdxs = idxs[:, mask]
        return torch.sparse.FloatTensor(newIdxs, newVals, adj.shape)
