import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """

    def __init__(self, n_hops, n_users, interact_mat,
                 edge_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(GraphConv, self).__init__()

        self.interact_mat = interact_mat
        self.n_users = n_users
        self.n_hops = n_hops
        self.edge_dropout_rate = edge_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def forward(self, user_embed, item_embed,
                mess_dropout=True, edge_dropout=True):
        # user_embed: [n_users, channel]
        # item_embed: [n_items, channel]

        # all_embed: [n_users+n_items, channel]
        all_embed = torch.cat([user_embed, item_embed], dim=0)
        agg_embed = all_embed
        embs = [all_embed]

        for hop in range(self.n_hops):
            interact_mat = self._sparse_dropout(self.interact_mat,
                                                self.edge_dropout_rate) if edge_dropout \
                else self.interact_mat

            agg_embed = torch.sparse.mm(interact_mat, agg_embed)
            if mess_dropout:
                agg_embed = self.dropout(agg_embed)
            # agg_embed = F.normalize(agg_embed)
            embs.append(agg_embed)
        embs = torch.stack(embs, dim=1)  # [n_entity, n_hops+1, emb_size]
        return embs[:self.n_users, :], embs[self.n_users:, :]


class LightGCN(nn.Module):
    def __init__(self, data_config, args_config, adj_mat):
        super(LightGCN, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.adj_mat = adj_mat

        self.decay = args_config.l2
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.edge_dropout = args_config.edge_dropout
        self.edge_dropout_rate = args_config.edge_dropout_rate
        self.pool = args_config.pool
        self.epoch = args_config.epoch
        self.n_negs = args_config.n_negs
        self.ns = args_config.ns
        self.K = args_config.K
        self.topk = args_config.topk

        self.simi = args_config.simi
        self.gamma = args_config.gamma
        self.warmup = args_config.warmup
        self.p = args_config.p
        self.alpha = args_config.alpha
        self.alpha_co = args_config.alpha_co
        self.alpha_se = args_config.alpha_se
        self.beta = args_config.beta

        self.device = torch.device(f"cuda:{args_config.gpu_id}") if args_config.cuda else torch.device("cpu")

        self._init_weight()
        self.user_embed = nn.Parameter(self.user_embed)
        self.item_embed = nn.Parameter(self.item_embed)

        self.user_gate = nn.Linear(self.emb_size, self.emb_size).to(self.device)
        self.item_gate = nn.Linear(self.emb_size, self.emb_size).to(self.device)

        self.pos_gate = nn.Linear(self.emb_size, self.emb_size).to(self.device)
        self.neg_gate = nn.Linear(self.emb_size, self.emb_size).to(self.device)

        self.sigmoid = nn.Sigmoid()
        self.gcn = self._init_model()

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.user_embed = initializer(torch.empty(self.n_users, self.emb_size))
        self.item_embed = initializer(torch.empty(self.n_items, self.emb_size))

        # [n_users+n_items, n_users+n_items]
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)

    def _init_model(self):
        return GraphConv(n_hops=self.context_hops,
                         n_users=self.n_users,
                         interact_mat=self.sparse_norm_adj,
                         edge_dropout_rate=self.edge_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def forward(self, cur_epoch, batch=None):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']  # [batch_size, n_negs * K]
        # user_gcn_emb: [n_users, channel]
        # item_gcn_emb: [n_users, channel]
        user_gcn_emb, item_gcn_emb = self.gcn(self.user_embed,
                                              self.item_embed,
                                              edge_dropout=self.edge_dropout,
                                              mess_dropout=self.mess_dropout)



        if self.ns == 'rns':  # n_negs = 1
            neg_gcn_embs = item_gcn_emb[neg_item[:, :self.K]]

        elif self.ns == 'dns_mn':
            neg_gcn_embs = []
            for k in range(self.K):
                neg_gcn_embs.append(self.dynamic_mn_negative_sampling(user_gcn_emb, item_gcn_emb,
                                                                      user,
                                                                      neg_item[:,
                                                                      k * self.n_negs: (k + 1) * self.n_negs],
                                                                      pos_item))
            neg_gcn_embs = torch.stack(neg_gcn_embs, dim=1)
        elif self.ns == 'mix':
            neg_gcn_embs = []
            for k in range(self.K):
                neg_gcn_embs.append(self.mix_negative_sampling(user_gcn_emb, item_gcn_emb,
                                                               user,
                                                               neg_item[:, k * self.n_negs: (k + 1) * self.n_negs],
                                                               pos_item))
            neg_gcn_embs = torch.stack(neg_gcn_embs, dim=1)


        elif self.ns == 'dtl':
            neg_gcn_embs = []
            em_path = batch['em_path']
            co_path = batch['co_path']  # [batch_size, n_negs * K]
            for k in range(self.K):
                neg_gcn_embs.append(self.mix_negative_sampling_dtl(user_gcn_emb, item_gcn_emb,
                                                                user,
                                                                neg_item[:, k * self.n_negs: (k + 1) * self.n_negs],
                                                                pos_item,
                                                                em_path, co_path))

            neg_gcn_embs = torch.stack(neg_gcn_embs, dim=1)

        else:  # ahns
            neg_gcn_embs = []
            for k in range(self.K):
                neg_gcn_embs.append(self.adaptive_negative_sampling(user_gcn_emb, item_gcn_emb,
                                                                    user,
                                                                    neg_item[:, k * self.n_negs: (k + 1) * self.n_negs],
                                                                    pos_item))
            neg_gcn_embs = torch.stack(neg_gcn_embs, dim=1)

        return self.create_bpr_loss(user, user_gcn_emb[user], item_gcn_emb[pos_item], neg_gcn_embs)


    def adaptive_negative_sampling(self, user_gcn_emb, item_gcn_emb, user, neg_candidates, pos_item):
        batch_size = user.shape[0]
        s_e, p_e = user_gcn_emb[user], item_gcn_emb[pos_item]  # [batch_size, n_hops+1, channel]
        n_e = item_gcn_emb[neg_candidates]  # [batch_size, n_negs, n_hops+1, channel]

        s_e = s_e.mean(dim=1)  # [batch_size, channel]
        p_e = p_e.mean(dim=1)  # [batch_size, channel]
        n_e = n_e.mean(dim=2)  # [batch_size, n_negs, channel]

        p_scores = self.similarity(s_e, p_e).unsqueeze(dim=1)  # [batch_size, 1]
        n_scores = self.similarity(s_e.unsqueeze(dim=1), n_e)  # [batch_size, n_negs]

        scores = torch.abs(n_scores - self.beta * (p_scores + self.alpha).pow(self.p + 1))

        """adaptive negative sampling"""
        indices = torch.min(scores, dim=1)[1].detach()  # [batch_size]
        neg_item = torch.gather(neg_candidates, dim=1, index=indices.unsqueeze(-1)).squeeze()

        return item_gcn_emb[neg_item]



    def dynamic_mn_negative_sampling(self, user_gcn_emb, item_gcn_emb, user, neg_candidates, pos_item):
        batch_size = user.shape[0]
        s_e, p_e = user_gcn_emb[user], item_gcn_emb[pos_item]  # [batch_size, n_hops+1, channel]
        n_e = item_gcn_emb[neg_candidates]  # [batch_size, n_negs, n_hops+1, channel]

        s_e = s_e.mean(dim=1)  # [batch_size, channel]
        p_e = p_e.mean(dim=1)  # [batch_size, channel]
        n_e = n_e.mean(dim=2)  # [batch_size, n_negs, channel]

        """dynamic negative sampling"""
        scores = self.similarity(s_e.unsqueeze(dim=1), n_e)  # [batch_size, n_negs]
        indices = torch.topk(scores, self.topk, dim=1)[1].detach()  # [batch_size, topk]
        selected_indices = torch.randint(0, self.topk, (batch_size,)).to(p_e.device)  # [batch_size]
        result_indices = torch.gather(indices, dim=1, index=selected_indices.unsqueeze(1)).squeeze()  # [batch_size]

        neg_item = torch.gather(neg_candidates, dim=1, index=result_indices.unsqueeze(-1)).squeeze()

        return item_gcn_emb[neg_item]



    def batch_lcp_similarity(self, pos_codes, neg_codes, pad_value=-1):
        """
        pos_codes: [batch, depth]
        neg_codes: [batch, n_negs, depth]
        """
        B, N, D = neg_codes.shape

        # real lengths
        pos_len = (pos_codes != pad_value).sum(dim=1)  # [B]
        neg_len = (neg_codes != pad_value).sum(dim=2)  # [B, N]
        min_len = torch.minimum(pos_len.unsqueeze(1), neg_len)  # [B, N]

        # 1. broadcast
        pos = pos_codes.unsqueeze(1).expand_as(neg_codes)

        # 2. compare
        eq_raw = (pos == neg_codes)

        # 3. mask
        valid = (pos != pad_value) & (neg_codes != pad_value)
        eq = eq_raw & valid

        # 4. mismatch index
        mismatch = (~eq).int().argmax(dim=2)

        # 5. fully matched items
        fully_match = eq.all(dim=2)
        mismatch[fully_match] = min_len[fully_match]

        # final LCP
        lcp = mismatch.float()
        sim = lcp / min_len.clamp(min=1).float()  # 防止除0

        return sim  # [B, N]


    def mix_negative_sampling_dtl(self, user_gcn_emb, item_gcn_emb, user, neg_candidates, pos_item,
                                   emb_tensor, co_tensor):
        batch_size = user.shape[0]
        s_e, p_e = user_gcn_emb[user], item_gcn_emb[pos_item]  # [B, H, C]

        if self.pool != 'concat':
            s_e = self.pooling(s_e).unsqueeze(1)  # [B, 1, C]

        # ----- MixGCF mixing -----
        seed = torch.rand(batch_size, 1, p_e.shape[1], 1).to(p_e.device)  # [B,1,H,1]
        n_e = item_gcn_emb[neg_candidates]  # [B, N, H, C]
        n_e_ = seed * p_e.unsqueeze(1) + (1 - seed) * n_e  # [B, N, H, C]

        # ----- Mix score -----
        scores = (s_e.unsqueeze(1) * n_e_).sum(-1)  # [B, N, H]

        # ----- normalize score -----
        min_val = scores.min(dim=1, keepdim=True)[0]  # [B,1,H]
        max_val = scores.max(dim=1, keepdim=True)[0]  # [B,1,H]
        normalized_scores = (scores - min_val) / (max_val - min_val + 1e-8)  # [B,N,H]

        # ----- BSCN structural similarity -----
        pos_co = co_tensor[pos_item]  # [B, D]
        neg_co = co_tensor[neg_candidates]  # [B, N, D]
        pos_emb = emb_tensor[pos_item]  # [B, D2]
        neg_emb = emb_tensor[neg_candidates]  # [B, N, D2]

        sim_co = self.batch_lcp_similarity(pos_co, neg_co)  # [B, N]
        sim_emb = self.batch_lcp_similarity(pos_emb, neg_emb)  # [B, N]

        # broadcast to hop dim
        sim_co = sim_co.unsqueeze(-1)  # [B,N,1]
        sim_emb = sim_emb.unsqueeze(-1)  # [B,N,1]

        # ----- final score -----
        final_scores = normalized_scores + self.alpha_co * sim_co + self.alpha_se * sim_emb  # [B,N,H]

        # ----- pick hardest per hop -----
        indices = torch.max(final_scores, dim=1)[1].detach()  # [B, H]
        neg_items_emb_ = n_e_.permute([0, 2, 1, 3])  # [B, H, N, C]

        return neg_items_emb_[[[i] for i in range(batch_size)],
               range(neg_items_emb_.shape[1]), indices, :]


    def mix_negative_sampling(self, user_gcn_emb, item_gcn_emb, user, neg_candidates, pos_item):
        batch_size = user.shape[0]
        s_e, p_e = user_gcn_emb[user], item_gcn_emb[pos_item]  # [batch_size, n_hops+1, channel]
        if self.pool != 'concat':
            s_e = self.pooling(s_e).unsqueeze(dim=1)

        """positive mixing"""
        seed = torch.rand(batch_size, 1, p_e.shape[1], 1).to(p_e.device)  # (0, 1)
        n_e = item_gcn_emb[neg_candidates]  # [batch_size, n_negs, n_hops, channel]
        n_e_ = seed * p_e.unsqueeze(dim=1) + (1 - seed) * n_e  # mixing

        """hop mixing"""
        scores = (s_e.unsqueeze(dim=1) * n_e_).sum(dim=-1)  # [batch_size, n_negs, n_hops+1]
        indices = torch.max(scores, dim=1)[1].detach()
        neg_items_emb_ = n_e_.permute([0, 2, 1, 3])  # [batch_size, n_hops+1, n_negs, channel]

        # [batch_size, n_hops+1, channel]
        return neg_items_emb_[[[i] for i in range(batch_size)],
               range(neg_items_emb_.shape[1]), indices, :]

    def pooling(self, embeddings):
        # [-1, n_hops, channel]
        if self.pool == 'mean':
            return embeddings.mean(dim=1)
        elif self.pool == 'sum':
            return embeddings.sum(dim=1)
        elif self.pool == 'concat':
            return embeddings.view(embeddings.shape[0], -1)
        else:  # final
            return embeddings[:, -1, :]

    def similarity(self, user_embeddings, item_embeddings):
        # [-1, n_hops, channel]
        if self.simi == 'ip':
            return (user_embeddings * item_embeddings).sum(dim=-1)
        elif self.simi == 'cos':
            return F.cosine_similarity(user_embeddings, item_embeddings, dim=-1)
        elif self.simi == 'ed':
            return ((user_embeddings - item_embeddings) ** 2).sum(dim=-1)
        else:  # ip
            return (user_embeddings * item_embeddings).sum(dim=-1)

    def generate(self, split=True):
        user_gcn_emb, item_gcn_emb = self.gcn(self.user_embed, self.item_embed, edge_dropout=False, mess_dropout=False)
        user_gcn_emb, item_gcn_emb = self.pooling(user_gcn_emb), self.pooling(item_gcn_emb)
        if split:
            return user_gcn_emb, item_gcn_emb
        else:
            return torch.cat([user_gcn_emb, item_gcn_emb], dim=0)

    def rating(self, u_g_embeddings=None, i_g_embeddings=None):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def create_bpr_loss(self, user, user_gcn_emb, pos_gcn_embs, neg_gcn_embs):
        # user_gcn_emb: [batch_size, n_hops+1, channel]
        # pos_gcn_embs: [batch_size, n_hops+1, channel]
        # neg_gcn_embs: [batch_size, K, n_hops+1, channel]

        batch_size = user.shape[0]

        u_e = self.pooling(user_gcn_emb)
        pos_e = self.pooling(pos_gcn_embs)
        neg_e = self.pooling(neg_gcn_embs.view(-1, neg_gcn_embs.shape[2], neg_gcn_embs.shape[3])).view(batch_size,
                                                                                                       self.K, -1)

        pos_scores = (u_e * pos_e).sum(dim=-1)
        neg_scores = (u_e.unsqueeze(dim=1) * neg_e).sum(dim=-1)

        mf_loss = torch.mean(torch.log(1 + torch.exp(neg_scores - pos_scores.unsqueeze(dim=1)).sum(dim=1)))

        if self.ns == 'dens' and self.gamma > 0.:
            gate_pos = torch.sigmoid(self.item_gate(pos_gcn_embs) + self.user_gate(user_gcn_emb))
            gated_pos_e_r = pos_gcn_embs * gate_pos
            gated_pos_e_ir = pos_gcn_embs - gated_pos_e_r

            gate_neg = torch.sigmoid(self.neg_gate(neg_gcn_embs) + self.pos_gate(gated_pos_e_r).unsqueeze(1))
            gated_neg_e_r = neg_gcn_embs * gate_neg
            gated_neg_e_ir = neg_gcn_embs - gated_neg_e_r

            gated_pos_e_r = self.pooling(gated_pos_e_r)
            gated_neg_e_r = self.pooling(gated_neg_e_r.view(-1, neg_gcn_embs.shape[2], neg_gcn_embs.shape[3])).view(
                batch_size, self.K, -1)

            gated_pos_e_ir = self.pooling(gated_pos_e_ir)
            gated_neg_e_ir = self.pooling(gated_neg_e_ir.view(-1, neg_gcn_embs.shape[2], neg_gcn_embs.shape[3])).view(
                batch_size, self.K, -1)

            gated_pos_scores_r = torch.sum(torch.mul(u_e, gated_pos_e_r), axis=1)
            gated_neg_scores_r = torch.sum(torch.mul(u_e.unsqueeze(dim=1), gated_neg_e_r), axis=-1)  # [batch_size, K]

            gated_pos_scores_ir = torch.sum(torch.mul(u_e, gated_pos_e_ir), axis=1)
            gated_neg_scores_ir = torch.sum(torch.mul(u_e.unsqueeze(dim=1), gated_neg_e_ir), axis=-1)  # [batch_size, K]

            # BPR
            mf_loss += self.gamma * (
                        torch.mean(torch.log(1 + torch.exp(gated_pos_scores_ir - gated_pos_scores_r))) + torch.mean(
                    torch.log(1 + torch.exp(gated_neg_scores_r - gated_neg_scores_ir).sum(dim=1))) + torch.mean(
                    torch.log(1 + torch.exp(gated_neg_scores_r - gated_pos_scores_r.unsqueeze(dim=1)).sum(
                        dim=1))) + torch.mean(torch.log(
                    1 + torch.exp(gated_pos_scores_ir.unsqueeze(dim=1) - gated_neg_scores_ir).sum(dim=1)))) / 4

        # cul regularizer
        regularize = (torch.norm(user_gcn_emb[:, 0, :]) ** 2
                      + torch.norm(pos_gcn_embs[:, 0, :]) ** 2
                      + torch.norm(neg_gcn_embs[:, :, 0, :]) ** 2) / 2  # take hop=0
        emb_loss = self.decay * regularize / batch_size

        return mf_loss + emb_loss, mf_loss, emb_loss