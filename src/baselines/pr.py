from renode import index2dense
from .gnn import gnn
import torch


class pr(gnn):
    def __init__(self, args):
        super().__init__(args)
        ## ReNode method ##
        ## hyperparam ##
        pagerank_prob = 0.85

        # calculating the Personalized PageRank Matrix
        pr_prob = 1 - pagerank_prob
        A = index2dense(self.data.edge_index, self.n_sample)
        A_hat   = A.to(self.device) + torch.eye(A.size(0)).to(self.device) # add self-loop
        D       = torch.diag(torch.sum(A_hat,1))
        D       = D.inverse().sqrt()
        A_hat   = torch.mm(torch.mm(D, A_hat), D)
        self.Pi = pr_prob * ((torch.eye(A.size(0)).to(self.device) - (1 - pr_prob) * A_hat).inverse())
        # Pi = Pi.cpu()


        # calculating the ReNode Weight
        gpr_matrix = [] # the class-level influence distribution

        for c in range(self.n_cls):
            #iter_Pi = data.Pi[torch.tensor(target_data.train_node[iter_c]).long()]
            iter_Pi = self.Pi[self.mask(c) & self.mask('train')] # check! is it same with above line?
            iter_gpr = torch.mean(iter_Pi,dim=0).squeeze()
            gpr_matrix.append(iter_gpr)

        temp_gpr = torch.stack(gpr_matrix,dim=0)
        temp_gpr = temp_gpr.transpose(0,1)
        self.gpr = temp_gpr
