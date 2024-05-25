from .gnn import gnn
import torch


class pr(gnn):
    def __init__(self, args):
        super().__init__(args)
        self.Pi = self.pr(self.data.edge_index)

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

        # I_star = torch.zeros(self.n_sample)

        # for c in range(self.n_cls):
        #     Lc = (self.mask(c) & self.mask('train')).sum().item()
        #     Ic = torch.zeros(self.n_sample)
        #     Ic[self.mask(c) & self.mask('train')] = 1.0 / Lc
        #     if c == 0:
        #         I_star = Ic
        #     if c != 0:
        #         I_star = torch.vstack((I_star,Ic))

        # I_star = I_star.transpose(-1, -2).to(self.device)

        # self.pastel_gpr = torch.mm(self.Pi, I_star)

        # assert torch.all((self.gpr - self.pastel_gpr < 1e-3) & (self.pastel_gpr - self.gpr < 1e-3))
        # # True
