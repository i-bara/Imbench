from .gnn import gnn, GnnModel
from .nets.hyper import GCNConv, GATConv, SAGEConv
import torch
import numpy as np
from torch_geometric.utils import add_self_loops, remove_self_loops


class HyperimbaModel(GnnModel):
    def config_gnn(self):
        if self.args.dataset == 'PubMed':
            name = 'pubmed'
        elif self.args.dataset == 'Amazon-Photo':
            name = 'photo'
        elif self.args.dataset == 'Amazon-Computers':
            name = 'computers'
        elif self.args.dataset == 'CiteSeer':
            name = 'Citeseer'
        else:
            name = self.args.dataset

        prefix = 'src/'

        def num(strings):
            try:
                return int(strings)
            except ValueError:
                return float(strings)

        #ricci
        filename = prefix + 'hyperemb/' + name + '.edge_list'
        f=open(filename)
        cur_list=list(f)
        if name=='Cora' or name == 'Actor' or name=='chameleon' or name=='squirrel' or name == 'computers' or name=='pubmed':
            ricci_cur=[[] for i in range(len(cur_list))]
            for i in range(len(cur_list)):
                ricci_cur[i]=[num(s) for s in cur_list[i].split(' ',2)]
        else:
            ricci_cur=[[] for i in range(2*len(cur_list))]
            for i in range(len(cur_list)):
                ricci_cur[i]=[num(s) for s in cur_list[i].split(' ',2)]
                ricci_cur[i+len(cur_list)]=[ricci_cur[i][1],ricci_cur[i][0],ricci_cur[i][2]]
        ricci_cur=sorted(ricci_cur)
        k_ricci=[i[2] for i in ricci_cur]
        k_ricci=k_ricci+[0 for i in range(self.baseline.n_sample)]
        k_ricci=torch.tensor(k_ricci, dtype=torch.float)
        self.k_ricci=k_ricci.view(-1,1)
        self.n_components=1
        #poincare
        self.baseline.data.edge_index, _ = remove_self_loops(self.baseline.data.edge_index)
        keys=np.load(prefix + 'hyperemb/'+name+'_keys.npy')
        values=np.load(prefix + 'hyperemb/'+name+'_values.npy')
        e_poinc = dict(zip(keys, values))
        self.n_components_p = values.shape[1]
        alls = dict(enumerate(np.ones((self.baseline.n_sample, self.n_components_p)), 0))
        alls.update(e_poinc)
        e_poinc = torch.tensor(np.array([alls[i] for i in alls]))
        self.e_poinc = e_poinc.to(torch.float32)
        self.baseline.data.edge_index, _ = add_self_loops(self.baseline.data.edge_index,num_nodes=self.baseline.n_sample)

        self.k_ricci = self.k_ricci.to(self.baseline.device)
        self.e_poinc = self.e_poinc.to(self.baseline.device)

        conv_dict = {
            'GCN': GCNConv,
            'GAT': GATConv,
            'SAGE': SAGEConv,
        }
        gnn_kwargs = {
            'k_ricci': self.k_ricci,
            'e_poinc': self.e_poinc,
            'n_components': self.n_components,
            'n_components_p': self.n_components_p,
            'cached': True,
        }
        return conv_dict, gnn_kwargs


class hyperimba(gnn):
    def parse_args(parser):
        parser.add_argument("--loss_hp", type=float, default=1, help="Loss hyper-parameters (alpha). Default: 1")


    def __init__(self, args):
        super().__init__(args)
        self.use(HyperimbaModel)
        self.forward_kwargs['alpha_hp'] = self.args.loss_hp
