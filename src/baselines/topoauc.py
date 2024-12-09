from renode import IMB_LOSS, index2adj_bool
from utils.GraphAUC import ELossFN, GAUCLoss
from .gnnv3 import GnnModel, GNN
from .pr import pr
import torch
from torch import optim
from torch.nn import functional as F
from torch import nn
# from .nets.sha import GCNConv, GATConv, SAGEConv
from .nets import GCNConv, GATConv, SAGEConv
from .mixv3 import EnsGNN


class TopoaucModel(nn.Module):
    def __init__(self, args, baseline):
        super(TopoaucModel, self).__init__()
        self.args = args
        self.baseline = baseline
    
        conv_dict = {
            'GCN': GCNConv,
            'GAT': GATConv,
            'SAGE': SAGEConv,
        }

        self.encoder = GNN(Conv=conv_dict[self.args.net], 
                                n_feat=self.baseline.n_feat, n_hid=self.args.feat_dim, n_cls=self.baseline.n_cls, 
                                dropout=self.args.dropout, 
                                n_layer=self.args.n_layer,
                                encoder=True)
        
        self.reg_params = self.parameters()
        self.non_reg_params = list()


    def forward(self, x, edge_index, y, **kwargs):
        self.embed = self.encoder(x=x, edge_index=edge_index, **kwargs)        
        return self.embed

# class TopoaucModel(GnnModel):
#     def __init__(self, args, baseline):
#         super().__init__(args, baseline)
#         self.criterion_dict['warmup'] = self.warmup_criterion


#     def warmup_criterion(self, output, y):
#         loss = self.baseline.imb_loss(output, y)
#         return loss


#     def criterion(self, output, y):
#         # loss = self.baseline.imb_loss(output[mask], y[mask])
#         output = F.softmax(output, dim=1)
#         loss = self.baseline.auc_loss(output, y, w_values_dict=None)
#         return loss


class topoauc(pr):
    def parse_args(parser):
        parser.add_argument('--warmup', default=5, type=int, help='warmup epoch')
        parser.add_argument('--warmup_loss_name', default="focal", type=str,
                            choices=["ce", "focal", "re-weight", "cb-softmax"])
        parser.add_argument('--factor_focal', default=2.0,    type=float, help="alpha in Focal Loss")
        parser.add_argument('--factor_cb',    default=0.9999, type=float, help="beta  in CB Loss")

        parser.add_argument("--auc_loss_name", default="ExpGAUC", 
                            choices=["ExpGAUC", "HingeGAUC", "SqGAUC"])
        parser.add_argument('--pair_ner_diff', default=1, type=int, help="add our topology weight")

        parser.add_argument('--weight_sub_dim', default=64, type=int)
        parser.add_argument('--weight_inter_dim', default=64, type=int)
        parser.add_argument('--weight_global_dim', default=64, type=int)
        parser.add_argument('--topo_dim', default=64, type=int)
        parser.add_argument('--beta', default=0.5, type=float)
        parser.add_argument('--gamma', default=1.0, type=int)


    def __init__(self, args):
        super().__init__(args)
        # self.use(TopoaucModel)

        self.imb_loss = IMB_LOSS(args.warmup_loss_name, self.n_cls, self.num_list('train'), args.factor_focal, args.factor_cb)

        adj_bool = index2adj_bool(self.data.edge_index, self.n_sample).to(self.device)
        self.auc_loss = ELossFN(self.n_cls, self.n_sample, adj_bool, self.Pi, self.gpr, self.mask('train'), self.device, 
                                weight_sub_dim=args.weight_sub_dim,
                                weight_inter_dim=args.weight_inter_dim,
                                weight_global_dim=args.weight_global_dim,
                                beta=args.beta,
                                gamma=args.gamma,
                                is_ner_weight=args.pair_ner_diff,
                                loss_type=args.auc_loss_name)
        
        self.use(TopoaucModel, self.auc_loss)
        # self.model = TopoaucModel(args=self.args, baseline=self).to(self.device)
        # self.optimizer = optim.Adam([dict(params=self.model.classifier.reg_params, weight_decay=self.args.weight_decay), 
        #                              dict(params=self.model.classifier.non_reg_params, weight_decay=0),
        #                              dict(params=self.auc_loss.parameters(), weight_decay=self.args.weight_decay),], lr=self.args.lr)
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=100)


    # def train_epoch(self, epoch):
    #     self.model.train()
    #     self.auc_loss.train()
    #     self.optimizer.zero_grad()
    #     if self.args.warmup is not None and epoch < self.args.warmup:
    #         loss = self.model(x=self.data.x, edge_index=self.data.edge_index, y=self.data.y, mask=self.mask('train'), phase='warmup')
    #     else:
    #         loss = self.model(x=self.data.x, edge_index=self.data.edge_index, y=self.data.y, mask=self.mask('train'))
    #     loss.backward()
    #     self.optimizer.step()


    # @torch.no_grad()
    # def val_epoch(self, epoch):
    #     self.model.eval()
    #     self.auc_loss.eval()
    #     if self.args.warmup is not None and epoch < self.args.warmup:
    #         loss = self.model(x=self.data.x, edge_index=self.data.edge_index, y=self.data.y, mask=self.mask('val'), phase='warmup')
    #     else:
    #         loss = self.model(x=self.data.x, edge_index=self.data.edge_index, y=self.data.y, mask=self.mask('val'))
    #     self.scheduler.step(loss)
        
        
    def epoch_output(self, epoch):
        return self.model(x=self.data.x, edge_index=self.data.edge_index, y=self.data.y)
            

    def logits(self, output):
        return F.softmax(output, dim=1)


    def loss(self, output, y):
        pred = F.softmax(output, dim=1)
        if self.epoch >= self.args.warmup:
            return self.auc_loss(pred, y, mask=self.mask('train'))
        else:
            return self.imb_loss(pred, y)
            
        # return F.cross_entropy(pred, y)
        # return self.imb_loss(pred, y)
        # return self.auc_loss(pred, y, mask=self.mask(self.phase))
        #  + 0.3 * (torch.log(1.000001 - self.model.score_bad)).squeeze(dim=-1)
        # return F.cross_entropy(output, y, reduction='none') - 0.4 * (torch.log(self.model.score)).squeeze(dim=-1) - 0.4 * (torch.log(1.000001 - self.model.score_bad)).squeeze(dim=-1)
