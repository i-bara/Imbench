from renode import IMB_LOSS, get_renode_weight
from .gnn import GnnModel
from .pr import pr
import torch


class RenodeModel(GnnModel):
    def criterion(self, output, y, mask, weight):
        loss = self.baseline.imb_loss(output[mask], y[mask])
        loss = torch.sum(loss * self.baseline.rn_weight[mask]) / loss.size(0)

        return loss


class renode(pr):
    def parse_args(parser):
        parser.add_argument('--loss_name', default="ce", type=str, help="the training loss") #ce focal re-weight cb-softmax
        parser.add_argument('--factor_focal', default=2.0, type=float, help="alpha in Focal Loss")
        parser.add_argument('--factor_cb', default=0.9999, type=float, help="beta  in CB Loss")
        parser.add_argument('--rn_base', default=0.5, type=float, help="Lower bound of RN")
        parser.add_argument('--rn_max', default=1.5, type=float, help="Upper bound of RN")


    def __init__(self, args):
        super().__init__(args)
        self.use(RenodeModel)

        self.imb_loss = IMB_LOSS(args.loss_name, self.n_cls, self.num_list('train'), args.factor_focal, args.factor_cb)
        self.rn_weight =  get_renode_weight(self.data.y, self.Pi, self.gpr, self.mask('train'), args.rn_base, args.rn_max) #ReNode Weight
