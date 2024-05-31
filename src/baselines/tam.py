from .ens import ens
from tam import adjust_output as _adjust_output


class tam(ens):
    def parse_args(parser):
        parser.add_argument('--tam_alpha', type=float, default=2.5, help='coefficient of ACM')
        parser.add_argument('--tam_beta', type=float, default=0.5, help='coefficient of ADM')
        parser.add_argument('--temp_phi', type=float, default=1.2, help='classwise temperature')


    def adjust_output(self, output, epoch):
        return _adjust_output(self.args, output, self.data.edge_index, self.data.y,
                              self.mask(), self.aggregator, self.class_num_list, epoch)
