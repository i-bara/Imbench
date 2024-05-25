import numpy as np
import torch
import torch.nn.functional as F
import random
import os
import argparse
import datetime
from data_utils import get_dataset, get_longtail_split
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score


class Timer:
    def __init__(self, baseline) -> None:
        self.time = datetime.datetime.now()
        self.baseline = baseline


    def begin(self):
        self.time = datetime.datetime.now()


    def end(self, text=''):
        self.baseline.debug(f'time passed: {datetime.datetime.now() - self.time} ({text})')

    
    def tick(self, text=''):
        self.end(text=text)
        self.begin()


class Baseline:
    def parse_args(parser):
        pass


    def add_argument(parser, *args, **kwargs):
        try:
            parser.add_argument(*args, **kwargs)
        except argparse.ArgumentError:
            pass


    def debug(self, *args_):
        if self.args.debug:
            print(*args_)


    def debug_shape(self, tensor):
        if self.args.debug:
            print(tensor.shape)


    def debug_all(self, *args_):
        if self.args.debug:
            # torch.set_printoptions(threshold=10_000)
            torch.set_printoptions(profile="full")
            print(*args_)
            torch.set_printoptions(profile="default")

    
    def stat(self, rows=None, cols=['train', 'val', 'test']):
        if rows is None:
            rows = range(self.n_cls)

        # Output the split distribution
        st = '%-8s' % 'class'
        for col in cols:
            st += '%-8s' % str(col)
        st += '%-8s' % 'total'
        self.debug(st)
        for row in rows:
            st = '%-8s' % str(row)
            for col in cols:
                st += '%-8s' % str((self.mask(col) & self.mask(row)).sum().item())
            st += '%-8s' % str((self.mask(row)).sum().item())
            self.debug(st)
        st = '%-8s' % 'total'
        for col in cols:
            st += '%-8s' % str((self.mask(col)).sum().item())
        st += '%-8s' % str((self.mask()).sum().item())
        self.debug(st)


    def stat_gen(self, data, data_train_mask, data_val_mask, data_test_mask, data_gen_mask):
        idx_train = torch.tensor(range(data_train_mask.shape[0]), device=data_train_mask.device)[data_train_mask]
        idx_val = torch.tensor(range(data_val_mask.shape[0]), device=data_val_mask.device)[data_val_mask]
        idx_test = torch.tensor(range(data_test_mask.shape[0]), device=data_test_mask.device)[data_test_mask]
        idx_gen = torch.tensor(range(data_gen_mask.shape[0]), device=data_gen_mask.device)[data_gen_mask]

        # Output the split distribution
        self.debug('class   train   val     test    gen     total   ')
        for i in range(data.y.max().item() + 1):
            idx_train_i = idx_train[(data.y == i)[idx_train]]
            idx_val_i = idx_val[(data.y == i)[idx_val]]
            idx_test_i = idx_test[(data.y == i)[idx_test]]
            idx_gen_i = idx_gen[(data.y == i)[idx_gen]]
            self.debug('%-4d    %-8d%-8d%-8d%-8d%-8d' % (i, idx_train_i.shape[0], idx_val_i.shape[0], idx_test_i.shape[0], idx_gen_i.shape[0], idx_train_i.shape[0] + idx_val_i.shape[0] + idx_test_i.shape[0] + idx_gen_i.shape[0]))
        self.debug('total   %-8d%-8d%-8d%-8d%-8d' % (idx_train.shape[0], idx_val.shape[0], idx_test.shape[0], idx_gen.shape[0], idx_train.shape[0] + idx_val.shape[0] + idx_test.shape[0] + idx_gen.shape[0]))


    def mask(self, keys=None, mask=None, **kwargs):
        if keys is not None:
            if type(keys) not in [list, torch.Tensor]:
                keys = [keys]
            mask = torch.zeros(self.data.y.shape[0], dtype=bool, device=self.device)
            for i in keys:
                if type(i) in [int, torch.Tensor]:
                    mask |= self.data.y == i
                else:
                    mask |= self.masks[i]
            return mask

        set_ = kwargs.get('set')
        cls_ = kwargs.get('cls')

        if mask is None:
            mask = torch.ones(self.data.y.shape[0], dtype=bool, device=self.device)

        if set_ is not None:
            if type(set_) not in [list, torch.Tensor]:
                set_ = [set_]
            set_mask = torch.zeros(self.data.y.shape[0], dtype=bool, device=self.device)
            for i in set_:
                set_mask |= self.masks[i]
            mask &= set_mask

        if cls_ is not None:
            if type(cls_) not in [list, torch.Tensor]:
                cls_ = [cls_]
            cls_mask = torch.zeros(self.data.y.shape[0], dtype=bool, device=self.device)
            for i in cls_:
                cls_mask |= self.data.y == i
            mask &= cls_mask
        
        return mask


    def idx(self, keys=None, mask=None, **kwargs):
        if keys is not None:
            return torch.arange(self.data.y.shape[0], device=self.device)[self.mask(keys=keys)]

        if mask is None:
            return torch.arange(self.data.y.shape[0], device=self.device)[self.mask(**kwargs)]
        else:
            return torch.arange(self.data.y.shape[0], device=self.device)[mask]
    

    def inv(self, idx):
        inv_idx = -torch.ones(self.n_cls, dtype=idx.dtype, device=idx.device)
        for i in range(len(idx)):
            inv_idx[idx[i].item()] = i
        return inv_idx
    

    def num(self, keys=None, **kwargs):
        if keys is not None:
            if type(keys) == torch.Tensor and keys.dtype == torch.bool:
                return keys.sum().item()
            return self.mask(keys=keys).sum().item()

        return self.mask(**kwargs).sum().item()
    
    
    def num_list(self, keys=None, **kwargs):
        if keys is not None:
            return torch.tensor([self.num(self.mask(keys) & self.mask(cls_)) for cls_ in range(self.n_cls)], 
                            dtype=torch.int32, device=self.device)

        return torch.tensor([self.num(cls=cls_, **kwargs) for cls_ in range(self.n_cls)], 
                            dtype=torch.int32, device=self.device)


    def __init__(self, args):
        self.args = args
        self.seed = args.seed
        self.device = torch.device(args.device)

        torch.cuda.empty_cache()
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.debug(f'seed={self.seed}')

        path = args.data_path
        path = os.path.join(path, args.dataset)
        dataset = get_dataset(args.dataset, path, split_type='full')
        data_ = dataset[0]
        self.n_cls = data_.y.max().item() + 1
        self.n_sample = data_.x.shape[0]
        self.n_feat = data_.x.shape[1]
        self.data = data_.to(self.device)

        self.debug(args.dataset)

        self.masks = dict()
        self.edge_masks = dict()

        if args.dataset in ['Cora', 'CiteSeer', 'PubMed', 'chameleon', 'squirrel', 'Actor', 'Wisconsin', 'Coauthor-CS', 'Amazon-Computers', 'Amazon-Photo', 'ogbn-arxiv']:
            # Use original split

            # if args.dataset in ['Cora', 'CiteSeer', 'PubMed']:
            #     data_train_mask = data.train_mask.clone()
            #     data_val_mask = data.val_mask.clone()
            #     data_test_mask = data.test_mask.clone()
            # elif args.dataset in ['chameleon', 'squirrel', 'Actor', 'Wisconsin']:
            #     data_train_mask = data.train_mask[:, 0].clone()  # chameleon and squirrel dataset provides 10 masks
            #     data_val_mask = data.val_mask[:, 0].clone()
            #     data_test_mask = data.test_mask[:, 0].clone()
            # elif args.dataset == 'ogbn-arxiv':
            #     data.y = data.y[:, 0]

            self.masks['train'], self.masks['val'], self.masks['test'] = get_longtail_split(self.data, imb_ratio=args.imb_ratio, train_ratio=0.1, val_ratio=0.1)

            # data_train_mask, data_val_mask, data_test_mask, class_num_list, idx_info = step(data=data, imb_ratio=args.imb_ratio)

            self.debug(f'feature size: {self.data.x.shape[1]}')
            self.debug(f'number of edges: {self.data.edge_index.shape[1]}')
            self.stat()

            self.edge_masks['train'] = torch.ones(self.data.edge_index.shape[1], dtype=torch.bool)

            # train_node_mask = self.mask(set=['train', 'val', 'test'])
            # self.edge_masks['train'] = torch.ones(self.edge_index.shape[1], dtype=torch.bool)

            # self.idx_info = [torch.arange(data.y.shape[0], device=data.y.device)[(self.data.y == i) & data_train_mask] for i in range(self.n_cls)]
            # self.class_num_list = [idx_info[i].shape[0] for i in range(n_cls)]

            # stat(data, data_train_mask, data_val_mask, data_test_mask)
            # data_train_mask, train_node_mask, train_edge_mask, class_num_list, idx_info = lt(data=data, data_train_mask=data_train_mask, imb_ratio=args.imb_ratio)
            # stat(data, data_train_mask, data_val_mask, data_test_mask)

            # assert torch.all(train_node_mask == data_train_mask | data_val_mask | data_test_mask)
            # for i in range(train_edge_mask.shape[0]):
            #     row, col = data.edge_index[0][i], data.edge_index[1][i]
            #     if train_edge_mask[i]:  # edge in mask iff both nodes in mask
            #         assert train_node_mask[row] and train_node_mask[col]
            #     else:
            #         assert not train_node_mask[row] or not train_node_mask[col]

            # print(class_num_list, data_train_mask, idx_info, train_node_mask, train_edge_mask) 
        else:
            raise NotImplementedError
        
        self.score_func = lambda perf: (perf['acc'] + perf['f1']) / 2
        self.best_score = 0.
        self.test_acc = 0.
        self.test_bacc = 0.
        self.test_f1 = 0.
        self.test_auc = 0.

        self.timer = Timer(self)


    def run(self):
        output = self.train()
        if self.args.output is not None:
            torch.save(output, self.args.output)
        print('acc: {:.9f}, bacc: {:.9f}, f1: {:.9f}, auc: {:.9f}'.format(self.test_acc*100, self.test_bacc*100, self.test_f1*100, self.test_auc*100))


    def train(self):
        pass


    def test(self, logits, mask=None, **kwargs):
        if mask is None:
            mask = self.mask()
        mask &= self.mask(**kwargs)
        
        perf_val = self.perf(logits=logits, mask=mask & self.mask(set='val'))
        perf_test = self.perf(logits=logits, mask=mask & self.mask(set='test'))
        score = self.score_func(perf_val)
        if self.best_score < score:
            self.best_score = score
            self.test_acc = perf_test['acc']
            self.test_bacc = perf_test['bacc']
            self.test_f1 = perf_test['f1']
            self.test_auc = perf_test['auc']


    @torch.no_grad()
    def perf(self, logits, mask=None, **kwargs):
        if mask is None:
            mask = self.mask()
        mask &= self.mask(**kwargs)

        perf = dict()

        pred = logits[mask].max(1)[1]
        y_pred = pred.cpu().numpy()
        y_true = self.data.y[mask].cpu().numpy()
        perf['acc'] = pred.eq(self.data.y[mask]).sum().item() / mask.sum().item()
        perf['bacc'] = balanced_accuracy_score(y_true, y_pred)
        perf['f1'] = f1_score(y_true, y_pred, average='macro')
        try:
            perf['auc'] = roc_auc_score(y_true, F.softmax(logits[mask], dim=1).cpu().numpy(), average='macro', multi_class='ovr')
        except ValueError:  # Number of classes in y_true not equal to the number of columns in 'y_score'
            pass

        return perf


    # @torch.no_grad()
    # def test(self, logits):
    #     accs, baccs, f1s, aucs = dict(), dict(), dict(), dict()
    #     for set_ in ['train', 'val', 'test']:
    #         mask = self.masks[set_]
    #         pred = logits[mask].max(1)[1]
    #         y_pred = pred.cpu().numpy()
    #         y_true = self.data.y[mask].cpu().numpy()
    #         accs[set_] = pred.eq(self.data.y[mask]).sum().item() / mask.sum().item()
    #         baccs[set_] = balanced_accuracy_score(y_true, y_pred)
    #         f1s[set_] = f1_score(y_true, y_pred, average='macro')
    #         aucs[set_] = roc_auc_score(y_true, F.softmax(logits[mask], dim=1).cpu().numpy(), average='macro', multi_class='ovr')
    #     return accs, baccs, f1s, aucs