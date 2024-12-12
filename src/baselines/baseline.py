import wandb
import numpy as np
import torch
import torch.nn.functional as F
import random
import os
import json
import argparse
import datetime
from data_utils import get_dataset, get_longtail_split, get_step_split, get_natural_split
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score


wandb.login()

USE_WANDB = True
RUN_NAME = 'a'

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
            print(self.data.y.shape[0])
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
    

    def inv2(self, idx):
        inv_idx = -torch.ones(idx.shape, dtype=idx.dtype, device=idx.device)
        for i in range(len(idx)):
            inv_idx[idx[i].item()] = i
        return inv_idx
    
    
    def inv_of_permuted(self, permuted):
        inv_of_permuted = torch.zeros_like(permuted)
        inv_of_permuted[permuted] = torch.arange((len(permuted)), dtype=permuted.dtype, device=permuted.device)
        return inv_of_permuted
    

    def num(self, keys=None, mask=None, **kwargs):
        if keys is not None:
            if isinstance(keys, torch.Tensor) and keys.dtype == torch.bool:
                return keys.sum().item()
            return self.mask(keys=keys).sum().item()

        if mask is not None:
            return mask.sum().item()

        return self.mask(**kwargs).sum().item()
    
    
    def reweight(self, keys=None):
        return torch.tensor([self.num(self.mask(keys=keys) & self.mask(c)) for c in range(self.n_cls)], \
            dtype=torch.float32, device=self.device)
    
    
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
        self.n_node = self.n_sample
        self.n_feat = data_.x.shape[1]
        self.n_edge = data_.edge_index.shape[1]
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

            if args.split in ['long-tailed', 'lt']:
                self.masks['train'], self.masks['val'], self.masks['test'] = get_longtail_split(self.data, imb_ratio=args.imb_ratio, train_ratio=0.1, val_ratio=0.1)
            elif args.split in ['step', 'st']:
                self.masks['train'], self.masks['val'], self.masks['test'] = get_step_split(self.data, imb_ratio=args.imb_ratio, train_ratio=0.1, val_ratio=0.1)
            elif args.split in ['natural', 'nt']:
                self.masks['train'], self.masks['val'], self.masks['test'] = get_natural_split(self.data, imb_ratio=args.imb_ratio, train_ratio=0.1, val_ratio=0.1)
            else:
                raise NotImplementedError('Unknown split method')

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

        self._train = self.save_tensor(self.mask('train'))
        self._val = self.save_tensor(self.mask('val'))
        self._test = self.save_tensor(self.mask('test'))
        self.y = self.save_tensor(self.data.y)
        self._n_sample_ori = self.n_sample


    def save_tensor(self, tensor):
        tensor_dir = 'tensor/'
        if not os.path.isdir(tensor_dir):
            os.mkdir(tensor_dir)
        tensor_file = f'{str(datetime.datetime.now())}-{self.args.method}-{self.args.dataset}-{self.args.seed}.pt'
        tensor_path = os.path.join(tensor_dir, tensor_file)
        while os.path.isfile(tensor_path):
            tensor_file = f'{str(datetime.datetime.now())}-{self.args.method}-{self.args.dataset(lt=self.args.imb_ratio)}-{self.args.seed}.pt'
            tensor_path = os.path.join(tensor_dir, tensor_file)
        torch.save(tensor, tensor_path)
        return tensor_path


    def run(self):
        hyperparameter = dict()
        for arg in vars(self.args):
            if arg not in ['method', 'dataset', 'split', 'imb_ratio', 'seed', 'net', 'device', 'debug', 'output', 'data_path', 'n_head', 'project_name']:
                hyperparameter[arg] = getattr(self.args, arg)

        config = {
            'method': self.args.method,
            'dataset': self.args.dataset,
            'split': self.args.split,
            'imb_ratio': self.args.imb_ratio,
            'seed': self.args.seed,
            'device': str(self.device),
            'backbone': self.args.net,
            'hyperparameter': hyperparameter,
        }

        if USE_WANDB:
            run = wandb.init(
                # Set the project where this run will be logged
                project="imbench",
                name=RUN_NAME,
                # Track hyperparameters and run metadata
                config=config,
            )
            try:
                for key in wandb.config._items['hyperparameter'].keys():
                    self.args.__setattr__(key, wandb.config.__getattr__(key))
            except AttributeError:
                pass

        # Train
        begin_datetime = datetime.datetime.now()
        output = self.train()
        output = self.save_tensor(output[:self._n_sample_ori])

        end_datetime = datetime.datetime.now()

        # if self.args.output is not None:
        #     torch.save(output, self.args.output)

        system = {
            'begin_datetime': str(begin_datetime),
            'end_datetime': str(end_datetime),
            'time_erased': str(end_datetime - begin_datetime),
            'max_memory_allocated': torch.cuda.max_memory_allocated(device=self.device),
        }

        perf = {
            'acc': self.test_acc*100,
            'bacc': self.test_bacc*100,
            'f1': self.test_f1*100,
            'auc': self.test_auc*100,
            'train': self._train,
            'val': self._val,
            'test': self._test,
            'y': self.y,
            'output': output,
        }
        
        result = system | config | perf

        result = json.dumps(result, indent=4)
        print(f'result: {result}')
        
        wandb.finish()


    def train(self):
        pass


    def test(self, logits, mask=None, **kwargs):
        if mask is None:
            mask = self.mask()
        mask &= self.mask(**kwargs)
        
        perf_train = self.perf(logits=logits, mask=mask & self.mask(set='train'))
        perf_val = self.perf(logits=logits, mask=mask & self.mask(set='val'))
        perf_test = self.perf(logits=logits, mask=mask & self.mask(set='test'))
        if USE_WANDB:
            for mode, perf in zip(['train', 'val', 'test'], [perf_train, perf_val, perf_test]):
                log = dict()
                for score in perf:
                    log[mode + '_' + score] = perf[score]
                wandb.log(log)
        score = self.score_func(perf_val)
        if self.best_score < score:
            self.best_score = score
            self.test_acc = perf_test['acc']
            self.test_bacc = perf_test['bacc']
            self.test_f1 = perf_test['f1']
            try:
                self.test_auc = perf_test['auc']
            except KeyError:
                pass


    @torch.no_grad()
    def perf(self, logits, mask=None, **kwargs):
        if mask is None:
            mask = self.mask()
        mask &= self.mask(**kwargs)

        perf = dict()

        print(logits.shape, mask.shape)
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


    def save(self):
        self.data_original = self.data.clone().detach()
        self.masks_original = dict()
        for name, mask in self.masks.items():
            self.masks_original[name] = mask.clone().detach()
        self.n_sample_original = self.n_sample


    def restore(self):
        self.data = self.data_original.clone().detach()
        self.masks = dict()
        for name, mask in self.masks_original.items():
            self.masks[name] = mask.clone().detach()
        self.n_sample = self.n_sample_original


    def init_batch(self, minibatch_size):
        self.idx_batch = dict()
        self.now_batch = dict()
        self.each_batch = dict()
        for name in ['train', 'val', 'test']:
            self.idx_batch[name] = [self.idx(mask=self.mask(name) & self.mask(c)) for c in range(self.n_cls)]
            self.now_batch[name] = [0 for _ in range(self.n_cls)]
            self.each_batch[name] = [max(int(self.num(mask=self.mask(name) & self.mask(c)) / self.data.x.shape[0] * minibatch_size), 1)
                               for c in range(self.n_cls)]


    def batch(self, random=False):
        idx = []
        for name in ['train', 'val', 'test']:
            for c in range(self.n_cls):
                # random.shuffle(self.idx_batch[name][c])
                if random:
                    idx += np.random.choice(self.idx_batch[name][c].cpu().numpy(), self.each_batch[name][c], replace=False).tolist()
                else:
                    size = len(self.idx_batch[name][c])
                    begin = self.now_batch[name][c] % size
                    end = begin + self.each_batch[name][c] % size
                    if begin < end:
                        idx += self.idx_batch[name][c][begin:end]
                    else:
                        idx += self.idx_batch[name][c][begin:] + self.idx_batch[name][c][:end]
                self.now_batch[name][c] += self.each_batch[name][c]
        inv = -torch.ones(self.data.x.shape[0], dtype=torch.int64, device=self.device)
        inv[idx] = torch.arange(len(idx), dtype=torch.int64, device=self.device)
        idx = torch.tensor(idx, dtype=torch.int64, device=self.device)
        self.data.x = self.data.x[idx]
        self.data.edge_index = inv[self.data.edge_index]
        self.data.edge_index = self.data.edge_index[:, (self.data.edge_index[0] != -1) & (self.data.edge_index[1] != -1)]
        self.data.y = self.data.y[idx]
        for name, mask in self.masks.items():
            self.masks[name] = mask[idx]
        self.n_sample = self.data.x.shape[0]


    def _batch(self, epoch, minibatch_size):
        batch_size = self.data.x.shape[0]
        begin = epoch * minibatch_size % batch_size
        end = (epoch + 1) * minibatch_size % batch_size
        self.n_sample = minibatch_size
        if begin < end:
            self.data.x = self.data.x[begin:end]
            self.data.edge_index = self.data.edge_index[:, (self.data.edge_index[0] >= begin)
                                                         & (self.data.edge_index[0] < end)
                                                         & (self.data.edge_index[1] >= begin)
                                                         & (self.data.edge_index[1] < end)] - begin
            self.data.y = self.data.y[begin:end]
            for name, mask in self.masks.items():
                self.masks[name] = mask[begin:end]
        else:
            self.data.x = torch.cat((self.data.x[begin:], self.data.x[:end]))
            self.data.edge_index = self.data.edge_index[:, ((self.data.edge_index[0] >= begin)
                                                         | (self.data.edge_index[0] < end))
                                                         & ((self.data.edge_index[1] >= begin)
                                                         | (self.data.edge_index[1] < end))] - begin
            self.data.edge_index[self.data.edge_index < 0] += batch_size
            self.data.y = torch.cat((self.data.y[begin:], self.data.y[:end]))
            for name, mask in self.masks.items():
                self.masks[name] = torch.cat((mask[begin:], mask[:end]))
