from .baseline import Baseline
from data_utils import separator_ht, degree, adj_mse_loss, adj, idx, scheduler
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from . import layers
import tqdm
from copy import deepcopy


class Model(nn.Module):
    def __init__(self, args, baseline):
        super(Model, self).__init__()
        self.args = args
        self.baseline = baseline
        self.expert_dict = {}

        self.encoder = layers.GNN_Encoder(nfeat=args.nfeat, nhid=args.nhid, dropout=args.dropout)
        self.classifier_og = layers.GNN_Classifier(nhid=args.nhid, nclass=args.nclass, dropout=args.dropout)
        
        for sep in ['HH', 'H', 'TH', 'T']:
            num_class = baseline.cls_masks['H'].sum().item() if sep[0] == 'H' else baseline.cls_masks['T'].sum().item()
            self.expert_dict[sep] = layers.GNN_Classifier(nhid=args.nhid, nclass=num_class, dropout=args.dropout)


    def forward(self, x, y, edge_index, mask=None, classifier=None, embed=None, sep=None, cls_mask=None, teacher=None, weight=None, phase=None, logit=False):
        if embed == None:
            embed = self.encoder(x=x, edge_index=edge_index)

        if mask is None:
            mask = self.baseline.mask()

        if phase == 'embed':
            return embed

        elif phase == 'pretrain':
            generated_G = self.decoder(embed)
            loss_reconstruction = adj_mse_loss(generated_G, adj(x=x, edge_index=edge_index))
            return loss_reconstruction

        elif phase == 'og':
            output = self.classifier_og(x=embed, edge_index=edge_index)
            if logit:
                return output

            if self.args.class_weight:
                ce_loss = -F.cross_entropy(output[mask], y[mask], weight=weight)
                pt = torch.exp(-F.cross_entropy(output[mask], y[mask]))
                loss_nodeclassfication = -((1 - pt) ** self.args.gamma) * ce_loss
            else:
                ce_loss = -F.cross_entropy(output[mask], y[mask])
                pt = torch.exp(-F.cross_entropy(output[mask], y[mask]))
                loss_nodeclassfication = -((1 - pt) ** self.args.gamma) * self.args.alpha * ce_loss
            
            return loss_nodeclassfication
            # if self.args.rec:
            #     generated_G = self.decoder(embed)
            #     loss_reconstruction = adj_mse_loss(generated_G, adj(x=x, edge_index=edge_index))
            #     return loss_nodeclassfication, loss_reconstruction
            # else:
            #     return loss_nodeclassfication
        
        elif phase == 'expert':
            output = classifier(x=embed, edge_index=edge_index)
            if logit:
                return output

            y = ((y.view(-1,1) == cls_mask.nonzero().squeeze()).int().argmax(dim=1))

            loss_nodeclassfication = F.cross_entropy(output[mask], y[mask])

            return loss_nodeclassfication

        elif phase == 'student':
            # teacher
            teacher_head_degree = teacher[sep+'H']
            teacher_tail_degree = teacher[sep+'T']
            mask_head_degree = mask[sep+'H']
            mask_tail_degree = mask[sep+'T']
            mask_all = mask_head_degree | mask_tail_degree

            teacher_head_degree.eval()
            teacher_tail_degree.eval()
            
            out_head_teacher = teacher_head_degree(x=embed, edge_index=edge_index)[mask_head_degree]
            out_tail_teacher = teacher_tail_degree(x=embed, edge_index=edge_index)[mask_tail_degree]
                
            # student
            out_head_student = classifier(x=embed, edge_index=edge_index)[mask_head_degree]
            out_tail_student = classifier(x=embed, edge_index=edge_index)[mask_tail_degree]

            if logit:
                output = torch.zeros((x.shape[0], y.max().item() + 1), dtype=torch.float32, device=x.device)
                output[mask_head_degree][:, cls_mask] = out_head_student
                output[mask_tail_degree][:, cls_mask] = out_tail_student
                return output

            kd_head = F.kl_div(F.log_softmax(out_head_student / self.args.tau, dim=1), F.softmax(out_head_teacher / self.args.tau, dim=1), reduction='mean') * self.args.tau * self.args.tau
            kd_tail = F.kl_div(F.log_softmax(out_tail_student / self.args.tau, dim=1), F.softmax(out_tail_teacher / self.args.tau, dim=1), reduction='mean') * self.args.tau * self.args.tau
            
            y = ((y.view(-1,1) == cls_mask.nonzero().squeeze()).int().argmax(dim=1))

            ce_loss = F.cross_entropy(classifier(x=embed, edge_index=edge_index)[mask_all], y[mask_all])

            return kd_head, kd_tail, ce_loss

        else:
            raise NotImplementedError


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0, more=False):
        self.patience = patience
        self.min_delta = min_delta
        self.more = more

        self.counter = 0
        if self.more:
            self.min_validation_loss = -float('inf')
        else:
            self.min_validation_loss = float('inf')

    def reset(self):
        self.counter = 0
        if self.more:
            self.min_validation_loss = -float('inf')
        else:
            self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss, func=None):
        if self.more:
            if validation_loss > self.min_validation_loss:
                self.min_validation_loss = validation_loss
                if func is not None:
                    func()
                self.counter = 0
            elif validation_loss < (self.min_validation_loss - self.min_delta):
                self.counter += 1
                if self.counter >= self.patience:
                    return True
        else:
            if validation_loss < self.min_validation_loss:
                self.min_validation_loss = validation_loss
                if func is not None:
                    func()
                self.counter = 0
            elif validation_loss > (self.min_validation_loss + self.min_delta):
                self.counter += 1
                if self.counter >= self.patience:
                    return True
        return False


class lte4g(Baseline):
    def parse_args(parser):
        Baseline.add_argument(parser, "--tau", type=float, default=1, help="")


    def predict_ht(self, model):
        embed = model(x=self.data.x, y=self.data.y, edge_index=self.data.edge_index, phase='embed')
        output = model(x=self.data.x, y=self.data.y, edge_index=self.data.edge_index, logit=True, phase='og')
        prediction = F.softmax(output, dim=1)

        centroids = torch.empty((self.args.nclass, embed.shape[1])).to(embed.device)

        for i in range(self.n_cls):
            resources = []
            centers = list(map(int, self.idx(set='train', cls=i)))
            resources.extend(centers)
            adj_dense = adj(x=self.data.x, edge_index=self.data.edge_index)[centers]
            adj_dense[adj_dense>0] = 1

            similar_matrix = (F.normalize(self.data.x) @ F.normalize(self.data.x).T)[centers]
            similar_matrix -= adj_dense

            if self.args.criterion == 'mean':
                avg_num_candidates = int(self.num(set='train') / self.n_cls)
            elif self.args.criterion == 'median':
                avg_num_candidates = int(torch.median(self.num_list(set='train')))
            elif self.args.criterion == 'max':
                avg_num_candidates = max(self.num_list(set='train'))

            if self.num(set='train', cls=i) < avg_num_candidates:
                num_candidates_to_fill = avg_num_candidates - self.num(set='train', cls=i)
                neighbors = np.array(list(set(map(int, adj_dense.nonzero()[:,1])) - set(centers)))
                similar_nodes = np.array(list(set(map(int, similar_matrix.topk(10+1)[1][:,1:].reshape(-1)))))

                # Candidate Selection
                candidates_by_neighbors = prediction.cpu()[neighbors, i].sort(descending=True)[1][:num_candidates_to_fill]
                resource = neighbors[candidates_by_neighbors]
                if len(candidates_by_neighbors) != 0:
                    resource = [resource] if len(candidates_by_neighbors) == 1 else resource
                    resources.extend(resource)
                if len(resources) < num_candidates_to_fill:
                        num_candidates_to_fill = num_candidates_to_fill - len(resources)
                        candidates_by_similar_nodes = prediction.cpu()[similar_nodes, i].sort(descending=True)[1][:num_candidates_to_fill]
                        resource = similar_nodes[candidates_by_similar_nodes]
                        if len(candidates_by_similar_nodes) != 0:
                            resource = [resource] if len(candidates_by_similar_nodes) == 1 else resource
                            resources.extend(resource)

            resource = torch.tensor(resources)

            centroids[i, :] = embed[resource].mean(0)
        
        similarity = (F.normalize(embed) @ F.normalize(centroids).t())

        sim_top1 = torch.argmax(similarity, dim=1).long()

        self.masks['H_predicted'] = self.cls_masks['H'][sim_top1]
        self.masks['T_predicted'] = torch.logical_not(self.masks['H_predicted'])

        # Top-1 Similarity
        # sim_top1_val = torch.argmax(similarity[self.idx_val], 1).long() # top 1 similarity
        # sim_top1_test = torch.argmax(similarity[self.idx_test], 1).long() # top 1 similarity

        # idx_val_ht_pred = (sim_top1_val >= self.args.sep_point).long()
        # idx_test_ht_pred = (sim_top1_test >= self.args.sep_point).long()
        
        # idx_class = {}
        # for index in [self.idx_val, self.idx_test]:
        #     idx_class[index] = {}

        # idx_class[self.idx_val]['H'] = self.idx_val[(idx_val_ht_pred == 0)].detach().cpu()
        # idx_class[self.idx_val]['T'] = self.idx_val[(idx_val_ht_pred == 1)].detach().cpu()

        # idx_class[self.idx_test]['H'] = self.idx_test[(idx_test_ht_pred == 0)].detach().cpu()
        # idx_class[self.idx_test]['T'] = self.idx_test[(idx_test_ht_pred == 1)].detach().cpu()


    def __init__(self, args):
        super().__init__(args)
        print(f'train num list: {self.num_list(set='train')}')
        ht_dict = separator_ht(self.num_list(set='train'))
        print(f'ht_dict: {ht_dict}')
        self.masks['H'] = self.mask(ht_dict['H'])
        self.masks['T'] = torch.logical_not(self.masks['H'])
        
        deg = degree(self.data)
        # deg[::2] += 0.01  # Avoid all the same
        # self.debug_all(deg)
        
        # max_deg_train = deg[self.mask('train')].max().item()
        # min_deg_train = deg[self.mask('train')].min().item()
        # median = torch.median(deg[self.mask('train')])
        # print(max_deg_train, min_deg_train, median)
        # self.masks['.H'] = deg > median
        # self.masks['.T'] = torch.logical_not(self.masks['.H'])

        ht_dict_deg = separator_ht(deg[self.mask()], head_ratio=0.5)
        self.masks['.H'] = torch.zeros(self.n_sample, dtype=torch.bool, device=self.device)
        self.masks['.H'][ht_dict_deg['H']] = True
        self.masks['.T'] = torch.logical_not(self.masks['.H'])

        self.masks['HH'] = self.masks['H'] & self.masks['.H']
        self.masks['HT'] = self.masks['H'] & self.masks['.T']
        self.masks['TH'] = self.masks['T'] & self.masks['.H']
        self.masks['TT'] = self.masks['T'] & self.masks['.T']

        self.stat(rows=['HH', 'HT', 'TH', 'TT'])

        self.cls_masks = dict()
        self.cls_masks['H'] = torch.zeros(self.n_cls, dtype=torch.bool, device=self.device)
        self.cls_masks['H'][ht_dict['H']] = True
        self.cls_masks['T'] = torch.zeros(self.n_cls, dtype=torch.bool, device=self.device)
        self.cls_masks['T'][ht_dict['T']] = True

        self.class_weight = 1 / self.num_list(set='train')

        self.args.nfeat = self.data.x.shape[1]
        self.args.nhid = 64
        self.args.nclass = self.n_cls
        self.args.class_weight = True
        self.args.criterion = 'mean'
        self.args.gamma = 1
        self.model = Model(args=self.args, baseline=self).to(self.device)

        self.optimizer_fe = optim.Adam(self.model.encoder.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay) # feature extractor
        self.optimizer_cls_og = optim.Adam(self.model.classifier_og.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        self.f1_early_stopper = EarlyStopper(patience=50, min_delta=0.03, more=True)


    def train(self):
        # ======================================= Embed Training ======================================= #

        self.f1_early_stopper.reset()

        for _ in tqdm.tqdm(range(self.args.epoch)):
            model = self.model
            optimizer_fe = self.optimizer_fe
            optimizer_cls_og = self.optimizer_cls_og

            model.train()
            optimizer_fe.zero_grad()
            optimizer_cls_og.zero_grad()

            loss = model(x=self.data.x, y=self.data.y, edge_index=self.data.edge_index, mask=self.mask(set='train'), phase="og", weight=self.class_weight)
            loss.backward(retain_graph=True)
            
            optimizer_fe.step()
            optimizer_cls_og.step()

            model.eval()

            output = model(x=self.data.x, y=self.data.y, edge_index=self.data.edge_index, logit=True, phase="og", weight=self.class_weight)

            self.test(output)

            if self.f1_early_stopper.early_stop(self.perf(output, set='val')['f1']):             
                break
        
        model.eval()
        self.predict_ht(model=model)

        # =============================================================================================== #

        classifier_dict = {}
        mask_dict = {'HH': self.mask('HH') & self.mask('train'), 'HT': self.mask('HT') & self.mask('train'),
                     'TH': self.mask('TH') & self.mask('train'), 'TT': self.mask('TT') & self.mask('train')}

        # ======================================= Expert Training ======================================= #

        for sep in ['HH', 'HT', 'TH', 'TT']:
            self.f1_early_stopper.reset()
            # if degree belongs to tail, finetune head degree classifier
            classifier = deepcopy(classifier_dict[sep[0] + 'H']) if sep[1] == 'T' else model.expert_dict[sep].to(self.device)
            optimizer = optim.Adam(classifier.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            
            for epoch in tqdm.tqdm(range(self.args.epoch)):
                classifier.train()
                optimizer.zero_grad()

                loss = model(x=self.data.x, y=self.data.y, edge_index=self.data.edge_index, mask=self.mask(sep) & self.mask('train'), phase='expert', 
                             classifier=classifier, sep=sep, cls_mask=self.cls_masks[sep[0]])
                
                loss.backward(retain_graph=True)
                optimizer.step()

                classifier.eval()
                output = model(x=self.data.x, y=self.data.y, edge_index=self.data.edge_index, logit=True, phase='expert', 
                               classifier=classifier, sep=sep, cls_mask=self.cls_masks[sep[0]])

                def save():
                    classifier_dict[sep] = deepcopy(classifier)
                
                if self.f1_early_stopper.early_stop(self.perf(output, mask=self.mask(sep) & self.mask('val'))['f1'], save):             
                    break

                if epoch == self.args.epoch - 1:
                    save()

        # ======================================= Student Training ======================================= #

        for sep in ['H', 'T']:
            self.f1_early_stopper.reset()

            classifier = model.expert_dict[sep].to(self.device)
            optimizer = optim.Adam(classifier.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            
            for epoch in tqdm.tqdm(range(self.args.epoch)):
                classifier.train()
                optimizer.zero_grad()

                kd_head, kd_tail, ce_loss = \
                    model(x=self.data.x, y=self.data.y, edge_index=self.data.edge_index, mask=mask_dict, phase='student', 
                          classifier=classifier, sep=sep, cls_mask=self.cls_masks[sep[0]], teacher=classifier_dict)
                alpha = scheduler(epoch, self.args.epoch)

                # Head-to-Tail Curriculum Learning
                loss = ce_loss + (alpha * kd_head + (1-alpha) * kd_tail)

                loss.backward(retain_graph=True)
                optimizer.step()

                classifier.eval()
                output = model(x=self.data.x, y=self.data.y, edge_index=self.data.edge_index, mask=mask_dict, logit=True, phase='student', 
                               classifier=classifier, sep=sep, cls_mask=self.cls_masks[sep[0]], teacher=classifier_dict)

                def save():
                    classifier_dict[sep] = deepcopy(classifier)

                if self.f1_early_stopper.early_stop(self.perf(output, mask=self.mask(sep), set='val')['f1'], save):             
                    break

                if epoch == self.args.epoch - 1:
                    save()

        # ======================================= Inference Phase =======================================

        output = torch.zeros((self.data.x.shape[0], self.n_cls), dtype=torch.float32, device=self.device)

        for sep in ['H', 'T']:
            classifier = classifier_dict[sep]

            classifier.eval()
            output_sep = model(x=self.data.x, y=self.data.y, edge_index=self.data.edge_index, mask=mask_dict, logit=True, phase='student', 
                               classifier=classifier, sep=sep, cls_mask=self.cls_masks[sep], teacher=classifier_dict)
            
            output[self.masks[sep + '_predicted']][:, self.cls_masks[sep]] = output_sep[self.masks[sep + '_predicted']][:, self.cls_masks[sep]]  # , self.sep[sep]

        self.test(output)

        return output
