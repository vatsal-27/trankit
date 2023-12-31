import torch.nn.functional as F
from .base_models import *
from trankit.layers.crf_layer import CRFLoss, viterbi_decode
from ..utils.base_utils import *
from ..utils.conll import *
from ..utils.custom_classifiers import *
from ..utils.spear import probability, log_likelihood_loss, precision_loss, predict_gm_labels,kl_divergence
instance_fields = [
    'sent_index',
    'words', 'word_num', 'word_ids', 
    'piece_idxs', 'attention_masks','word_span_idxs', 'word_lens',
    'edit_type_idxs', 'upos_type_idxs', 'xpos_type_idxs', 
    'head_idxs', 'deprel_idxs', 'word_mask'
]
H = len(instance_fields)
batch_fields = [
    'sent_index','word_ids',
    'words', 'word_num',  
    'piece_idxs', 'attention_masks', 'word_lens','word_span_idxs',
    'edit_type_idxs', 'upos_type_idxs', 'xpos_type_idxs',
    'upos_ids', 'xpos_ids',
    'head_idxs', 'deprel_idxs', 'word_mask'
]
B = len(batch_fields)

for i in CLASS_NAMES:  #new_idxs
    instance_fields += [i+"_type_idxs"]
    batch_fields += [i+"_type_idxs"]
for i in CLASS_NAMES:  #new_idxs
    batch_fields += [i+"_ids"  ]

def get_labels(df,v):
    L = list(df["label"])
    R =[]
    for i in L:
        try:
            R+=[v[i]]   
        except:
            R+=[v["k1"]]
    return R

def get_scores(df,device):
    return torch.tensor(list(df["prec"]),device=device).double()

class NERClassifier(nn.Module):
    def __init__(self, config, language):
        super().__init__()
        self.config = config
        self.xlmr_dim = 768 if config.embedding_name == 'xlm-roberta-base' else 1024
        self.entity_label_stoi = config.ner_vocabs[language]  # BIOES tags
        self.entity_label_itos = {i: s for s, i in self.entity_label_stoi.items()}
        self.entity_label_num = len(self.entity_label_stoi)

        self.entity_label_ffn = Linears([self.xlmr_dim, config.hidden_num,
                                         self.entity_label_num],
                                        dropout_prob=config.linear_dropout,
                                        bias=config.linear_bias,
                                        activation=config.linear_activation)

        self.crit = CRFLoss(self.entity_label_num)

        if not config.training:
            # load pretrained weights
            self.initialized_weights = self.state_dict()
            self.pretrained_ner_weights = torch.load(os.path.join(self.config._cache_dir, self.config.embedding_name, language,
                                                                  '{}.ner.mdl'.format(
                                                                      language)), map_location=self.config.device)[
                'adapters']

            for name, value in self.pretrained_ner_weights.items():
                if name in self.initialized_weights:
                    self.initialized_weights[name] = value
            self.load_state_dict(self.initialized_weights)
            print('Loading NER tagger for {}'.format(language))

    def forward(self, batch, word_reprs):
        batch_size, _, _ = word_reprs.size()

        logits = self.entity_label_ffn(word_reprs)
        loss, trans = self.crit(logits, batch.word_mask, batch.entity_label_idxs)
        return loss

    def predict(self, batch, word_reprs):
        batch_size, _, _ = word_reprs.size()

        logits = self.entity_label_ffn(word_reprs)
        _, trans = self.crit(logits, batch.word_mask, batch.entity_label_idxs)
        # decode
        trans = trans.data.cpu().numpy()
        scores = logits.data.cpu().numpy()
        bs = logits.size(0)
        tag_seqs = []
        for i in range(bs):
            tags, _ = viterbi_decode(scores[i, :batch.word_num[i]], trans)
            tags = [self.entity_label_itos[t] for t in tags]
            tag_seqs += [tags]
        return tag_seqs


class PosDepClassifier(nn.Module):
    def __init__(self, config, treebank_name):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.vocabs = config.vocabs[treebank_name]
        self.xlmr_dim = 768 if config.embedding_name == 'xlm-roberta-base' else 1024
        self.upos_embedding = nn.Embedding(
            num_embeddings=len(self.vocabs[UPOS]),
            embedding_dim=50
        )
        self.n_lfs = self.config.n_lffs
        self.qc_ = 0.85
        self.qt_ = get_scores(self.config.df,self.config.device)
        # pos tagging
        self.upos_ffn = nn.Linear(self.xlmr_dim, len(self.vocabs[UPOS]))
        self.xpos_ffn = nn.Linear(self.xlmr_dim + 50, len(self.vocabs[XPOS]))

        for i in range(NUM_CLASS):
            setattr(self, CLASS_NAMES[i]+"_ffn", nn.Linear(self.xlmr_dim, len(self.vocabs[CLASS_NAMES[i]])))

        self.down_dim = self.xlmr_dim // 4
        self.down_project = nn.Linear(self.xlmr_dim, self.down_dim)
        # dependency parsing
        self.unlabeled = Deep_Biaffine(self.down_dim, self.down_dim,
                                       self.down_dim, 1)
        self.deprel = Deep_Biaffine(self.down_dim, self.down_dim,
                                    self.down_dim, len(self.vocabs[DEPREL]))

        # loss function
        self.criteria = torch.nn.CrossEntropyLoss()
        if integrate_spear:
            self.pi = torch.nn.Parameter(torch.rand((len(self.vocabs[DEPREL]), self.n_lfs), device = self.config.device).double())
            (self.pi).requires_grad = True
            self.theta = torch.nn.Parameter(torch.rand((len(self.vocabs[DEPREL]), self.n_lfs), device = self.config.device).double())
            (self.theta).requires_grad = True
            self.k = torch.tensor(get_labels(self.config.df,self.vocabs[DEPREL]),device=self.config.device)
        # print("k=",self.k)
        
            self.continuous_mask  = torch.tensor([0]*len(self.config.df),device=self.config.device)
        # print([i for i,j in self.named_parameters()])
        if not config.training:
            # load pretrained weights
            self.initialized_weights = self.state_dict()
            language = treebank2lang[treebank_name]
            self.pretrained_tagger_weights = torch.load(os.path.join(self.config._cache_dir, self.config.embedding_name, language,
                                                                     '{}.tagger.mdl'.format(
                                                                         language)), map_location=self.config.device)[
                'adapters']
            for name, value in self.pretrained_tagger_weights.items():
                if name in self.initialized_weights:
                    self.initialized_weights[name] = value
            self.load_state_dict(self.initialized_weights)
            print('Loading tagger for {}'.format(language))

    def forward(self, batch, word_reprs, cls_reprs):

        upos_scores = self.upos_ffn(word_reprs)
        upos_scores = upos_scores.view(-1, len(self.vocabs[UPOS]))

        # xpos
        xpos_reprs = torch.cat(
            [word_reprs, self.upos_embedding(batch.upos_ids)], dim=2
        )
        xpos_scores = self.xpos_ffn(xpos_reprs)
        xpos_scores = xpos_scores.view(-1, len(self.vocabs[XPOS]))
        
        
        loss = self.criteria(upos_scores, batch.upos_type_idxs) + \
               self.criteria(xpos_scores, batch.xpos_type_idxs) 
        loss = self.criteria(upos_scores, batch.upos_type_idxs) + \
               self.criteria(xpos_scores, batch.xpos_type_idxs) 
        if(ignore_upos_xpos):
            loss = 0
        for i in range(NUM_CLASS):
            x = getattr(self,CLASS_NAMES[i]+"_ffn").to(self.device)(word_reprs)
            loss += self.criteria(x.view(-1,len(self.vocabs[CLASS_NAMES[i]])),batch[16+i])
        if integrate_spear:
            s = batch.label_fns_score
            l = batch.label_fns_tau

            loss1 = log_likelihood_loss(
                self.theta, 
                self.pi, 
                l, 
                s,
                self.k, 
                len(self.vocabs[DEPREL]), 
                self.continuous_mask, 
                self.qc_, 
                self.config.device
                )

            loss2 = precision_loss(self.theta, self.k, len(self.vocabs[DEPREL]), self.qt_, self.config.device)
        
        # head
        dep_reprs = torch.cat(
            [cls_reprs, word_reprs], dim=1
        )  # [batch size, 1 + max num words, xlmr dim] # cls serves as ROOT node
        dep_reprs = self.down_project(dep_reprs)
        unlabeled_scores = self.unlabeled(dep_reprs, dep_reprs).squeeze(3)

        diag = torch.eye(batch.head_idxs.size(-1) + 1, dtype=torch.bool).to(self.device).unsqueeze(0)
        unlabeled_scores.masked_fill_(diag, -float('inf'))

        unlabeled_scores = unlabeled_scores[:, 1:, :]  
        unlabeled_scores = unlabeled_scores.masked_fill(batch.word_mask.unsqueeze(1), -float('inf'))
        unlabeled_target = batch.head_idxs.masked_fill(batch.word_mask[:, 1:], -100)
        loss += self.criteria(unlabeled_scores.contiguous().view(-1, unlabeled_scores.size(2)),
                              unlabeled_target.view(-1))
        # deprel
        deprel_scores = self.deprel(dep_reprs, dep_reprs)
        # print(deprel_scores.shape)
        deprel_scores = deprel_scores[:, 1:]  
        
        deprel_scores = torch.gather(deprel_scores, 2,
                                     batch.head_idxs.unsqueeze(2).unsqueeze(3).expand(-1, -1, -1, len(
                                         self.vocabs[DEPREL]))).view(
            -1, len(self.vocabs[DEPREL]))
        # # print(deprel_scores.shape)
        # print("dperel scores = ",deprel_scores)
        # print("deprel scores are between ",torch.min(deprel_scores),torch.max(deprel_scores))
        if integrate_spear:
            deprel_gm_scores = probability(self.theta, self.pi, batch.label_fns_tau, batch.label_fns_score, self.k, len(self.vocabs[DEPREL]), self.continuous_mask, self.qc_, self.config.device)
        deprel_fm_scores = torch.nn.Softmax(dim = 1)(deprel_scores)
        # print("gm scores are between ",torch.min(deprel_gm_scores),torch.max(deprel_gm_scores))
        # print("fm scores are between ",torch.min(deprel_fm_scores),torch.max(deprel_fm_scores))
        if integrate_spear:
            loss3 = kl_divergence(deprel_gm_scores+1e-18,deprel_fm_scores+1e-18)
        deprel_target = batch.deprel_idxs.masked_fill(batch.word_mask[:, 1:], -100)
        loss += self.criteria(deprel_scores.contiguous(), deprel_target.view(-1))
        # print("loss loop ended")
        # print(loss,loss1)4
        
        if not integrate_spear:
            loss4 = loss
        else: 
            loss4 = loss + loss1 + loss2 + loss3
        if torch.isnan(loss4):
            loss4 = loss
        # print("full loss", loss4.item())
        # print('loss 1 is ', loss1.item())
        # print('loss 2 is ', loss2.item())
        # print('loss 3 is ', loss3.item())
        # print('trankit loss is ', loss.item())
        # if loss == nan:
        #     return 0
        return loss4

    def forward_without_grad(self, batch, word_reprs, cls_reprs):
        # upos
        with torch.no_grad():
            upos_scores = self.upos_ffn(word_reprs)
            upos_scores = upos_scores.view(-1, len(self.vocabs[UPOS]))

            # xpos
            xpos_reprs = torch.cat(
                [word_reprs, self.upos_embedding(batch.upos_ids)], dim=2
            )
            xpos_scores = self.xpos_ffn(xpos_reprs)
            xpos_scores = xpos_scores.view(-1, len(self.vocabs[XPOS]))
            
            
            loss = self.criteria(upos_scores, batch.upos_type_idxs) + \
                self.criteria(xpos_scores, batch.xpos_type_idxs) 
            loss = self.criteria(upos_scores, batch.upos_type_idxs) + \
                self.criteria(xpos_scores, batch.xpos_type_idxs) 
            if(ignore_upos_xpos):
                loss = 0
            for i in range(NUM_CLASS):
                print(i)
                x = getattr(self,CLASS_NAMES[i]+"_ffn").to(self.device)(word_reprs)
                try:
                    loss += self.criteria(x.view(-1,len(self.vocabs[CLASS_NAMES[i]])),batch[16+i])
                except:
                    print(x)
                    print(batch)

            s = batch.label_fns_score
            l = batch.label_fns_tau

            loss1 = log_likelihood_loss(
                self.theta, 
                self.pi, 
                l, 
                s,
                self.k, 
                len(self.vocabs[DEPREL]), 
                self.continuous_mask, 
                self.qc_, 
                self.config.device
                )

            loss2 = precision_loss(self.theta, self.k, len(self.vocabs[DEPREL]), self.qt_, self.config.device)
            
            # head
            dep_reprs = torch.cat(
                [cls_reprs, word_reprs], dim=1
            )  # [batch size, 1 + max num words, xlmr dim] # cls serves as ROOT node
            dep_reprs = self.down_project(dep_reprs)
            unlabeled_scores = self.unlabeled(dep_reprs, dep_reprs).squeeze(3)

            diag = torch.eye(batch.head_idxs.size(-1) + 1, dtype=torch.bool).to(self.device).unsqueeze(0)
            unlabeled_scores.masked_fill_(diag, -float('inf'))

            unlabeled_scores = unlabeled_scores[:, 1:, :]  
            unlabeled_scores = unlabeled_scores.masked_fill(batch.word_mask.unsqueeze(1), -float('inf'))
            unlabeled_target = batch.head_idxs.masked_fill(batch.word_mask[:, 1:], -100)
            loss += self.criteria(unlabeled_scores.contiguous().view(-1, unlabeled_scores.size(2)),
                                unlabeled_target.view(-1))
            # deprel
            deprel_scores = self.deprel(dep_reprs, dep_reprs)
            print(deprel_scores.shape)
            deprel_scores = deprel_scores[:, 1:]  
            deprel_scores = torch.gather(deprel_scores, 2,
                                        batch.head_idxs.unsqueeze(2).unsqueeze(3).expand(-1, -1, -1, len(
                                            self.vocabs[DEPREL]))).view(
                -1, len(self.vocabs[DEPREL]))
            print(deprel_scores.shape)
            deprel_gm_scores = probability(self.theta, self.pi, batch.label_fns_tau, batch.label_fns_score, self.k, len(self.vocabs[DEPREL]), self.continuous_mask, self.qc_, self.config.device)
            deprel_fm_scores = torch.nn.Softmax(dim = 1)(deprel_scores)
            print("gm scores are between ",torch.min(deprel_gm_scores),torch.max(deprel_gm_scores))
            print("fm scores are between ",torch.min(deprel_fm_scores),torch.max(deprel_fm_scores))
            loss3 = kl_divergence(deprel_fm_scores, deprel_gm_scores)
            deprel_target = batch.deprel_idxs.masked_fill(batch.word_mask[:, 1:], -100)
            loss += self.criteria(deprel_scores.contiguous(), deprel_target.view(-1))
            # print("loss loop ended")
            print(loss,loss1,loss3,loss2)
            print("losses")
            print("theta")
            print(torch.max(self.theta))
            print("pi")
            print(torch.max(self.pi))
            return loss+loss1+loss3+loss2

    def predict(self, batch, word_reprs, cls_reprs):

        # upos
        upos_scores = self.upos_ffn(word_reprs)
        predicted_upos = torch.argmax(upos_scores, dim=2)
        # edits
        xpos_reprs = torch.cat(
            [word_reprs, self.upos_embedding(predicted_upos)], dim=2
        )  # [batch size, num words, xlmr dim + 50]
        # xpos
        xpos_scores = self.xpos_ffn(xpos_reprs)
        predicted_xpos = torch.argmax(xpos_scores, dim=2)
        P = []
        for i in range(NUM_CLASS):
            x = getattr(self,CLASS_NAMES[i]+"_ffn").to(self.device)(word_reprs)
            
            P += [torch.argmax(x,dim=2)]
        

        # head
        dep_reprs = torch.cat(
            [cls_reprs, word_reprs], dim=1
        )  # [batch size, 1 + max num words, xlmr dim] # cls serves as ROOT node
        dep_reprs = self.down_project(dep_reprs)
        unlabeled_scores = self.unlabeled(dep_reprs, dep_reprs).squeeze(3)

        diag = torch.eye(batch.head_idxs.size(-1) + 1, dtype=torch.bool).unsqueeze(0).to(self.config.device)
        unlabeled_scores.masked_fill_(diag, -float('inf'))

        # deprel
        deprel_scores = self.deprel(dep_reprs, dep_reprs)
        # print(batch.label_fns_score.shape,batch.label_fns_tau.shape,batch.)
        dep_preds = []
        dep_preds.append(F.log_softmax(unlabeled_scores, 2).detach().cpu().numpy())
        
        # y = predict_gm_labels(self.theta, self.pi, batch.label_fns_tau, batch.label_fns_score, self.k, len(self.vocabs[DEPREL]), self.continuous_mask, self.qc_, self.config.device)
        # y = y.reshape(len(word_reprs),len(word_reprs[0]))
        # print(y)
        # print(batch.label_fns_tau,batch.label_fns_score)
        # print(max(batch.label_fns_tau))
        # print(y.shape)
        # print(P[0].shape)
        # print(predicted_upos.shape)
        # dep_preds.append(y)
        # print(dep_preds[0].shape)
        # print(dep_preds[1].shape)
        # y2 = deprel_scores.max(3)[1].detach().cpu().numpy()
        # print(y2.shape)
        #dep_preds.append(predict_gm_labels(self.theta, self.pi, batch.label_fns_tau, batch.label_fns_score, self.k, len(self.vocabs[DEPREL]), self.continuous_mask, self.qc_, self.config.device))
        dep_preds.append(deprel_scores.max(3)[1].detach().cpu().numpy())
        return [predicted_upos, predicted_xpos ] + P  +[dep_preds]




class TokenizerClassifier(nn.Module):
    def __init__(self, config, treebank_name):
        super().__init__()
        self.config = config
        self.xlmr_dim = 768 if config.embedding_name == 'xlm-roberta-base' else 1024
        self.tokenizer_ffn = nn.Linear(self.xlmr_dim, 5)

        # loss function
        self.criteria = torch.nn.CrossEntropyLoss()

        if not config.training:
            language = treebank2lang[treebank_name]
            # load pretrained weights
            self.pretrained_tokenizer_weights = torch.load(os.path.join(self.config._cache_dir, self.config.embedding_name, language,
                                                                        '{}.tokenizer.mdl'.format(
                                                                            language)),
                                                           map_location=self.config.device)[
                'adapters']
            self.initialized_weights = self.state_dict()

            for name, value in self.pretrained_tokenizer_weights.items():
                if name in self.initialized_weights:
                    self.initialized_weights[name] = value

            self.load_state_dict(self.initialized_weights)
            print('Loading tokenizer for {}'.format(language))

    def forward(self, wordpiece_reprs, batch):
        wordpiece_scores = self.tokenizer_ffn(wordpiece_reprs)
        wordpiece_scores = wordpiece_scores.view(-1, 5)
        token_type_idxs = batch.token_type_idxs.view(-1)
        loss = self.criteria(wordpiece_scores, token_type_idxs)

        return loss

    def predict(self, batch, wordpiece_reprs):
        wordpiece_scores = self.tokenizer_ffn(wordpiece_reprs)
        predicted_wordpiece_labels = torch.argmax(wordpiece_scores, dim=2)  # [batch size, num wordpieces]

        return predicted_wordpiece_labels, batch.wordpiece_ends, batch.paragraph_index
