import torch
import torch.nn as nn
import torch.nn.functional as F
from GradientReverseLayer import ReverseLayerF
import random
import copy

# The ABP module
class Attention(nn.Module):
    def __init__(self, cuda, input_dim):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        if cuda:
            self.w_linear = nn.Parameter(torch.randn(input_dim, input_dim).cuda())
            self.u_linear = nn.Parameter(torch.randn(input_dim).cuda())
        else:
            self.w_linear = nn.Parameter(torch.randn(input_dim, input_dim))
            self.u_linear = nn.Parameter(torch.randn(input_dim))

    def forward(self, x, batch_size, time_steps):
        x_reshape = torch.Tensor.reshape(x, [-1, self.input_dim])
        attn_softmax = F.softmax(torch.mm(x_reshape, self.w_linear)+ self.u_linear,1)
        res = torch.mul(attn_softmax, x_reshape)
        res = torch.Tensor.reshape(res, [batch_size, time_steps, self.input_dim])
        return res

class LSTM(nn.Module):
    def __init__(self, input_dim=310, output_dim=64, layers=2, location=-1):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, output_dim, num_layers=layers, batch_first=True)
        self.location = location
    def forward(self, x):
        # self.lstm.flatten_parameters()
        feature, (hn, cn) = self.lstm(x)
        return feature[:, self.location, :], hn, cn

class Encoder(nn.Module):
    def __init__(self, input_dim=310, hid_dim=64, n_layers=2):
        super(Encoder, self).__init__()
        self.theta = LSTM(input_dim, hid_dim, n_layers)
    def forward(self, x):
        x_h = self.theta(x)
        return x_h

class Decoder(nn.Module):
    def __init__(self, input_dim=310, hid_dim=64, n_layers=2,output_dim=310):
        super(Decoder, self).__init__()
        self.rnn = nn.LSTM(input_dim, hid_dim, n_layers)
        self.fc_out = nn.Linear(hid_dim, output_dim)
    def forward(self, input, hidden, cell, time_steps):
        out =[]
        out1 = self.fc_out(input)
        out.append(out1)
        out1= out1.unsqueeze(0)  # input = [batch size] to [1, batch size]
        for i in range(time_steps-1):
            output, (hidden, cell) = self.rnn(out1,
                                              (hidden, cell))  # output =[seq len, batch size, hid dim* ndirection]
            out_cur = self.fc_out(output.squeeze(0))  # prediction = [batch size, output dim]
            out.append(out_cur)
            out1 = out_cur.unsqueeze(0)
        out.reverse()
        out = torch.stack(out)
        out = out.transpose(1,0) #batch first
        return out, hidden, cell


#namely The Subject Classifier SD
class DomainClassifier(nn.Module):
    def __init__(self, input_dim =64, output_dim=14):
        super(DomainClassifier, self).__init__()
        self.classifier = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.classifier(x)
        return x

# The MSE loss
class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n
        return mse


def timeStepsShuffle(source_data):
    source_data_1 = source_data.clone()
    #retain the last time step
    curTimeStep_1 = source_data_1[:, -1, :]
    # get data of other time steps
    dim_size = source_data[:, :-1, :].size(1)
    # generate a random sequence
    idxs_1 = list(range(dim_size))
    # generate a shuffled sequence
    random.shuffle(idxs_1)
    # get data corresponding to the shuffled sequence
    else_1 = source_data_1[:, idxs_1, :]
    # add the origin last time step
    result_1 = torch.cat([else_1, curTimeStep_1.unsqueeze(1)], dim=1)
    return result_1

def maskTimeSteps(source_data, rate):
    source_data_1 = source_data.clone()
    num_zeros = int(source_data.size(1) * rate)
    #mask certain rate of time steps ignoring the last
    zero_indices_1 = torch.randperm(source_data_1.size(1)-1)[:num_zeros]
    source_data_1[:, zero_indices_1,:] = 0
    return source_data_1

def maskChannels(source_data, args, rate):
    # reshape for operating the channel dimension
    source_data_reshaped = source_data.reshape(args.batch_size, args.time_steps, 5, 62)
    source_data_reshaped_1 = source_data_reshaped.clone()
    num_zeros = int(source_data_reshaped.size(-1) * rate)
    # mask certain rate of channels
    zero_indices_1 = torch.randperm(source_data_reshaped_1.size(-1))[:num_zeros]
    source_data_reshaped_1[..., zero_indices_1] = 0
    source_data_reshaped_1 = source_data_reshaped_1.reshape(args.batch_size, args.time_steps, 310)
    return source_data_reshaped_1

def shuffleChannels(source_data, args):
    # reshape for operating the channel dimension
    source_data_reshaped = source_data.reshape(args.batch_size, args.time_steps, 5, 62)
    source_data_reshaped_1 = source_data_reshaped.clone()
    dim_size = source_data_reshaped[..., :].size(-1)
    # # generate a random sequence
    idxs_1 = list(range(dim_size))
    random.shuffle(idxs_1)
    # shuffle channels
    source_data_reshaped_1 = source_data_reshaped_1[..., idxs_1]
    result_1 = source_data_reshaped_1.reshape(args.batch_size, args.time_steps, 310)
    return result_1

# proposed DMMR model
class DMMRPreTrainingModel(nn.Module):
    def __init__(self, cuda, number_of_source=14, number_of_category=3, batch_size=10, time_steps=15):
        super(DMMRPreTrainingModel, self).__init__()
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.number_of_source = number_of_source
        self.attentionLayer = Attention(cuda, input_dim=310)
        self.sharedEncoder = Encoder(input_dim=310, hid_dim=64, n_layers=1)
        self.mse = MSE()
        self.domainClassifier = DomainClassifier(input_dim=64, output_dim=14)
        for i in range(number_of_source):
            exec('self.decoder' + str(i) + '=Decoder(input_dim=310, hid_dim=64, n_layers=1, output_dim=310)')
    def forward(self, x, corres, subject_id, m=0, mark=0):
        # Noise Injection, with the proposed method Time Steps Shuffling
        x = timeStepsShuffle(x)
        # The ABP module
        x = self.attentionLayer(x, x.shape[0], self.time_steps)
        # Encoder the weighted features with one-layer LSTM
        shared_last_out, shared_hn, shared_cn = self.sharedEncoder(x)
        # The DG_DANN module
        # The GRL layer in the first stage
        reverse_feature = ReverseLayerF.apply(shared_last_out, m)
        # The Subject Discriminator
        subject_predict = self.domainClassifier(reverse_feature)
        subject_predict = F.log_softmax(subject_predict,dim=1)
        # The domain adversarial loss
        sim_loss = F.nll_loss(subject_predict, subject_id)

        # Build Supervision for Decoders, the inputs are also weighted
        corres = self.attentionLayer(corres, corres.shape[0], self.time_steps)
        splitted_tensors = torch.chunk(corres, self.number_of_source, dim=0)
        rec_loss = 0
        mixSubjectFeature = 0
        for i in range(self.number_of_source):
            # Reconstruct features in the first stage
            x_out, *_ = eval('self.decoder' + str(i))(shared_last_out, shared_hn, shared_cn, self.time_steps)
            # The proposed mix method for data augmentation
            mixSubjectFeature += x_out
        shared_last_out_2, shared_hn_2, shared_cn_2 = self.sharedEncoder(mixSubjectFeature)
        for i in range(self.number_of_source):
            # Reconstruct features in the second stage
            x_out, *_ = eval('self.decoder' + str(i))(shared_last_out_2, shared_hn_2, shared_cn_2, self.time_steps)
            # Compute the reconstructive loss in the second stage only
            rec_loss += self.mse(x_out, splitted_tensors[i])
        return rec_loss, sim_loss
class DMMRFineTuningModel(nn.Module):
    def __init__(self, cuda, baseModel, number_of_source=14, number_of_category=3, batch_size=10, time_steps=15):
        super(DMMRFineTuningModel, self).__init__()
        self.baseModel = copy.deepcopy(baseModel)
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.number_of_source = number_of_source
        # The ABP module and sharedEncoder are from the pretrained model
        self.attentionLayer = self.baseModel.attentionLayer
        self.sharedEncoder = self.baseModel.sharedEncoder
        # Add a new emotion classifier for emotion recognition
        self.cls_fc = nn.Sequential(nn.Linear(64, 64, bias=False), nn.BatchNorm1d(64),
                               nn.ReLU(inplace=True), nn.Linear(64, number_of_category, bias=True))
        self.mse = MSE()
        for i in range(number_of_source):
            exec('self.decoder' + str(i) + '=Decoder(input_dim=310, hid_dim=64, n_layers=1, output_dim=310)')
    def forward(self, x, label_src=0):
        x = self.attentionLayer(x, x.shape[0], self.time_steps)
        shared_last_out, shared_hn, shared_cn = self.sharedEncoder(x)
        x_logits = self.cls_fc(shared_last_out)
        x_pred = F.log_softmax(x_logits, dim=1)
        cls_loss = F.nll_loss(x_pred, label_src.squeeze())
        return x_pred, x_logits, cls_loss

class DMMRTestModel(nn.Module):
    def __init__(self, baseModel):
        super(DMMRTestModel, self).__init__()
        self.baseModel = copy.deepcopy(baseModel)
    def forward(self, x):
        x = self.baseModel.attentionLayer(x, self.baseModel.batch_size, self.baseModel.time_steps)
        shared_last_out, shared_hn, shared_cn = self.baseModel.sharedEncoder(x)
        x_shared_logits = self.baseModel.cls_fc(shared_last_out)
        return x_shared_logits

############## other noisy injection methods ##############
class PreTrainingWithMaskTimeSteps(nn.Module):
    def __init__(self, cuda, number_of_source=14, number_of_category=3, batch_size=10, time_steps=15):
        super(PreTrainingWithMaskTimeSteps, self).__init__()
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.number_of_source = number_of_source
        self.attentionLayer = Attention(cuda, input_dim=310)
        self.sharedEncoder = Encoder(input_dim=310, hid_dim=64, n_layers=1)
        self.mse = MSE()
        self.domainClassifier = DomainClassifier(input_dim=64, output_dim=14)
        for i in range(number_of_source):
            exec('self.decoder' + str(i) + '=Decoder(input_dim=310, hid_dim=64, n_layers=1, output_dim=310)')
    def forward(self, x, corres, subject_id, args, m=0, mark=0):
        x = maskTimeSteps(x, 0.2)
        x = self.attentionLayer(x, x.shape[0], self.time_steps)
        shared_last_out, shared_hn, shared_cn = self.sharedEncoder(x)

        reverse_feature = ReverseLayerF.apply(shared_last_out, m)
        subject_predict = self.domainClassifier(reverse_feature)
        subject_predict = F.log_softmax(subject_predict,dim=1)
        sim_loss = F.nll_loss(subject_predict, subject_id)

        corres = self.attentionLayer(corres, corres.shape[0], self.time_steps)
        splitted_tensors = torch.chunk(corres, self.number_of_source, dim=0)
        rec_loss = 0
        mixSubjectFeature = 0
        for i in range(self.number_of_source):
            x_out, *_ = eval('self.decoder' + str(i))(shared_last_out, shared_hn, shared_cn, self.time_steps)
            mixSubjectFeature += x_out
        shared_last_out_2, shared_hn_2, shared_cn_2 = self.sharedEncoder(mixSubjectFeature)
        for i in range(self.number_of_source):
            x_out, *_ = eval('self.decoder' + str(i))(shared_last_out_2, shared_hn_2, shared_cn_2, self.time_steps)
            rec_loss += self.mse(x_out, splitted_tensors[i])
        return rec_loss, sim_loss

class PreTrainingWithMaskChannels(nn.Module):
    def __init__(self, cuda, number_of_source=14, number_of_category=3, batch_size=10, time_steps=15):
        super(PreTrainingWithMaskChannels, self).__init__()
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.number_of_source = number_of_source
        self.attentionLayer = Attention(cuda, input_dim=310)
        self.sharedEncoder = Encoder(input_dim=310, hid_dim=64, n_layers=1)
        self.mse = MSE()
        self.domainClassifier = DomainClassifier(input_dim=64, output_dim=14)
        for i in range(number_of_source):
            exec('self.decoder' + str(i) + '=Decoder(input_dim=310, hid_dim=64, n_layers=1, output_dim=310)')
    def forward(self, x, corres, subject_id, args, m=0, mark=0):
        x = maskChannels(x, args, 0.2)
        x = self.attentionLayer(x, x.shape[0], self.time_steps)
        shared_last_out, shared_hn, shared_cn = self.sharedEncoder(x)

        reverse_feature = ReverseLayerF.apply(shared_last_out, m)
        subject_predict = self.domainClassifier(reverse_feature)
        subject_predict = F.log_softmax(subject_predict,dim=1)
        sim_loss = F.nll_loss(subject_predict, subject_id)

        corres = self.attentionLayer(corres, corres.shape[0], self.time_steps)
        splitted_tensors = torch.chunk(corres, self.number_of_source, dim=0)
        rec_loss = 0
        mixSubjectFeature = 0
        for i in range(self.number_of_source):
            x_out, *_ = eval('self.decoder' + str(i))(shared_last_out, shared_hn, shared_cn, self.time_steps)
            mixSubjectFeature += x_out
        shared_last_out_2, shared_hn_2, shared_cn_2 = self.sharedEncoder(mixSubjectFeature)
        for i in range(self.number_of_source):
            x_out, *_ = eval('self.decoder' + str(i))(shared_last_out_2, shared_hn_2, shared_cn_2, self.time_steps)
            rec_loss += self.mse(x_out, splitted_tensors[i])
        return rec_loss, sim_loss

class PreTrainingWithChannelsShuffling(nn.Module):
    def __init__(self, cuda, number_of_source=14, number_of_category=3, batch_size=10, time_steps=15):
        super(PreTrainingWithChannelsShuffling, self).__init__()
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.number_of_source = number_of_source
        self.attentionLayer = Attention(cuda, input_dim=310)
        self.sharedEncoder = Encoder(input_dim=310, hid_dim=64, n_layers=1)
        self.mse = MSE()
        self.domainClassifier = DomainClassifier(input_dim=64, output_dim=14)
        for i in range(number_of_source):
            exec('self.decoder' + str(i) + '=Decoder(input_dim=310, hid_dim=64, n_layers=1, output_dim=310)')
    def forward(self, x, corres, subject_id, args, m=0, mark=0):
        x = shuffleChannels(x, args)
        x = self.attentionLayer(x, x.shape[0], self.time_steps)
        shared_last_out, shared_hn, shared_cn = self.sharedEncoder(x)

        reverse_feature = ReverseLayerF.apply(shared_last_out, m)
        subject_predict = self.domainClassifier(reverse_feature)
        subject_predict = F.log_softmax(subject_predict,dim=1)
        sim_loss = F.nll_loss(subject_predict, subject_id)

        corres = self.attentionLayer(corres, corres.shape[0], self.time_steps)
        splitted_tensors = torch.chunk(corres, self.number_of_source, dim=0)
        rec_loss = 0
        mixSubjectFeature = 0
        for i in range(self.number_of_source):
            x_out, *_ = eval('self.decoder' + str(i))(shared_last_out, shared_hn, shared_cn, self.time_steps)
            mixSubjectFeature += x_out
        shared_last_out_2, shared_hn_2, shared_cn_2 = self.sharedEncoder(mixSubjectFeature)
        for i in range(self.number_of_source):
            x_out, *_ = eval('self.decoder' + str(i))(shared_last_out_2, shared_hn_2, shared_cn_2, self.time_steps)
            rec_loss += self.mse(x_out, splitted_tensors[i])
        return rec_loss, sim_loss

class PreTrainingWithDropout(nn.Module):
    def __init__(self, cuda, number_of_source=14, number_of_category=3, batch_size=10, time_steps=15, dropout_rate=0.2):
        super(PreTrainingWithDropout, self).__init__()
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.number_of_source = number_of_source
        self.attentionLayer = Attention(cuda, input_dim=310)
        self.sharedEncoder = Encoder(input_dim=310, hid_dim=64, n_layers=1)
        self.mse = MSE()
        self.domainClassifier = DomainClassifier(input_dim=64, output_dim=14)
        self.dropout = nn.Dropout(dropout_rate)  # noise
        for i in range(number_of_source):
            exec('self.decoder' + str(i) + '=Decoder(input_dim=310, hid_dim=64, n_layers=1, output_dim=310)')
    def forward(self, x, corres, subject_id, args, m=0, mark=0):
        x = self.dropout(x)
        x = self.attentionLayer(x, x.shape[0], self.time_steps)
        shared_last_out, shared_hn, shared_cn = self.sharedEncoder(x)

        reverse_feature = ReverseLayerF.apply(shared_last_out, m)
        subject_predict = self.domainClassifier(reverse_feature)
        subject_predict = F.log_softmax(subject_predict,dim=1)
        sim_loss = F.nll_loss(subject_predict, subject_id)

        corres = self.attentionLayer(corres, corres.shape[0], self.time_steps)
        splitted_tensors = torch.chunk(corres, self.number_of_source, dim=0)
        rec_loss = 0
        mixSubjectFeature = 0
        for i in range(self.number_of_source):
            x_out, *_ = eval('self.decoder' + str(i))(shared_last_out, shared_hn, shared_cn, self.time_steps)
            mixSubjectFeature += x_out
        shared_last_out_2, shared_hn_2, shared_cn_2 = self.sharedEncoder(mixSubjectFeature)
        for i in range(self.number_of_source):
            x_out, *_ = eval('self.decoder' + str(i))(shared_last_out_2, shared_hn_2, shared_cn_2, self.time_steps)
            rec_loss += self.mse(x_out, splitted_tensors[i])
        return rec_loss, sim_loss


############## noiseInjectionMethods stydy ##############
#w/o mix
class PreTrainingWithoutMix(nn.Module):
    def __init__(self, cuda, number_of_source=14, number_of_category=3, batch_size=10, time_steps=15):
        super(PreTrainingWithoutMix, self).__init__()
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.number_of_source = number_of_source
        self.attentionLayer = Attention(cuda, input_dim=310)
        self.sharedEncoder = Encoder(input_dim=310, hid_dim=64, n_layers=1)
        self.mse = MSE()
        self.domainClassifier = DomainClassifier(input_dim=64, output_dim=14)
        for i in range(number_of_source):
            exec('self.decoder' + str(i) + '=Decoder(input_dim=310, hid_dim=64, n_layers=1, output_dim=310)')
    def forward(self, x, corres, subject_id, m=0, mark=0):
        x = timeStepsShuffle(x)
        x = self.attentionLayer(x, x.shape[0], self.time_steps)
        shared_last_out, shared_hn, shared_cn = self.sharedEncoder(x)

        reverse_feature = ReverseLayerF.apply(shared_last_out, m)
        subject_predict = self.domainClassifier(reverse_feature)
        subject_predict = F.log_softmax(subject_predict,dim=1)
        sim_loss = F.nll_loss(subject_predict, subject_id)

        corres = self.attentionLayer(corres, corres.shape[0], self.time_steps)
        splitted_tensors = torch.chunk(corres, self.number_of_source, dim=0)
        rec_loss = 0
        for i in range(self.number_of_source):
            x_out, *_ = eval('self.decoder' + str(i))(shared_last_out, shared_hn, shared_cn, self.time_steps)
            rec_loss += self.mse(x_out, splitted_tensors[i])
        return rec_loss, sim_loss
#w/o noise
class PreTrainingWithoutNoise(nn.Module):
    def __init__(self, cuda, number_of_source=14, number_of_category=3, batch_size=10, time_steps=15):
        super(PreTrainingWithoutNoise, self).__init__()
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.number_of_source = number_of_source
        self.attentionLayer = Attention(cuda, input_dim=310)
        self.sharedEncoder = Encoder(input_dim=310, hid_dim=64, n_layers=1)
        self.mse = MSE()
        self.domainClassifier = DomainClassifier(input_dim=64, output_dim=14)
        for i in range(number_of_source):
            exec('self.decoder' + str(i) + '=Decoder(input_dim=310, hid_dim=64, n_layers=1, output_dim=310)')
    def forward(self, x, corres, subject_id, m=0, mark=0):
        x = self.attentionLayer(x, x.shape[0], self.time_steps)
        shared_last_out, shared_hn, shared_cn = self.sharedEncoder(x)

        reverse_feature = ReverseLayerF.apply(shared_last_out, m)
        subject_predict = self.domainClassifier(reverse_feature)
        subject_predict = F.log_softmax(subject_predict,dim=1)
        sim_loss = F.nll_loss(subject_predict, subject_id)

        corres = self.attentionLayer(corres, corres.shape[0], self.time_steps)
        splitted_tensors = torch.chunk(corres, self.number_of_source, dim=0)
        rec_loss = 0
        mixSubjectFeature = 0
        for i in range(self.number_of_source):
            x_out, *_ = eval('self.decoder' + str(i))(shared_last_out, shared_hn, shared_cn, self.time_steps)
            mixSubjectFeature += x_out
        shared_last_out_2, shared_hn_2, shared_cn_2 = self.sharedEncoder(mixSubjectFeature)
        for i in range(self.number_of_source):
            x_out, *_ = eval('self.decoder' + str(i))(shared_last_out_2, shared_hn_2, shared_cn_2, self.time_steps)
            rec_loss += self.mse(x_out, splitted_tensors[i])
        return rec_loss, sim_loss
#w/o both
class PreTrainingWithoutBothMixAndNoise(nn.Module):
    def __init__(self, cuda, number_of_source=14, number_of_category=3, batch_size=10, time_steps=15):
        super(PreTrainingWithoutBothMixAndNoise, self).__init__()
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.number_of_source = number_of_source
        self.attentionLayer = Attention(cuda, input_dim=310)
        self.sharedEncoder = Encoder(input_dim=310, hid_dim=64, n_layers=1)
        self.mse = MSE()
        self.domainClassifier = DomainClassifier(input_dim=64, output_dim=14)
        for i in range(number_of_source):
            exec('self.decoder' + str(i) + '=Decoder(input_dim=310, hid_dim=64, n_layers=1, output_dim=310)')
    def forward(self, x, corres, subject_id, m=0, mark=0):
        x = self.attentionLayer(x, x.shape[0], self.time_steps)
        shared_last_out, shared_hn, shared_cn = self.sharedEncoder(x)

        reverse_feature = ReverseLayerF.apply(shared_last_out, m)
        subject_predict = self.domainClassifier(reverse_feature)
        subject_predict = F.log_softmax(subject_predict,dim=1)
        sim_loss = F.nll_loss(subject_predict, subject_id)

        corres = self.attentionLayer(corres, corres.shape[0], self.time_steps)
        splitted_tensors = torch.chunk(corres, self.number_of_source, dim=0)
        rec_loss = 0
        for i in range(self.number_of_source):
            x_out, *_ = eval('self.decoder' + str(i))(shared_last_out, shared_hn, shared_cn, self.time_steps)
            rec_loss += self.mse(x_out, splitted_tensors[i])
        return rec_loss, sim_loss

#return feature of shared feature for T_SNE plots
class ModelReturnFeatures(nn.Module):
    def __init__(self, baseModel, time_steps=15):
        super(ModelReturnFeatures, self).__init__()
        self.baseModel = baseModel
        self.time_steps = time_steps
    def forward(self, x):
        x = self.baseModel.attentionLayer(x, x.shape[0], self.time_steps)
        shared_last_out, shared_hn, shared_cn = self.baseModel.sharedEncoder(x)
        return x, shared_last_out


