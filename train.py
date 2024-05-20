import os
import time
from model import *
import numpy as np
from test import *
from collections import defaultdict
import random
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def trainDMMR(data_loader_dict, optimizer_config, cuda, args, iteration, writer, one_subject):
    # data of source subjects, which is used as the training set
    source_loader = data_loader_dict['source_loader']
    # The pre-training phase
    preTrainModel = DMMRPreTrainingModel(cuda, number_of_source=len(source_loader), number_of_category=args.cls_classes, batch_size=args.batch_size, time_steps=args.time_steps)
    if cuda:
        preTrainModel = preTrainModel.cuda()
    source_iters = []
    for i in range(len(source_loader)):
        source_iters.append(iter(source_loader[i]))
    optimizer_PreTraining = torch.optim.Adam(preTrainModel.parameters(), **optimizer_config)

    acc_final = 0
    for epoch in range(args.epoch_preTraining):
        print("epoch: "+str(epoch))
        start_time_pretrain = time.time()
        preTrainModel.train()
        data_set_all = 0
        for i in range(1, iteration + 1):
            p = float(i + epoch * iteration) / args.epoch_preTraining / iteration
            m = 2. / (1. + np.exp(-10 * p)) - 1 # for the gradient reverse layer (GRL)
            batch_dict = defaultdict(list) #Pre-fetch a batch of data for each subject in advance and store them in this dictionary.
            data_dict = defaultdict(list) #Store the data of each subject in the current batch
            label_dict = defaultdict(list) #Store the labels corresponding to the data of each subject in the current batch
            label_data_dict = defaultdict(set)
            for j in range(len(source_iters)):
                try:
                    batch_dict[j] = next(source_iters[j])
                except:
                    source_iters[j] = iter(source_loader[j])
                    batch_dict[j]= next(source_iters[j])
                index = 0
                for o in batch_dict[j][1]:
                    cur_label = o[0].item()
                    data_dict[j].append(batch_dict[j][0][index])
                    label_dict[j].append(cur_label)
                    index+=1

            for j in range(len(source_iters)):
                # Assign a unique ID to each source subject
                subject_id = torch.ones(args.batch_size)
                subject_id = subject_id * j
                subject_id = subject_id.long()
                #the input of the model
                source_data, source_label = batch_dict[j]
                # Prepare corresponding new batch of each subject, the new batch has same label with current batch.
                label_data_dict_list = []
                for one_index in range(args.source_subjects):
                    cur_data_list = data_dict[one_index]
                    cur_label_list = label_dict[one_index]
                    for one in range(args.batch_size):
                        label_data_dict[cur_label_list[one]].add(cur_data_list[one])
                    label_data_dict_list.append(label_data_dict)
                    label_data_dict = defaultdict(set)
                # Store the corresponding new batch of each subject, providing the supervision for different decoders
                corres_batch_data = []
                for i in range(len(label_data_dict_list)):
                    for one in source_label:
                        label_cur = one[0].item()
                        corres_batch_data.append(random.choice(list(label_data_dict_list[i][label_cur])))
                corres_batch_data = torch.stack(corres_batch_data)
                if cuda:
                    source_data = source_data.cuda()
                    source_label = source_label.cuda()
                    subject_id = subject_id.cuda()
                    corres_batch_data = corres_batch_data.cuda()
                data_set_all += len(source_label)
                optimizer_PreTraining.zero_grad()
                # Call the pretraining model
                rec_loss, sim_loss = preTrainModel(source_data, corres_batch_data, subject_id, m, mark=j)
                # The loss of the pre-training phase, beta is the balancing hyperparameter
                loss_pretrain = rec_loss + args.beta * sim_loss
                loss_pretrain.backward()
                optimizer_PreTraining.step()
        print("data set amount: "+str(data_set_all))
        writer.add_scalars('subject: '+str(one_subject+1)+' '+'train DMMR/loss',
                           {'loss_pretrain':loss_pretrain.data,'rec_loss':rec_loss.data,'sim_loss':sim_loss.data}, epoch + 1)
        end_time_pretrain = time.time()
        pretrain_epoch_time = end_time_pretrain - start_time_pretrain
        print("The time required for one pre-training epoch is：", pretrain_epoch_time, "second")
        print("rec_loss: "+str(rec_loss))

    # The fine-tuning phase
    source_iters2 = []
    for i in range(len(source_loader)):
        source_iters2.append(iter(source_loader[i]))
    #Load the ABP module, the encoder from pretrained model and build a new model for the fine-tuning phase
    fineTuneModel = DMMRFineTuningModel(cuda, preTrainModel, number_of_source=len(source_loader),
                                    number_of_category=args.cls_classes, batch_size=args.batch_size,
                                    time_steps=args.time_steps)

    optimizer_FineTuning = torch.optim.Adam(fineTuneModel.parameters(), **optimizer_config)
    if cuda:
        fineTuneModel = fineTuneModel.cuda()
    for epoch in range(args.epoch_fineTuning):
        print("epoch: " + str(epoch))
        start_time = time.time()
        fineTuneModel.train()
        count = 0
        data_set_all = 0  
        for i in range(1, iteration + 1):
            batch_dict = defaultdict(list)
            data_dict = defaultdict(list)
            label_dict = defaultdict(list)

            for j in range(len(source_iters2)):
                try:
                    batch_dict[j] = next(source_iters2[j])
                except:
                    source_iters2[j] = iter(source_loader[j])
                    batch_dict[j] = next(source_iters2[j])
                index = 0
                for o in batch_dict[j][1]:
                    cur_label = o[0].item()
                    data_dict[j].append(batch_dict[j][0][index])
                    label_dict[j].append(cur_label)
                    index += 1
            for j in range(len(source_iters)):
                source_data, source_label = batch_dict[j]
                if cuda:
                    source_data = source_data.cuda()
                    source_label = source_label.cuda()
                data_set_all += len(source_label)
                optimizer_FineTuning.zero_grad()
                # Call the fine-tuning model
                x_pred, x_logits, cls_loss = fineTuneModel(source_data, source_label)
                cls_loss.backward()
                optimizer_FineTuning.step()
                _, pred = torch.max(x_pred, dim=1)
                count += pred.eq(source_label.squeeze().data.view_as(pred)).sum()
        end_time = time.time()
        epoch_time = end_time - start_time
        print("The time required for one fine-tuning epoch is：", epoch_time, "second")
        print("data set amount: " + str(data_set_all))
        acc = float(count) / data_set_all
        writer.add_scalars('subject: ' + str(one_subject + 1) + ' ' + 'train DMMR/loss',
                           {'cls_loss': cls_loss.data}, epoch + 1)
        writer.add_scalar('subject: ' + str(one_subject + 1) + ' ' + 'train DMMR/train accuracy', acc, epoch + 1)
        print("acc: " + str(acc))
        # test the fine-tuned model with the data of unseen target subject
        testModel = DMMRTestModel(fineTuneModel)
        acc_DMMR = testDMMR(data_loader_dict["test_loader"], testModel, cuda, args.batch_size)
        print("acc_DMMR: " + str(acc_DMMR))
        writer.add_scalars('subject: ' + str(one_subject + 1) + ' ' + 'test DMMR/test acc',
                           {'test acc': acc_DMMR}, epoch + 1)
        if acc_DMMR > acc_final:
            acc_final = acc_DMMR
            best_pretrain_model = copy.deepcopy(preTrainModel.state_dict())
            best_tune_model = copy.deepcopy(fineTuneModel.state_dict())
            best_test_model = copy.deepcopy(testModel.state_dict())
    modelDir = "model/" + args.way + "/" + args.index + "/"
    try:
        os.makedirs(modelDir)
    except:
        pass
    # save models
    torch.save(best_pretrain_model, modelDir + str(one_subject) + '_pretrain_model.pth')
    torch.save(best_tune_model, modelDir + str(one_subject) + '_tune_model.pth')
    torch.save(best_test_model, modelDir + str(one_subject) + '_test_model.pth')
    return acc_final

############## Ablation studies ##############
# w/o mix
def trainDMMR_WithoutMix(data_loader_dict, optimizer_config, cuda, args, iteration, writer, one_subject):
    source_loader = data_loader_dict['source_loader']
    # The pre-training phase
    preTrainModel = PreTrainingWithoutMix(cuda, number_of_source=len(source_loader), number_of_category=args.cls_classes, batch_size=args.batch_size, time_steps=args.time_steps)
    if cuda:
        preTrainModel = preTrainModel.cuda()
    source_iters = []
    for i in range(len(source_loader)):
        source_iters.append(iter(source_loader[i]))
    optimizer_PreTraining = torch.optim.Adam(preTrainModel.parameters(), **optimizer_config)

    acc_final = 0
    for epoch in range(args.epoch_preTraining):
        print("epoch: "+str(epoch))
        start_time_pretrain = time.time()
        preTrainModel.train()
        count = 0
        data_set_all = 0
        for i in range(1, iteration + 1):
            p = float(i + epoch * iteration) / args.epoch_preTraining / iteration
            m = 2. / (1. + np.exp(-10 * p)) - 1
            batch_dict = defaultdict(list)
            data_dict = defaultdict(list)
            label_dict = defaultdict(list)
            label_data_dict = defaultdict(set)

            for j in range(len(source_iters)):
                try:
                    batch_dict[j] = next(source_iters[j])
                except:
                    source_iters[j] = iter(source_loader[j])
                    batch_dict[j]= next(source_iters[j])
                index = 0
                for o in batch_dict[j][1]:
                    cur_label = o[0].item()
                    data_dict[j].append(batch_dict[j][0][index])
                    label_dict[j].append(cur_label)
                    index+=1

            for j in range(len(source_iters)):
                subject_id = torch.ones(args.batch_size)
                subject_id = subject_id * j
                subject_id = subject_id.long()

                source_data, source_label = batch_dict[j]
                # prepare corresponding new batch, the new batch has same label with current batch
                label_data_dict_list = []
                for one_index in range(args.source_subjects):
                    cur_data_list = data_dict[one_index]
                    cur_label_list = label_dict[one_index]
                    for one in range(args.batch_size):
                        label_data_dict[cur_label_list[one]].add(cur_data_list[one])
                    label_data_dict_list.append(label_data_dict)
                    label_data_dict = defaultdict(set)
                corres_batch_data = []
                for i in range(len(label_data_dict_list)):
                    for one in source_label:
                        label_cur = one[0].item()
                        corres_batch_data.append(random.choice(list(label_data_dict_list[i][label_cur])))
                corres_batch_data = torch.stack(corres_batch_data)
                if cuda:
                    source_data = source_data.cuda()
                    source_label = source_label.cuda()
                    subject_id = subject_id.cuda()
                    corres_batch_data =corres_batch_data.cuda()
                data_set_all+=len(source_label)
                optimizer_PreTraining.zero_grad()
                rec_loss, sim_loss = preTrainModel(source_data, corres_batch_data, subject_id, m, mark=j)
                loss_pretrain = rec_loss + args.beta*sim_loss
                loss_pretrain.backward()
                optimizer_PreTraining.step()
        print("data set amount: "+str(data_set_all))
        writer.add_scalars('subject: '+str(one_subject+1)+' '+'train DMMR/loss',
                           {'loss_pretrain':loss_pretrain.data,'rec_loss':rec_loss.data,'sim_loss':sim_loss.data}, epoch + 1)
        end_time_pretrain = time.time()
        pretrain_epoch_time = end_time_pretrain - start_time_pretrain
        print("The time required for one pre-training epoch is：", pretrain_epoch_time, "second")
        print("rec_loss: "+str(rec_loss))

    # The fine-tuning phase
    source_iters2 = []
    for i in range(len(source_loader)):
        source_iters2.append(iter(source_loader[i]))
    fineTuneModel = DMMRFineTuningModel(cuda, preTrainModel, number_of_source=len(source_loader),
                                    number_of_category=args.cls_classes, batch_size=args.batch_size,
                                    time_steps=args.time_steps)

    optimizer_FineTuning = torch.optim.Adam(fineTuneModel.parameters(), **optimizer_config)
    if cuda:
        fineTuneModel = fineTuneModel.cuda()
    for epoch in range(args.epoch_fineTuning):
        print("epoch: " + str(epoch))
        start_time = time.time()
        fineTuneModel.train()
        count = 0
        data_set_all = 0  
        for i in range(1, iteration + 1):
            p = float(i + epoch * iteration) / args.epoch_preTraining / iteration
            m = 2. / (1. + np.exp(-10 * p)) - 1
            batch_dict = defaultdict(list)
            data_dict = defaultdict(list)
            label_dict = defaultdict(list)
            label_data_dict = defaultdict(set)

            for j in range(len(source_iters2)):
                try:
                    batch_dict[j] = next(source_iters2[j])
                except:
                    source_iters2[j] = iter(source_loader[j])
                    batch_dict[j] = next(source_iters2[j])
                index = 0
                for o in batch_dict[j][1]:
                    cur_label = o[0].item()
                    data_dict[j].append(batch_dict[j][0][index])
                    label_dict[j].append(cur_label)
                    index += 1
            for j in range(len(source_iters)):
                source_data, source_label = batch_dict[j]
                if cuda:
                    source_data = source_data.cuda()
                    source_label = source_label.cuda()
                data_set_all += len(source_label)
                optimizer_FineTuning.zero_grad()
                x_pred, x_logits, cls_loss = fineTuneModel(source_data, source_label)
                cls_loss.backward()
                optimizer_FineTuning.step()
                _, pred = torch.max(x_pred, dim=1)
                count += pred.eq(source_label.squeeze().data.view_as(pred)).sum()
        end_time = time.time()
        epoch_time = end_time - start_time
        print("The time required for one fine-tuning epoch is：", epoch_time, "second")
        print("data set amount: " + str(data_set_all))
        acc = float(count) / data_set_all
        writer.add_scalars('subject: ' + str(one_subject + 1) + ' ' + 'train DMMR/loss',
                           {'cls_loss': cls_loss.data}, epoch + 1)
        writer.add_scalar('subject: ' + str(one_subject + 1) + ' ' + 'train DMMR/train accuracy', acc, epoch + 1)
        print("acc: " + str(acc))

        testModel = DMMRTestModel(fineTuneModel)
        acc_DMMR = testDMMR(data_loader_dict["test_loader"], testModel, cuda, args.batch_size)
        print("acc_DMMR: " + str(acc_DMMR))
        writer.add_scalars('subject: ' + str(one_subject + 1) + ' ' + 'test DMMR/test acc',
                           {'test acc': acc_DMMR}, epoch + 1)
        if acc_DMMR > acc_final:
            acc_final = acc_DMMR
            best_pretrain_model = copy.deepcopy(preTrainModel.state_dict())
            best_tune_model = copy.deepcopy(fineTuneModel.state_dict())
            best_test_model = copy.deepcopy(testModel.state_dict())
    modelDir = "model/" + args.way + "/" + args.index + "/"
    try:
        os.makedirs(modelDir)
    except:
        pass
    torch.save(best_pretrain_model, modelDir + str(one_subject) + '_pretrain_model.pth')
    torch.save(best_tune_model, modelDir + str(one_subject) + '_tune_model.pth')
    torch.save(best_test_model, modelDir + str(one_subject) + '_test_model.pth')
    return acc_final
# w/o noise
def trainDMMR_WithoutNoise(data_loader_dict, optimizer_config, cuda, args, iteration, writer, one_subject):
    source_loader = data_loader_dict['source_loader']
    # The pre-training phase
    preTrainModel = PreTrainingWithoutNoise(cuda, number_of_source=len(source_loader), number_of_category=args.cls_classes, batch_size=args.batch_size, time_steps=args.time_steps)
    if cuda:
        preTrainModel = preTrainModel.cuda()
    source_iters = []
    for i in range(len(source_loader)):
        source_iters.append(iter(source_loader[i]))
    optimizer_PreTraining = torch.optim.Adam(preTrainModel.parameters(), **optimizer_config)

    acc_final = 0
    for epoch in range(args.epoch_preTraining):
        print("epoch: "+str(epoch))
        start_time_pretrain = time.time()
        preTrainModel.train()
        count = 0
        data_set_all = 0
        for i in range(1, iteration + 1):
            p = float(i + epoch * iteration) / args.epoch_preTraining / iteration
            m = 2. / (1. + np.exp(-10 * p)) - 1
            batch_dict = defaultdict(list)
            data_dict = defaultdict(list)
            label_dict = defaultdict(list)
            label_data_dict = defaultdict(set)

            for j in range(len(source_iters)):
                try:
                    batch_dict[j] = next(source_iters[j])
                except:
                    source_iters[j] = iter(source_loader[j])
                    batch_dict[j]= next(source_iters[j])
                index = 0
                for o in batch_dict[j][1]:
                    cur_label = o[0].item()
                    data_dict[j].append(batch_dict[j][0][index])
                    label_dict[j].append(cur_label)
                    index+=1

            for j in range(len(source_iters)):
                subject_id = torch.ones(args.batch_size)
                subject_id = subject_id * j
                subject_id = subject_id.long()

                source_data, source_label = batch_dict[j]
                # prepare corresponding new batch, the new batch has same label with current batch
                label_data_dict_list = []
                for one_index in range(args.source_subjects):
                    cur_data_list = data_dict[one_index]
                    cur_label_list = label_dict[one_index]
                    for one in range(args.batch_size):
                        label_data_dict[cur_label_list[one]].add(cur_data_list[one])
                    label_data_dict_list.append(label_data_dict)
                    label_data_dict = defaultdict(set)
                corres_batch_data = []
                for i in range(len(label_data_dict_list)):
                    for one in source_label:
                        label_cur = one[0].item()
                        corres_batch_data.append(random.choice(list(label_data_dict_list[i][label_cur])))
                corres_batch_data = torch.stack(corres_batch_data)
                if cuda:
                    source_data = source_data.cuda()
                    source_label = source_label.cuda()
                    subject_id = subject_id.cuda()
                    corres_batch_data =corres_batch_data.cuda()
                data_set_all+=len(source_label)
                optimizer_PreTraining.zero_grad()
                rec_loss, sim_loss = preTrainModel(source_data, corres_batch_data, subject_id, m, mark=j)
                loss_pretrain = rec_loss + args.beta*sim_loss
                loss_pretrain.backward()
                optimizer_PreTraining.step()
        print("data set amount: "+str(data_set_all))
        writer.add_scalars('subject: '+str(one_subject+1)+' '+'train DMMR/loss',
                           {'loss_pretrain':loss_pretrain.data,'rec_loss':rec_loss.data,'sim_loss':sim_loss.data}, epoch + 1)
        end_time_pretrain = time.time()
        pretrain_epoch_time = end_time_pretrain - start_time_pretrain
        print("The time required for one pre-training epoch is：", pretrain_epoch_time, "second")
        print("rec_loss: "+str(rec_loss))

    # The fine-tuning phase
    source_iters2 = []
    for i in range(len(source_loader)):
        source_iters2.append(iter(source_loader[i]))
    fineTuneModel = DMMRFineTuningModel(cuda, preTrainModel, number_of_source=len(source_loader),
                                    number_of_category=args.cls_classes, batch_size=args.batch_size,
                                    time_steps=args.time_steps)

    optimizer_FineTuning = torch.optim.Adam(fineTuneModel.parameters(), **optimizer_config)
    if cuda:
        fineTuneModel = fineTuneModel.cuda()
    for epoch in range(args.epoch_fineTuning):
        print("epoch: " + str(epoch))
        start_time = time.time()
        fineTuneModel.train()
        count = 0
        data_set_all = 0  
        for i in range(1, iteration + 1):
            p = float(i + epoch * iteration) / args.epoch_preTraining / iteration
            m = 2. / (1. + np.exp(-10 * p)) - 1
            batch_dict = defaultdict(list)
            data_dict = defaultdict(list)
            label_dict = defaultdict(list)
            label_data_dict = defaultdict(set)

            for j in range(len(source_iters2)):
                try:
                    batch_dict[j] = next(source_iters2[j])
                except:
                    source_iters2[j] = iter(source_loader[j])
                    batch_dict[j] = next(source_iters2[j])
                index = 0
                for o in batch_dict[j][1]:
                    cur_label = o[0].item()
                    data_dict[j].append(batch_dict[j][0][index])
                    label_dict[j].append(cur_label)
                    index += 1
            for j in range(len(source_iters)):
                source_data, source_label = batch_dict[j]
                if cuda:
                    source_data = source_data.cuda()
                    source_label = source_label.cuda()
                data_set_all += len(source_label)
                optimizer_FineTuning.zero_grad()
                x_pred, x_logits, cls_loss = fineTuneModel(source_data, source_label)
                cls_loss.backward()
                optimizer_FineTuning.step()
                _, pred = torch.max(x_pred, dim=1)
                count += pred.eq(source_label.squeeze().data.view_as(pred)).sum()
        end_time = time.time()
        epoch_time = end_time - start_time
        print("The time required for one fine-tuning epoch is：", epoch_time, "second")
        print("data set amount: " + str(data_set_all))
        acc = float(count) / data_set_all
        writer.add_scalars('subject: ' + str(one_subject + 1) + ' ' + 'train DMMR/loss',
                           {'cls_loss': cls_loss.data}, epoch + 1)
        writer.add_scalar('subject: ' + str(one_subject + 1) + ' ' + 'train DMMR/train accuracy', acc, epoch + 1)
        print("acc: " + str(acc))

        testModel = DMMRTestModel(fineTuneModel)
        acc_DMMR = testDMMR(data_loader_dict["test_loader"], testModel, cuda, args.batch_size)
        print("acc_DMMR: " + str(acc_DMMR))
        writer.add_scalars('subject: ' + str(one_subject + 1) + ' ' + 'test DMMR/test acc',
                           {'test acc': acc_DMMR}, epoch + 1)
        if acc_DMMR > acc_final:
            acc_final = acc_DMMR
            best_pretrain_model = copy.deepcopy(preTrainModel.state_dict())
            best_tune_model = copy.deepcopy(fineTuneModel.state_dict())
            best_test_model = copy.deepcopy(testModel.state_dict())
    modelDir = "model/"+args.way+"/"+args.index+"/"
    try:
        os.makedirs(modelDir)
    except:
        pass
    torch.save(best_pretrain_model, modelDir+str(one_subject)+'_pretrain_model.pth')
    torch.save(best_tune_model, modelDir+str(one_subject)+'_tune_model.pth')
    torch.save(best_test_model, modelDir+str(one_subject)+'_test_model.pth')
    return acc_final
# w/o both
def trainDMMR_WithoutBothMixAndNoise(data_loader_dict, optimizer_config, cuda, args, iteration, writer, one_subject):
    source_loader = data_loader_dict['source_loader']
    # The pre-training phase
    preTrainModel = PreTrainingWithoutBothMixAndNoise(cuda, number_of_source=len(source_loader), number_of_category=args.cls_classes, batch_size=args.batch_size, time_steps=args.time_steps)
    if cuda:
        preTrainModel = preTrainModel.cuda()
    source_iters = []
    for i in range(len(source_loader)):
        source_iters.append(iter(source_loader[i]))
    optimizer_PreTraining = torch.optim.Adam(preTrainModel.parameters(), **optimizer_config)

    acc_final = 0
    for epoch in range(args.epoch_preTraining):
        print("epoch: "+str(epoch))
        start_time_pretrain = time.time()
        preTrainModel.train()
        count = 0
        data_set_all = 0
        for i in range(1, iteration + 1):
            p = float(i + epoch * iteration) / args.epoch_preTraining / iteration
            m = 2. / (1. + np.exp(-10 * p)) - 1
            batch_dict = defaultdict(list)
            data_dict = defaultdict(list)
            label_dict = defaultdict(list)
            label_data_dict = defaultdict(set)

            for j in range(len(source_iters)):
                try:
                    batch_dict[j] = next(source_iters[j])
                except:
                    source_iters[j] = iter(source_loader[j])
                    batch_dict[j]= next(source_iters[j])
                index = 0
                for o in batch_dict[j][1]:
                    cur_label = o[0].item()
                    data_dict[j].append(batch_dict[j][0][index])
                    label_dict[j].append(cur_label)
                    index+=1

            for j in range(len(source_iters)):
                subject_id = torch.ones(args.batch_size)
                subject_id = subject_id * j
                subject_id = subject_id.long()

                source_data, source_label = batch_dict[j]
                # prepare corresponding new batch, the new batch has same label with current batch
                label_data_dict_list = []
                for one_index in range(args.source_subjects):
                    cur_data_list = data_dict[one_index]
                    cur_label_list = label_dict[one_index]
                    for one in range(args.batch_size):
                        label_data_dict[cur_label_list[one]].add(cur_data_list[one])
                    label_data_dict_list.append(label_data_dict)
                    label_data_dict = defaultdict(set)
                corres_batch_data = []
                for i in range(len(label_data_dict_list)):
                    for one in source_label:
                        label_cur = one[0].item()
                        corres_batch_data.append(random.choice(list(label_data_dict_list[i][label_cur])))
                corres_batch_data = torch.stack(corres_batch_data)
                if cuda:
                    source_data = source_data.cuda()
                    source_label = source_label.cuda()
                    subject_id = subject_id.cuda()
                    corres_batch_data =corres_batch_data.cuda()
                data_set_all+=len(source_label)
                optimizer_PreTraining.zero_grad()
                rec_loss, sim_loss = preTrainModel(source_data, corres_batch_data, subject_id, m, mark=j)
                loss_pretrain = rec_loss + args.beta*sim_loss
                loss_pretrain.backward()
                optimizer_PreTraining.step()
        print("data set amount: "+str(data_set_all))
        writer.add_scalars('subject: '+str(one_subject+1)+' '+'train DMMR/loss',
                           {'loss_pretrain':loss_pretrain.data,'rec_loss':rec_loss.data,'sim_loss':sim_loss.data}, epoch + 1)
        end_time_pretrain = time.time()
        pretrain_epoch_time = end_time_pretrain - start_time_pretrain
        print("The time required for one pre-training epoch is：", pretrain_epoch_time, "second")
        print("rec_loss: "+str(rec_loss))

    # The fine-tuning phase
    source_iters2 = []
    for i in range(len(source_loader)):
        source_iters2.append(iter(source_loader[i]))
    fineTuneModel = DMMRFineTuningModel(cuda, preTrainModel, number_of_source=len(source_loader),
                                                number_of_category=args.cls_classes, batch_size=args.batch_size,
                                                time_steps=args.time_steps)

    optimizer_FineTuning = torch.optim.Adam(fineTuneModel.parameters(), **optimizer_config)
    if cuda:
        fineTuneModel = fineTuneModel.cuda()
    for epoch in range(args.epoch_fineTuning):
        print("epoch: " + str(epoch))
        start_time = time.time()
        fineTuneModel.train()
        count = 0
        data_set_all = 0  
        for i in range(1, iteration + 1):
            p = float(i + epoch * iteration) / args.epoch_preTraining / iteration
            m = 2. / (1. + np.exp(-10 * p)) - 1
            batch_dict = defaultdict(list)
            data_dict = defaultdict(list)
            label_dict = defaultdict(list)
            label_data_dict = defaultdict(set)

            for j in range(len(source_iters2)):
                try:
                    batch_dict[j] = next(source_iters2[j])
                except:
                    source_iters2[j] = iter(source_loader[j])
                    batch_dict[j] = next(source_iters2[j])
                index = 0
                for o in batch_dict[j][1]:
                    cur_label = o[0].item()
                    data_dict[j].append(batch_dict[j][0][index])
                    label_dict[j].append(cur_label)
                    index += 1
            for j in range(len(source_iters)):
                source_data, source_label = batch_dict[j]
                if cuda:
                    source_data = source_data.cuda()
                    source_label = source_label.cuda()
                data_set_all += len(source_label)
                optimizer_FineTuning.zero_grad()
                x_pred, x_logits, cls_loss = fineTuneModel(source_data, source_label)
                cls_loss.backward()
                optimizer_FineTuning.step()
                _, pred = torch.max(x_pred, dim=1)
                count += pred.eq(source_label.squeeze().data.view_as(pred)).sum()
        end_time = time.time()
        epoch_time = end_time - start_time
        print("The time required for one fine-tuning epoch is：", epoch_time, "second")
        print("data set amount: " + str(data_set_all))
        acc = float(count) / data_set_all
        writer.add_scalars('subject: ' + str(one_subject + 1) + ' ' + 'train DMMR/loss',
                           {'cls_loss': cls_loss.data}, epoch + 1)
        writer.add_scalar('subject: ' + str(one_subject + 1) + ' ' + 'train DMMR/train accuracy', acc, epoch + 1)
        print("acc: " + str(acc))

        testModel = DMMRTestModel(fineTuneModel)
        acc_DMMR = testDMMR(data_loader_dict["test_loader"], testModel, cuda, args.batch_size)
        print("acc_DMMR: " + str(acc_DMMR))
        writer.add_scalars('subject: ' + str(one_subject + 1) + ' ' + 'test DMMR/test acc',
                           {'test acc': acc_DMMR}, epoch + 1)
        if acc_DMMR > acc_final:
            acc_final = acc_DMMR
            best_pretrain_model = copy.deepcopy(preTrainModel.state_dict())
            best_tune_model = copy.deepcopy(fineTuneModel.state_dict())
            best_test_model = copy.deepcopy(testModel.state_dict())
    modelDir = "model/" + args.way + "/" + args.index + "/"
    try:
        os.makedirs(modelDir)
    except:
        pass
    torch.save(best_pretrain_model, modelDir + str(one_subject) + '_pretrain_model.pth')
    torch.save(best_tune_model, modelDir + str(one_subject) + '_tune_model.pth')
    torch.save(best_test_model, modelDir + str(one_subject) + '_test_model.pth')
    return acc_final

############## Other noise injection methods ##############
def trainDMMR_Noise_MaskChannels(data_loader_dict, optimizer_config, cuda, args, iteration, writer, one_subject):
    source_loader = data_loader_dict['source_loader']
    # The pre-training phase
    preTrainModel = PreTrainingWithMaskChannels(cuda, number_of_source=len(source_loader), number_of_category=args.cls_classes, batch_size=args.batch_size, time_steps=args.time_steps)
    if cuda:
        preTrainModel = preTrainModel.cuda()
    source_iters = []
    for i in range(len(source_loader)):
        source_iters.append(iter(source_loader[i]))
    optimizer_PreTraining = torch.optim.Adam(preTrainModel.parameters(), **optimizer_config)

    acc_final = 0
    for epoch in range(args.epoch_preTraining):
        print("epoch: "+str(epoch))
        start_time_pretrain = time.time()
        preTrainModel.train()
        count = 0
        data_set_all = 0
        for i in range(1, iteration + 1):
            p = float(i + epoch * iteration) / args.epoch_preTraining / iteration
            m = 2. / (1. + np.exp(-10 * p)) - 1
            batch_dict = defaultdict(list)
            data_dict = defaultdict(list)
            label_dict = defaultdict(list)
            label_data_dict = defaultdict(set)

            for j in range(len(source_iters)):
                try:
                    batch_dict[j] = next(source_iters[j])
                except:
                    source_iters[j] = iter(source_loader[j])
                    batch_dict[j]= next(source_iters[j])
                index = 0
                for o in batch_dict[j][1]:
                    cur_label = o[0].item()
                    data_dict[j].append(batch_dict[j][0][index])
                    label_dict[j].append(cur_label)
                    index+=1

            for j in range(len(source_iters)):
                subject_id = torch.ones(args.batch_size)
                subject_id = subject_id * j
                subject_id = subject_id.long()

                source_data, source_label = batch_dict[j]
                # prepare corresponding new batch, the new batch has same label with current batch
                label_data_dict_list = []
                for one_index in range(args.source_subjects):
                    cur_data_list = data_dict[one_index]
                    cur_label_list = label_dict[one_index]
                    for one in range(args.batch_size):
                        label_data_dict[cur_label_list[one]].add(cur_data_list[one])
                    label_data_dict_list.append(label_data_dict)
                    label_data_dict = defaultdict(set)
                corres_batch_data = []
                for i in range(len(label_data_dict_list)):
                    for one in source_label:
                        label_cur = one[0].item()
                        corres_batch_data.append(random.choice(list(label_data_dict_list[i][label_cur])))
                corres_batch_data = torch.stack(corres_batch_data)
                if cuda:
                    source_data = source_data.cuda()
                    source_label = source_label.cuda()
                    subject_id = subject_id.cuda()
                    corres_batch_data =corres_batch_data.cuda()
                data_set_all+=len(source_label)
                optimizer_PreTraining.zero_grad()
                rec_loss, sim_loss = preTrainModel(source_data, corres_batch_data, subject_id, args, m, mark=j)
                loss_pretrain = rec_loss + args.beta*sim_loss
                loss_pretrain.backward()
                optimizer_PreTraining.step()
        print("data set amount: "+str(data_set_all))
        writer.add_scalars('subject: '+str(one_subject+1)+' '+'train DMMR/loss',
                           {'loss_pretrain':loss_pretrain.data,'rec_loss':rec_loss.data,'sim_loss':sim_loss.data}, epoch + 1)
        end_time_pretrain = time.time()
        pretrain_epoch_time = end_time_pretrain - start_time_pretrain
        print("The time required for one pre-training epoch is：", pretrain_epoch_time, "second")
        print("rec_loss: "+str(rec_loss))

    # The fine-tuning phase
    source_iters2 = []
    for i in range(len(source_loader)):
        source_iters2.append(iter(source_loader[i]))
    fineTuneModel = DMMRFineTuningModel(cuda, preTrainModel, number_of_source=len(source_loader),
                                    number_of_category=args.cls_classes, batch_size=args.batch_size,
                                    time_steps=args.time_steps)

    optimizer_FineTuning = torch.optim.Adam(fineTuneModel.parameters(), **optimizer_config)
    if cuda:
        fineTuneModel = fineTuneModel.cuda()
    for epoch in range(args.epoch_fineTuning):
        print("epoch: " + str(epoch))
        start_time = time.time()
        fineTuneModel.train()
        count = 0
        data_set_all = 0  
        for i in range(1, iteration + 1):
            p = float(i + epoch * iteration) / args.epoch_preTraining / iteration
            m = 2. / (1. + np.exp(-10 * p)) - 1
            batch_dict = defaultdict(list)
            data_dict = defaultdict(list)
            label_dict = defaultdict(list)
            label_data_dict = defaultdict(set)

            for j in range(len(source_iters2)):
                try:
                    batch_dict[j] = next(source_iters2[j])
                except:
                    source_iters2[j] = iter(source_loader[j])
                    batch_dict[j] = next(source_iters2[j])
                index = 0
                for o in batch_dict[j][1]:
                    cur_label = o[0].item()
                    data_dict[j].append(batch_dict[j][0][index])
                    label_dict[j].append(cur_label)
                    index += 1
            for j in range(len(source_iters)):
                source_data, source_label = batch_dict[j]
                if cuda:
                    source_data = source_data.cuda()
                    source_label = source_label.cuda()
                data_set_all += len(source_label)
                optimizer_FineTuning.zero_grad()
                x_pred, x_logits, cls_loss = fineTuneModel(source_data, source_label)
                cls_loss.backward()
                optimizer_FineTuning.step()
                _, pred = torch.max(x_pred, dim=1)
                count += pred.eq(source_label.squeeze().data.view_as(pred)).sum()
        end_time = time.time()
        epoch_time = end_time - start_time
        print("The time required for one fine-tuning epoch is：", epoch_time, "second")
        print("data set amount: " + str(data_set_all))
        acc = float(count) / data_set_all
        writer.add_scalars('subject: ' + str(one_subject + 1) + ' ' + 'train DMMR/loss',
                           {'cls_loss': cls_loss.data}, epoch + 1)
        writer.add_scalar('subject: ' + str(one_subject + 1) + ' ' + 'train DMMR/train accuracy', acc, epoch + 1)
        print("acc: " + str(acc))

        testModel = DMMRTestModel(fineTuneModel)
        acc_DMMR = testDMMR(data_loader_dict["test_loader"], testModel, cuda, args.batch_size)
        print("acc_DMMR: " + str(acc_DMMR))
        writer.add_scalars('subject: ' + str(one_subject + 1) + ' ' + 'test DMMR/test acc',
                           {'test acc': acc_DMMR}, epoch + 1)
        if acc_DMMR > acc_final:
            acc_final = acc_DMMR
            best_pretrain_model = copy.deepcopy(preTrainModel.state_dict())
            best_tune_model = copy.deepcopy(fineTuneModel.state_dict())
            best_test_model = copy.deepcopy(testModel.state_dict())
    modelDir = "model/" + args.way + "/" + args.index + "/"
    try:
        os.makedirs(modelDir)
    except:
        pass
    torch.save(best_pretrain_model, modelDir + str(one_subject) + '_pretrain_model.pth')
    torch.save(best_tune_model, modelDir + str(one_subject) + '_tune_model.pth')
    torch.save(best_test_model, modelDir + str(one_subject) + '_test_model.pth')
    return acc_final

def trainDMMR_Noise_MaskTimeSteps(data_loader_dict, optimizer_config, cuda, args, iteration, writer, one_subject):
    source_loader = data_loader_dict['source_loader']
    # The pre-training phase
    preTrainModel = PreTrainingWithMaskTimeSteps(cuda, number_of_source=len(source_loader), number_of_category=args.cls_classes, batch_size=args.batch_size, time_steps=args.time_steps)
    if cuda:
        preTrainModel = preTrainModel.cuda()
    source_iters = []
    for i in range(len(source_loader)):
        source_iters.append(iter(source_loader[i]))
    optimizer_PreTraining = torch.optim.Adam(preTrainModel.parameters(), **optimizer_config)

    acc_final = 0
    for epoch in range(args.epoch_preTraining):
        print("epoch: "+str(epoch))
        start_time_pretrain = time.time()
        preTrainModel.train()
        count = 0
        data_set_all = 0
        for i in range(1, iteration + 1):
            p = float(i + epoch * iteration) / args.epoch_preTraining / iteration
            m = 2. / (1. + np.exp(-10 * p)) - 1
            batch_dict = defaultdict(list)
            data_dict = defaultdict(list)
            label_dict = defaultdict(list)
            label_data_dict = defaultdict(set)

            for j in range(len(source_iters)):
                try:
                    batch_dict[j] = next(source_iters[j])
                except:
                    source_iters[j] = iter(source_loader[j])
                    batch_dict[j]= next(source_iters[j])
                index = 0
                for o in batch_dict[j][1]:
                    cur_label = o[0].item()
                    data_dict[j].append(batch_dict[j][0][index])
                    label_dict[j].append(cur_label)
                    index+=1

            for j in range(len(source_iters)):
                subject_id = torch.ones(args.batch_size)
                subject_id = subject_id * j
                subject_id = subject_id.long()

                source_data, source_label = batch_dict[j]
                # prepare corresponding new batch, the new batch has same label with current batch
                label_data_dict_list = []
                for one_index in range(args.source_subjects):
                    cur_data_list = data_dict[one_index]
                    cur_label_list = label_dict[one_index]
                    for one in range(args.batch_size):
                        label_data_dict[cur_label_list[one]].add(cur_data_list[one])
                    label_data_dict_list.append(label_data_dict)
                    label_data_dict = defaultdict(set)
                corres_batch_data = []
                for i in range(len(label_data_dict_list)):
                    for one in source_label:
                        label_cur = one[0].item()
                        corres_batch_data.append(random.choice(list(label_data_dict_list[i][label_cur])))
                corres_batch_data = torch.stack(corres_batch_data)
                if cuda:
                    source_data = source_data.cuda()
                    source_label = source_label.cuda()
                    subject_id = subject_id.cuda()
                    corres_batch_data =corres_batch_data.cuda()
                data_set_all+=len(source_label)
                optimizer_PreTraining.zero_grad()
                rec_loss, sim_loss = preTrainModel(source_data, corres_batch_data, subject_id, args, m, mark=j)
                loss_pretrain = rec_loss + args.beta*sim_loss
                loss_pretrain.backward()
                optimizer_PreTraining.step()
        print("data set amount: "+str(data_set_all))
        writer.add_scalars('subject: '+str(one_subject+1)+' '+'train DMMR/loss',
                           {'loss_pretrain':loss_pretrain.data,'rec_loss':rec_loss.data,'sim_loss':sim_loss.data}, epoch + 1)
        end_time_pretrain = time.time()
        pretrain_epoch_time = end_time_pretrain - start_time_pretrain
        print("The time required for one pre-training epoch is：", pretrain_epoch_time, "second")
        print("rec_loss: "+str(rec_loss))

    # The fine-tuning phase
    source_iters2 = []
    for i in range(len(source_loader)):
        source_iters2.append(iter(source_loader[i]))
    fineTuneModel = DMMRFineTuningModel(cuda, preTrainModel, number_of_source=len(source_loader),
                                    number_of_category=args.cls_classes, batch_size=args.batch_size,
                                    time_steps=args.time_steps)

    optimizer_FineTuning = torch.optim.Adam(fineTuneModel.parameters(), **optimizer_config)
    if cuda:
        fineTuneModel = fineTuneModel.cuda()
    for epoch in range(args.epoch_fineTuning):
        print("epoch: " + str(epoch))
        start_time = time.time()
        fineTuneModel.train()
        count = 0
        data_set_all = 0  
        for i in range(1, iteration + 1):
            p = float(i + epoch * iteration) / args.epoch_preTraining / iteration
            m = 2. / (1. + np.exp(-10 * p)) - 1
            batch_dict = defaultdict(list)
            data_dict = defaultdict(list)
            label_dict = defaultdict(list)
            label_data_dict = defaultdict(set)

            for j in range(len(source_iters2)):
                try:
                    batch_dict[j] = next(source_iters2[j])
                except:
                    source_iters2[j] = iter(source_loader[j])
                    batch_dict[j] = next(source_iters2[j])
                index = 0
                for o in batch_dict[j][1]:
                    cur_label = o[0].item()
                    data_dict[j].append(batch_dict[j][0][index])
                    label_dict[j].append(cur_label)
                    index += 1
            for j in range(len(source_iters)):
                source_data, source_label = batch_dict[j]
                if cuda:
                    source_data = source_data.cuda()
                    source_label = source_label.cuda()
                data_set_all += len(source_label)
                optimizer_FineTuning.zero_grad()
                x_pred, x_logits, cls_loss = fineTuneModel(source_data, source_label)
                cls_loss.backward()
                optimizer_FineTuning.step()
                _, pred = torch.max(x_pred, dim=1)
                count += pred.eq(source_label.squeeze().data.view_as(pred)).sum()
        end_time = time.time()
        epoch_time = end_time - start_time
        print("The time required for one fine-tuning epoch is：", epoch_time, "second")
        print("data set amount: " + str(data_set_all))
        acc = float(count) / data_set_all
        writer.add_scalars('subject: ' + str(one_subject + 1) + ' ' + 'train DMMR/loss',
                           {'cls_loss': cls_loss.data}, epoch + 1)
        writer.add_scalar('subject: ' + str(one_subject + 1) + ' ' + 'train DMMR/train accuracy', acc, epoch + 1)
        print("acc: " + str(acc))

        testModel = DMMRTestModel(fineTuneModel)
        acc_DMMR = testDMMR(data_loader_dict["test_loader"], testModel, cuda, args.batch_size)
        print("acc_DMMR: " + str(acc_DMMR))
        writer.add_scalars('subject: ' + str(one_subject + 1) + ' ' + 'test DMMR/test acc',
                           {'test acc': acc_DMMR}, epoch + 1)
        if acc_DMMR > acc_final:
            acc_final = acc_DMMR
            best_pretrain_model = copy.deepcopy(preTrainModel.state_dict())
            best_tune_model = copy.deepcopy(fineTuneModel.state_dict())
            best_test_model = copy.deepcopy(testModel.state_dict())
    modelDir = "model/" + args.way + "/" + args.index + "/"
    try:
        os.makedirs(modelDir)
    except:
        pass
    torch.save(best_pretrain_model, modelDir + str(one_subject) + '_pretrain_model.pth')
    torch.save(best_tune_model, modelDir + str(one_subject) + '_tune_model.pth')
    torch.save(best_test_model, modelDir + str(one_subject) + '_test_model.pth')
    return acc_final

def trainDMMR_Noise_ChannelsShuffling(data_loader_dict, optimizer_config, cuda, args, iteration, writer, one_subject):
    source_loader = data_loader_dict['source_loader']
    # The pre-training phase
    preTrainModel = PreTrainingWithChannelsShuffling(cuda, number_of_source=len(source_loader), number_of_category=args.cls_classes, batch_size=args.batch_size, time_steps=args.time_steps)
    if cuda:
        preTrainModel = preTrainModel.cuda()
    source_iters = []
    for i in range(len(source_loader)):
        source_iters.append(iter(source_loader[i]))
    optimizer_PreTraining = torch.optim.Adam(preTrainModel.parameters(), **optimizer_config)

    acc_final = 0
    for epoch in range(args.epoch_preTraining):
        print("epoch: "+str(epoch))
        start_time_pretrain = time.time()
        preTrainModel.train()
        count = 0
        data_set_all = 0
        for i in range(1, iteration + 1):
            p = float(i + epoch * iteration) / args.epoch_preTraining / iteration
            m = 2. / (1. + np.exp(-10 * p)) - 1
            batch_dict = defaultdict(list)
            data_dict = defaultdict(list)
            label_dict = defaultdict(list)
            label_data_dict = defaultdict(set)

            for j in range(len(source_iters)):
                try:
                    batch_dict[j] = next(source_iters[j])
                except:
                    source_iters[j] = iter(source_loader[j])
                    batch_dict[j]= next(source_iters[j])
                index = 0
                for o in batch_dict[j][1]:
                    cur_label = o[0].item()
                    data_dict[j].append(batch_dict[j][0][index])
                    label_dict[j].append(cur_label)
                    index+=1

            for j in range(len(source_iters)):
                subject_id = torch.ones(args.batch_size)
                subject_id = subject_id * j
                subject_id = subject_id.long()

                source_data, source_label = batch_dict[j]
                # prepare corresponding new batch, the new batch has same label with current batch
                label_data_dict_list = []
                for one_index in range(args.source_subjects):
                    cur_data_list = data_dict[one_index]
                    cur_label_list = label_dict[one_index]
                    for one in range(args.batch_size):
                        label_data_dict[cur_label_list[one]].add(cur_data_list[one])
                    label_data_dict_list.append(label_data_dict)
                    label_data_dict = defaultdict(set)
                corres_batch_data = []
                for i in range(len(label_data_dict_list)):
                    for one in source_label:
                        label_cur = one[0].item()
                        corres_batch_data.append(random.choice(list(label_data_dict_list[i][label_cur])))
                corres_batch_data = torch.stack(corres_batch_data)
                if cuda:
                    source_data = source_data.cuda()
                    source_label = source_label.cuda()
                    subject_id = subject_id.cuda()
                    corres_batch_data =corres_batch_data.cuda()
                data_set_all+=len(source_label)
                optimizer_PreTraining.zero_grad()
                rec_loss, sim_loss = preTrainModel(source_data, corres_batch_data, subject_id, args, m, mark=j)
                loss_pretrain = rec_loss + args.beta*sim_loss
                loss_pretrain.backward()
                optimizer_PreTraining.step()
        print("data set amount: "+str(data_set_all))
        writer.add_scalars('subject: '+str(one_subject+1)+' '+'train DMMR/loss',
                           {'loss_pretrain':loss_pretrain.data,'rec_loss':rec_loss.data,'sim_loss':sim_loss.data}, epoch + 1)
        end_time_pretrain = time.time()
        pretrain_epoch_time = end_time_pretrain - start_time_pretrain
        print("The time required for one pre-training epoch is：", pretrain_epoch_time, "second")
        print("rec_loss: "+str(rec_loss))

    # The fine-tuning phase
    source_iters2 = []
    for i in range(len(source_loader)):
        source_iters2.append(iter(source_loader[i]))
    fineTuneModel = DMMRFineTuningModel(cuda, preTrainModel, number_of_source=len(source_loader),
                                    number_of_category=args.cls_classes, batch_size=args.batch_size,
                                    time_steps=args.time_steps)

    optimizer_FineTuning = torch.optim.Adam(fineTuneModel.parameters(), **optimizer_config)
    if cuda:
        fineTuneModel = fineTuneModel.cuda()
    for epoch in range(args.epoch_fineTuning):
        print("epoch: " + str(epoch))
        start_time = time.time()
        fineTuneModel.train()
        count = 0
        data_set_all = 0  
        for i in range(1, iteration + 1):
            p = float(i + epoch * iteration) / args.epoch_preTraining / iteration
            m = 2. / (1. + np.exp(-10 * p)) - 1
            batch_dict = defaultdict(list)
            data_dict = defaultdict(list)
            label_dict = defaultdict(list)
            label_data_dict = defaultdict(set)

            for j in range(len(source_iters2)):
                try:
                    batch_dict[j] = next(source_iters2[j])
                except:
                    source_iters2[j] = iter(source_loader[j])
                    batch_dict[j] = next(source_iters2[j])
                index = 0
                for o in batch_dict[j][1]:
                    cur_label = o[0].item()
                    data_dict[j].append(batch_dict[j][0][index])
                    label_dict[j].append(cur_label)
                    index += 1
            for j in range(len(source_iters)):
                source_data, source_label = batch_dict[j]
                if cuda:
                    source_data = source_data.cuda()
                    source_label = source_label.cuda()
                data_set_all += len(source_label)
                optimizer_FineTuning.zero_grad()
                x_pred, x_logits, cls_loss = fineTuneModel(source_data, source_label)
                cls_loss.backward()
                optimizer_FineTuning.step()
                _, pred = torch.max(x_pred, dim=1)
                count += pred.eq(source_label.squeeze().data.view_as(pred)).sum()
        end_time = time.time()
        epoch_time = end_time - start_time
        print("The time required for one fine-tuning epoch is：", epoch_time, "second")
        print("data set amount: " + str(data_set_all))
        acc = float(count) / data_set_all
        writer.add_scalars('subject: ' + str(one_subject + 1) + ' ' + 'train DMMR/loss',
                           {'cls_loss': cls_loss.data}, epoch + 1)
        writer.add_scalar('subject: ' + str(one_subject + 1) + ' ' + 'train DMMR/train accuracy', acc, epoch + 1)
        print("acc: " + str(acc))

        testModel = DMMRTestModel(fineTuneModel)
        acc_DMMR = testDMMR(data_loader_dict["test_loader"], testModel, cuda, args.batch_size)
        print("acc_DMMR: " + str(acc_DMMR))
        writer.add_scalars('subject: ' + str(one_subject + 1) + ' ' + 'test DMMR/test acc',
                           {'test acc': acc_DMMR}, epoch + 1)
        if acc_DMMR > acc_final:
            acc_final = acc_DMMR
            best_pretrain_model = copy.deepcopy(preTrainModel.state_dict())
            best_tune_model = copy.deepcopy(fineTuneModel.state_dict())
            best_test_model = copy.deepcopy(testModel.state_dict())
    modelDir = "model/" + args.way + "/" + args.index + "/"
    try:
        os.makedirs(modelDir)
    except:
        pass
    torch.save(best_pretrain_model, modelDir + str(one_subject) + '_pretrain_model.pth')
    torch.save(best_tune_model, modelDir + str(one_subject) + '_tune_model.pth')
    torch.save(best_test_model, modelDir + str(one_subject) + '_test_model.pth')
    return acc_final

def trainDMMR_Noise_Dropout(data_loader_dict, optimizer_config, cuda, args, iteration, writer, one_subject):
    source_loader = data_loader_dict['source_loader']
    # The pre-training phase
    preTrainModel = PreTrainingWithDropout(cuda, number_of_source=len(source_loader), number_of_category=args.cls_classes, batch_size=args.batch_size, time_steps=args.time_steps, dropout_rate=0.2)
    if cuda:
        preTrainModel = preTrainModel.cuda()
    source_iters = []
    for i in range(len(source_loader)):
        source_iters.append(iter(source_loader[i]))
    optimizer_PreTraining = torch.optim.Adam(preTrainModel.parameters(), **optimizer_config)

    acc_final = 0
    for epoch in range(args.epoch_preTraining):
        print("epoch: "+str(epoch))
        start_time_pretrain = time.time()
        preTrainModel.train()
        count = 0
        data_set_all = 0
        for i in range(1, iteration + 1):
            p = float(i + epoch * iteration) / args.epoch_preTraining / iteration
            m = 2. / (1. + np.exp(-10 * p)) - 1
            batch_dict = defaultdict(list)
            data_dict = defaultdict(list)
            label_dict = defaultdict(list)
            label_data_dict = defaultdict(set)

            for j in range(len(source_iters)):
                try:
                    batch_dict[j] = next(source_iters[j])
                except:
                    source_iters[j] = iter(source_loader[j])
                    batch_dict[j]= next(source_iters[j])
                index = 0
                for o in batch_dict[j][1]:
                    cur_label = o[0].item()
                    data_dict[j].append(batch_dict[j][0][index])
                    label_dict[j].append(cur_label)
                    index+=1

            for j in range(len(source_iters)):
                subject_id = torch.ones(args.batch_size)
                subject_id = subject_id * j
                subject_id = subject_id.long()

                source_data, source_label = batch_dict[j]
                # prepare corresponding new batch, the new batch has same label with current batch
                label_data_dict_list = []
                for one_index in range(args.source_subjects):
                    cur_data_list = data_dict[one_index]
                    cur_label_list = label_dict[one_index]
                    for one in range(args.batch_size):
                        label_data_dict[cur_label_list[one]].add(cur_data_list[one])
                    label_data_dict_list.append(label_data_dict)
                    label_data_dict = defaultdict(set)
                corres_batch_data = []
                for i in range(len(label_data_dict_list)):
                    for one in source_label:
                        label_cur = one[0].item()
                        corres_batch_data.append(random.choice(list(label_data_dict_list[i][label_cur])))
                corres_batch_data = torch.stack(corres_batch_data)
                if cuda:
                    source_data = source_data.cuda()
                    source_label = source_label.cuda()
                    subject_id = subject_id.cuda()
                    corres_batch_data =corres_batch_data.cuda()
                data_set_all+=len(source_label)
                optimizer_PreTraining.zero_grad()
                rec_loss, sim_loss = preTrainModel(source_data, corres_batch_data, subject_id, args, m, mark=j)
                loss_pretrain = rec_loss + args.beta*sim_loss
                loss_pretrain.backward()
                optimizer_PreTraining.step()
        print("data set amount: "+str(data_set_all))
        writer.add_scalars('subject: '+str(one_subject+1)+' '+'train DMMR/loss',
                           {'loss_pretrain':loss_pretrain.data,'rec_loss':rec_loss.data,'sim_loss':sim_loss.data}, epoch + 1)
        end_time_pretrain = time.time()
        pretrain_epoch_time = end_time_pretrain - start_time_pretrain
        print("The time required for one pre-training epoch is：", pretrain_epoch_time, "second")
        print("rec_loss: "+str(rec_loss))

    # The fine-tuning phase
    source_iters2 = []
    for i in range(len(source_loader)):
        source_iters2.append(iter(source_loader[i]))
    fineTuneModel = DMMRFineTuningModel(cuda, preTrainModel, number_of_source=len(source_loader),
                                    number_of_category=args.cls_classes, batch_size=args.batch_size,
                                    time_steps=args.time_steps)

    optimizer_FineTuning = torch.optim.Adam(fineTuneModel.parameters(), **optimizer_config)
    if cuda:
        fineTuneModel = fineTuneModel.cuda()
    for epoch in range(args.epoch_fineTuning):
        print("epoch: " + str(epoch))
        start_time = time.time()
        fineTuneModel.train()
        count = 0
        data_set_all = 0  
        for i in range(1, iteration + 1):
            p = float(i + epoch * iteration) / args.epoch_preTraining / iteration
            m = 2. / (1. + np.exp(-10 * p)) - 1
            batch_dict = defaultdict(list)
            data_dict = defaultdict(list)
            label_dict = defaultdict(list)
            label_data_dict = defaultdict(set)

            for j in range(len(source_iters2)):
                try:
                    batch_dict[j] = next(source_iters2[j])
                except:
                    source_iters2[j] = iter(source_loader[j])
                    batch_dict[j] = next(source_iters2[j])
                index = 0
                for o in batch_dict[j][1]:
                    cur_label = o[0].item()
                    data_dict[j].append(batch_dict[j][0][index])
                    label_dict[j].append(cur_label)
                    index += 1
            for j in range(len(source_iters)):
                source_data, source_label = batch_dict[j]
                if cuda:
                    source_data = source_data.cuda()
                    source_label = source_label.cuda()
                data_set_all += len(source_label)
                optimizer_FineTuning.zero_grad()
                x_pred, x_logits, cls_loss = fineTuneModel(source_data, source_label)
                cls_loss.backward()
                optimizer_FineTuning.step()
                _, pred = torch.max(x_pred, dim=1)
                count += pred.eq(source_label.squeeze().data.view_as(pred)).sum()
        end_time = time.time()
        epoch_time = end_time - start_time
        print("The time required for one fine-tuning epoch is：", epoch_time, "second")
        print("data set amount: " + str(data_set_all))
        acc = float(count) / data_set_all
        writer.add_scalars('subject: ' + str(one_subject + 1) + ' ' + 'train DMMR/loss',
                           {'cls_loss': cls_loss.data}, epoch + 1)
        writer.add_scalar('subject: ' + str(one_subject + 1) + ' ' + 'train DMMR/train accuracy', acc, epoch + 1)
        print("acc: " + str(acc))

        testModel = DMMRTestModel(fineTuneModel)
        acc_DMMR = testDMMR(data_loader_dict["test_loader"], testModel, cuda, args.batch_size)
        print("acc_DMMR: " + str(acc_DMMR))
        writer.add_scalars('subject: ' + str(one_subject + 1) + ' ' + 'test DMMR/test acc',
                           {'test acc': acc_DMMR}, epoch + 1)
        if acc_DMMR > acc_final:
            acc_final = acc_DMMR
            best_pretrain_model = copy.deepcopy(preTrainModel.state_dict())
            best_tune_model = copy.deepcopy(fineTuneModel.state_dict())
            best_test_model = copy.deepcopy(testModel.state_dict())
    modelDir = "model/" + args.way + "/" + args.index + "/"
    try:
        os.makedirs(modelDir)
    except:
        pass
    torch.save(best_pretrain_model, modelDir + str(one_subject) + '_pretrain_model.pth')
    torch.save(best_tune_model, modelDir + str(one_subject) + '_tune_model.pth')
    torch.save(best_test_model, modelDir + str(one_subject) + '_test_model.pth')
    return acc_final

############## T-SNE plots ##############
class FeatureVisualize(object):
    '''
    Visualize features by TSNE
    '''

    def __init__(self, features, labels):
        '''
        features: (m,n)
        labels: (m,)
        '''
        self.features = features
        self.labels = labels

    def plot_tsne(self, save_filename, save_eps=False):
        ''' Plot TSNE figure. Set save_eps=True if you want to save a .eps file.
        '''
        tsne = TSNE(n_components=2, init='pca', random_state=0)
        features = tsne.fit_transform(self.features)
        x_min, x_max = np.min(features, 0), np.max(features, 0)
        data = (features - x_min) / (x_max - x_min)
        del features
        for i in range(data.shape[0]):
            colors = plt.cm.tab20.colors
            plt.scatter(data[i, 0], data[i, 1], color=colors[self.labels[i]])
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.title('T-SNE')
        if save_eps:
            plt.savefig('tsne.eps', dpi=600, format='eps')
        plt.savefig(save_filename, dpi=600)
        plt.show()
def TSNEForDMMR(data_loader_dict, cuda, args):
    source_loader = data_loader_dict['source_loader']
    # The pre-training phase
    target_loader = data_loader_dict["test_loader"]
    preTrainModel = DMMRPreTrainingModel(cuda,
                                         number_of_source=len(source_loader),
                                         number_of_category=args.cls_classes,
                                         batch_size=args.batch_size,
                                         time_steps=args.time_steps)
    #load the pretrained model
    preTrainModel.load_state_dict(torch.load("T-SNE/model/1_pretrain_model.pth", map_location='cpu'))
    preTrainModel.eval()
    pretrainReturnFeature = ModelReturnFeatures(preTrainModel, time_steps=args.time_steps)
    fineTuneModel = DMMRFineTuningModel(cuda, preTrainModel, number_of_source=len(source_loader),
                                    number_of_category=args.cls_classes,
                                    batch_size=args.batch_size,
                                    time_steps=args.time_steps)
    # load the fine-tuned model
    fineTuneModel.load_state_dict(torch.load("T-SNE/model/1_tune_model.pth", map_location='cpu'))
    fineTuneModel.eval()
    fineTuneModelReturnFeauters = ModelReturnFeatures(fineTuneModel, time_steps=args.time_steps)
    fineTuneModelReturnFeauters.eval()

    source_iters = []
    for i in range(len(source_loader)):
        source_iters.append(iter(source_loader[i]))

    origin_features_list = []
    origin_subject_id_list = []
    label_list = []
    pretrain_shared_features_list = []
    shared_features_list = []
    for i in range(1, 2):
        for j in range(len(source_iters)):
            try:
                source_data, source_label = next(source_iters[j])
            except:
                source_iters[j] = iter(source_loader[j])
                source_data, source_label = next(source_iters[j])
            subject_id = torch.ones(args.batch_size)
            subject_id = subject_id * j
            subject_id = subject_id.long()

            _, pretrain_shared_feature = pretrainReturnFeature(source_data)
            _, shared_feature = fineTuneModelReturnFeauters(source_data)

            num_samples = 50
            # 50 samples are taken from each individual subject data
            source_data_narray = source_data.numpy()
            label_data_narray = source_label.squeeze().numpy()
            # Reshape for sampling
            source_data_narray = source_data_narray.reshape(512, 30 * 310)
            # Randomly select 50 samples from it to obtain a tensor of size (50, 310).
            random_indices = np.random.choice(source_data_narray.shape[0], num_samples, replace=False)
            source_data_narray_50 = source_data_narray[random_indices]
            subject_narray = np.full((num_samples,), j)
            label_data_narray_50 = label_data_narray[random_indices]
            #origin feature
            origin_features_list.append(source_data_narray_50)
            origin_subject_id_list.append(subject_narray)
            label_list.append(label_data_narray_50)

            # pretrained feature
            pretrain_shared_feature_narray = pretrain_shared_feature.detach().numpy()
            pretrain_shared_feature_narray_50 = pretrain_shared_feature_narray[random_indices]
            pretrain_shared_features_list.append(pretrain_shared_feature_narray_50)
            #fine-tuned feature
            shared_feature_narray = shared_feature.detach().numpy()
            shared_feature_narray_50 = shared_feature_narray[random_indices]
            shared_features_list.append(shared_feature_narray_50)

        #generate target data
        target_data, target_label = next(iter(target_loader))
        _, target_pretrain_shared_feature = pretrainReturnFeature(target_data)
        _, target_shared_feature = fineTuneModelReturnFeauters(target_data)
        target_data_narray = target_data.numpy()
        target_label = target_label.squeeze().numpy()
        target_data_narray = target_data_narray.reshape(512, 30 * 310)
        random_indices_target = np.random.choice(target_data_narray.shape[0], num_samples, replace=False)
        target_data_narray_50 = target_data_narray[random_indices_target]
        target_subject_id = np.full((num_samples,), 14)
        target_label_narray_50 = target_label[random_indices]


        #add target subject data
        origin_features_list.append(target_data_narray_50)
        origin_subject_id_list.append(target_subject_id)
        label_list.append(target_label_narray_50)

        target_pretrain_shared_feature_narray = target_pretrain_shared_feature.detach().numpy()
        target_pretrain_shared_feature_narray_50 = target_pretrain_shared_feature_narray[random_indices]
        pretrain_shared_features_list.append(target_pretrain_shared_feature_narray_50)

        target_shared_feature_narray = target_shared_feature.detach().numpy()
        target_shared_feature_narray_50 = target_shared_feature_narray[random_indices]
        shared_features_list.append(target_shared_feature_narray_50)


        #concat for later norm
        origin_stacked_feature = np.concatenate(origin_features_list, axis=0)
        stacked_subject_id = np.concatenate(origin_subject_id_list, axis=0)
        stacked_label = np.concatenate(label_list, axis=0)

        # T-SNE
        #origin data
        vis_pretrain_shared = FeatureVisualize(origin_stacked_feature, stacked_subject_id)
        vis_pretrain_shared.plot_tsne('T-SNE/plot/origin_subject.jpg',save_eps=False)
        vis_pretrain_shared = FeatureVisualize(origin_stacked_feature, stacked_label)
        vis_pretrain_shared.plot_tsne("T-SNE/plot/origin_label.jpg",save_eps=False)

        # pretrained feature
        pretrain_shared_stacked_feature = np.concatenate(pretrain_shared_features_list, axis=0)
        vis_pretrain_shared = FeatureVisualize(pretrain_shared_stacked_feature, stacked_subject_id)
        vis_pretrain_shared.plot_tsne('T-SNE/plot/pretrain_subject.jpg',save_eps=False)
        vis_pretrain_shared = FeatureVisualize(pretrain_shared_stacked_feature, stacked_label)
        vis_pretrain_shared.plot_tsne("T-SNE/plot/pretrain_label.jpg",save_eps=False)
        # fine tuned data
        shared_stacked_feature = np.concatenate(shared_features_list, axis=0)
        vis_shared = FeatureVisualize(shared_stacked_feature, stacked_subject_id)
        vis_shared.plot_tsne("T-SNE/plot/tune_subject.jpg",save_eps=False)
        vis_shared_label = FeatureVisualize(shared_stacked_feature, stacked_label)
        vis_shared_label.plot_tsne("T-SNE/plot/tune_label.jpg",save_eps=False)
        return 0