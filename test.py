import torch
from torch.autograd import Variable

def testDMMR(dataLoader, DMMRTestModel, cuda, batch_size):
    print("testing DMMR")
    index = 0
    count = 0
    data_set_all = 0
    if cuda:
        DMMRTestModel = DMMRTestModel.cuda()
    DMMRTestModel.eval()
    with torch.no_grad():
        for _, (test_input, label) in enumerate(dataLoader):
            if cuda:
                test_input, label = test_input.cuda(), label.cuda()
            test_input, label = Variable(test_input), Variable(label)
            data_set_all += len(label)
            x_shared_pred = DMMRTestModel(test_input)
            _, pred = torch.max(x_shared_pred, dim=1)
            count += pred.eq(label.squeeze().data.view_as(pred)).sum()
            index += batch_size
    acc = float(count) / data_set_all
    return acc
