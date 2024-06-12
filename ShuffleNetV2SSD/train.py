import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from ShuffleNetV2SSD import ShuffleNetV2SSD, MultiBoxLoss
from Dataloader import Dataset, DataLoader
from utils import *

# 数据参数
data_folder = './'  # 数据文件夹路径
keep_difficult = True  # 是否使用被认为难以检测的对象？

# 模型参数
# 这里没有太多参数，因为SSD300具有非常特定的结构
n_classes = len(label_map)  # 不同类型对象的数量
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设备：如果有cuda则使用cuda，否则使用cpu

# 学习参数
checkpoint = None  # 模型检查点路径，如果没有则为None
batch_size = 8  # 批量大小
iterations = 12000 # 训练迭代次数
workers = 4  # DataLoader中用于加载数据的工作进程数
print_freq = 200  # 每__批次打印训练状态
lr = 1e-3  # 学习率
decay_lr_at = [8000, 10000]  # 在这些迭代后降低学习率
decay_lr_to = 0.1  # 将学习率降低到现有学习率的这个比例
momentum = 0.9  # 动量
weight_decay = 5e-4  # 权重衰减
grad_clip = None  # 如果梯度爆炸，则剪裁，这可能发生在较大的批量大小（有时在32）- 通过MuliBox损失计算中的排序错误来识别

cudnn.benchmark = True  # 在每次迭代中自动寻找最适合当前硬件的最优算法，加速训练



def main():
    """
    Training.
    """
    global start_epoch, label_map, epoch, checkpoint, decay_lr_at

    # Initialize model or load checkpoint
    if checkpoint is None:
        start_epoch = 0
        model = ShuffleNetV2SSD(n_classes=n_classes)
        # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                    lr=lr, momentum=momentum, weight_decay=weight_decay)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # Move to default device
    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

    # Custom dataloaders
    #train_dataset = parse_xml_and_adjust_images()
    '''train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here'''
    train_dataset = Dataset('300wboxes1.xml')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_dataset.collate_fn)


    # Calculate total number of epochs to train and the epochs to decay learning rate at (i.e. convert iterations to epochs)
    # To convert iterations to epochs, divide iterations by the number of iterations per epoch
    # The paper trains for 120,000 iterations with a batch size of 32, decays after 80,000 and 100,000 iterations
    epochs = iterations // (len(train_dataset) // 32)
    decay_lr_at = [it // (len(train_dataset) // 32) for it in decay_lr_at]

    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate at particular epochs
        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, decay_lr_to)

        # One epoch's training
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch)

        # Save checkpoint
        save_checkpoint(epoch, model, optimizer)
    torch.save(model.state_dict(), 'featuredetection.pth')

def train(train_loader, model, criterion, optimizer, epoch):
    """
    训练一次epoch

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()

    # Batches
    for i, (images, boxes, labels) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # Forward prop.
        predicted_locs, predicted_scores = model(images) # (N, 8732, 4), (N, 8732, n_classes)
        # Loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))
    del predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored


if __name__ == '__main__':
    main()