import torch
import torch.nn as nn
import torch.nn.functional as F


class GradReverse(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """
    @staticmethod
    def forward(ctx, x, hp_lambda=1.0):
        ctx.hp_lambda = hp_lambda
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.hp_lambda
        return grad_output, None

    def grad_reverse(x, hp_lambda=1.0):
        return GradReverse.apply(x, hp_lambda)


class Extractor(nn.Module):

    def __init__(self):
        super(Extractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 48, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()

    def forward(self, input):
        input = input.expand(input.data.shape[0], 3, 32, 32)
        x = F.relu(F.max_pool2d(self.conv1(input), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 48 * 5 * 5)

        return x


class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(48 * 5 * 5, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, input):
        logits = F.relu(self.fc1(input))
        logits = self.fc2(F.dropout(logits))
        logits = F.relu(logits)
        logits = self.fc3(logits)

        return F.log_softmax(logits, 1)


class Discrimimator(nn.Module):

    def __init__(self):
        super(Discrimimator, self).__init__()
        self.fc1 = nn.Linear(48 * 5 * 5, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, input, hp_lambda):
        input = GradReverse.grad_reverse(input, hp_lambda)
        logits = F.relu(self.fc1(input))
        logits = F.log_softmax(self.fc2(logits), 1)

        return logits
