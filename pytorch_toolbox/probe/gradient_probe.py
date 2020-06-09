'''
Simple way to monitor/manipulate gradient.

Usage:
  gradient_record = GradientRecordHook(name='record')
  gradient_scale = GradientScale(name='scale')
  def net.forward(input):  # your forward function
    fc = base_layers(input)
    # Check gradient:
    fc = gradient_record(fc)
    # invert gradient (e.g. domain adapt.)
    fc = gradient_scale(fc, -1)
    output = estimator_layers(fc)
  Afterwards, we can plot these records to check the gradient.  
'''
import torch


class GradientScaleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = lambda_ * grads
        return dx, None


class GradientScale(torch.nn.Module):
    def __init__(self, name=None):
        super(GradientScale, self).__init__()
        self.name = name
        self.lambdar = 0

    def forward(self, x, lambdar):
        self.lambdar = lambdar
        return GradientScaleFunction.apply(x, lambdar)


class GradientRecordHook(torch.nn.Module):
    '''
    Simple way to record gradient
    '''

    def __init__(self, name=None):
        super(GradientRecordHook, self).__init__()
        self.lambdar = 0
        self.gradients = []
        self.mag = None
        self.std = None
        self.name = name

    def hook_fun(self, grad):
        self.mag = torch.mean(torch.abs(grad)).item()
        self.std = torch.std(grad).item()

    def forward(self, x):
        '''
        Do Nothing
        '''
        if self.training:
            x.register_hook(self.hook_fun)
        return x

