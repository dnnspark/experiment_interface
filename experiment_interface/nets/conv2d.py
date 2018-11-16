from torch import nn

def compute_eff_kernel_size(kernel_size, dilation):
    return dilation * (kernel_size-1) + 1

def compute_padding_size(kernel_size, dilation):
    K_eff = compute_eff_kernel_size(kernel_size, dilation)
    return (K_eff-1) // 2

class Conv2D(nn.Module):

    def __init__(self,
        in_channel, out_channel, kernel_size, stride=1, dilation=1, 
        padding=None, use_bias=True, use_batchnorm=False, act_fn=None, 
        ):

        super().__init__()

        if use_batchnorm:
            use_bias = False
        self.use_batchnorm = use_batchnorm

        if kernel_size % 2 == 0 and padding is None:
            raise ValueError('conv2d only accepts odd kernel size, if padding is undetermined.')

        if padding is None:
            padding = compute_padding_size(kernel_size, dilation)

        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, dilation, bias=use_bias) 
        if use_batchnorm:
            self.batchnorm = nn.BatchNorm2d(out_channel)

        if act_fn is not None:
            if act_fn == 'relu':
                self.act_fn = nn.ReLU(inplace=True) 
            elif act_fn == 'tanh':
                self.act_fn = nn.Tanh()
            elif act_fn == 'sigmoid':
                self.act_fn = nn.Sigmoid() 
            else:
                raise ValueError('%s activation fn is not supported.' % act_fn)
        self.use_act_fn = act_fn is not None

    def forward(self, x):
        x = self.conv(x)
        if self.use_batchnorm:
            x = self.batchnorm(x)
        if self.use_act_fn:
            x = self.act_fn(x)

        return x



