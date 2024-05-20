from torch.autograd import Function

#gradient reversal
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, m):
        ctx.m = m
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.m
        return output, None