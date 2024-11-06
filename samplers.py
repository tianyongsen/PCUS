#This file is under the MIT license from
#https://github.com/alisiahkoohi/Langevin-dynamics/tree/master
"""
MIT License

Copyright (c) 2020 SLIM group @ Georgia Institute of Technology

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
import numpy as np

class LangevinDynamics(object):
    """
    LangevinDynamics class for performing Langevin dynamics optimization.

    Args:
        x (torch.Tensor): Initial parameter values.
        func (callable): The loss function to be optimized.
        lr (float, optional): Initial learning rate. Default is 1e-2.
        lr_final (float, optional): Final learning rate. Default is 1e-4.
        max_itr (int, optional): Maximum number of iterations. Default is 1e4.
        device (str, optional): Device to perform computations on ('cpu' or
            'cuda'). Default is 'cpu'.

    Attributes:
        x (torch.Tensor): Current parameter values.
        optim (torch.optim.Optimizer): Optimizer for updating parameters.
        lr (float): Initial learning rate.
        lr_final (float): Final learning rate.
        max_itr (int): Maximum number of iterations.
        func (callable): The loss function.
        lr_fn (callable): Learning rate decay function.
        counter (float): Iteration counter.
    """

    def __init__(self,
                 x: np.ndarray,
                #  func: callable,
                 lr: float = 1e-2,
                 lr_final: float = 1e-4,
                 max_itr: int = 10000,
                 device: str = 'cpu'):
        super(LangevinDynamics, self).__init__()
        self.x = torch.from_numpy(x)
        self.x.requires_grad = True

        self.optim = pSGLD([self.x], lr, weight_decay=0.0)
        # self.optim=SGLD([self.x],lr)   #use SGLD optimizer
        self.lr = lr
        self.lr_final = lr_final
        self.max_itr = max_itr
        # self.func = func
        self.lr_fn = self.decay_fn(lr=lr, lr_final=lr_final, max_itr=max_itr)
        self.counter = 0.0

    def sample_step(self,loss_grad:np.ndarray)->bool:
        """
        Perform a Langevin dynamics step.

        Returns:
            tuple: A tuple containing the current parameter values and the loss
                value.
        """
        self.optim.zero_grad()

        loss_grad=loss_grad.astype(np.float32)    # convert numpy array to float32
        self.x.grad=torch.from_numpy(loss_grad)   # manually set the gradient of x
        self.lr_decay()
        # loss = self.func(self.x)
        # loss.backward()
        self.optim.step()                         #x has been updated. And the cefiled
        self.counter += 1
        # return copy.deepcopy(self.x.data)         #return a copy of x
        return True

    def decay_fn(self,
                 lr: float = 1e-2,
                 lr_final: float = 1e-4,
                 max_itr: int = 10000) -> callable:
        """
        Calculate the learning rate decay function.

        Args:
            lr (float): Initial learning rate.
            lr_final (float): Final learning rate.
            max_itr (int): Maximum number of iterations.

        Returns:
            callable: Learning rate decay function.
        """
        gamma = -0.55
        b = max_itr / ((lr_final / lr)**(1 / gamma) - 1.0)
        a = lr / (b**gamma)

        def lr_fn(t: float,
                  a: float = a,
                  b: float = b,
                  gamma: float = gamma) -> float:
            """
            Calculate the learning rate based on the iteration number.

            Args:
                t (float): Current iteration number.
                a (float): Scaling factor.
                b (float): Scaling factor.
                gamma (float): Exponent factor.

            Returns:
                float: Learning rate at the given iteration.
            """
            return a * ((b + t)**gamma)

        return lr_fn

    def lr_decay(self):
        """
        Update the learning rate of the optimizer based on the current
        iteration.
        """
        for param_group in self.optim.param_groups:
            param_group['lr'] = self.lr_fn(self.counter)

"""
The class of the Optimizers
"""
import torch
from torch.optim.optimizer import Optimizer


class pSGLD(Optimizer):
    """Implements pSGLD algorithm based on https://arxiv.org/pdf/1512.07666.pdf

    Built on the PyTorch RMSprop implementation
    (https://pytorch.org/docs/stable/_modules/torch/optim/rmsprop.html#RMSprop)
    """

    def __init__(self,
                 params,
                 lr: float = 1e-2,
                 beta: float = 0.99,
                 Lambda: float = 1e-15,
                 weight_decay: float = 0,
                 centered: bool = False):
        """
        Initializes the pSGLD optimizer.

        Args:
            params (iterable): Iterable of parameters to optimize.
            lr (float, optional): Learning rate. Default is 1e-2.
            beta (float, optional): Exponential moving average coefficient.
                Default is 0.99.
            Lambda (float, optional): Epsilon value. Default is 1e-15.
            weight_decay (float, optional): Weight decay coefficient. Default
                is 0.
            centered (bool, optional): Whether to use centered gradients.
                Default is False.
        """
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= Lambda:
            raise ValueError("Invalid epsilon value: {}".format(Lambda))
        if not 0.0 <= weight_decay:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= beta:
            raise ValueError("Invalid beta value: {}".format(beta))

        defaults = dict(lr=lr,
                        beta=beta,
                        Lambda=Lambda,
                        centered=centered,
                        weight_decay=weight_decay)
        super(pSGLD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(pSGLD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('centered', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.

        Returns:
            float: Value of G (as defined in the algorithm) after the step.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'pSGLD does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['V'] = torch.zeros_like(p.data)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p.data)

                V = state['V']
                beta = group['beta']
                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                V.mul_(beta).addcmul_(grad, grad, value=1 - beta)

                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.mul_(beta).add_(1 - beta, grad)
                    G = V.addcmul(grad_avg, grad_avg,
                                  value=-1).sqrt_().add_(group['Lambda'])
                else:
                    G = V.sqrt().add_(group['Lambda'])

                p.data.addcdiv_(grad, G, value=-group['lr'])

                noise_std = 2 * group['lr'] / G
                noise_std = noise_std.sqrt()
                noise = p.data.new(p.data.size()).normal_(mean=0,
                                                          std=1) * noise_std
                p.data.add_(noise)

        return G


import torch
from torch.optim.optimizer import Optimizer, required

class SGLD(Optimizer):
    """Implements SGLD algorithm based on
        https://www.ics.uci.edu/~welling/publications/papers/stoclangevin_v6.pdf

    Built on the PyTorch SGD implementation
    (https://github.com/pytorch/pytorch/blob/v1.4.0/torch/optim/sgd.py)
    """

    def __init__(self,
                 params,
                 lr=required,
                 momentum=0,
                 dampening=0,
                 weight_decay=0,
                 nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr,
                        momentum=momentum,
                        dampening=dampening,
                        weight_decay=weight_decay,
                        nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                "Nesterov momentum requires a momentum and zero dampening")
        super(SGLD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGLD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(
                            d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(d_p, alpha=-group['lr'])
                # noise_std = torch.tensor([2 * group['lr']])
                noise_std = torch.tensor([2 * group['lr']])          #add
                noise_std = noise_std.sqrt()
                noise = p.data.new(p.data.size()).normal_(mean=0,
                                                          std=1) * noise_std
                p.data.add_(noise)

        return 1.0

