# -*- coding: utf-8 -*-
import torch
import numpy as np

class Optimizer:  
    OPT_METHODS=["adagrad",
                 "rmsprop",
                 "nesterov",
                 "sgd","adam",
                 "adadelta"]
    def __init__(self,z_var,opt_method="adgrad",lr=0.1):
        """Initialize the optimizer with the given parameters.
            z_var: the variable to be optimized. np.array.
            opt_method: the optimization method to be used. string.
            lr: the learning rate. float.\n
            OPT_METHODS=["adgrad","rmsprop","nesterov","sgd","adam","adadelta"]
        """
        self.z_var=torch.from_numpy(z_var)  # convert numpy array to tensor
        self.z_var.requires_grad=True       # set requires_grad to True for z_var

        if opt_method not in self.OPT_METHODS:
            raise ValueError("Invalid optimization method.")
        
        if opt_method=="adagrad":
            self.optimizer=torch.optim.Adagrad([self.z_var], lr=lr)
        elif opt_method=="rmsprop":
            self.optimizer=torch.optim.RMSprop([self.z_var], lr=lr,momentum=0.1)
        elif opt_method=="nesterov":
            self.optimizer=torch.optim.nesterov([self.z_var],lr=lr)
        elif opt_method=="sgd":
            self.optimizer=torch.optim.SGD([self.z_var], lr=lr)
        elif opt_method=="adam":
            self.optimizer=torch.optim.Adam([self.z_var], lr=lr)
        elif opt_method=="adadelta":
            self.optimizer=torch.optim.Adadelta([self.z_var], lr=lr)
        else:
            raise ValueError("Please check the optimization method.")

    def step(self,loss_grad):
        """Perform one optimization step with the given loss gradient.
            loss_grad: the gradient of the loss function w.r.t. z_var. np.array.
        """
        loss_grad=loss_grad.astype(np.float32)  # convert numpy array to float32
        self.optimizer.zero_grad()  # clear the gradient buffer
        self.z_var.grad=torch.from_numpy(loss_grad)   # manually set the gradient of z_var

        # perform one optimization step.Beacause the z_var(tensor) and z_var(numpy array) have the same memory address, 
        # so the z_var(numpy) will be automatically updated in the optimizer.
        self.optimizer.step() 



