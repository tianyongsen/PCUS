import numpy as np
class Loss:
    LOSS_TYPE={0: "likelihood: i.i.d. Gaussian, prior: i.i.d. Gaussian",
               1: "likelihood: multivariate Gaussian, prior: i.i.d. Gaussian",
               2: "likelihood: multivariate Gaussian, prior: multivariate Gaussian",
               #case 3: the loss class just need the covariance value (float like). 
               #        Because the prior is uniform distribution.
               3: "likelihood: i.i.d. Gaussian, prior: uniform distribution",     
               4: "likelihood: i.i.d. Gaussian, prior: 1_norm regularization",
               5: "likelihood: multivariate Gaussian, prior: 1_norm regularization",
               6: "likelihood: Cauchy distribution,   prior: i.i.d. Gaussian",
               7: "likelihood: i.i.d. Gaussian,       prior: 2_norm regularization",
               8: "likelihood: i.i.d. Gaussian,       prior: 1_norm regularization and TV regularization",
               #6:....According to the requirement, more loss functions can be added here.
               }
    def __init__(self,loss_type,**params):
        if loss_type not in Loss.LOSS_TYPE:
            raise ValueError("loss_type should be in {}".format(Loss.LOSS_TYPE))
        self.loss_result_list=[]    #Add a result every time the criterion() is called
        self.backward_epoch=0       #Add a epoch number every time the backward() is called

        #************important attributes************
        self.grad=None  #gradient of the loss function w.r.t. z
        #********************************************
        

        #--------loss type and parameters----------
        self.loss_type=loss_type
        #gamma is for likelihood, sigma is for prior.
        if loss_type==0:
            self.gamma=params['Gaussian_Gamma']  #covariance value
            self.sigma=params['Gaussian_Sigma']  #coavariance value
            self.z_mean=params['z_mean']         #Gaussian mean 
        if loss_type==3: 
            self.gamma=params['Gaussian_Gamma']  #covariance value
        if loss_type==1:
            self.Gamma=params['Gaussian_Gamma']  #covariance matrix
            self.precision_Gamma=np.linalg.inv(self.Gamma)  #precision matrix
            self.sigma=params['Gaussian_Sigma']  #coavariance value
            self.z_mean=params['z_mean']         #Gaussian mean
        if loss_type==2:
            self.Gamma=params['Gaussian_Gamma']  #covariance matrix
            self.precision_Gamma=np.linalg.inv(self.Gamma)  #precision matrix
            self.Sigma=params['Gaussian_Sigma']  #coavariance value
            self.precision_Sigma=np.linalg.inv(self.Gamma)  #precision matrix
            self.z_mean=params['z_mean']         #Gaussian mean

        if loss_type==4:   #norm 1 regularization
            self.gamma=params['Gaussian_Gamma']  #covariance value
            self.scheme=['sign','huber','proximal']  #the handling scheme of the norm 1 regularization
            self.sel_scheme='huber'                                     #the selected handling scheme of the norm 1 regularization
            self.Continuity_strategy=True     #the Continuity strategy for LASSO problem, see __mu_scheduler__()
           
            #for Continuity_strategy
            self.mu_up=100.                      #mu,see __mu_scheduler__()
            self.mu_low=0.01                      #mu,see __mu_scheduler__()
            self.factor=0.1                        #mu,see __mu_scheduler__()
            self.mu_t=self.mu_up                   #mu,see __mu_scheduler__()
            
            #for huber
            self.delta=0.1 if self.sel_scheme=='huber' else None  #delta,see __prior_grad__()
            
            #for proximal
        if loss_type==7:   #norm 2 regularization
            self.gamma=params['Gaussian_Gamma']  #covariance value
            #for Continuity_strategy
            self.mu_up=100.                      #mu,see __mu_scheduler__()
            self.mu_low=0.01                      #mu,see __mu_scheduler__()
            self.factor=0.1                        #mu,see __mu_scheduler__()
            self.mu_t=self.mu_up                   #mu,see __mu_scheduler__()
        if loss_type==8:   #1_norm regularization and  TV regularization
            self.gamma=params['Gaussian_Gamma']  #covariance value
            self.Norm1_index=params['Norm1_index']  #the variable index in z of NORM1 regularization
            self.TV_index=params['TV_index']        #the variable index in z of TV regularization
            self.mu_up=100.                      #mu,see __mu_scheduler__()
            self.mu_low=0.01                      #mu,see __mu_scheduler__()
            self.factor=0.1                        #mu,see __mu_scheduler__()
            self.mu_t=self.mu_up                   #mu,see __mu_scheduler__()

        #if loss_type==...:

    def __likelihood_loss__(self,y_pred,y_obs):
        """calculate the likelihood loss of y_pred given y_obs"""
        dy=y_pred-y_obs
        if self.loss_type==0 or self.loss_type==3 or self.loss_type==4 or \
            self.loss_type==7 or self.loss_type==8: #i.i.d Gaussion
            return 0.5*np.sum(dy*dy)/self.gamma   
        if self.loss_type==1 or self.loss_type==2:                      #multivariate Gaussion
            return 0.5*np.sum(dy* (self.precision_Gamma@ dy)) # 1/2 \sum_t{(y_pred_t-y_obs_t)^T Gamma^{-1}(y_pred_t-y_obs_t)}

    def __prior_loss__(self,z):   
        """calculate the prior loss of the model"""
        if self.loss_type==3:                       #prior is uniform distribution
            return 0.  #prior is uniform distribution,so the prior loss is 0, because we have on knowledge about the z.
        if self.loss_type==0 or self.loss_type==1:  #i.i.d Gaussion
            return 0.5*np.sum((z-self.z_mean)**2)/self.sigma
        if self.loss_type==2:                      #multivariate Gaussion
            return 0.5*np.sum((z-self.z_mean)*self.precision_Sigma @ (z-self.z_mean))
        if self.loss_type==4:                      #norm 1 regularization
            return np.sum(np.abs(z))
        if self.loss_type==7:                      #norm 2 regularization
            return np.sum(z**2)
        if self.loss_type==8:                      #norm1 and TV regularization
            tv_a=z[self.TV_index].copy()        
            tv_b=tv_a
            tv_b[:-1]=tv_a[1:]
            tv_b[-1]=tv_a[0]
            return np.sum(np.abs(z[self.Norm1_index]))+np.sum(np.abs(tv_a-tv_b))
        

    
    def criterion(self,y_pred,y_obs,z=None):
        """calculate the loss between y_pred and y_obs.
            y_pred: the predicted value of y.       dim=|obs_states|*times
            y_obs: the observed value of y.         dim=|obs_states|*times
            z: the latent variable of the model.    dim=|latent_states|
            z_mean: the mean of the prior distribution of z. If prior is uniform distribution or some regularization, z_mean is None.
        """
        loss=self.__likelihood_loss__(y_pred,y_obs)+self.__prior_loss__(z)
        self.loss_result_list.append(loss)
        
        return loss
    
    def backward(self,y_pred,y_obs,z,y_pred_grad):
        """calculate the gradient of the loss function w.r.t. z"""
        self.backward_epoch+=1
        likelihood_grad=self.__likelihood_grad__(y_pred,y_obs,y_pred_grad) #compute the likelihood gradient w.r.t. z
        prior_grad=self.__prior_grad__(z)   #compute the prior gradient w.r.t. z
        self.grad=likelihood_grad+self.__mu_scheduler__()*prior_grad  

    def __likelihood_grad__(self,y_pred,y_obs,y_pred_grad):
        """calculate the gradient of the likelihood loss
            y_pred: the predicted value of y.  shape=(|obs_states|,times)
            y_obs: the observed value of y.    shape=(|obs_states|,times)
            y_pred_grad: the gradient of y_pred w.r.t. z. shape=(|obs_states|,|z_var|,times)
        """
        times=y_pred.shape[1]
        dy=y_pred-y_obs    #max=6 min=-8
        grad_times=np.zeros((y_pred_grad.shape[1],times))  #shape=(|z_var|,times)

        if self.loss_type==0 or self.loss_type==3 or self.loss_type==4 or\
            self.loss_type==7  or self.loss_type==8: #i.i.d Gaussion
            for t in range(times):
                grad_times[:,t]=y_pred_grad[:,:,t].T @ dy[:,t]
            return 0.5*np.sum(grad_times,axis=1)/self.gamma    #sum time axis   
        
        if self.loss_type==1 or self.loss_type==2: #multivariate Gaussion
            for t in range(times):
                grad_times[:,t]=y_pred_grad[:,:,t].T @ (self.precision_Gamma@ dy[:,t])
            return 0.5*np.sum(grad_times,axis=1) # 1/2 \sum_t{(y_pred_t-y_obs_t)^T Gamma^{-1}(y_pred_t-y_obs_t)}
        
    
    def __prior_grad__(self,z):
        """calculate the gradient of the prior loss of the model"""
        if self.loss_type==3:                       #prior is uniform distribution
            return 0.  #prior is uniform distribution,so the prior loss gradient is 0, because we have on knowledge about the z.
        if self.loss_type==0 or self.loss_type==1:  #i.i.d Gaussion
            return (z-self.z_mean)/self.sigma
        if self.loss_type==2:                      #multivariate Gaussion
            return self.precision_Sigma@(z-self.z_mean)
        if self.loss_type==4:                      #norm 1 regularization
            if self.sel_scheme=='sign':
            #策略1：直接取符号函数
                Norm1_grad=np.sign(z)   
            elif self.sel_scheme=='huber':
                #策略2：LASSO 问题的Huber光滑化梯度方法，see: http://faculty.bicmr.pku.edu.cn/~wenzw/optbook/pages/lasso_grad/LASSO_grad_huber_inn.html
                # delta=0.1 #default
                Norm1_grad=np.sign(z)   
                Huber_index=np.abs(z)<=self.delta
                Norm1_grad[Huber_index]=z[Huber_index]/self.delta
            #elif ...:
            return Norm1_grad
        if self.loss_type==7:                      #norm 2 regularization
            return 2*z
        if self.loss_type==8:                      #norm1 and TV regularization
            grad=np.zeros(z.shape)
            grad[self.Norm1_index]=np.sign(z[self.Norm1_index])  #norm 1 regularization
            #TV regularization
            tv_a=z[self.TV_index].copy()        
            tv_b=tv_a.copy()
            tv_b[:-1]=tv_a[1:]
            tv_b[-1]=tv_a[0]

            tv_sign_a=np.sign(tv_a-tv_b)
            tv_sign_b=tv_sign_a.copy()
            tv_sign_b[1:]=tv_sign_a[0:-1]   #turn
            tv_sign_b[0]=tv_sign_a[-1]    
            grad[self.TV_index]=tv_sign_a-tv_sign_b  #TV regularization
            return grad         


    def __mu_scheduler__(self):
        """In regularization type, schedule the prior lambda,see
           http://faculty.bicmr.pku.edu.cn/~wenzw/optbook/pages/LASSO_con/LASSO_con.html
           Note: There are some differences between ours and the link above.
           If in non-regularization type, the mu_scheduler() is not used.
        """
        if self.loss_type!=4 and self.loss_type!=7 and self.loss_type!=8:
            return 1.
        
        self.mu_t = max(self.mu_t*self.factor, self.mu_low)
        return self.mu_t 

    def convergence(self):
        """
        Stopping criterion or convergence criterion.
        """
        #loss convergence criterion
        if len(self.loss_result_list)>2:
            loss_relative_error=np.abs(self.loss_result_list[-1]-self.loss_result_list[-2])/self.loss_result_list[-2]
            print("loss_relative_error:",loss_relative_error)
            if loss_relative_error<1e-5: 
                print("loss convergence")
                return True
        #gradient convergence criterion
        grad_norm=np.linalg.norm(self.grad,ord=2)
        print("grad_norm:",grad_norm)
        if grad_norm<1e-5:
            print("gradient convergence")
            return True
        return False


    