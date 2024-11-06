from scipy.integrate import solve_ivp
from scipy.integrate._ivp import RK45
from cellfield import CellField
import numpy as np
from load_data import DATYPE_f,DATYPE_int

class OdeModel():
    """ ODE model. 
        the function is to get the h(t) and grad_h_z(t)
    """    
    METHODS = ('RK23',     #0
                'RK45',    #1
                'DOP853',  #2
                'Radau',   #3
                'BDF',     #4
                'LSODA',   #5
                'my_solve_ivp')   #6

    def __init__(self,cell_field:CellField):
        self.cell_field=cell_field
        self.ode_method=OdeModel.METHODS[1]


    def predict(self,h_0,z,t0,t_end,t_eval,
                require_grad=False):
        """ Predict the h and/or h_grad given the current state z.
            h_0: initial state
            z: latent state
            t0: initial time
            t_end: final time or end time
            t_eval: the time moment of the prediction(h)
            require_grad: whether to compute the gradient h w.r.t. z            
        """
        h,grad_h_z=None,None
        if not require_grad:
            #simultaneously compute the h and the gradient h w.r.t. z
            ode_fun=self.cell_field.ode_fun   
            z=self.cell_field.z
            if self.ode_method in ['RK45' ,'RK23', 'DOP853']:   
                sol=solve_ivp(ode_fun, [t0, t_end],h_0,args=[z],t_eval=t_eval, 
                            method=self.ode_method,rtol=1e-3,atol=1e-5,max_step=20. )  
            elif self.ode_method in ['Radau' , 'BDF' , 'LSODA']:       #need jaca_sparsity. not stabel in some cases.
                sol=solve_ivp(ode_fun, [t0, t_end],h_0,args=[z],t_eval=t_eval, 
                            method=self.ode_method,rtol=1e-3,atol=1e-5,max_step=20.,
                            jac_sparsity=self.cell_field.jacobian_sparce_matrix)
            else:                                                      #for test only.
                sol=self.my_solve_ivp(ode_fun, [t0, t_end],h_0,t_eval,args=[z],eps=1e-3,step=0.5 )
            
            h=sol.y       #has dim_h*len(t_eval),consistent with column vector conventions
        else:            #only compute the h
            #initialize the grad_h_z_init and y0  
            grad_h_z_init=self.cell_field.require_grad()    #initialize the gradient
            y0=np.concatenate((h_0,grad_h_z_init))          #initial state
            
            ode_fun_with_grad=self.cell_field.ode_fun_with_grad   

            #====================****************============================
            sol=solve_ivp(ode_fun_with_grad, [t0, t_end],y0,\
                            args=[z],t_eval=t_eval, method=self.ode_method,\
                            rtol=1e-3,atol=1e-5,max_step=20. )
            #====================****************==============================

            result=sol.y   #has shape(|h|+|h|*|z_val|,t_eval.shape[0]), column vector
            h,grad_h_z=self.cell_field.handle_result_from_ivp(result)
        return h,grad_h_z     


    def my_solve_ivp(self,fun, t_span, y0, t_eval,args,eps=1e-4,step=0.01,max_iter=10):
        """user defined ode slover """
        #给出初值问题的梯形方法 
        class solution:
            def __init__(self):
                self.y=[]
            
        sol=solution()
        t,t_end=t_span
        
        k=0
        yt=y0
        half_step=step*0.5
        while t<t_end:
            t_next=t+step
            
            #predict the next state
            f_value=fun(t,yt,args[0])
            y_next=yt+step*f_value
            if y_next.min()<0:
                # from matplotlib import pyplot as plt
                # rows=201
                # cols=483
                # fig,axs=plt.subplots(2,2)
                # axs[0,0].imshow(y_next.reshape((rows,-1)),cmap='coolwarm')
                # axs[0,1].imshow(yt.reshape((rows,-1)),cmap='coolwarm')
                # axs[1,0].imshow((f_value*step).reshape((rows,-1)),cmap='coolwarm')
                # axs[1,1].imshow(self.cell_field.load_data.dem_orginal,cmap='coolwarm')
                # plt.show()

                y_next[y_next<0.]=0.

            temp=yt+half_step*f_value

            #iterate to converge
            y_next_s=temp+half_step*fun(t_next,y_next,args[0])
            if y_next_s.min()<0:
                y_next_s[y_next_s<0.]=0.
                
            iter_count=0
            while np.max(np.abs(y_next_s-y_next))>eps and iter_count<max_iter:
                y_next=y_next_s
                y_next_s=temp+half_step*fun(t_next,y_next_s,args[0])
                # if y_next_s.min()<0:
                #     y_next_s[y_next_s<0.]=0.
                # if y_next_s.min()<0:
                iter_count+=1

            #update the time and state
            t=t_next
            yt=y_next_s

            #--report
            if t>=t_eval[k]:
                sol.y.append(yt)
                k+=1
                print("current time:",t,"time step:",step)
        sol.y=np.array(sol.y).T
        return sol
                


                




        



    





    
        


        
        
        
