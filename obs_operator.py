from abc import abstractmethod
import numpy as np
from load_data import LoadData     #发挥着数据池（数据库)的功能.

class Observation_Operator:
    """ project the prediction or the gradient of the prediction to the observation space.
        The "obs_operator" and the "grad_obs_operator" functions must be inherited in subclasses
    """
    def __init__(self,load_data:LoadData):
        self.load_data=load_data
        

    @abstractmethod           
    def obs_operator(self,h,mode):  
        """For y=O(h). geiven the prediction h, return the observation y."""
        raise NotImplementedError

    @abstractmethod 
    def grad_obs_operator(self,grad_h_z,mode):
        """For \nabla_h O(h) . geiven the prediction h, return the gradient of the observation y w.r.t. h."""
        raise NotImplementedError

    
class Linear_Observation_Operator(Observation_Operator):
    """ project the prediction to the observation space using a linear operator"""
    MODE=('cell_to_cell',
          '4Neighbours_to_cell',
          '8Neighbours_to_cell')    #observation operator mode.

    def __init__(self,load_data:LoadData):
        super().__init__(load_data)
        self.obs_index=self.load_data.obs_index   #the index of the observation in the prediction vector. 
        self.default_mode=self.MODE[0]            #default observation operator mode.
        #---for 4Neighbours_to_cell mode.
        self.FourNeighbours_weights=None  #shape=(5|y|,). Generate when called
        self.FourNeighbours_index=None    #shape=(5|y|,). Generate when called

        #---for 8Neighbours_to_cell mode.
        self.EightNeighbours_index=None   #shape=(9|y|,). Generate when called
        self.EightNeighbours_weights=None #shape=(9|y|,). Generate when called
    def __call__(self,h,h_grad=None,mode=None,require_grad=False):
        if require_grad:
            return self.obs_operator(h,mode),self.grad_obs_operator(h_grad,mode)
        else:
            return self.obs_operator(h,mode)


    def obs_operator(self,h,mode=None):
        """For y=O(h). geiven the prediction h, return the observation y.
            h: the prediction state vector.  shape=(|h|,) or (|h|,times),times are the observation moments. 
            mode: string or np.ndarray containing strings. Indicate the observation operator mode,see MODE.
            return: the likelihood of y.

            MODE=('cell_to_cell','4Neighbours_to_cell','8Neighbours_to_cell')  
        """
        if mode is None:
            mode=self.default_mode
        #---if the mode is string
        if isinstance(mode,str):
            if mode not in self.MODE:
                raise ValueError('Invalid observation operator mode.')

            if mode==self.MODE[0]:
                #cell_to_cell. Just slice the observation from the prediction vector.
                if len(h.shape)==1: return h[self.obs_index]   #shape=(|y|,)
                else: return h[self.obs_index,:]   #shape=(|h|,|y|,times) 
            
            elif mode==self.MODE[1]: 
                #4Neighbours_to_cell.
                return self.__4Neighbours_to_cell__(h)
            
            elif mode==self.MODE[2]: 
                #8Neighbours_to_cell.
                return self.__8Neighbours_to_cell__(h)

        #---if the mode is array like of string.  Indicating each moment has its own observation operator.
        else:
            if mode.shape!=self.obs_index.shape or not all(m in self.MODE for m in mode): 
                raise ValueError('Invalid observation operator mode.') 
            
            if len(h.shape)==1:
                y=np.zeros_like(mode)     #shape=(|y|,) 
                for i,m in enumerate(mode):
                    if m==self.MODE[0]:
                        y[i]=h[self.obs_index[i]]
                    elif m==self.MODE[1]:
                        y[i]=self.__4Neighbours_to_cell__(h[i],i)
                    elif m==self.MODE[2]:
                        y[i]=self.__8Neighbours_to_cell__(h[i],i)
                return y
            else:
                y=np.zeros((mode.shape[0],h.shape[1]))     #shape=(|y|,times)
                for i,m in enumerate(mode):
                    if m==self.MODE[0]:
                        y[i,:]= h[self.obs_index[i],:]
                    elif m==self.MODE[1]:
                        y[i,:]= self.__4Neighbours_to_cell__(h[i,:],i)
                    elif m==self.MODE[2]:
                        y[i,:]= self.__8Neighbours_to_cell__(h[i,:],i)
                return y 

    def grad_obs_operator(self,grad_h_z,mode=None):
        """For \nabla_h O(h) @ \nabla_z M(h).
            grad_h_z:  \nabla_z M(h)  shape=(|h|,|z_var|) or (|h|,|z_var|,times)
            mode: string or np.ndarray containing strings. Indicate the observation operator mode,see MODE.
            Note: this funciton fully utilizes the processing mechanism for the time dimension of the "obs_operator"
        """
        if mode is None:
            mode=self.default_mode

        if len(grad_h_z.shape)==2:   #without time dimension.
            return self.obs_operator(grad_h_z,mode)  #Utilize the processing mechanism for the time dimension of the "obs_operator"
        else:
            y_grad=np.zeros((self.obs_index.shape[0],grad_h_z.shape[1],grad_h_z.shape[2])) #shape=(|y|,|z_var|,times)
            for i in range(grad_h_z.shape[1]):
                y_grad[:,i,:]=self.obs_operator(grad_h_z[:,i,:],mode)
            return y_grad  #shape=(|y|,|z_var|,times)


    def __cell_to_cell__(self,h):
        """just for logic clearness. we don't need to add this function spending in obs_operator function."""
        pass

    def __4Neighbours_to_cell__(self,h,y_index=None):
        """Weighted information from the four neighborhoods of a cell to obtain the observation value of the cell.
        """
        if self.FourNeighbours_index is None:
            self.FourNeighbours_index,self.FourNeighbours_weights=self.load_data.get_observation_4Neighbours()  #shape=(5|y|,)
        if y_index==None:        
            if len(h.shape)==1: 
                temp=(self.FourNeighbours_weights*h[self.FourNeighbours_index]).reshape(-1,5) #shape=(|y|,5)
                return np.sum(temp,axis=1)   #shape=(|y|,)
            else: 
                temp=(np.expand_dims(self.FourNeighbours_weights,-1)*h[self.FourNeighbours_index,:]).reshape(-1,5,h.shape[1]) #shape=(|y|,5,times)
                return np.sum(temp,axis=1)   #shape=(|y|,times)
        else:
            if len(h.shape)==1: 
                temp=(self.FourNeighbours_weights[y_index*4:y_index*4+4]*h[self.FourNeighbours_index[y_index*4:y_index*4+4]]) #shape=(5,)
                return np.sum(temp)   #shape=(1,)
            else: 
                temp=(np.expand_dims(self.FourNeighbours_weights[y_index*4:y_index*4+4],-1)*h[self.FourNeighbours_index[y_index*4:y_index*4+4],:]) #shape=(5,times)
                return np.sum(temp,axis=0)   #shape=(times,)
            
    def __8Neighbours_to_cell__(self,h,y_index=None):
        """Weighted information from the eight neighborhoods of a cell to obtain the observation value of the cell"""
        if self.EightNeighbours_index is None:
            self.EightNeighbours_index,self.EightNeighbours_weights=self.load_data.get_observation_8Neighbours()  #shape=(9|y|,)
        if y_index==None:
            if len(h.shape)==1: 
                temp=(self.EightNeighbours_weights*h[self.EightNeighbours_index]).reshape(-1,9) #shape=(|y|,9)
                return np.sum(temp,axis=1)   #shape=(|y|,)
            else: 
                temp=(np.expand_dims(self.EightNeighbours_weights,-1)*h[self.EightNeighbours_index,:]).reshape(-1,9,h.shape[1]) #shape=(|y|,9,times)
                return np.sum(temp,axis=1)   #shape=(|y|,times)
        else:
            if len(h.shape)==1: 
                temp=(self.EightNeighbours_weights[y_index*8:y_index*8+8]*h[self.EightNeighbours_index[y_index*8:y_index*8+8]]) #shape=(9,)
                return np.sum(temp)   #shape=(1,)
            else: 
                temp=(np.expand_dims(self.EightNeighbours_weights[y_index*8:y_index*8+8],-1)*h[self.EightNeighbours_index[y_index*8:y_index*8+8],:]) #shape=(9,times)
                return np.sum(temp,axis=0)   #shape=(times,)

#this class is not used in this version 0.0.1 .
class Nonlinear_Observation_Operator(Observation_Operator):
    """ project the prediction to the observation space using a nonlinear operator"""
    def __init__(self,f):
        pass
    def obs_operator(self,h):
        pass

    def grad_obs_operator(self,grad_h_z):
        pass