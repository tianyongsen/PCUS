import numpy as np
from scipy.interpolate import interp1d
from scipy.sparse import csc_matrix


from load_data import LoadData,DATYPE_f,DATYPE_int
#unit: mm and sec

class CellField:
    def __init__(self,load_data:LoadData):
        self.load_data=load_data
        self.h0=load_data.init_h
        self.v_dim=self.h0.shape[0]       
        self.z=load_data.z    
        self.z_var_index=load_data.z_var_index

        #***********z_var. the optimization variable of the latent variables.***********
        self.z_var=self.z[self.z_var_index].copy() 
        #===============================================================================     

        self.rain_obs=Rain(load_data.rain_map,load_data.rains)
        self.infil_obs=Infiltration(load_data.infil) if load_data.infil is not None else None
        
        runoff_input=[self.v_dim,load_data.dem_vector,load_data.cell_length,load_data.A,load_data.z,load_data.link_n,
                        load_data.cell_link,load_data.cell_link_dirc,
                        load_data.hi_index1,load_data.hi_index2,load_data.hj_index1,
                        load_data.hj_index2,load_data.adjacency_vector,load_data.z_var_index]
        self.runoff_obs=Runoff(*runoff_input)

        self.jacobian_sparce_matrix=\
                self.__jacobian_sparse_matrix(load_data.adjacency_vector,load_data.dem_orginal.shape)
        
        self._require_grad=False   

        #for as_one and localization
        self.as_one=False
        self.z_var_as_one=None
        self.localization_flag=False
        self.buffer_cells_index=None 
        self.buffer_z_index=None
    def get_opt_variable(self):
        if self.as_one:
            return self.z_var_as_one
        else:
            return self.z_var 
    def constraint_opt_variable(self):
        #--optimize:Projection operator under constraint optimization. The constraint is z>=0.
        if self.as_one:
            if self.z_var_as_one<0.:
                self.z_var_as_one=0.   
        else:
            self.z_var[self.z_var<0.]=0.
        
    def step(self):
        """将潜在更新的z_var值更新z"""
        if self.as_one:
            self.z_var[:]=self.z_var_as_one[0]     #self.z_var_as_one's type is np.array([num]).
            self.z[self.z_var_index]=self.z_var
        else:
            self.z[self.z_var_index]=self.z_var    
        
    def require_grad(self):   
        """ set the require_grad flag to be True and init some variables"""
        if self._require_grad!=True:
            self._require_grad=True
            self.runoff_obs.require_grad() #Create new data
        #infiltation.require_grad()
        #evaporation.require_grad()
        #boundary_condition.require_grad()

        #======set the initial gradient state=========
        if self.as_one:
            if self.localization_flag:
                grad_h_z_init=np.zeros(self.buffer_cells_index.shape[0],dtype=DATYPE_f)
            else:
                grad_h_z_init=np.zeros(self.v_dim,dtype=DATYPE_f)
        else:
            grad_h_z_init=np.zeros(self.v_dim*self.z_var_index.shape[0],dtype=DATYPE_f) #initial gradient state=0.
        return grad_h_z_init        
        
    def handle_result_from_ivp(self,result):
        h=result[:self.v_dim,:]
        t_times=result.shape[1]
        if self.as_one:
            if self.localization_flag:
                grad_h_z_buffer=result[self.v_dim:,:]
                grad_h_z=np.zeros_like(h) 
                grad_h_z[self.buffer_cells_index,:]=grad_h_z_buffer
                grad_h_z=grad_h_z.reshape(self.v_dim,-1,t_times)      #Upgrade dimension
            else:
                grad_h_z=result[self.v_dim:,:].reshape(self.v_dim,-1,t_times)
        else:
            grad_h_z=result[self.v_dim:,:].reshape(self.v_dim,-1,t_times)
        return h,grad_h_z

    def set_as_one(self,as_one=False,**paras):
        """        
        as_one: whether to see the all latent vars as one variable
                if as_one=True,the paras can be:
                    localization: whether to use the localization method. Just for as_one=True now.
                    buffer_type: ['rectangular','square',circular','buffer_file']
                    buffer_size: the buffer size for the localization method. 
                        'rectangular': [width,height], must be even numbers.
                        'square': [size], must be even numbers.    #is engough for now.
                        'circular': [radius].  #Not implemented yet.
                        'buffer_file': the path of the buffer file. #not implemented yet.
                    local_boudanry: float (1.>=*>=0) 
                        0.: zero value boudanry.
                        1.: equal value boudanry.
                        other values: fraction value boudanry.
        """
        self.as_one=as_one  
        if as_one:     
            #***********z_var. the optimization variable of the latent variables in as_one mode.***********
            self.z_var_as_one=np.array([self.z_var[0]])  # think all the z_var are the same in defalut.
            #==============================================================================================
            
            if paras['localization']:
                if 'buffer_type' not in paras or 'buffer_size' not in paras:
                    raise ValueError("buffer_type and buffer_size must be provided for localization method.")
                self.set_localization(paras['buffer_type'],paras['buffer_size'])
                self.runoff_obs.set_as_one(as_one=True,localization=True,buffer_cells_index=self.buffer_cells_index,
                                            buffer_z_index=self.buffer_z_index)
            else: 
                self.runoff_obs.set_as_one(as_one=True)
                #self.infiltation.set_as_one(True,buffer_cells_index=self.buffer_cells_index)
                #self.evaporation.set_as_one(True,buffer_cells_index=self.buffer_cells_index)
                #self.boundary_condition.set_as_one(True,buffer_cells_index=self.buffer_cells_index)

    def set_localization(self,buffer_type,buffer_size_or_file):
        """
        yeild the localization.
        buffer_type: ['rectangular','square',circular','buffer_file']
        buffer_size_or_file: the buffer information for the localization method. 
            'rectangular': [width,height], .
            'square': half of length, i.e. the cell numbers up or left of the z_var    #is engough for now.
            'circular': radius.  #Not implemented yet.
            'buffer_file': the path of the buffer file. #not implemented yet.
        resutl: the number of buffer cells and the buffer cells index array in the cellfield。
        Note: the funciton can be accelerated, but it is not necessary for now.  
        """
        # self.hi_index=np.concatenate((self.load_data.hi_index1,self.load_data.hi_index2),axis=0)
        # self.hj_index=np.concatenate((self.load_data.hj_index1,self.load_data.hj_index2),axis=0)
        hi_index1=self.load_data.hi_index1
        hi_index2=self.load_data.hi_index2
        buffer_cells=np.zeros(self.v_dim,dtype=DATYPE_int)
        dem=self.load_data.dem_orginal
        to_dem_index=lambda x:(x//dem.shape[1],x%dem.shape[1])
        to_vector_index=lambda x: x[0]*dem.shape[1]+x[1]
        def generate_buffer_by_corner(corner,rows,cols):
            x0,y0=corner
            index_pairs = [(i, j) for i in range(rows) for j in range(cols)]
            return [(x0+i,y0+j) for i,j in index_pairs]   
        if buffer_type=='rectangular':
            pass
        elif buffer_type=='square':
            #just for dem which has rectangular shape for now. Other module can be for inhomogeneous shape.
            half_num=buffer_size_or_file
            ordinary_condition=lambda row,col:row-half_num+1>0 and row+half_num<dem.shape[0] \
                    and col-half_num+1>0 and col+half_num<dem.shape[1]
            in_condition=lambda row,col:row>=0 and row<dem.shape[0] and col>=0 and col<dem.shape[1] 
            for k in self.z_var_index:
                if k< hi_index1.shape[0]:     #vertical direction 
                    cell=hi_index1[k]  #the left cell of the z edge
                else:                         #horizontal direction
                    cell=hi_index2[k-hi_index1.shape[0]]  #the upper cell of the z edge
                row,col=to_dem_index(cell)
                corner=(row-half_num+1,col-half_num+1)
                buffer_cells_dem=generate_buffer_by_corner(corner,2*half_num,2*half_num)
                if ordinary_condition(row,col):   #ordinary case.
                    buffer_cells_index=np.array([to_vector_index(t) for t in buffer_cells_dem])
                else:    #special case special handling.
                    buffer_cells_dem=[ t for t in buffer_cells_dem if in_condition(t[0],t[1])]
                    buffer_cells_index=np.array([to_vector_index(t) for t in buffer_cells_dem])
                buffer_cells[buffer_cells_index]=1    #label the buffer cells as 1.
        elif buffer_type=='circular':
            pass 
        elif buffer_type=='buffer_file':
            pass
        else:
            raise ValueError("buffer_type must be in ['rectangular','square',circular','buffer_file']")
        #--get the buffer_cells_index
        self.buffer_cells_index=np.where(buffer_cells==1)[0]
        
        #--for bufffer_z_index
        cell_link=self.load_data.cell_link.reshape(self.v_dim,-1)
        cell_link_dirc=self.load_data.cell_link_dirc.reshape(self.v_dim,-1)
        links=cell_link[self.buffer_cells_index]            #shape=|buffer_cells_index|*4
        links_dirc=cell_link_dirc[self.buffer_cells_index]  
        abandon_index=np.where(links_dirc!=0.)              #shape=(*,)
        links=links[abandon_index]                      #shape=(*,)
        links=np.unique(links)                              #duplicate removal
        #--get the buffer_z_index
        self.buffer_z_index=links  
        #--set the localization_flag, and indicate that the localization will be used.
        self.localization_flag=True   
        return 
    
    def ode_fun(self,t,h,z):  
        """ The right-hand side of the ODE system,dh/dt=f(t,h,z). 
            The funcition f is a Parametric equation,so it has put the z at the important role.
            t: time
            h: the state variable
            z: latent variables
        """
        #这部分是和RungeKutta算法耦合的，也与ivp求解器算法是耦合的。
        # ode_fun在ivp模块的调用见 rk.py中的rk_step()函数.
        # 因为调用ode_fun时，引用传递h，故可以在这里修改h值，以满足非负性条件。
        # 调制准则：如果h_i<0,则令h_i=0.; 
        #           若h_i接近于0且总通量为负值，则令总通量为0.
        # 要结合RungeKutta算法,理解这样调制的原因。
        # h[h<0.]=0.
        
        f_value=self.runoff(t,h,z)+self.precipitation(t)-self.evaporation(t,h)-\
                self.infiltration(t,h)+self.boundary_condition(t,h)      
 
        # epsilon=2 # 2mm the threshold for the manunally setting  
        # index=np.where(h<epsilon)
        # f_value[index][f_value[index]<0.]=0.
        return f_value
    
    def ode_fun_with_grad(self,t,y,z):
        """ The right-hand side of the ODE system of the gradient of h w.r.t. z,
            d(\nabla_z h)/dt=f(t,\nabla_z h,h,z).
            t: time
            h:the state variable
            z:latent variables
        """
        f_value=self.runoff_with_grad(t,y,z)+self.infiltration_with_grad(t,y,z)
        +self.boundary_condition_with_grad(t,y,z) 
        f_value[:self.v_dim]+=self.precipitation(t)-self.evaporation(t,y)
        return f_value

    def precipitation(self, t):
        """the function of Precipitation"""
        return self.rain_obs(t)
    def infiltration(self, t,h):
        """the function of Infiltration"""
        return self.infil_obs(t,h) if self.infil_obs is not None else 0.
    
    def evaporation(self, t,h):
        """the function of Evaporation.No evaporation in this version"""
        return 0.
    def boundary_condition(self,t,h):
        """the function of boundary condition"""
        return 0.  #no boundary condition in this version
    
    def runoff(self,t,h,z):
        """ the function of Runoff """
        return self.runoff_obs(t,h,z)   #has dim_h

    #--for gradient calculation--
    def runoff_with_grad(self,t,y,z):
        """ the gradient of Runoff w.r.t. z """
        return self.runoff_obs.runoff_adjonit_grad(t,y,z)   #has dim_h*dim_z
    
    def infiltration_with_grad(self,t,h,z):
        """ the gradient of Infiltration w.r.t. z """
        return 0.   #has dim_h*dim_z
    
    def evaporation_with_grad(self,t,h,z):
        """ the gradient of Evaporation w.r.t. z """
        return 0.   #has dim_h*dim_z  
    
    def boundary_condition_with_grad(self,t,h,z):
        """ the gradient of the boundary condition w.r.t. z """
        return 0.   #has dim_h*dim_z

    def __jacobian_sparse_matrix(self,adjacency_vector,dem_shape):
        """construct the sparse link matrix"""
        adjacency_vector=adjacency_vector.reshape((self.v_dim,-1))
        adjacency_vector[adjacency_vector==0]=-1
        adjacency_vector[1,2]=0    # right dowm  left up 
        adjacency_vector[dem_shape[1],3]=0
        my2my=np.arange(adjacency_vector.shape[0])
        adjacency_vector=np.column_stack((adjacency_vector,my2my))

        #transform the adjacency_vector to a CSR sparse matrix
        index=np.where(adjacency_vector!=-1)
        row=index[0]
        col=adjacency_vector[index]
        data=np.zeros(len(row))
        s=csc_matrix((data,(row,col)),shape=(self.v_dim,self.v_dim)) #Compressed Sparse Row matrix.
        # print(s.shape)
        return s
    

#from scipy.interpolate import interp1d
class Rain:
    class Chicago_Design_Storm:
        """
        Given the Intensity-Duration-Frequency(IDF):i=a/(T+b)^n ang the Peaking-Time-Ratio r,
        We can get the Chicago_Design_Storm using the following formula: 
            i_t= a*( (t_p - t)/r * (1-n)+b) / (t_p-t)/r +b)^{n+1}  , if t<t_p 
            i_t= a*( (t - t_p)/(1-r) * (1-n) + b) / (t-t_p)/(1-r) +b)^{n+1}  , if t>t_p 
            where t_p=r*T is the peak time.
        Args:
            see the above formula.
            there is a example of Beijing 2 district storm using the following IDF: 
                Take the IDF as i=2001*(1+0.811* lg10)/ (167*(t+8)^0.711)   mm/min
                then a,b and n can be get.
                The T and r is user-defined, for example, T=30min, r=0.5 
        return:
            the intensity of the Chicago_Design_Storm at time t.
        Refeerences:
            https://www.bilibili.com/read/cv26311056/ 
            https://docs.bentley.com/LiveContent/web/Bentley%20StormCAD%20SS5-v1/en/GUID-A0096667-D870-426C-A77A-233BE0A41A32.html
            or more academic references.
            https://doi.org/10.3390/ijerph20054245
            Chen, J.; Li, Y.; Zhang, C. The Effect of Design Rainfall Patterns on Urban Flooding Based on the Chicago Method. Int. J. Environ. Res. Public Health 2023, 20, 4245.
        """
        def __init__(self,a,b,n,T,r):
            self.a=a
            self.b=b
            self.n=n
            self.T=T       # unit: min
            self.r=r       
            self.tp=r*T
        def __call__(self,t):
            """ return the intensity of the Chicago_Design_Storm at time t."""
            if t<self.tp:
                return self.__general_formula__((self.tp-t)/self.r)    #unit mm/min
            else:
                return self.__general_formula__((t-self.tp)/(1-self.r)) #unit mm/min
            
        def __general_formula__(self,t):
            return self.a*( t* (1-self.n)+self.b) / np.power(t +self.b,self.n+1) 
        
        def Accumulated_rainfall(self):
            from scipy.integrate import quad
            result,error=quad(self.__call__,0,self.T)  
            return result   #unit: mm
    
    def __init__(self, rain_map, rains):
        """
        rain_map: np.ndarray |h|*1  
        rains:{index: {time:t,intensity:I}}   unit:mm/min.  
       """
        self.rain_map = rain_map
        self.rains=rains  
        self.Intensity=[]  #Intensity function for each rain
        self.mode=None    #mode of the rain field, "all" or "map"
        # self.curr_p=None  #current precipitation
        # self.acc_p=None   #accumulated precipitation
    
        if rain_map.shape[0]==len(rains):
            self.mode="all"   #this mode means that every cell has its own rain
            self.Intensity= [interp1d(v["time or symbol"], v["intensity"], kind='linear',fill_value="extrapolate")
                        for _,v in self.rains]
            #if need more precise interpolation, use cubic instead of linear interpolation
            #self.Intensity = [interp1d(v["time"], v["intensity"], kind='cubic',fill_value="extrapolate")
            #                for _,v in self.rains]
        else:
            self.mode="map"   #this mode means that there are just a few rains
            unique = np.unique(rain_map)
            #filter out the rains that are not in the rain_map
            self.rains={index:self.rains[index] for index in unique}
            for v in self.rains.values():
                #if have Chicago_Design_Storm, create the object directly
                # if v["time or symbol"]==['a','b','n','T','r']: 
                if v["time or symbol"][0]=='a':
                    self.Intensity.append(Rain.Chicago_Design_Storm(*v["intensity"]))
                else:  
                    #create interpolation functions for each rain
                    self.Intensity.append(interp1d(v["time or symbol"], v["intensity"], kind='linear',fill_value="extrapolate"))                       
            #record the index of each rain in the rain_map
            self.index=[np.where(rain_map == ele) for ele in unique]
    def __call__(self, t):
        return self.precipitation(t)
    
    def precipitation(self, t):
        """t: unit=s"""
        t=t/60.                             #unit: s->min
        if self.mode=="all":
            return self.__all_mode__(t)/60. #unit: mm/min->mm/s
        elif self.mode=="map":
            return self.__map_mode__(t)/60. #unit: mm/min->mm/s
        
    def __all_mode__(self, t):
        return np.array([i(t) for i in self.Intensity]) 
    
    def __map_mode__(self, t):
        curr_p=np.zeros(self.rain_map.shape)
        p_s=np.array([i(t) for i in self.Intensity])
        for id_arr,p in zip(self.index,p_s):
            curr_p[id_arr]=p
        return curr_p
            
class Infiltration:
    
    def __init__(self,infil,method=0):
        """
        infil: np.ndarray dim=|paras|*|h| or |paras|*|dem[0]*dem[1]|
        this version only supports the Horton method
       """
        self.METHOD={0: "HORTON",                      #Horton infiltration
                     1:"MOD_HORTON",                   # Modified Horton infiltration
                     2:"GREEN_AMPT",                   # Green-Ampt infiltration
                     3:"MOD_GREEN_AMPT",               # Modified Green-Ampt infiltration
                     4:"CURVE_NUMBER"};                # SCS Curve Number infiltration
        self.method=method
        self.infil = infil
        self.f0=infil[0]
        self.fmin=infil[1]
        self.decay=infil[2]
        self.tp=np.zeros(self.f0.shape)
        #------辅助变量--------
        self.f0_fmin=self.f0-self.fmin
        self.t_last=0.                             #上次询问的时间
    def __call__(self, t,h):
        return self.infiltration(t,h)
    
    def infiltration(self,t,h):
        return self.__hotton__(t,h)

    def __hotton__(self,t,h):
        """
        Horton infiltration model. the unit is cm/min.
        """
        index=np.where(h>0.)       
        if not index.any(): #if index is empty, rate= 0
            rate=np.zeros(h.shape,dtype=DATYPE_f)
        elif index.shape==h.shape: #if all h_i is positive
            self.tp+=t-self.t_last
            rate=self.f0+self.f0_fmin*np.exp(-self.decay*self.tp)    
        else:
            rate=np.zeros(h.shape,dtype=DATYPE_f)
            self.t_p[index]+=t-self.t_last
            rate[index]=self.f0[index]+self.f0_fmin[index]*np.exp(-self.decay[index]*self.t_p[index]) #at t_p moment
        self.t_last=t    # update the query time
        return rate      # |h|*1, unit:mm/s


import time     

class Runoff:
    RULES_NAME={0: "Manning_formula",
            1: "Weir_formula",
            2: "dynamic_wave"}


    def __init__(self,v_dim,dem,cell_length,A,z,link_n,cell_link,cell_link_dirc,
                 hi_index1,hi_index2,hj_index1,hj_index2,adjacency_vector,
                 z_var_index=np.array([])):
        """
            v_dim: the number of cells. scalar, int
            dem: the elevation of each cell. column vector, dim=|h|
            cell_length: the length of each cell. scalar, unit:m
            A: the rule of each link. column vector, dim=|h|
            z: the latend variable of each link. column vector, dim=|h|
            cell_link:  dim=|h|*4, the links index in z or A of each cell. If the links num is less than 4, the filling num is 0.
            cell_link_dirc: dim=|h|*4, the direction of the links in each cell. If the links num is less than 4, the filling num is 0.
            link_n:    dim=|z|*1, the fraction of each link, link_n=(n_i+n_j)/2
            hi_index1 and hi_index2: np.cancatenate(h[hi_index1],h[hi_index2]) is for the h_i in the A(h_i,h_j).
            hj_index1 and hj_index2: np.cancatenate(h[hj_index1],h[hj_index2]) is for the h_j in the A(h_i,h_j).
            adjacency_vector: the vector form of the adjacency matrix of the cells. shape=(4|h|,)
            
            z_var_index: the index of the variable in z, which is used to calculate the gradient of h w.r.t. z. If None, the gradient is not calculated.
                
            So the hi_index1, hi_index2, hj_index1, hj_index2 are used to calculate the A(h_i,h_j) in the Runoff class.
            We do this, because the main computational workload is in the A(h_i,h_j) calculation.
            We want to use the numpy array operations in A(h_i,h_j) calculation, to make the code more efficient.
        """
        self.dem=dem
        self.v_dim=v_dim                                      #=|h|
        self.cell_length=np.float32(cell_length)
        self.cell_link=cell_link
        self.cell_link_dirc=cell_link_dirc
        self.init_z=z
        self.A=A
        self.link_n=link_n
        self.hi_index=np.concatenate((hi_index1,hi_index2),axis=0)
        self.hj_index=np.concatenate((hj_index1,hj_index2),axis=0)
        self.adjacency_vector=adjacency_vector
        self.z_var_index=z_var_index
        

        self.link_flux=np.zeros(A.shape,dtype=DATYPE_f)
        self.link_flux_without_z=np.zeros(self.A.shape,dtype=DATYPE_f)

        #------auxiliary terms--------
        self.area=self.cell_length*self.cell_length   #unit:m^2
        self.slected_rules=np.sort(np.unique(A))
        self.rules_obs=self.__get_rules_obs__(self.slected_rules)
        self.rules_to_links=[np.where(A==i)  for i in self.slected_rules ]
        self.hi_index_list=[self.hi_index[index] for index in self.rules_to_links]
        self.hj_index_list=[self.hj_index[index] for index in self.rules_to_links]
        self.link_n_list=[self.link_n[index] for index in self.rules_to_links]     
        self.demi_list=[self.dem[index] for index in self.hi_index_list] 
        self.demj_list=[self.dem[index] for index in self.hj_index_list] 

        #for gradient calculation
        self._require_grad=False

        #for as_one        
        self.as_one=False    #for as_one
        self.h_connect_index=None
        self.h_connect_cell_link=None
        self.h_connect_cell_link_dirc=None

        #for localization
        self.localization=False  #for localization 
        self.local_z_index=None
        self.local_cells_index=None
        self.loc_cell_link=None

        #------for manning formula--------
        if 0 in self.slected_rules:  # 0 in RULES_NAME is manning formula
            self.manning_coe= np.sqrt(self.cell_length)/self.area/np.power(1000,7./6.)/self.link_n_list[0]
            self.delta_dem= (self.demi_list[0]-self.demj_list[0])*1000. #unit: m->mm
        #//////////////////////
        self.t1=time.time()
        self.t2=time.time()
    def set_as_one(self,as_one=True,localization=False,buffer_cells_index=None,buffer_z_index=None):
        self.as_one=as_one
        self.localization=localization
        
        if as_one:  #==config the relative paras
            #==config the the relative paras
            self.h_connect_index=np.unique(np.concatenate((self.hi_index[self.z_var_index],self.hj_index[self.z_var_index])))    
            self.h_connect_cell_link=self.cell_link.reshape(self.v_dim,-1)[self.h_connect_index,:].reshape(-1)
            self.h_connect_cell_link_dirc=self.cell_link_dirc.reshape(self.v_dim,-1)[self.h_connect_index,:].reshape(-1)
            in_array=np.isin(self.h_connect_cell_link,self.z_var_index)
            not_in_array=np.where(in_array==False)[0]
            #--
            self.h_connect_cell_link_dirc[not_in_array]=0.
            
            if localization:
                self.localization=True
                self.local_z_index=buffer_z_index
                self.local_cells_index=buffer_cells_index
                
                self.loc_cell_link=self.cell_link.reshape(self.v_dim,-1)[self.local_cells_index,:].reshape(-1)
                loc_cell_link_dirc=self.cell_link_dirc.reshape(self.v_dim,-1)[self.local_cells_index,:].reshape(-1)
                
                #localization postition
                for i in range(self.loc_cell_link.shape[0]): 
                    if self.loc_cell_link[i]!=0:
                        k=np.where(self.local_z_index==self.loc_cell_link[i])[0][0]
                        self.loc_cell_link[i]=k

                self.loc_cell_link_dirc_expand=np.column_stack((loc_cell_link_dirc,loc_cell_link_dirc))   

                #localization postition
                self.loc_h_connect_index= np.array([np.where(self.local_cells_index==t)[0][0] for t in self.h_connect_index] )    
                
                #localization postition
                  # shape=(|h|,4)
                loc_adjacency_vector=self.adjacency_vector.reshape(self.v_dim,-1)[self.local_cells_index,:].reshape(-1)
                loc_in_array=np.isin(loc_adjacency_vector,self.local_cells_index)
                self.in_loc_region_index=np.where(loc_in_array==True)[0]
                loc_adj_h_index=loc_adjacency_vector[self.in_loc_region_index]
                self.coresponding_in_loc_region_index=np.array([np.where(self.local_cells_index==t)[0][0] for t in loc_adj_h_index] )    
                 


    def require_grad(self):
        """set some variables for the gradient calculation"""
        if self._require_grad==True:
            return 
        if self.z_var_index.shape[0]==0:
            raise ValueError("z_var_index is empty, can not calculate the gradient of h w.r.t. z")
        self._require_grad=True

        self.rules_obs_grad=self.__get_rules_obs_grad__(self.slected_rules)
        self.grad_coes=np.zeros((self.A.shape[0],2),dtype=DATYPE_f)      #shape=(|z|,2)

        self.cell_link_dirc_expand=np.column_stack((self.cell_link_dirc,self.cell_link_dirc))

        source_index_row=np.concatenate((self.hi_index[self.z_var_index],self.hj_index[self.z_var_index])) #shape=(2|z_var|,1)
        source_index_col=np.concatenate((np.arange(0,self.z_var_index.shape[0]),np.arange(0,self.z_var_index.shape[0])))  
        self.source_index=(source_index_row,source_index_col)         

        
        
    def __call__(self,t,h,z):
        return self.runoff(t,h,z)
    def __get_rules_obs__(self,rules_labels):
        rules_obs=[]
        for i in rules_labels:
            if i==0: rules_obs.append(self.__manning_formula__) 
            elif i==1: rules_obs.append(self.__weir_formula__)
            elif i==2: rules_obs.append(self.__dynamic_wave__)
        return rules_obs
    
    def __get_rules_obs_grad__(self,rules_labels):
        rules_obs_grad=[]
        for i in rules_labels:
            if i==0: rules_obs_grad.append(self.__manning_formula_with_grad__) 
            elif i==1: rules_obs_grad.append(self.__weir_formula_grad__)
            elif i==2: rules_obs_grad.append(self.__dynamic_wave_grad__)
       
        return rules_obs_grad
    
    def runoff(self,t,h,z):
        """
        given the h,z and t, caiculate the (Z/circ A)1
        """
        #1. calculate the link flux,（这段之后用cuda或者pytorch的并行来写，试一试哪个快？） 
        for k in range(len(self.rules_obs)):
            self.link_flux[self.rules_to_links[k]]=      \
                z[self.rules_to_links[k]]*\
                self.rules_obs[k](h[self.hi_index_list[k]],h[self.hj_index_list[k]],self.demi_list[k],self.demj_list[k],self.link_n_list[k],self.cell_length)
        
        # self.t1=time.time()
        # print('time_other',self.t1-self.t2)
        #2. calculate the runoff
        temp=(self.link_flux[self.cell_link]*self.cell_link_dirc).reshape(h.shape[0],-1)
        runoff=np.sum(temp,axis=1)


        # b=[self.link_flux[self.cell_link[i]]*self.cell_link_dirc[i] for i in range(h.shape[0])]
        # temp=np.array(b)
        # runoff=np.sum(temp,axis=1)
        # runoff=np.array([np.sum(self.link_flux[self.cell_link[i]]*self.cell_link_dirc[i])  
        #                  for i in range(h.shape[0])])
        # self.t2=time.time()
        # print('time_sum',self.t2-self.t1) 
        # print(self.link_flux)
        return runoff
    
    def runoff_adjonit_grad(self,t,y,z):
        """
        Simultaneously calculate the gradients of runoff and runoff.
        Note: Because it is the core code, in order to ensure logical clarity, 
                there is a set of code for each implementation scenario
        """ 
        if self.as_one:
            if self.localization:
                #=================for as_one and localization case====================
                return self.__runoff_adjonit_grad_as_one_and_loacalization__(t,y,z)
            #=================for as_one case====================
            return self.__runoff_adjonit_grad_as_one__(t,y,z)
    
        #=================for general case===============================
        h=y[0:self.v_dim]        #water depth vetor for runoff calculation
        grad_h_z=y[self.v_dim:].reshape(self.v_dim,-1)  #gradient of water depth vector w.r.t. z

        #---calculate the flux and gradient of flux w.r.t. z for each link
        for k in range(len(self.rules_obs)):
            aij,aij_hi,aij_hj=self.rules_obs_grad[k](h[self.hi_index_list[k]],h[self.hj_index_list[k]],self.demi_list[k],self.demj_list[k],self.link_n_list[k],self.cell_length,self.delta_dem,self.manning_coe)
            self.link_flux[self.rules_to_links[k]]=z[self.rules_to_links[k]]*aij  
            self.link_flux_without_z[self.rules_to_links[k]]=aij          
            self.grad_coes[self.rules_to_links[k],0]=z[self.rules_to_links[k]]*aij_hi
            self.grad_coes[self.rules_to_links[k],1]=z[self.rules_to_links[k]]*aij_hj
        
        #=========================calculate the runoff==================================
        temp=(self.link_flux[self.cell_link]*self.cell_link_dirc).reshape(h.shape[0],-1)
        runoff=np.sum(temp,axis=1)  
        #===============================================================================
        
        #================calculate the gradient of runoff w.r.t. z=======================
        #like: dy/dt=Ay+b, where A is sparse matrix
        #--cal the Ay+b

        temp1=(self.grad_coes[self.cell_link,:]*self.cell_link_dirc_expand).reshape(self.v_dim,4,2)  #shape=(|h|,4,2). 如果是虚假边连接，则归零。

        #a. first calculate the diagonal elements of A
        diagonal=np.sum((temp1[:,0,0],temp1[:,1,0],temp1[:,2,1],temp1[:,3,1]),axis=0)
        diagonal=np.expand_dims(diagonal,axis=-1)   # shape=(|h|,)-->(|h|,1)

        source=np.concatenate((-self.link_flux_without_z[self.z_var_index],
                          self.link_flux_without_z[self.z_var_index])) #shape=(2|z_var|,) ,这里的正负号不要反了
        diagonal_and_source_term=diagonal*grad_h_z   
        # temp=source[self.source_index]          
        diagonal_and_source_term[self.source_index]+=source   #这里是+=，不是等于。上面的正负号加上这里，debug了一天。
    
        #b. then calculate the off-diagonal elements of A
        temp1[:,:2,0]=temp1[:,:2,1]      # reuse 
        off_diagonal=np.expand_dims(temp1[:,:,0],axis=-1)   # shape=(|h|,4)-->(|h|,4,1)
        temp2=grad_h_z[self.adjacency_vector,:].reshape(self.v_dim,-1,self.z_var_index.shape[0]) #shape=(|h|,4,|z_var|)
        off_sum=np.sum(off_diagonal*temp2,axis=1)       #shape=(|h|,|z_var|)

        #c. calculate the Ay+b 
        Ay_b=off_sum+diagonal_and_source_term                             #shape=(|h|,|z_var|)       
        Ay_b_v=Ay_b.reshape(-1)    #shape=(|h|*|z_var|,)   
        
        f_value=np.concatenate((runoff,Ay_b_v)) #shape=(|h|+|h|*|z_var|,)
        
        

        return f_value 
    def __runoff_adjonit_grad_as_one__(self,t,y,z):
        if not self.as_one:
            raise ValueError("The as_one flag is not set, can not use this function")
        grad_h_z=y[self.v_dim:]   
        
        #=================Most of them are the same as general case==============================
        h=y[0:self.v_dim]        #water depth vetor for runoff calculation
        
        #---calculate the flux and gradient of flux w.r.t. z for each link
        for k in range(len(self.rules_obs)):
            aij,aij_hi,aij_hj=self.rules_obs_grad[k](h[self.hi_index_list[k]],h[self.hj_index_list[k]],self.demi_list[k],self.demj_list[k],self.link_n_list[k],self.cell_length,self.delta_dem,self.manning_coe)
            self.link_flux[self.rules_to_links[k]]=z[self.rules_to_links[k]]*aij            
            self.link_flux_without_z[self.rules_to_links[k]]=aij          
            self.grad_coes[self.rules_to_links[k],0]=z[self.rules_to_links[k]]*aij_hi
            self.grad_coes[self.rules_to_links[k],1]=z[self.rules_to_links[k]]*aij_hj
        
        #=========================calculate the runoff==================================
        temp=(self.link_flux[self.cell_link]*self.cell_link_dirc).reshape(h.shape[0],-1)
        runoff=np.sum(temp,axis=1)  
        #===============================================================================
        
        #================calculate the gradient of runoff w.r.t. z=======================
        #like: dy/dt=Ay+b, where A is sparse matrix
        #--cal the Ay+b

        temp1=(self.grad_coes[self.cell_link,:]*self.cell_link_dirc_expand).reshape(self.v_dim,4,2)  #shape=(|h|,4,2). 如果是虚假边连接，则归零。

        #a. first calculate the diagonal elements of A
        diagonal=np.sum((temp1[:,0,0],temp1[:,1,0],temp1[:,2,1],temp1[:,3,1]),axis=0) #shape=(|h|,)
        diagonal_and_source_term=diagonal*grad_h_z       #shape=(|h|,)

        #b. then calculate the source elements  
        A_ij=(self.link_flux_without_z[self.h_connect_cell_link]*self.h_connect_cell_link_dirc).reshape(self.h_connect_index.shape[0],-1)
        source_as_one=np.sum(A_ij,axis=1)   #sum over colunms ,shape=(|h|,)
        diagonal_and_source_term[self.h_connect_index]+=source_as_one #shape=(|h|,)
    
        #c. then calculate the off-diagonal elements of A
        temp1[:,:2,0]=temp1[:,:2,1]      # reuse 
        off_diagonal=temp1[:,:,0]        # shape=(|h|,4)
        temp2=grad_h_z[self.adjacency_vector].reshape(self.v_dim,-1) #shape=(|h|,4)
        off_sum=np.sum(off_diagonal*temp2,axis=1)       #shape=(|h|,)

        #d. calculate the Ay+b 
        Ay_b=off_sum+diagonal_and_source_term                         #shape=(|h|,)       
        # Ay_b_v=Ay_b.reshape(-1)    #shape=(|h|*|z_var|,)   
        
        f_value=np.concatenate((runoff,Ay_b)) #shape=(|h|+|h|,)
        
        return f_value                  
             
       

    def __runoff_adjonit_grad_as_one_and_loacalization__(self,t,y,z): 
        """
        Simultaneously calculate the local gradients of runoff and runoff.
        Note: Because it is the core code, in order to ensure logical clarity, 
                there is a set of code for each implementation scenario
        """ 
        # if not self.as_one:
        #     raise ValueError("The as_one flag is not set, can not use this function")
        # if not self.localization:
        #     raise ValueError("The localization flag is not set, can not use this function")
        grad_h_z=y[self.v_dim:]       #as_one
        
        #=================Most of them are the same as general case==============================
        h=y[0:self.v_dim]        #water depth vetor for runoff calculation
        
        #---calculate the flux and gradient of flux w.r.t. z for each link
        # just for Manning formula
        k=0
        aij,loc_aij_hi,loc_aij_hj=self.rules_obs_grad[k](h[self.hi_index_list[k]],h[self.hj_index_list[k]],self.demi_list[k],self.demj_list[k],self.link_n_list[k],self.cell_length,self.delta_dem,self.manning_coe)
        self.link_flux[self.rules_to_links[k]]=z[self.rules_to_links[k]]*aij            
        self.link_flux_without_z[self.rules_to_links[k]]=aij   
        loc_grad_coes=np.zeros((self.local_z_index.shape[0],2))   
        loc_grad_coes[:,0]=z[self.rules_to_links[k]][self.local_z_index]*loc_aij_hi  #不动z 
        loc_grad_coes[:,1]=z[self.rules_to_links[k]][self.local_z_index]*loc_aij_hj
        
        
        #=========================calculate the runoff==================================
        temp=(self.link_flux[self.cell_link]*self.cell_link_dirc).reshape(h.shape[0],-1)
        runoff=np.sum(temp,axis=1)  
        #===============================================================================
        
        #================calculate the gradient of runoff w.r.t. z=======================
        #like: dy/dt=Ay+b, where A is sparse matrix
        #--cal the Ay+b
        
        temp1=(loc_grad_coes[self.loc_cell_link,:]*self.loc_cell_link_dirc_expand).reshape(self.local_cells_index.shape[0],4,2)

        #a. first calculate the diagonal elements of A
        diagonal=np.sum((temp1[:,0,0],temp1[:,1,0],temp1[:,2,1],temp1[:,3,1]),axis=0) #
        diagonal_and_source_term=diagonal*grad_h_z       #shape=(|loc_cells|,)

        #b. then calculate the source elements  
        A_ij=(self.link_flux_without_z[self.h_connect_cell_link]*self.h_connect_cell_link_dirc).reshape(self.h_connect_index.shape[0],-1)
        source_as_one=np.sum(A_ij,axis=1)   #sum over colunms ,shape=(|h|,)
        diagonal_and_source_term[self.loc_h_connect_index]+=source_as_one #shape=(|h|,)
    
        #c. then calculate the off-diagonal elements of A
        temp1[:,:2,0]=temp1[:,:2,1]      # reuse
        off_diagonal=temp1[:,:,0]        # shape=(|loc_cells|,4)
        tian=np.zeros((self.local_cells_index.shape[0]*4))
        tian[self.in_loc_region_index]=grad_h_z[self.coresponding_in_loc_region_index] 
        tian=tian.reshape(self.local_cells_index.shape[0],-1)  #shape=(|loc_cells|,4)
        off_sum=np.sum(off_diagonal*tian,axis=1)       #shape=(|loc_cells|,)

        # temp2=grad_h_z[self.loc_adjacency_vector].reshape(self.local_cells_index.shape[0],-1) #shape=(|loc_cells|,4)
        # off_sum=np.sum(off_diagonal*temp2,axis=1)       #shape=(|loc_cells|,)

        #d. calculate the Ay+b
        Ay_b=off_sum+diagonal_and_source_term                         #shape=(|h|,)       
        # Ay_b_v=Ay_b.reshape(-1)    #shape=(|h|*|z_var|,)   
        
        f_value=np.concatenate((runoff,Ay_b)) #shape=(|h|+|h|,)

        return f_value       

    

    def __manning_formula__(self,hi,hj,dem_i,dem_j,link_n,dx):
        '''
        Note: 函数参数是为了适应统一接口，里面的参数不全使用。
              因为这块是计算的核心，可以说80%的计算量集中这里，所以为了优化速度，必须在可读性和效率之间做取舍。
              为了实现更快速地计算，在init实现了部分预计算。
              hi,hj,unit:mm
        '''

        wl_delta=hi-hj+self.delta_dem
        sign=np.sign(wl_delta)        
        hr=hi
        hr[sign==-1]=hj[sign==-1]
        hr[hr<0.]=0.
        
        # abs_delta=np.abs(wl_delta)                   #add
        # index=np.where(abs_delta<10.)                 #add
        # index2=np.where(hr[index]>abs_delta[index])  #add
        # hr[index][index2]=abs_delta[index][index2]                   #add 用水位差代替水头差，解决振荡问题。

        hr_power=np.power(hr,1.66)
        sqrt_slope=np.sqrt(np.abs(wl_delta))

        # index=np.where(abs_delta<1.)                #add
        # sqrt_slope[index]=0.1*abs_delta[index]   #add
        #-------修订------------
        # condition=(hr>300.) &  (hr<400.)
        # index1=np.where(condition)   #修订
        # index2=np.where(hr>=500.)              #修订

        value=sign*self.manning_coe*hr_power*sqrt_slope #unit: mm/s
        # value[index1]*=0.5                          #修订
        # value[index2]*=0.01                          #修订
        return value  

    
    def __manning_formula_with_grad__(self,hi,hj,dem_i,dem_j,link_n,dx,delta_dem,manning_coe):
        """
        Simultaneously calculate the gradients of runoff and runoff, and support localization.
        """
        #--------for runoff calculation----------
        wl_delta=hi-hj+self.delta_dem
        sign=np.sign(wl_delta)
        hr=hi
        hr[sign==-1]=hj[sign==-1]
        hr[hr<0.]=0.
        hr_2_3=np.power(hr,0.66)                  #hr^{2/3}
        hr_2_3_coe=self.manning_coe*hr_2_3   #c*hr^{2/3}
        sqrt_slope=np.sqrt(np.abs(wl_delta))      #sqrt(|Hi-Hj|)
        hr_2_3_coe_sqrt_slope=hr_2_3_coe*sqrt_slope      #c*hr^{2/3}*sqrt(|Hi-Hj|)
        aij=sign*hr*hr_2_3_coe_sqrt_slope          #  c*hr^{5/3}*sqrt(|Hi-Hj|)  unit: m/s->mm/s

        #---------for gradient calculation----------
        #calculate the \partial a_{ij}/\partial h_i and \partial a_{ij}/\partial h_j
        #--for \partial a_{ij}/\partial h_i. for the 1st term
        if self.localization:
            loc_equal_term_in_ij=5./3.*hr_2_3_coe_sqrt_slope[self.local_z_index]
            term1_hi=np.copy(loc_equal_term_in_ij)
            loc_sign=sign[self.local_z_index]
            term1_hi[loc_sign==-1]=0.                #H_i<H_j
            #for the 2nd term 
            loc_sqrt_slope=sqrt_slope[self.local_z_index]
            loc_sqrt_slope[loc_sqrt_slope<1.]=1.              #避免水位相平的情况,special case
            loc_hr=hr[self.local_z_index]
            loc_hr_2_3_coe=hr_2_3_coe[self.local_z_index]
            term2=loc_hr_2_3_coe*loc_hr/loc_sqrt_slope/2.              #共同项
            loc_aij_hi=term1_hi+term2                #\partial a_{ij}/\partial h_i

            #--for \partial a_{ij}/\partial h_j. for the 1st term
            term1_hj=loc_equal_term_in_ij
            term1_hj[loc_sign==1]=0.
            loc_aij_hj=-(term1_hj+term2)                    #\partial a_{ij}/\partial h_j
            return aij,loc_aij_hi,loc_aij_hj      
        else:
            equal_term_in_ij=5./3.*hr_2_3_coe_sqrt_slope
            term1_hi=np.copy(equal_term_in_ij)
            term1_hi[sign==-1]=0.                #H_i<H_j
            #for the 2nd term 
            sqrt_slope[sqrt_slope<1.]=1.         #避免水位相平的情况,special case
            term2=hr_2_3_coe*hr/sqrt_slope/2.              #共同项
            aij_hi=term1_hi+term2                #\partial a_{ij}/\partial h_i

            #--for \partial a_{ij}/\partial h_j. for the 1st term
            term1_hj=equal_term_in_ij
            term1_hj[sign==1]=0.
            aij_hj=-(term1_hj+term2)                    #\partial a_{ij}/\partial h_j

            return aij,aij_hi,aij_hj

  

    def __weir_formula__(self,hi,hj,link_n,dx):
        """薄堰公式（Sharp-crested weir）:Q=C_d* L *H^{3/2} /1.5
            Q: 堰流量,单位为m^3/s
            C_d: 堰截面形状系数,取值0.01-0.05,越大越平滑,一般取0.02-0.03
            L: 堰长度,单位为m
            H: 堰上游的水头，单位为m
        """
        
        pass
    def __dynamic_wave__(self,hi,hj,link_n,dx):
        pass
    
    def __weir_formula_grad__(self,hi,hj,link_n,dx):
        pass
    def __dynamic_wave_grad__(self,hi,hj,link_n,dx):
        pass

    
