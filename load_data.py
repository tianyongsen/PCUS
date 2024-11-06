#--------------------------------------------------------------------------------------------------
#For parsing the input data to construct the ode and grad_ode models, and the observation operator 
#--------------------------------------------------------------------------------------------------
import numpy as np

RULES_NAME={0: "Manning_formula",
            1: "Weir_formula",
            2: "dynamic_wave"}

DATYPE_f=np.float32   #for general data type
DATYPE_int=np.int8   #for type data


class LoadData:
    Noddata_Value=-999
                    
    def __init__(self):

        #structure information stroed in the dem
        self.dem_orginal=None  
        self.cell_length=None
        

        #for the initial condition
        self.init_h=None

        #for the latent variables (z)
        self.z_var_index=None

        #for the rainfall
        self.rain_map=None
        self.rains=None

        #for the infiltration
        self.infil=None       
    
        #for the boundary condition
        self.Boundary=None

        #for the runff
        self.dem_vector=None         #column vector, shape=(|h|,)
        self.z=None                  #column vector, shape=(|z|,)
        self.A=None                  #column vector, shape=(|A|,)
        self.link_n=None             #column vector, shape=(|A|,)
        self.cell_link=None          #column vector, dim=|h|*_     
        self.cell_link_dirc=None     #column vector, dim=|h|*_
        self.hi_index1=None          #column vector, dim=depends on the |A|
        self.hi_index2=None          #column vector, dim=depends on the |A|
        self.hj_index1=None          #column vector, dim=depends on the |A|     
        self.hj_index2=None          #column vector, dim=depends on the |A|
        self.adjacency_vector=None   #column vector, shape=(4|h|,), 

        #for observation
        self.obs_times=None              #column vector, shape=(times,)
        self.obs_index_original=None #column vector, shape=(|obs|,2)  
        self.obs_index=None          #column vector, shape=(|obs|,)
        self.obs_value=None          #column vector, shape=(|obs|,times)
        self.FourNeightbors_Weights=np.array([0.4,0.15,0.15,0.15,0.15])  #1+4
        self.EightNeightbors_Weights=np.array([0.2,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])#1+8

        #some helper variables       
        self.M_ij_to_Vk=None         #a map from the original dem index to the state vector index
    
    def load_model_data(self,file_path_list):
        """ 
        Load the input data from files.
        file_path_list: a list of file paths, including:
            dem_f: the path of the DEM file
            z_f: the path of the z file
            A_f: the path of the A file
            init_h_f: the path of the initial h file
            n_f: the path of the n file
            rain_map_f: the path of the rain map file
            rain_f: the path of the rain file
            infil_f: the path of the infiltration file
            B_f: the path of the boundary condition file
            z_var_f: the path of the z_var file
        i.e. file_path_list=
            ['dem.txt', 'z.txt', 'A.txt', 'init_h.txt', 
            'n.txt', 'rain_map.txt', 'rain.txt', 'infil.txt', 
            'boundary.txt', 'z_var.txt'] 
            len(file_path_list)=10
        Note: the infil.txt and boundary.txt are optional, can be None
        """
        if len(file_path_list)!=10:
            raise ValueError("file_path_list should have 10 elements, \
                             including: dem_f, z_f, A_f, init_h_f, n_f,\
                              rain_map_f, rain_f, infil_f, B_f,z_var")
        dem_f,z_f,A_f,init_h_f,n_f,rain_map_f,rain_f,infil_f,B_f,z_var_f=file_path_list
        
        #parser the input files and transform the data into the form required for cellfield.
        self.parser_files_for_CellField(dem_f,z_f,A_f,init_h_f,n_f,
                                                    rain_map_f,rain_f,infil_f,B_f,z_var_f)

    def generate_data_from_dem(self):
        #
        pass
    def laod_data_quickly_from_dem(self,file_path_list):
        pass

    def parser_files_for_CellField(self,
                    dem_f,
                    z_f,
                    A_f,
                    init_h_f,
                    n_f,
                    rain_map_f,
                    rain_f,
                    infil_f=None,
                    B_f=None,z_var_f=None):
        """
            Parse the input files and transform the data into the form required for cellfield.
        """
        #--parser dem file
        self.dem_orginal= np.loadtxt(dem_f,dtype=DATYPE_f,comments='#') #matrix form
        dim=self.dem_orginal.shape
        with open(dem_f, 'r') as f:
            for line in f:
                if line.startswith('#cellsize'): 
                    self.cell_length=float(line.split()[1])
                    break
            else: raise ValueError("cellsize not found in dem file")
        # remove the nodata value
            #Note:tuple(row_array,clo_array), 
            # ex. row_array=[0,0,0,1,1] col_array=[0,1,2,1,2] -->（0,0）（0,1）
            # The elements in the index are arranged line by line 
        index_orginal=np.where(self.dem_orginal!=LoadData.Noddata_Value)  
            # Automatically flatten to a 1d array 
        self.dem_vector=self.dem_orginal[index_orginal]          #|h|*1

        #construct the dictionary to turn the cell index in original dem to the index in state vector
        self.M_ij_to_Vk={(i,j):k  for k,(i,j) in enumerate(zip(index_orginal[0],index_orginal[1]))}
        
        
        index=index_orginal
        if dim[0]*dim[1]==self.dem_vector.shape[0]:  #if regular grid,set the nodata_value to None
            index=None
        
        #--parser init_h file.For the initial condition,  |h|*1
        self.init_h= self.parser_single_file(init_h_f,dim,index)

        #--parser n file. For the runoff |h|*1
        n= self.parser_single_file(n_f,dim,index)
        
        #--parser rain_map file. For the rainfall |h|*1
        self.rain_map= self.parser_single_file(rain_map_f,dim,index,dtype_=DATYPE_int)

        #--parser rain file.  |rains|
        self.rains=self.parser_rain_file(rain_f)

        #--parser infiltration file. |h|*1 or None
        if infil_f is not None:
            self.infil= self.parser_infil_file(infil_f,dim,index)
        else:
            self.infil=None

        #parser z file  |z|*1
        self.z=self.parser_link_file(z_f,dim,index,link_info='z')

        #parser z_var file 
        if z_var_f is not None:
            self.z_var_index=self.parser_link_file(z_var_f,dim,index,link_info='z_var')
        else: 
            self.z_var_index=None
        #parser A file  |A|*1
        self.A=self.parser_link_file(A_f,dim,index,link_info='A')

        #parser B file
        if B_f is not None:
            self.Boundary= self.parser_single_file(B_f,dim,index)
        else:
            self.Boundary=None
        
        #For the runoff
        self.__for_cellfiled_runoff(self.dem_vector.shape[0],index_orginal,dim,self.z.shape[0],n)
    
    def __for_cellfiled_runoff(self,dim_vector,index,dim,link_dim,n):
        """
        For the runoff, we need to know the following information:
            cell_link:  dim=|h|*4, the links index in z or A of each cell. If the links num is less than 4, the filling num is 0.
            cell_link_dirc: dim=|h|*4, the direction of the links in each cell. If the links num is less than 4, the filling num is 0.
            link_n:    dim=|z|*1, the fraction of each link, link_n=(n_i+n_j)/2
            hi_index1 and hi_index2: np.cancatenate(h[hi_index1],h[hi_index2]) is for the h_i in the A(h_i,h_j).
            hj_index1 and hj_index2: np.cancatenate(h[hj_index1],h[hj_index2]) is for the h_j in the A(h_i,h_j).
            So the hi_index1, hi_index2, hj_index1, hj_index2 are used to calculate the A(h_i,h_j) in the Runoff class.
            We do this, because the main computational workload is in the A(h_i,h_j) calculation.
            We want to use the numpy array operations in A(h_i,h_j) calculation, to make the code more efficient.
        """
        
        self.hi_index1=np.array([i for i in range(index[0].shape[0]) if index[1][i]!=dim[1]-1]) #column vector 
        self.hj_index1=np.array([i for i in range(index[0].shape[0]) if index[1][i]!=0])
        self.hi_index2=np.array([i for i in range(index[0].shape[0]) if index[0][i]!=dim[0]-1])
        self.hj_index2=np.array([i for i in range(index[0].shape[0]) if index[0][i]!=0])

        #there are two map:    link--> the cell index at the h_i side of the link,  
                             # link--> the cell index at the h_j side of the link.  a(hi,hj)
        hi=np.concatenate((self.hi_index1,self.hi_index2),axis=0)  # column vector ,dim=|z|
        hj=np.concatenate((self.hj_index1,self.hj_index2),axis=0)  # column vector ,dim=|z|

        #--for the cell_link, cell_link_dirc and link_n
        cell_link=np.zeros((dim_vector,4),dtype=int)  #shape=(|h|,4)
        cell_link_dirc=np.zeros((dim_vector,4),dtype=DATYPE_f)  #shape=(|h|,4)
        adjacency_vector=np.zeros((dim_vector,4),dtype=int)  #shape=(|h|,4)
        count=np.zeros(dim_vector,dtype=int)  #shape=(|h|,)
        for k in range(link_dim):
            cell_link[hi[k],count[hi[k]]]=k
            cell_link_dirc[hi[k],count[hi[k]]]=-1.  #-1 means the outflow  
            adjacency_vector[hi[k],count[hi[k]]]=hj[k]
            count[hi[k]]+=1
            
        #The following row code has been annotated because it is implemented by default, 
        # but it is actually very important.
        # cell_link[count<1]=0            #0 means nothing
        # cell_link_dirc[count<1]=0.      #0. is important, meaning the location is meaningless.
        # adjacency_vector[count<1]=0     #0 means nothing
        count[:]=2
        for k in range(link_dim):
            cell_link[hj[k],count[hj[k]]]=k
            cell_link_dirc[hj[k],count[hj[k]]]=1.  #1 means the inflow  
            adjacency_vector[hj[k],count[hj[k]]]=hi[k]
            count[hj[k]]+=1

        self.cell_link=np.array(cell_link).reshape(-1)   #shape=(4|h|,)
        self.cell_link_dirc=np.array(cell_link_dirc).reshape(-1)    #shape(4|h|,1)
        self.adjacency_vector=np.array(adjacency_vector).reshape(-1)  #shape=(4|h|,)
        self.link_n=np.array([(n[hi[k]]+n[hj[k]])/2. for k in range(link_dim)])
        
    def parser_single_file(self,file_path,dim,index,dtype_=DATYPE_f):
        """
        parse the general input file. could skip comments and empty lines.
        """
        # read data from file
        data = np.loadtxt(file_path,dtype=dtype_,comments='#')
        if data.shape != dim:  #check the dim of data is equal to the dim of DEM
            raise ValueError("data shape is not equal to the dim of DEM")
        if index is not None:
            data=data[index]   #remove the nodata value according to the nodata index of DEM. Auto flatten to 1d array.
        else:
            data = data.reshape(-1)    #|data|*1   
        return data
        
    def parser_rain_file(self,rain_f):
        """ parse rain file.could skip comments and empty lines.
            paras: rain_f: the path of the rain file.
            content form:
                #mm/min time intensity
                #rain1
                0 1 2 3 4 5 6 7 8 300
                0 0 6.66667 6.66667 6.66667 0 0 0 0 0
                #rain2
                0 1 2 3 4 5 6 7 8 9 10 11 12 13
                0 0 0 1.33333 1.33333 1.33333 0 0 0 0 0 0 0 0
                #rain3  Chicago_Design_Storm
                a      b    n      T   r   
                21.377  8.   0.711  20  0.5
                ...
            Note:If you want to set different rainfall time series for each cell, 
                it is recommended to add a dedicated function.
        """
        rains=[]
        with open(rain_f, 'r') as f:
            for line in f:
                #skip comments and empty lines
                if line.startswith('#') or line.strip()=='': 
                    continue
                else:
                    content=line.strip().split()
                    if content[0]=='a' and content[1]=='b' and content[2]=='n' and content[3]=='T' and content[4]=='r':
                        rains.append(['a','b','n','T','r'])    # for Chicago_Design_Storm
                    else:
                        rains.append(np.array(line.strip().split(),dtype=DATYPE_f))
        rain_data={}
        index=0      
        for i in range(0,len(rains),2):
            rain_data[index]={"time or symbol":rains[i],"intensity":rains[i+1]}
            index+=1
        return rain_data

    def parser_infil_file(self,infil_f,dim,index):
        """ parse infiltration file. Could skip comments and empty lines.
            Every para represents a coe of the corresponding infiltration model.
            for example, in Horton infiltration model, there are 3 paras:
                initial infiltration rare f_s -- corresponding para1;            
                steady state infiltration rate f_0 -- coresponding para2;
                decay coefficient \alpha -- corresponding para3.
            The content form in txt file:
                #para1
                1 2 3 4 5 6 7
                11 12 13 14 15 16 17
                #para2
                1 2 3 4 5 6 7
                11 12 13 14 15 16 17
                #para3
                ...
                #para4
                ...
                #para5
                ...
        """
        infil_data=np.loadtxt(infil_f,dtype=DATYPE_f,comments='#')
        #check dim of infil_data
        if infil_data.shape[1]!=dim[1] and infil_data.shape[0]%dim[0]!=0:
            raise ValueError("infil_data shape is not equal to the dim of DEM")
        if index is not None: #remove the nodata value according to the nodata index of DEM
            temp=infil_data[i*dim[0]:(i+1)*dim[0],:][index]
            for i in range(1,infil_data.shape[0]//dim[0]):
                temp=np.concatenate((temp,infil_data[i*dim[0]:(i+1)*dim[0],:][index]),axis=1)
            infil_data=temp   #|h|*|paras|
        else: #reshape the dim of infil_data to 2D array 
            infil_data=infil_data.reshape((dim[0]*dim[1],-1))
        return infil_data    #|h|*|paras| 

    def parser_link_file(self,file,dim,index,link_info='z'):
        """ parse z or z_var or A file. could skip comments and empty lines.
            (if the dim of dem is N*M,then the dim of 'right' part of z or h is (N)*(M),
            and the dim of 'dowm' part is (N)*(M).
            same for the 'dowm' part).
            link_info:'z' or 'A' or "z_var"
            content form:
            #right: 
            1. 1. 1. 1.  or(for A) 0,1,2,3
            1. 1. 1. 1.            1,1,1,1
            #dowm:
            1. 1. 1. 1.  or (for A) 0,1,2,3
            1. 1. 1. 1.             1,1,1,1

            graph example:  the right part of z is (1,2,3,4) columns ,the down part of z is (a,b,c) rows.
                            Here, 4 and c are boundary edges and do not belong to internal edges, so we will remove them.
                            Adding edges 4 and c to the txt file facilitates data preparation and maintains consistency across various data dimensions.
                 0  1  2  3  4
            0    -------------
                 |  |  |  |  |
            a    -------------
                 |  |  |  |  |
            b    -------------
                 |  |  |  |  |
            c    -------------
            
        """
        if link_info not in ["z","z_var","A"]:
            raise ValueError("link_info should be 'z' or 'A' or 'z_var'")
        
        if link_info=='A' or link_info=='z_var':
            data=np.loadtxt(file,dtype=DATYPE_int,comments='#')        
        else:
            data=np.loadtxt(file,dtype=DATYPE_f,comments='#')
        
        #check dim of data
        if data.shape[1]!=dim[1] or data.shape[0]!=(dim[0])*2:
            raise ValueError("z_data or A_data shape does not match to the dim of DEM")
        if index==None:
            #remove the boundary edges 
            data_right=data[:dim[0],:dim[1]-1].reshape(-1)  #remove the most right column
            data_bottom=data[dim[0]:2*dim[0]-1,:].reshape(-1) #remove the most bottom row      
            data=np.concatenate((data_right,data_bottom),axis=0)    #|z|*1 
        else:
           temp_right=np.where(index[1]!=dim[1]-1) #remove the most right column
           right_index=([index[0][temp_right],index[1][temp_right]])
           data_right=data[right_index]

           temp_bottom=np.where(index[0]!=dim[0]-1) #remove the most bottom row
           bottom_index=([index[0][temp_bottom],index[1][temp_bottom]])
           data_bottom=data[bottom_index]

           data=np.concatenate((data_right,data_bottom),axis=0)    #|z|*1 

        if link_info=='A': 
            if np.max(data) not in RULES_NAME.keys() or np.min(data) not in RULES_NAME.keys():
                raise ValueError("A_data value is not in the Rules dictionary")
        elif link_info=='z_var':
            if np.any(data[data!=0]!=1):
                raise ValueError("z_var must be 0 or 1")
        else: # check the the non-negative of z_data 
            if np.min(data)<0.:
                raise ValueError("z_data value is negative")
        
        if link_info=="z_var":
            data=np.where(data==1)[0]    #just require the index of elment 1
        return data # shape=(|z|,) for z or A ; shape=(|z_var|,) for z_var

    #==================For observation=========================
    def load_observation(self,obs_f):
        """ load the observation data from file."""
        return self.parser_obs_file(obs_f)

    def parser_obs_file(self,obs_f):
        """ parse observation file. could skip comments and empty lines.\n
            content form:  \n
            #row, col, t0, t1,...,tn \n
            -1,-1, 5,10,...100               #1.the first row is the time time monments.\n
            100,200,0.1,0.3,...,0.6          #2.then the after rows are the observation data.\n
            100,400,0.2,0.4,...,0.8          #3.the each row is the observation data of the corresponding cell.\n
            300,500,0.3,0.5,...,0.7          #4.row and col are the index of the cell in the original DEM.\n
        """
        obs_data=np.loadtxt(obs_f,dtype=DATYPE_f,comments='#')
        #extract the time moments
        self.obs_times=obs_data[0,2:]
        
        #extract the observation index and data
        self.obs_index_original= obs_data[1:,0:2].astype(int)#

        if self.M_ij_to_Vk==None:
            raise ValueError("Please load the model file first,then call this function")
        
        self.obs_index=np.array([ self.M_ij_to_Vk[tuple(row)] for row in self.obs_index_original[:,:]])

        self.obs_value=obs_data[1:,2:]

        return self.obs_value   #just obs_value for external use.

    def get_observation_4Neighbours(self):
        """Obtain the 4 neighborhood indices of the observation point cells and the weighted values of these five cells \n
            default weights (see the self.FourNeightbors_Weights): \n
            \t w_center=2/5,w_above=3/20,w_below=3/20,w_left=3/20,w_right=3/20 \n
        """
        if self.obs_index is None:
            raise ValueError("Please load the observation data first,then call this function")

        
        FourNeighbours_index=np.zeros((self.obs_index.shape[0],5),dtype=int)
        FourNeighbours_weights=np.zeros((self.obs_index.shape[0],5),dtype=DATYPE_f)
        FourNeighbours_index[:,0]=self.obs_index
        
        for k,ck in enumerate(self.obs_index_original):
            i,j=ck     #unpackage the index of the observation cell
            FourNeighbours_weights[k,:]=self.FourNeightbors_Weights    #default weights

            #--the above
            if i-1<0:   #no above cell
                FourNeighbours_index[k,1]=0
                FourNeighbours_weights[k,0]+=FourNeighbours_weights[k,1] #add the weight to the central cell
                FourNeighbours_weights[k,1]=0.
            else:
                FourNeighbours_index[k,1]=self.M_ij_to_Vk[(i-1,j)]
            
            #--the left
            if j-1<0:   #no left cell
                FourNeighbours_index[k,2]=0
                FourNeighbours_weights[k,0]+=FourNeighbours_weights[k,2] #add the weight to the central cell
                FourNeighbours_weights[k,2]=0.
            else:
                FourNeighbours_index[k,2]=self.M_ij_to_Vk[(i,j-1)]
            
            #--the right
            if j+1>=self.dem_orginal.shape[1]:   #no right cell
                FourNeighbours_index[k,3]=0
                FourNeighbours_weights[k,0]+=FourNeighbours_weights[k,3] #add the weight to the central cell
                FourNeighbours_weights[k,3]=0.
            else:
                FourNeighbours_index[k,3]=self.M_ij_to_Vk[(i,j+1)]

            #--blow
            if i+1>=self.dem_orginal.shape[0]:   #no below cell
                FourNeighbours_index[k,4]=0
                FourNeighbours_weights[k,0]+=FourNeighbours_weights[k,4] #add the weight to the central cell
                FourNeighbours_weights[k,4]=0. 
            else:
                FourNeighbours_index[k,4]=self.M_ij_to_Vk[(i+1,j)]
        
        return FourNeighbours_index.reshape(-1),FourNeighbours_weights.reshape(-1)  #shape=(|obs_index|*5,)
        
    def get_observation_8Neighbours(self):
        """Obtain the 8 neighborhood indices of the observation point cells and the weighted values of these five cells \n
            default weights (see the self.FourNeightbors_Weights): \n
            \t w_center=2/5,w_above=3/20,w_below=3/20,w_left=3/20,w_right=3/20 \n
        """
        if self.obs_index is None:
            raise ValueError("Please load the observation data first,then call this function")
        EightNeighbours_index=np.zeros((self.obs_index.shape[0],9),dtype=int)
        EightNeighbours_weights=np.zeros((self.obs_index.shape[0],9),dtype=DATYPE_f)
        EightNeighbours_index[:,0]=self.obs_index
        special_case=[lambda i,j:i-1<0 or j-1<0,                                         #the upper left
                      lambda i,j:i-1<0,                                                  #the directly upper
                      lambda i,j:i-1<0 or j+1>=self.dem_orginal.shape[1],                      #the upper right
                      lambda i,j:j-1<0,                                                  #the directly left
                      lambda i,j:j+1>=self.dem_orginal.shape[1],                               #the directly right
                      lambda i,j:i+1>=self.dem_orginal.shape[0] or j-1<0,                      #the below left
                      lambda i,j:i+1>=self.dem_orginal.shape[0],                               #the directly below
                      lambda i,j:i+1>=self.dem_orginal.shape[0] or j+1>=self.dem_orginal.shape[1],   #the below right
                      ]
        loc_index=[lambda i,j:(i-1,j-1),lambda i,j:(i-1,j),lambda i,j:(i-1,j+1),lambda i,j:(i,j-1),
                    lambda i,j:(i,j+1),lambda i,j:(i+1,j-1),lambda i,j:(i+1,j),lambda i,j:(i+1,j+1)]
        for k,ck in enumerate(self.obs_index_original):
            i,j=ck     #unpackage the index of the observation cell
            EightNeighbours_weights[k,:]=self.EightNeightbors_Weights    #default weights
            for loc in range(8):
                if special_case[loc](i,j):   # if it is a special case 
                    EightNeighbours_index[k,loc]=0
                    EightNeighbours_weights[k,0]+=EightNeighbours_weights[k,loc] #add the weight to the central cell
                    EightNeighbours_weights[k,loc]=0. 
                else:
                    EightNeighbours_index[k,1]=self.M_ij_to_Vk[loc_index[loc](i,j)]

        return EightNeighbours_index.reshape(-1),EightNeighbours_weights.reshape(-1)  #shape=(|obs_index|*9,)
               
        

    




        




