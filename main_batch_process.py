import numpy as np
import time
import argparse
from load_data import LoadData
from cellfield import CellField
from model import OdeModel
from obs_operator import Linear_Observation_Operator
from loss_module import Loss
from optimizer import Optimizer
from samplers import LangevinDynamics
from post_process import PostProcess
from samplers import MetropolisAdjustedLangevin
#=============================basic structure=================================

#     ------------------------>|projector|------>
#     |                                          |   
#|data pool|--->|celldiled|--->  |model|  ------>|loss|--->|optim|  
#                       ^                                     |         
#                       <-----------------<-------------------
#============================================================================
#unit: mm for water depth h

def main_DA(file_path_list,path,obs_file,result_files):
    #-------------------------------------------------------------------------
    # tian=["/tian_dem.txt","/tian_z_prior.txt","/tian_A.txt","/tian_init_h.txt","/tian_n.txt",
                # "/tian_rain_map.txt","/tian_rain.txt",None,None,"/tian_z_var.txt"] 
    z_var_posterior_file=result_files['z_var_posterior_file']
    loss_file=result_files['loss_file']
    sum_grad_info_file=result_files['sum_grad_info_file']                #if None,the grad_info will not be saved
    y_pred_grad_info_file=result_files['y_pred_grad_info_file']  #if None,the grad_info will not be saved

    parser = argparse.ArgumentParser(description='Data Assimilation')
    parser.add_argument('--obs_file_path',    type=str,     default=path+obs_file,     help=" the observation file")
    # parser.add_argument('--result_h_path',    type=str,     default=floder+'/tian_h_DA.txt',  help="the resutl file path")
    parser.add_argument('--result_z_var_path',type=str,     default=path+z_var_posterior_file, help="the resutl file path")
    parser.add_argument('--result_loss_path', type=str,     default=path+loss_file, help="the loss resutl file path")
    # parser.add_argument('--y_pred_and_y_obs', type=str,     default=floder+'/tian_y_pred_and_y_obs.txt',help="the time sequence of the simulation")
    parser.add_argument('--sum_grad_info',        type=str,     default=path+sum_grad_info_file,help="the gradient information")
    parser.add_argument('--y_pred_grad_info', type=str,     default=path+y_pred_grad_info_file,help="the gradient information of y_pred")
    parser.add_argument('--opt_or_sample',    type=str,     default="opt",    help="the optimize method or the sample method")  #"sample" or "optimize"
    parser.add_argument('--epoch',            type=int,     default=40,     help="the optimaziation times")
    # parser.add_argument('--opt_method',       type=str,     default="adagrad",help="see OPT_METHODS in Optimizer.py ")
    parser.add_argument('--opt_method',       type=str,     default="rmsprop",help="see OPT_METHODS in Optimizer.py ")
    parser.add_argument('--loss_type',        type=int,     default=3,        help="see LOSS_TYPE in Loss.py")
    parser.add_argument('--Gaussian_Gamma',   type=float,   default=1.,        help="the Likelihood covarance parameters,see LOSS_TYPE in Loss.py")  #depends on the loss_type
    parser.add_argument('--Gaussian_Sigma',   type=float,   default=0.4,        help="the prior covarance parameters,see LOSS_TYPE in Loss.py")  #depends on the loss_type
    parser.add_argument('--lr',               type=float,   default=0.1,     help="the learning rate of the optimizer")
    parser.add_argument('--obs_mode',         type=str,     default='cell_to_cell',help="see the MODE in Linear_Observation_Operator")

    parser.add_argument('--as_one',           type=bool,    default=True,     help="whether to use the as_one mode,see cellfield.set_as_one")
    parser.add_argument('--localization',     type=bool,    default=False,     help="whether to use the localization mode,see cellfield.set_as_one")
    parser.add_argument('--buffer_type',      type=str,     default='square',  help="see the buffer_type in the cellfield.set_as_one ")
    parser.add_argument('--buffer_size',      type=int,     default=8,        help="see the buffer_size in the cellfield.set_as_one")
    parser.add_argument('--local_boundary',    type=float,   default=0.,     help="see the local_boudary in the cellfield.set_as_one")
    Norm1_index=np.arange(15,45)                    #for LOSS_TYPE=8
    TV_index=np.concatenate((np.arange(0,15),np.arange(45,60)))       #for LOSS_TYPE=8
    # parser.add_argument('--require_grad',    type=bool,    default=True,     help="whether to require the gradient of the loss w.r.t. z")
    args = parser.parse_args()

    time_start=time.time()

    #+========================================
    #--create LoadData object as the data pool
    load_data=LoadData()    

    #--load the model data
    load_data.load_model_data(file_path_list)  

    #--create cellfield object and set the as_one mode
    cellfield=CellField(load_data)    
    cellfield.set_as_one(as_one=args.as_one,
                            localization=True,
                            buffer_type=args.buffer_type,
                            buffer_size=args.buffer_size,
                            local_boundary=args.local_boundary)   
    
    #--create ode model object
    ode_model=OdeModel(cellfield)

    #--load the observation data
    y_obs=load_data.load_observation(args.obs_file_path)

    #--construct the observation time sequence
    t_eval=load_data.obs_times
    t_end=t_eval[-1]
    t0=0.                      #must be 0.
    print("The data assimilation time sequence is:\n",t_eval)
    #--create Observation_Operator object
    projector=Linear_Observation_Operator(load_data)

    
    # a=projector.obs_operator(cellfield.h0,mode='8Neighbours_to_cell')

    #--create loss object

    loss=Loss(loss_type=args.loss_type,
            Gaussian_Gamma=args.Gaussian_Gamma,
            Gaussian_Sigma=args.Gaussian_Sigma, #for LOSS_TYPE=0
            z_mean=cellfield.z_var.copy(),       #for LOSS_TYPE=0
            Norm1_index=Norm1_index,                      #for LOSS_TYPE=8
            TV_index=TV_index         #for LOSS_TYPE=8
    )
    
    #--create optimizer object or Sampler object
    if args.opt_or_sample=="opt":
        optimizer=Optimizer(cellfield.get_opt_variable(),opt_method=args.opt_method,lr=args.lr)
    elif args.opt_or_sample=="sample":
        sampler=LangevinDynamics(cellfield.z_var)     
    
    #--create post process object
    postprocess=PostProcess()  

    #save the prior z_var 
    postprocess.save_z_var(cellfield.z_var,0,file_name=args.result_z_var_path)

    #some result information  
    # output_grad
    # y_pred_grad_info=np.zeros((args.epoch*y_obs.shape[0],cellfield.z_var.shape[0]))
    
    #grad information

    #--data assimilation
    for epoch in range(args.epoch):
        t_epoch=time.time()
        print("The epoch is:",epoch)
        #--predict states and the states gradient w.r.t. z
        h,h_grad=ode_model.predict(cellfield.h0,cellfield.z,t0, t_end,t_eval,require_grad=True) 
        
        #get the sum of the gradient of h w.r.t. z 
        if result_files['sum_grad_info_file'] is not None:
        #--save the grad information
            sum_grad=np.sum(h_grad,axis=0)
            postprocess.save_sum_grad(sum_grad,epoch,path_file=args.sum_grad_info)

        #--reserve the h file
        # postprocess.save_h(h,t_eval,file_name=args.result_h_path)
    
        #--project
        y_pred,y_pred_grad=projector(h,h_grad,mode=args.obs_mode,require_grad=True)      #project the h and the h_grad to the observation space
        
        #--save the grad info of y_pred w.r.t. z
        if result_files['y_pred_grad_info_file'] is not None:   
            postprocess.save_y_pred_grad(y_pred_grad,epoch,path_file=args.y_pred_grad_info)

        #--cal the loss
        loss.criterion(y_pred,y_obs,cellfield.z_var)                      #cal the loss

        #--cal the gradient of loss w.r.t. z
        loss.backward(y_pred,y_obs,cellfield.z_var,y_pred_grad)         #cal the gradient   

        #--optimize or sample: Forwad a step or sample a step
        if args.opt_or_sample=="opt":
            #----check the convergence
            if loss.convergence():
                print("The optimization is converged!")
                break
            #----optimize: forwad a step
            optimizer.step(loss.grad)    
                
        elif args.opt_or_sample=="sample":                      
            sampler.sample_step(loss.grad)        

        #--optimize:Projection operator under constraint optimization. The constraint is z>=0.
        cellfield.constraint_opt_variable()    

        #--update the z in thecellfiled
        cellfield.step()        

        #--save the z_var
        postprocess.save_z_var(cellfield.z_var,epoch+1,file_name=args.result_z_var_path)

        #--save the h-DA
        # postprocess.save_h(h,t_eval,file_name=args.result_h_path)

        #--save the loss
        postprocess.save_loss(loss.loss_result_list,file_name=args.result_loss_path)

        # postprocess.error_MAE(y_pred,y_obs,file_name=args.result_loss_path)
        # postprocess.save_y_pred_and_y_obs(y_pred,y_obs,file_name=args.y_pred_and_y_obs)

        t_epoch_end=time.time()
        print("The epoch running time is:",t_epoch_end-t_epoch)
    else:   #if the loop is all done, do the final step
        h,h_grad=ode_model.predict(cellfield.h0,cellfield.z,t0,t_end,t_eval,require_grad=True) 
        y_pred,y_pred_grad=projector(h,h_grad,mode=args.obs_mode,require_grad=True)      #project the h and the h_grad to the observation space
        loss.criterion(y_pred,y_obs,cellfield.z_var)                   #cal the loss
        postprocess.save_loss(loss.loss_result_list,file_name=args.result_loss_path)
        sum_grad=np.sum(h_grad,axis=0)
        postprocess.save_sum_grad(sum_grad,epoch,path_file=args.sum_grad_info)


        print("The all epochs have been finished!")

    #--save the grad information
    # postprocess.save_grad(sum_grad,y_pred_grad_info,file_name=args.grad_info)
    

    time_end=time.time()
    print("The data assimilation running time is:",time_end-time_start)
    #--post-processing


def main_simulation(file_path_list,simu_paras,h_simu_f_p=None):
    """
    Simulation function.
    Args:
        file_path_list: the input file for the model.
                        (example) Path+["tian_dem.txt","tian_z_true.txt","tian_A.txt","tian_init_h.txt","tian_n.txt",
                        "tian_rain_map.txt","tian_rain.txt",None,None,"tian_z_var.txt"] 
        simu_paras: the control parameters of the simulation. 
                    (example) {"t0":0,"t_end":1200,"t_interval":20}
        h_simu_f_p: the result file path of the simulation.
                    (example) Path+"tian_h_simu.txt"
    """
    #-------------------------------------------------------------------------
    #control parameters
    t0=simu_paras["t0"]                 #the start time of the simulation
    t_end=simu_paras["t_end"]           #the end time of the simulation
    t_interval=simu_paras["t_interval"] #the report interval of the simulation
    h_file_path=h_simu_f_p              #the result file path of the simulation


    #--set the report time moments
    t_eval=np.arange(t0+t_interval,t_end+t_interval,t_interval)  
    print("the report time moments are : (unit:second) \n",t_eval)   

    #main 函数

    #+========================================
    #--create LoadData object as the data pool
    load_data=LoadData()    

    #--load the model data
    load_data.load_model_data(file_path_list)  

    #--create cellfield object
    cellfield=CellField(load_data)       

    #--create ode model object
    ode_model=OdeModel(cellfield)

    time_start=time.time()
    h,_=ode_model.predict(cellfield.h0,cellfield.z,t0,t_end,t_eval,require_grad=False)  
    time_end=time.time()
    print("The run time is:",(time_end-time_start))     

    #--post-processing
    if h_file_path is not None:
        post=PostProcess()
        post.save_h(h,t_eval,h_file_path)
    
    return h    #shape=(|h|+1,|t_eval|) 

def make_obs(path,h_file,dem_shape,obs_file=None,obs_index=None,obs_interval=None):
        """make the observation file. Should be called after the simulation,i.e. after having the h.txt file.

        Args:
            floder (str): the floder path
            h_file (str): the h.txt file path in the floder
            dem_shape (tuple, optional): the shape of the DEM, row and col. Defaults to None.
            obs_file (str, optional): the observation file path in the floder. Defaults to None.
            obs_index (numpy.ndarray, optional): the observation index. Defaults to None.
        """
        h_info=np.loadtxt(path+h_file,dtype=np.float32)

        if obs_index is None:
            #==============================**********==============================================
            # # obs_index=np.array([(12,22),(22,37),(27,12),(37,27)])  #四个观测点坐标. Number from zero.
            obs_index=np.array([(7,22),(22,7)])  #两个观测点坐标. Number from zero.
            # obs_index=np.array([(7,7),(22,22)])  #两个观测点坐标. Number from zero.
            # obs_index=np.array([(7,7),(22,7),(7,22)])  #两个观测点坐标. Number from zero. s 
            # obs_index=np.array([(7,7),(22,7),(7,22),(22,22)])  #两个观测点坐标. Number from zero. s 
            # obs_index=np.array([(7,7),(22,7),(7,22),(22,22)])  #两个观测点坐标. Number from zero. s 

            #======================================================================================
            # obs_index=np.array([(4,4),(4,14),(15,5),(15,15)])  #四个观测点坐标. Number from zero.

            # obs_index=np.array([(5,14),(14,5)])  #两个观测点坐标. Number from zero.

            # obs_index=np.array([(30,13),(19,36)])  #两个观测点坐标. Number from zero. s 
            
            # obs_index=np.array([(7,7),(22,7),(7,22),(22,22)])  #两个观测点坐标. Number from zero. s 

        h_interval=h_info[0,1]-h_info[0,0]    
        if obs_interval % h_interval !=0:
            raise ValueError("the obs_interval should be a multiple of the h_interval")
        end_time=h_info[0,-1]
        if end_time % obs_interval !=0:
            raise ValueError("the end_time should be a multiple of the obs_interval")
        
        obs_num=int(end_time/obs_interval)
        obs=np.zeros((obs_index.shape[0]+1,obs_num+2),dtype=np.float32)  #shape=(|obs|+1,2+obs_num)

        time_index=np.array(range(1,int(end_time/obs_interval)+1),dtype=np.int32)
        time_index=time_index*(int(obs_interval/h_interval))-1

        #the first row
        obs[0,:2]=-1
        obs_times=h_info[0,time_index]   #obs_times
        obs[0,2:]=obs_times    #obs_times

        #the rest rows
        obs[1:,:2]=obs_index   #obs_index
        
        obs_index_in_hvector=obs_index[:,0]*dem_shape[1]+obs_index[:,1]   #the index of the observation points in the h 
        
        obs[1:,2:]=h_info[obs_index_in_hvector+1,:][:,time_index]            #   obs_index_in_hvector+1 is the index of the observation points in the h_file.
        
        obs=obs
        content_form="""content form:  
                        row, col, t0, t1,...,tn 
                        -1,-1, 5,10,...100               #1.the first row is the time time monments.
                        100,200,0.1,0.3,...,0.6          #2.then the after rows are the observation data.
                        100,400,0.2,0.4,...,0.8          #3.the each row is the observation data of the corresponding cell.
                        300,500,0.3,0.5,...,0.7          #4.row and col are the index of the cell in the original DEM."""
        
        if obs_file!=None:
            obs_file_path=path+obs_file
        else:
            obs_file_path=path+'tian_obs.txt'

        header=obs_file_path+'\n'+'(obs_num,times)=: {}\n'.format(obs.shape[0]-1,obs.shape[1])+'\n'+content_form+'\n'

        np.savetxt(obs_file_path,obs,fmt='%.2f',
                    header=header,comments='#')
        print("{} has been created successfully".format(obs_file_path))   


def test_by_rains(path,true_file_path_list,paras):
    """select the different rains to test 
        #only test the C1T1 and C1T2 cases with different rains
    """
    test_start=time.time()

    # ori_file_path_list=true_file_path_list
    t0=paras["t0"]                 #the start time of the simulation
    t_end=paras["t_end"]           #the end time of the simulation
    t_interval=paras["t_interval"] #the report interval of the simulation
    obs_mode=paras["obs_mode"]     #the observation mode,example: 'cell_to_cell'
    loss_type=paras["loss_type"]   #the loss type,example: 3
    Gaussian_Gamma=paras["Gaussian_Gamma"]   #the Gaussian_Gamma parameter,example: 1
    dem_shape=paras["dem_shape"]   #the shape of the DEM,example: (30,30)
    #==========================================
    obs_index=paras["obs_index"]
    z_posterior_file=paras["z_posterior_file"]
    h_simu_file=paras["h_simu_true_file"]
    obs_file=paras["obs_true_file"]
    test_loss_file=paras["test_loss_file"]
    #==========================================
    # Gaussian_Sigma=0.4                  #the Gaussian_Sigma parameter,example: 0.4
    t_eval=np.arange(t0+t_interval,t_end+t_interval,t_interval)   #the report time moments

    #--replace the rain_map file with the test rain
    # rain_map=np.loadtxt(test_file_path_list[5])      #'/rain_map.txt'
    # rain_map[:,:]=rain_index                                       # the index of the rain 
    # np.savetxt(path+'tian_rain_map_test.txt',rain_map) 
    # test_file_path_list[5]=path+'tian_rain_map_test.txt'    #'/tian_rain_map_test.txt'
    
    #--replace the z_true file with the z_posterior file
    z_posterior=np.loadtxt(path+z_posterior_file)
    epochs=z_posterior.shape[1]               #epochs

    #===========================*****=================================
    #--create LoadData object as the data pool
    load_data=LoadData()    
    # test_load_data=LoadData()    

    #--load the model data
    load_data.load_model_data(true_file_path_list)  
    # test_load_data.load_model_data(test_file_path_list)  

    #--create cellfield object
    cellfield=CellField(load_data)       
    # test_cellfield=CellField(test_load_data)       

    #--create ode model object
    ode_model=OdeModel(cellfield)
    # test_ode_model=OdeModel(test_cellfield)
    


    #--simulate the true data
    h_simu_true,_=ode_model.predict(cellfield.h0,cellfield.z,t0,t_end,t_eval,require_grad=False)   #the true simulation data
    # test_h_simu_true,_=test_ode_model.predict(test_cellfield.h0,test_cellfield.z,t0,t_end,t_eval,require_grad=False)   #the true simulation data    
    #--save the true simulation data
    # h_simu_file='tian_h_simu_true_test.txt'
    # test_h_simu_file='tian_test_h_simu_test.txt'

    np.savetxt(path+h_simu_file,np.concatenate((np.expand_dims(t_eval,axis=0),h_simu_true),axis=0),fmt='%.4f')
    # np.savetxt(path+test_h_simu_file,np.concatenate((np.expand_dims(t_eval,axis=0),test_h_simu_true),axis=0),fmt='%.4f')

    make_obs(path,h_simu_file,dem_shape,obs_file=obs_file,obs_index=obs_index,obs_interval=t_interval)
    # make_obs(path,test_h_simu_file,dem_shape,obs_file='obs_test_temp.txt',obs_index=obs_index,obs_interval=t_interval)

    #--create the observation operator object
    y_obs_true=load_data.load_observation(path+obs_file)     #the true data
    # test_y_obs_true=test_load_data.load_observation(path+'obs_test_temp.txt')             #the true data  
    #--create Observation_Operator object
    projector=Linear_Observation_Operator(load_data)    
    # test_projector=Linear_Observation_Operator(test_load_data)


    loss=Loss(loss_type=loss_type,
        Gaussian_Gamma=Gaussian_Gamma,
        # Gaussian_Sigma=Gaussian_Gamma,      #for LOSS_TYPE=0
        # z_mean=cellfield.z_var.copy(),
        # Norm1_index=None, 
        # TV_index=None
        )
    
    postprocess=PostProcess()
    
    obs_loss_list=[]      #the loss at observation locations 
    total_loss_list=[]        #the total loss of the cellfield

    for epoch in range(epochs):
        #--update the z_var in the cellfiled
        cellfield.z_var=z_posterior[:,epoch]
        cellfield.step()                      #update the z in the cellfield
        
        #--predict states and the states gradient w.r.t. z
        h_posterior,_=ode_model.predict(cellfield.h0,cellfield.z,t0,t_end,t_eval,require_grad=False) 

        #--project
        y_posterior=projector(h_posterior,mode=obs_mode,require_grad=False)

        #--cal the observation loss
        loss.criterion(y_posterior,y_obs_true)                      #cal the loss
        
        obs_mae=postprocess.error_MAE(y_obs_true,y_posterior)
        obs_rmse=postprocess.error_RMSE(y_obs_true,y_posterior)
        obs_mre=postprocess.error_MRE(y_obs_true,y_posterior)
        obs_loss_list.append((loss.loss_result_list[-1],obs_mae,obs_rmse,obs_mre))     

        
        #--cal the total loss
        total_mae=postprocess.error_MAE(h_simu_true,h_posterior)   
        total_rmse=postprocess.error_RMSE(h_simu_true,h_posterior)
        total_mre=postprocess.error_MRE(h_simu_true,h_posterior) 
        total_loss_list.append((total_mae,total_rmse,total_mre))           #the total loss of the cellfield

    #--save the loss data
    obs_loss_file=np.array(obs_loss_list)
    total_loss_file=np.array(total_loss_list)
    loss=np.column_stack((obs_loss_file,total_loss_file))
    
    header='obs_loss,obs_MAE,obs_RMSE,obs_MRE,total_loss,total_MAE,total_RMSE,total_MRE'
    np.savetxt(path+test_loss_file,loss,fmt='%.4f',header=header,comments='#')
    print("{} has been created successfully".format(path+test_loss_file))   

    test_end=time.time()
    print("The test time is {:.2f}s".format(test_end-test_start))   


def main_1():
    """for Case 1"""
    simu_paras={"t0":0,
                "t_end":1200,
                "t_interval":60 }

    path='tian_case30/'
    tian=["tian_dem.txt","tian_z_true.txt","tian_A.txt","tian_init_h.txt","tian_n.txt",
            "tian_rain_map.txt","tian_rain.txt",None,None,"tian_z_var.txt"] 
    z_prior='tian_z_prior.txt'
    for i,file in enumerate(tian): 
        if file is not None: tian[i]=path+file
    file_path_list=tian                   #the simulaiton input files

    h_simu_file='tian_h_simu.txt'
    h_simu_file_path=path+h_simu_file          #the output of simulation

    #Different cases of Data Assimilation.
    case1_suffix=['_C1T1.txt','_C1T2.txt','_C1T3.txt','_C1T4.txt']          #for interval=60
    case1_suffix=case1_suffix+['_C1T5.txt','_C1T6.txt','_C1T7.txt','_C1T8.txt'] #for interval=120

    #obseration file for data assimilation
    obs_file=['tian_obs'+s for s in case1_suffix]  

    #posterior file and loss file for data assimilation
    z_posterior_file=['tian_z_var_posterior'+s for s in case1_suffix]
    loss_file=['tian_loss'+s for s in case1_suffix]

    #grad_info
    sum_grad_info_file=['tian_sum_grad'+s for s in case1_suffix]
    y_pred_grad_info_file=['tian_y_pred_grad'+s for s in case1_suffix]


    #=================****===============================
    simulation=True
    simulation=False  
    if simulation:
        main_simulation(file_path_list,simu_paras,h_simu_file_path)
    
    make_obs_=True          #set
    make_obs_=False
    obs_type=[0,8]
    obs_index_list=[
        np.array([(7,22),(22,7)]),                      #两个观测点坐标. Number from zero.
        np.array([(7,7),(22,22)]),                      #两个观测点坐标. Number from zero.
        np.array([(7,7),(22,7),(7,22)]),                #两个观测点坐标. Number from zero. s 
        np.array([(7,7),(22,7),(7,22),(22,22)]),        #两个观测点坐标. Number from zero. s 
    ]
    if make_obs_:
        for i in range(obs_type[0],obs_type[1]):
            if i<4: 
                obs_interval=60
                make_obs(path,
                        h_simu_file,
                        dem_shape=(30,30),
                        obs_file=obs_file[i],
                        obs_index=obs_index_list[i],
                        obs_interval=obs_interval)
            if i>=4 and i<8: 
                obs_interval=120
                make_obs(path,
                           h_simu_file,
                           dem_shape=(30,30),
                           obs_file=obs_file[i],
                           obs_index=obs_index_list[i-4],
                           obs_interval=obs_interval)   
    
                
    DA=True
    # DA=False
    DA_file_path_list=file_path_list.copy()
    DA_file_path_list[1]=path+z_prior              #repalce the z_true file with the z_prior file
    if DA:
        single_case=True
        # single_case=False
        if single_case:
            i=0
            result_files={'z_var_posterior_file':z_posterior_file[i],
                'loss_file':loss_file[i],
                'sum_grad_info_file':sum_grad_info_file[i],
                'y_pred_grad_info_file':y_pred_grad_info_file[i],                
                }
            main_DA(DA_file_path_list,path,obs_file[i],result_files)

        else:
            for i in range(0,8):
                result_files={'z_var_posterior_file':z_posterior_file[i],      #nescesary
                            'loss_file':loss_file[i],                          #nescesary                                  
                            'sum_grad_info_file':sum_grad_info_file[i],        #or can be None that will be not saved
                            'y_pred_grad_info_file':y_pred_grad_info_file[i]}  #or can be None that will be not saved
                main_DA(DA_file_path_list,path,obs_file[i],result_files)

    rains_test=True
    rains_test=False
    if rains_test:
        rains=["1","2","3"]
        obs_plan_numbers=[0,1]
        for number in obs_plan_numbers:
            for r in rains:
                rain=r
                obs_plan_number=number
                file_path_list[5]=path+'tian_rain_map_R'+rain+'.txt'
                test_paras={"t0":0,
                            "t_end":1200,
                            "t_interval":60,                
                            "obs_mode":'cell_to_cell',
                            "loss_type":3,
                            "Gaussian_Gamma":1.0,
                            "dem_shape":(30,30),
                            "obs_index": obs_index_list[obs_plan_number],          
                            "z_posterior_file":z_posterior_file[obs_plan_number],
                            "h_simu_true_file":"h_sim_true_rain"+rain+case1_suffix[obs_plan_number],
                            "obs_true_file":"obs_true_rain"+rain+case1_suffix[obs_plan_number],
                            "test_loss_file":"test_loss_rain"+rain+case1_suffix[obs_plan_number]
                            }
                # obs_file_path=path+'tian_obs_C1T5.txt'
                test_by_rains(path,file_path_list,test_paras)


def main_2():
    """for Case 2"""
    simu_paras={"t0":0,
                "t_end":240,
                "t_interval":30 }

    path='eat8a/'
    eat8a=["eat8a_dem.txt","eat8a_z_true.txt","eat8a_A.txt","eat8a_init_h.txt","eat8a_n.txt",
            "eat8a_rain_map.txt","eat8a_rain.txt",None,None,"eat8a_z_var.txt"] 
    z_prior='eat8a_z_prior_4.txt'             #***
    for i,file in enumerate(eat8a): 
        if file is not None: eat8a[i]=path+file
    file_path_list=eat8a                   #the simulaiton input files

    h_simu_file='eat8a_h_simu_self_240.txt'
    h_simu_file_path=path+h_simu_file          #the output of simulation


    #Different cases of Data Assimilation.
    case=['1_24_self_rms_lr01','4_24_self_rms_lr01','1_12_self_rms_lr01','4_12_self_rms_lr01',
    '1_24_hec_rms_lr01','4_24_hec_rms_lr01','1_12_hec_rms_lr01','4_12_hec_rms_lr01']
    case1_suffix=['_T1.txt','_T2.txt','_T3.txt','_T4.txt']         
    case1_suffix=case1_suffix+['_T5.txt','_T6.txt','_T7.txt','_T8.txt'] 

    #obseration file for data assimilation
    obs_file=['eat8a_obs'+s for s in case1_suffix]  

    #posterior file and loss file for data assimilation
    z_posterior_file=['eat8a_z_var_posterior'+s for s in case1_suffix]
    loss_file=['eat8a_loss'+s for s in case1_suffix]

    #grad_info
    sum_grad_info_file=['eat8a_sum_grad'+s for s in case1_suffix]
    y_pred_grad_info_file=['eat8a_y_pred_grad'+s for s in case1_suffix]


    #=================****===============================
    simulation=True
    simulation=False              
    if simulation:
        main_simulation(file_path_list,simu_paras,h_simu_file_path)
    
    make_obs_=True          #set
    make_obs_=False       

    #--get the observation index
    lu_corner_x,lu_corner_y=[263976,664810]  #left up corner of the DEM

    #***
    observation_point=np.loadtxt('eat8a/observation.txt',comments='#',delimiter=',')[:,1:]
    # print(observation_point)
    observation_point[:,0]-=lu_corner_x
    observation_point[:,1]-=lu_corner_y
    obs_index=observation_point//2
    obs_index[:,1]*=-1
    obs_index=obs_index.astype(int)
    obs_index=np.array([obs_index[:,1],obs_index[:,0]]).T
    print(obs_index)
    if make_obs_:
        obs_interval=60
        make_obs(path,
                h_simu_file,
                dem_shape=(201,483),
                obs_file=obs_file[1],           #****
                obs_index=obs_index,
                obs_interval=obs_interval)    
                
    DA=True
    # DA=False
    DA_file_path_list=file_path_list.copy()
    DA_file_path_list[1]=path+z_prior              #repalce the z_true file with the z_prior file
    if DA:
        single_case=True
        # single_case=False
        if single_case:
            i=1      #***
            result_files={'z_var_posterior_file':z_posterior_file[i],
                'loss_file':loss_file[i],
                'sum_grad_info_file':sum_grad_info_file[i],
                'y_pred_grad_info_file':y_pred_grad_info_file[i],                
                }
            main_DA(DA_file_path_list,path,obs_file[i],result_files)

        else:
            for i in range(0,8):
                result_files={'z_var_posterior_file':z_posterior_file[i],      #nescesary
                            'loss_file':loss_file[i],                          #nescesary                                  
                            'sum_grad_info_file':sum_grad_info_file[i],        #or can be None that will be not saved
                            'y_pred_grad_info_file':y_pred_grad_info_file[i]}  #or can be None that will be not saved
                main_DA(DA_file_path_list,path,obs_file[i],result_files)

    rains_test=True
    rains_test=False
    if rains_test:
        rains=["1","2","3"]
        obs_plan_numbers=[0,1]
        for number in obs_plan_numbers:
            for r in rains:
                rain=r
                obs_plan_number=number
                file_path_list[5]=path+'tian_rain_map_R'+rain+'.txt'
                test_paras={"t0":0,
                            "t_end":1200,
                            "t_interval":60,                
                            "obs_mode":'cell_to_cell',
                            "loss_type":3,
                            "Gaussian_Gamma":1.0,
                            "dem_shape":(30,30),
                            "obs_index": obs_index,          
                            "z_posterior_file":z_posterior_file[obs_plan_number],
                            "h_simu_true_file":"h_sim_true_rain"+rain+case1_suffix[obs_plan_number],
                            "obs_true_file":"obs_true_rain"+rain+case1_suffix[obs_plan_number],
                            "test_loss_file":"test_loss_rain"+rain+case1_suffix[obs_plan_number]
                            }
                # obs_file_path=path+'tian_obs_C1T5.txt'
                test_by_rains(path,file_path_list,test_paras)



if __name__ == '__main__':
    main_1()
    main_2()




