import numpy as np
import matplotlib.pyplot as plt
# import os


class PostProcess:
    def __init__(self):
        pass

    def save_h(self,h,t_eval,file_name='h.txt'):
        header='Time (s)\tHeight (mm)\t rows and cols {}'.format(h.shape)+\
              'the first row is time, the rest are heights'  #create the header for the file
        h_info=np.vstack((t_eval,h))  #combine time and h into a 2D array
        np.savetxt(file_name,h_info,fmt='%0.4f',header=header,comments='#')  #save the predicted h to file
        print("h saved to file: {}".format(file_name))

    def save_z_var(self,z_var,epoch,file_name='z_var_posterior.txt'):
        if epoch==0:
            np.savetxt(file_name,z_var,fmt='%0.5f',header='the posterior values of z_var,epoch: {}'.format(epoch),comments='#')
        else:
            z_var_old=np.loadtxt(file_name,comments='#')
            z_var_new=np.column_stack((z_var_old,z_var))  #combine the old and new z_var into a 2D array
            np.savetxt(file_name,z_var_new,fmt='%0.5f',header='the posterior values of z_var,epoch: {}'.format(epoch),comments='#')
        print("z_var saved to file: {}".format(file_name))
        
    def save_loss(self,loss,file_name='loss.txt'):
        loss=np.array(loss)
        np.savetxt(file_name,loss,fmt='%0.5f',header='the loss values during training',comments='#')
        print("loss saved to file: {}".format(file_name))

    def save_y_pred_and_y_obs(self,y_pred,y_obs,file_name='y_pred_and_y_obs.txt'):
        y_pred_and_y_obs=np.vstack((y_pred,y_obs))  #combine y_pred and y_obs into a 2D array
        header='the predicted and observed values of y,repectively,y_pred.shape=y_obs.shape=\n{}'.format(y_pred.shape)
        np.savetxt(file_name,y_pred_and_y_obs,fmt='%0.5f',header=header,comments='#')
        print("y_pred_and_y_obs saved to file: {}".format(file_name))
    def save_sum_grad(self,sum_grad,epoch,as_one=False,path_file='grad_info.txt'):
        #shape of sum_grad: (|z_var|,t_eval)  
        if epoch==0:
            np.savetxt(path_file,sum_grad,fmt='%0.8f',header='the gradient sum',comments='#')
        else:
            if epoch==1 and as_one:
                content=np.expand_dims(np.loadtxt(path_file,comments='#'),-1).T
            else:
                content=np.loadtxt(path_file,comments='#') 
            content=np.concatenate((content,sum_grad),axis=0)  #combine the old and new sum_grad into a 2D array
            header='the gradient sum. Shape:{}, epoch: {}'.format(sum_grad.shape,epoch)
            np.savetxt(path_file,content,fmt='%0.8f',header=header,comments='#')
    def save_y_pred_grad(self,y_pred_grad,epoch,path_file='y_pred_grad_info.txt'):
        shape_old=y_pred_grad.shape
        shape_new=(shape_old[0]*shape_old[1],shape_old[2])
        y_pred_grad=np.reshape(y_pred_grad,shape_new)

        if epoch==0:
            np.savetxt(path_file,y_pred_grad,fmt='%0.8f',header='the gradient of y_pred,epoch: {}'.format(epoch),comments='#')
        else:
            content=np.loadtxt(path_file,comments='#')        
            content=np.concatenate((content,y_pred_grad),axis=0)  #combine the old and new y_pred_grad into a 2D array
            header='the gradient of y_pred. Shape:{}, epoch: {}'.format(shape_new,epoch)
            np.savetxt(path_file,content,fmt='%0.8f',header=header,comments='#')
    
    def plot_loss_paper(self,loss):
        """plot the loss curve with time for the paper"""

        fig, ax = plt.subplots()
        ax.plot(loss,'o-')
            # 设置对数尺度
        # ax.set_xscale('log')    
        ax.set_yscale('log')

        # 添加图表标题和坐标轴标签
        # ax.set_title('Logarithmic Scale Plot')
        ax.set_ylabel('Y-axis (log scale)')
        # ax.set_xlabel('Epoch')
        # plt.ylabel('Loss')
        # ax.title('Loss Curve')
        plt.show()
    def plot_z_var(self,z_var_posterior):
        """plot the z_var posterior with time"""
        time=np.arange(z_var_posterior.shape[1])
        plt.plot(time,z_var_posterior.transpose())
        plt.ylim([0,1.])
        plt.xlabel('Epoch')
        plt.ylabel('z_var')
        plt.title('Posterior of z_var')
        plt.show()

    def plo_h_form1(self,h_sim,h_DA,h_slice_sim,h_slice_DA,slice_time):
        # plot the water height with the form1 style
        #create the figure and four axes objects
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        #1. plot the simulated water height heatmap
        axes[0,0].imshow(h_sim,cmap='jet')
        axes[0,0].set_title('Simulated Water Height')
        axes[0,0].set_xlabel('X Coordinate')
        axes[0,0].set_ylabel('Y Coordinate')

        #2. plot the DA water height heatmap
        axes[0,1].imshow(h_DA,cmap='jet')
        axes[0,1].set_title('DA Water Height')
        axes[0,1].set_xlabel('X Coordinate')
        axes[0,1].set_ylabel('Y Coordinate')

        #3. plot the difference between h_sim and h_DA
        axes[1,1].imshow(h_sim-h_DA,cmap='jet')
        axes[1,1].set_title('Water Height Difference')
        axes[1,1].set_xlabel('X Coordinate')
        axes[1,1].set_ylabel('Y Coordinate')


        #4. plot the h_slice_sim and h_slice_DA at the slice_time.散点图
        axes[1,0].scatter(range(len(h_slice_sim)),h_slice_sim,label='h_slice_sim')
        axes[1,0].scatter(range(len(h_slice_DA)),h_slice_DA,label='h_slice_DA')
        axes[1,0].set_title('Water Height difference at Time {}'.format(slice_time))

        plt.show()



    #calculate the error between y_pred and y_obs,or h_simulation and h_DA. shape=(|*|,) or (|*|,times)
    def error_RMSE(self,y1,y2): 
        """calculate the root mean square error of y_pred and y_obs"""
        return np.sqrt(np.mean((y1-y2)**2))   #sqrt(1/n \sum_t (y_pred_t-y_obs_t)^2)

    def error_MAE(self,y1,y2):
        """calculate the mean absolute error of y_pred and y_obs"""
        return np.mean(np.abs(y1-y2))   #1/n \sum_t |y_pred_t-y_obs_t|
    def error_MRE(self,y1,y2):
        """calculate the Maximum Relative Error of y_obs and y_pred
            y1:  y_obs; y2: y_pred        
        """
        return np.max(np.abs(y1-y2)/np.abs(y1))       #max_k_l |y_pred_k_l-y_obs_k_l|/|y_obs_k_l|
        
    def error_MSE(self,y1,y2):
        """calculate the mean square error of y_pred and y_obs"""
        return np.mean((y1-y2)**2)   #1/n \sum_t (y_pred_t-y_obs_t)^2  


def plot_water_level(floder,h_f='/tian_h_DA.txt',dem_f='/tian_dem.txt'):
    dem_f=floder+dem_f
    h_f=floder+h_f
    dem_data = np.loadtxt(dem_f,comments='#')
    tian_h = np.loadtxt(h_f,comments='#')[1:,:]
    tian_h = tian_h.reshape(dem_data.shape[0],dem_data.shape[0],tian_h.shape[1])  # reshape the 1D array to 2D array

    # 绘制水深和DEM
    for i in range(40,41):
        plt.imshow(tian_h[:,:,i]/1000.+dem_data)  # 使用'terrain'颜色映射来模拟地形
        plt.colorbar()  # 显示颜色条以表示高度
        plt.title('Water Depth Image. Time Step: {}'.format(i))
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.show()

def plot_tian_case_z_var(floder,z_var_f='/tian_z_var_posterior.txt'):
    z_var_f=floder+z_var_f
    z_var_data = np.loadtxt(z_var_f,comments='#')
    z_var_data = z_var_data.reshape(2,-1)  # reshape the 1D array to 1D array
    
    plt.imshow(z_var_data,cmap='jet')
    plt.colorbar()
    plt.title('Posterior of z_var')
    
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    plt.show()
    

if __name__ == '__main__':
    floder='tian_case30'
    dem_shape=(30,30)
    # plot_water_level(floder,h_f='/tian_h_simu.txt')
    post_process=PostProcess()
    
    # plot_tian_case_z_var(floder)


    loss_data=np.loadtxt(floder+'/case1and2/tian_loss_C2T1.txt',comments='#')
    # loss_data_2000=np.loadtxt(floder+'/tian_loss_2000.txt',comments='#')
    # loss_data_Norm1=np.loadtxt(floder+'/tian_loss_Norm1.txt',comments='#')
    #对比绘图loss_data与loss_data_2000
    plt.plot(loss_data,color='b')
    # plt.plot(loss_data_2000,color='r')
    # plt.plot(loss_data_Norm1,color='g')
    plt.show()
    post_process.plot_loss_paper(loss_data)


    z_var_posterior_data=np.loadtxt(floder+'/case1and2/tian_z_var_posterior_C2T1.txt',comments='#')
    # z_var_posterior_data_NORM1=np.loadtxt(floder+'/tian_z_var_posterior_NORM1.txt',comments='#')
    # z_var_posterior_data_2000=np.loadtxt(floder+'/tian_z_var_posterior_2000.txt',comments='#')
    #对比绘图z_var_posterior_data与z_var_posterior_data_NORM1
    plt.plot(z_var_posterior_data.transpose(),color='b')
    # plt.plot(z_var_posterior_data_NORM1.transpose(),color='r')
    # plt.plot(z_var_posterior_data_2000.transpose(),color='g')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('z_var')
    plt.title('Posterior of z_var')
    plt.show()  #结论：更好地避免过拟合，提供了更多的信息，使得边远边也拟合的较好

    y_pred_and_y_obs_data=np.loadtxt(floder+'/tian_y_pred_and_y_obs.txt',comments='#')
    y_pred=y_pred_and_y_obs_data[0:y_pred_and_y_obs_data.shape[0]//2,:]
    y_obs=y_pred_and_y_obs_data[y_pred_and_y_obs_data.shape[0]//2:,:]
    # MAE=error_MAE(y_pred,y_obs)
    # RMSE=error_RMSE(y_pred,y_obs)
    # MSE=error_MSE(y_pred,y_obs)
    # print('MAE:',MAE,'MSE:',MSE,'RMSE:',RMSE)

    #绘制某个时刻模拟水深与数据同化水深的对比，
    time_slice=30
    h_sim_allTime=np.loadtxt(floder+'/tian_h_simu.txt',comments='#')
    h_DA_allTime=np.loadtxt(floder+'/tian_h_DA.txt',comments='#')
    h_sim_column=h_sim_allTime[1:,time_slice]
    h_DA_column=h_DA_allTime[1:,time_slice]
    h_sim_rectangle=h_sim_column.reshape(dem_shape[0],dem_shape[1])
    h_DA_rectangle=h_sim_column.reshape(dem_shape[0],dem_shape[1])
    post_process.plo_h_form1(h_sim_rectangle,h_DA_rectangle,h_sim_column,h_DA_column,time_slice)
    print("the inf norm between h_sim and h_DA at time {} is {}".format(time_slice,np.linalg.norm(h_sim_column-h_DA_column,np.inf)))
    
    
    

#z值升高对整个影响不大，反倒是减小到0影响很大，可能是z值升高的不够高。如果是扩大3倍哪。