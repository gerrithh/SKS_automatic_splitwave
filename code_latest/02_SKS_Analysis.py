
# coding: utf-8

# In[2]:


## Process AlpArray Switzerland for SKS-Splitting
## Gerrit Hein
######################################################
######### LOAD IN THE MODULES
######################################################
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:75% !important; }</style>"))

import matplotlib.pyplot as plt
import numpy as np

import os
import obspy
from obspy import read
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from obspy.taup import TauPyModel
from obspy.geodetics.base import gps2dist_azimuth
from obspy.geodetics import locations2degrees
from obspy.taup import plot_travel_times
from obspy.geodetics import locations2degrees
from obspy.signal.rotate import rotate_ne_rt
from obspy.signal.polarization import particle_motion_odr

from obspy.signal.util import next_pow_2

import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit

from matplotlib.mlab import specgram
from scipy import stats

from itertools import izip_longest as zip_longest
#from itertools import zip_longest as zip_longest


from tqdm import tqdm
import time

import multiprocessing

import splitwavepy as sw
######################################################
### 
######################################################



# In[123]:


### FUNCTIONS TO LOAD IN THE RESULTS

# def read_SKS_files(path,filename):
#     filename = '{0}/{1}'.format(path,filename) #removed first two header lines
#     with open(filename) as f:
#         content = f.readlines()

#     station = ['']*len(content) 
#     dt = ['']*len(content) 
#     dtlag  = ['']*len(content) 
#     fast_dir = ['']*len(content) 
#     dfast_dir = ['']*len(content) 

#     for i in range(1,len(content)-1):
#         data = zip_longest(*(x.split(' ') for x in content[i].splitlines()), fillvalue='\t')

#         for row in zip(*data):
#             new_data = tuple(np.nan if x == '' else x for x in row)
#             line = new_data
#             station[i] = line[0][1:-1]            
#             dt[i] = float(line[1][1:-2])
#             dtlag[i] = float(line[2][1:-2])          
#             fast_dir[i] = float(line[3][1:-2])            
#             dfast_dir[i] = float(line[4][1:-2])                        

#     station = np.asarray(station[1:-1])
#     dt = np.asarray(dt[1:-1])
#     dtlag = np.asarray(dtlag[1:-1])
#     fast_dir = np.asarray(fast_dir[1:-1])
#     ## convert from -90-90 to 0-180    
#     fast_dir = (fast_dir+180)%180    
#     dfast_dir = np.asarray(dfast_dir[1:-1])    
        
#     return station,dt,dtlag,fast_dir,dfast_dir

def calc_u_v(r,phi):
    
    if (phi>=0 and phi<=90):
        phi=90-phi
        u = r*np.cos(np.deg2rad(phi))
        v = r*np.sin(np.deg2rad(phi))
    elif (phi>90 and phi<=180):
        phi=180-phi
        u = r*np.sin(np.deg2rad(phi))
        v = -r*np.cos(np.deg2rad(phi))    
    elif (phi>180 and phi<=270):
        phi=270-phi
        u = -r*np.cos(np.deg2rad(phi))
        v = -r*np.sin(np.deg2rad(phi))    
    elif (phi>270 and phi<=360):
        phi=360-phi
        u = -r*np.sin(np.deg2rad(phi))
        v = r*np.cos(np.deg2rad(phi))    
    elif (phi>=-90 and phi<0):
        phi=abs(phi)
        u = -r*np.sin(np.deg2rad(phi))
        v = r*np.cos(np.deg2rad(phi))            
    elif (phi>=-180 and phi<-90):
        phi=180-abs(phi)
        u = -r*np.sin(np.deg2rad(phi))
        v = -r*np.cos(np.deg2rad(phi))                    
        
    return u,v

def read_SKS_files(path,filename):
    filename = '{0}/{1}'.format(path,filename) #removed first two header lines
    with open(filename) as f:
        content = f.readlines()

    station = ['']*len(content) 
    dt = ['']*len(content) 
    dtlag  = ['']*len(content) 
    fast_dir = ['']*len(content) 
    dfast_dir = ['']*len(content)
    best_twin = ['']*len(content)
    chi_phi = ['']*len(content) 
    chi_dt = ['']*len(content) 
    p_phi = ['']*len(content) 
    p_dt = ['']*len(content) 
    n_events = ['']*len(content) 
    

    for i in range(1,len(content)):
        data = zip_longest(*(x.split(' ') for x in content[i].splitlines()), fillvalue='\t')

        for row in zip(*data):
            new_data = tuple(np.nan if x == '' else x for x in row)
            line = new_data
            station[i] = line[0][1:-1]            
            fast_dir[i] = float(line[1][1:-2])            
            dfast_dir[i] = float(line[2][1:-2])                                    
            dt[i] = float(line[3][1:-2])
            dtlag[i] = float(line[4][1:-2])          
            best_twin[i] = float(line[5][1:-2])          
            chi_phi[i] = float(line[6][1:-2]) 
            p_phi[i] = float(line[7][1:-2]) 
            chi_dt[i] = float(line[8][1:-2]) 
            p_dt[i] = float(line[9][1:-2])  
            n_events[i] = float(line[10][1:-1])

    station = np.asarray(station[1:])
    dt = np.asarray(dt[1:])
    dtlag = np.asarray(dtlag[1:])
    fast_dir = np.asarray(fast_dir[1:])
    ## convert from -90-90 to 0-180    
    fast_dir = (fast_dir+180)%180    
    dfast_dir = np.asarray(dfast_dir[1:])    
    best_twin = np.asarray(best_twin[1:])    
    chi_phi = np.asarray(chi_phi[1:])
    chi_dt = np.asarray(chi_dt[1:])
    p_phi = np.asarray(p_phi[1:])
    p_dt = np.asarray(p_dt[1:]) 
    n_events = np.asarray(n_events[1:]) 

        
    return station,dt,dtlag,fast_dir,dfast_dir,best_twin,chi_phi,p_phi,chi_dt,p_dt,n_events



# Function to read in OTHER SKS Files 

def read_SKS_methods(save_loc,method,station):
    
    filename= 'SKS_Splitting_{0}_{1}.txt'.format(station,method)
    filepath = '{0}/{1}/{2}'.format(save_loc,method,filename)

    
#    filename = '{0}/{1}'.format(path,filename) #removed first two header lines
    with open(filepath) as f:
        content = f.readlines()

    station = ['']*len(content) 
    st_lat  = ['']*len(content)
    st_lon = ['']*len(content)
    ev_time = ['']*len(content)
    ev_depth = ['']*len(content)
    ev_mag = ['']*len(content)
    ev_lat = ['']*len(content)
    ev_lon = ['']*len(content)
    fast_dir = ['']*len(content)
    dfast_dir = ['']*len(content)
    lag = ['']*len(content)
    dlag = ['']*len(content)
    SNR  = ['']*len(content)
    ### 
    window_param_t1 = ['']*len(content)
    window_param_t2 = ['']*len(content)
    chi_phi = ['']*len(content) 
    chi_dt = ['']*len(content) 
    p_phi = ['']*len(content) 
    p_dt = ['']*len(content)
        
    
    for i in range(1,len(content)):
        data = zip_longest(*(x.split(' ') for x in content[i].splitlines()), fillvalue='\t')
        for row in zip(*data):
            new_data = tuple(np.nan if x == '' else x for x in row)
            line = new_data
            station[i] = line[0][1:-1]            
#            print(station)
            st_lat[i] = float(line[1][1:-1])
            st_lon[i] = float(line[2][1:-1]) 
            ev_time[i] = UTCDateTime(float(line[3][1:-1]))          
            ev_depth[i] = float(line[4][1:-1])  
            ev_mag[i] = float(line[5][1:-1])  
            ev_lat[i] = float(line[6][1:-1])  
            ev_lon[i] = float(line[7][1:-1])  
            fast_dir[i] = float(line[8][1:-1])  
            dfast_dir[i] = float(line[9][1:-1])  
            lag[i] = float(line[10][1:-1])  
            dlag[i] = float(line[11][1:-1])        
            SNR[i] = float(line[12][1:-1]) 
            window_param_t1[i] = float(line[13][1:-1])             
            window_param_t2[i] = float(line[14][1:-1])             
            chi_phi[i] = float(line[15][1:-1])             
            p_phi[i] = float(line[16][1:-1])             
            chi_dt[i] = float(line[17][1:-1])             
            p_dt[i] = float(line[18][1:-1])                    


            
#             if line[11][1:-1]=='nan':
#                 SNR[i] = np.nan
#             else:
#                 SNR[i] = float(line[12][1:-1])

    station = np.asarray(station[1:])
    print(station)    
    st_lat = np.asarray(st_lat[1:])
    st_lon = np.asarray(st_lon[1:])
    ev_time = np.asarray(ev_time[1:])
    ev_depth = np.asarray(ev_depth[1:])
    ev_mag = np.asarray(ev_mag[1:])
    ev_lat = np.asarray(ev_lat[1:])
    ev_lon = np.asarray(ev_lon[1:])
    
    fast_dir = np.asarray(fast_dir[1:])
    ## convert from -90-90 to 0-180
    fast_dir = (fast_dir+180)%180
    
    dfast_dir = np.asarray(dfast_dir[1:])
    lag = np.asarray(lag[1:])
    dlag = np.asarray(dlag[1:])
#    print(SNR)    
    SNR = np.asarray(SNR[1:])
#    print(SNR)    

    window_param_t1 = np.asarray(window_param_t1[1:])
    window_param_t2 = np.asarray(window_param_t2[1:])
    chi_phi = np.asarray(chi_phi[1:])
    p_phi = np.asarray(p_phi[1:])            
    chi_dt = np.asarray(chi_dt[1:])

    p_dt = np.asarray(p_dt[1:]) 
    
    
    return station,st_lat,st_lon,ev_time,ev_depth,ev_mag,ev_lat,ev_lon,fast_dir,dfast_dir,lag,dlag,SNR,window_param_t1,window_param_t2,chi_phi,p_phi,chi_dt,p_dt


# In[124]:


for istation in tqdm(station_list[0:1]):
# for istation in tqdm(station_list[0:8]):       40
    method = ['CrossC','TransM','EigM']
    method = method[0]
    print(istation)
    station_meth,st_lat,st_lon,ev_time,ev_depth,ev_mag,ev_lat,ev_lon,fast_dir,dfast_dir,lag,dlag,SNR,window_param_t1,window_param_t2,chi_phi,p_phi,chi_dt,p_dt = read_SKS_methods(save_dir,method,istation)
    print(st_lat)#,st_lon,ev_time,ev_depth,ev_mag,ev_lat,ev_lon,fast_dir,dfast_dir,lag,dlag,SNR,window_param_t1,window_param_t2,chi_phi,p_phi,chi_dt,p_dt)
#    print(station_meth[0])


# In[13]:



### FIRST LOAD IN THE BARRUOL RESULTS
## load in Barruol values
import csv

def load_in_Barruol_table(path):  

    station = ['']*60
    lat = ['']*60
    lon = ['']*60
    fast = ['']*60
    dfast= ['']*60
    lag= ['']*60
    dlag= ['']*60
    nSKS = ['']*60

    with open(path+'Barruol_values.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        for row in csv_reader:
    #        print(row)
            if line_count == 0:
                print('HEADER')
                print(row)
                line_count += 1
            else:
                station[line_count] = row[0]
                lat[line_count] = float(row[1])
                lon[line_count] = float(row[2])
                a = ['1','2','3','4','5','6','7','8','9','0']
                if row[3][0] in a:
                    fast[line_count] = float(row[3][-5:])
                else:
                    fast[line_count] = -1*float(row[3][-5:])

                dfast[line_count] = float(row[4])
                lag[line_count] = float(row[5])
                dlag[line_count] = float(row[6])
                nSKS[line_count] = float(row[7])            

                line_count += 1
    #     print(f'Processed {line_count} lines.')
    station = filter(None, station)
    lat = filter(None, lat)
    lon = filter(None, lon)
    fast = filter(None, fast)
    dfast = filter(None, dfast)
    lag = filter(None, lag)
    dlag = filter(None, dlag)
    nSKS = filter(None, nSKS)
    
    return station,lat,lon,fast,dfast,lag,dlag,nSKS



# In[14]:





# In[5]:


## Tasks: 
## make a Plan:

### after having best window run the chi squared test for chevrot 
## --> read results, take best timewindow
## --> read Waveforms
## --> run Chevrot again, with variable number of Events
## --> calc chi_squared of the fit for that 

## --> https://stackoverflow.com/questions/52591979/how-to-obtain-the-chi-squared-value-as-an-output-of-scipy-optimize-curve-fit
## --> write out Chi-Square for Hyp: Vals fit Gaussian Sinosoid, 

## plot number of Events vs chi-Squared


### comparison with tw analysis from others
## weighted histograms of 3 methods with best fit of Chevrot and Barruol
## get some statement about it 



### plot Switzerland maps
## best Chevrot vs Barruol
## best of the other three
## directions of fast axis
## length of dt
## Balls of Chi-Squared


### AND THEN START WRITING IT DOWN AGAIN
### READ MORE, GET NEW IDEAS


# In[6]:



#plt.hist(p_phi,range=(0,1),color='b')
#plt.hist(p_dt,range=(0,1),color='r')
## only look at upper 80 %

#print(p_dt[np.where(n_events==np.min(n_events))])


# In[15]:


print(station)


# In[6]:


## adapt 


# In[7]:


### READ IN THE OTHER RESULTS


save_loc = '/media/hein/home2/SplitWave_Data'
#save_loc = '/home/rud/Dropbox/PhD/code_PhD/SplitWave'
station_list = os.listdir(save_loc)

save_dir = '/media/hein/home2/SplitWave_Results/Splitws/Time_Window_Test/TXT/'
#save_dir = '/home/rud/Dropbox/PhD/code_PhD/SplitWave_Results/Comparison'
#station = station_list[np.random.randint(0,len(station_list))]

#for station in tqdm(station_list):
for istation in tqdm(station_list[0:8]):    
#    print(station)

    method = ['CrossC','TransM','EigM']
    method = method[1]
    station,st_lat,st_lon,ev_time,ev_depth,ev_mag,ev_lat,ev_lon,fast_dir,dfast_dir,lag,dlag,SNR,window_param_t1,window_param_t2,chi_phi,p_phi,chi_dt,p_dt = read_SKS_methods(save_dir,method,istation)

    fig = plt.figure(figsize=(16,9))
    plt.subplot(1,3,1)
    plt.hist(fast_dir,range=(0,180),color='blue',alpha=0.5,normed=True,label='{0} not-weighted'.format(method))
    plt.hist(fast_dir,weights=1/chi_phi*SNR,color='green',alpha=0.5,normed=True,label='weighted')
    plt.xlabel('Fast dir [degrees]')    
    plt.legend()
    
    method = ['CrossC','TransM','EigM']
    method = method[0]
    station,st_lat,st_lon,ev_time,ev_depth,ev_mag,ev_lat,ev_lon,fast_dir,dfast_dir,lag,dlag,SNR,window_param_t1,window_param_t2,chi_phi,p_phi,chi_dt,p_dt = read_SKS_methods(save_dir,method,istation)
    plt.subplot(1,3,2)
    plt.hist(fast_dir,color='blue',alpha=0.5,normed=True,label='{0} not-weighted'.format(method))
    plt.hist(fast_dir,weights=1/chi_phi*SNR,color='green',alpha=0.5,normed=True,label='weighted')
    plt.xlabel('Fast dir [degrees]')    
    plt.legend()    
    
    method = ['CrossC','TransM','EigM']
    method = method[2]
    station,st_lat,st_lon,ev_time,ev_depth,ev_mag,ev_lat,ev_lon,fast_dir,dfast_dir,lag,dlag,SNR,window_param_t1,window_param_t2,chi_phi,p_phi,chi_dt,p_dt = read_SKS_methods(save_dir,method,istation)
    plt.subplot(1,3,3)
    plt.hist(fast_dir,color='blue',alpha=0.5,normed=True,label='{0} not-weighted'.format(method))
    plt.hist(fast_dir,weights=1/chi_phi*SNR,color='green',alpha=0.5,normed=True,label='weighted')
    plt.xlabel('Fast dir [degrees]')
    plt.suptitle(station[0])
    plt.legend()
    plt.savefig('/media/hein/home2/SplitWave_Results/Project_images/Quick_plot_{0}.png'.format(station[0]))


# In[190]:




print(station_list[1])
istation=station_list[0]
station,st_lat,st_lon,ev_time,ev_depth,ev_mag,ev_lat,ev_lon,fast_dir,dfast_dir,lag,dlag,SNR,window_param_t1,window_param_t2,chi_phi,p_phi,chi_dt,p_dt = read_SKS_methods(save_dir,method,istation)
plt.hist(fast_dir,color='blue',alpha=0.5,range=(0,180))
plt.hist(fast_dir,weights=1/chi_phi*SNR,color='green',alpha=0.5)
plt.hist(fast_dir,weights=p_phi*SNR,color='red',alpha=0.5)

# plt.hist(fast_dir,color='blue',alpha=0.5)
# plt.hist(fast_dir,weights=p_phi*SNR,color='green',alpha=0.5)

# plt.hist(fast_dir,color='blue',alpha=0.5)
# plt.hist(fast_dir,weights=p_phi*SNR,color='green',alpha=0.5)



#plt.hist(lag,color='blue',alpha=0.5)
#plt.hist(lag,weights=p_dt*SNR,color='green',alpha=0.5)


#plt.hist(window_param_t1)
#plt.hist(window_param_t2)


# In[ ]:



save_loc = '/media/hein/home2/SplitWave_Data'
print(save_loc)


FMT = 'SAC'
Splitting_windows = True
plot_SplitW = False
mod = 'ak135' ##mod = 'iasp91'
model = TauPyModel(model=mod)

new_stat_list = os.listdir(save_loc) ## Get A list of downloaded Stations

### CHOOSE STATION and load all waveforms 
method= 'Chevrot'
write_head_CHEV(txt_path, method,header2)

########################################################################################################################################################################
########################################################################################################################################################################
#for istation in range(0,len(new_stat_list)):
#for istation in range(32,len(new_stat_list)):    
for istation in range(0,1):    
### LOOOP AROUND    
    station = new_stat_list[istation]

    print('Loading in Waveforms')
    st_ev = obspy.Stream()
    st_ev,st_lat_l,st_lon_l,ev_lat_l,ev_lon_l,ev_time_l,ev_depth_l,ev_mag_l,ev_dist_l,back_azimut_l, t_SKS_l,t_SKKS_l,t_PP_l = read_station_event_data(station)
    ########################################################################################################################################################################
    ########################################################################################################################################################################
    ### MAIN FUNCTION TO FIRST CALCULATE CHEVROT FOR ALL EVENTS

    ntwindows = 60  ### test 50 timw windows

    Az_l  = np.zeros((len(ev_time_l),ntwindows))
    SV_Az_l = np.zeros((len(ev_time_l),ntwindows))



    for nevent in range(0,len(ev_time_l)):
    #for nevent in range(0,3):    

        st_cut = n_event_st_cut(nevent)
        t_SKS_real = get_real_SKS_arrival(st_cut)
        ### BLOCK FOR CHEVROT

        twindows = np.linspace(1,90,ntwindows)
        
        #twindows = 0.1

        for iwindow in range(0,len(twindows)):
        #    print(twindows[iwindow])

            Az,SV_Az = SKS_Intensity_Chevrot(st_cut,ev_time_l[nevent],float(t_SKS_real)-float(ev_time_l[nevent]),back_azimut_l[nevent],twindows[iwindow],plot=False)
            Az_l[nevent,iwindow] = Az
            SV_Az_l[nevent,iwindow] = SV_Az



    val_4 = np.zeros((ntwindows,4))

    for iwin in range(0,len(twindows)):

        dt,phi,std_dt,std_phi = get_best_dt_and_phi(Az_l[:,iwin],SV_Az_l[:,iwin],st_cut[0].stats.station)
        val_4[iwin,:] = (dt,phi,std_dt,std_phi)
            ## write out results for Chevrot

    #### CHI2 TEST
    chisquare_dt4,p_dt4 = stats.chisquare(val_4[:,0])
    chisquare_phi4,p_phi4 = stats.chisquare((val_4[:,1]+360)%180)                           


    vals = [str(st_cut[0].stats.station),
           val_4[np.argmin(val_4[:,3]),1],
           val_4[np.argmin(val_4[:,3]),3],
           val_4[np.argmin(val_4[:,2]),0],
           val_4[np.argmin(val_4[:,2]),2],
           chisquare_phi4,
           p_phi4,
           chisquare_dt4,
           p_phi4,
           len(ev_time_l)]
    method = 'Chevrot'    
    write_SKS_Results_CHEV(txt_path, method, vals, header2) 
    


# In[124]:


## improve Map
## make Topographic one
##### MOVE ALL THE PLOTTING DOWN HERE
### to make a map of study area
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import shapely.geometry as sgeom
#from scalebar import scale_bar

figpath='/media/hein/home2/SplitWave_Results/Project_images'


## use topography map?
plt.figure(figsize=(16,9))

proj = ccrs.PlateCarree()

ax = plt.axes(projection=proj)
#ax = plt.axes([0.2,0.2,0.75,0.75])
#ax.set_extent([5, 11, 45, 48],proj)
ax.set_extent([4, 11, 42, 48],proj)

#ax = fig.add_axes([0.2,0.2,0.75,0.75])    


#ax.set_extent([0, 45, 0, 90],proj)

places =  cfeature.NaturalEarthFeature('cultural','populated_places','10m',facecolor='black')
land = cfeature.NaturalEarthFeature('physical','land','10m',
            edgecolor='k',facecolor='lightgoldenrodyellow',)

rivers = cfeature.NaturalEarthFeature(category='physical',name='rivers_lake_centerlines',scale='110m')

graticules = cfeature.NaturalEarthFeature(category='physical',name='graticules_1',scale='110m',facecolor='gray')
bounding_box = cfeature.NaturalEarthFeature(category='physical',name='wgs84_bounding_box',scale='10m',facecolor='none')
physical_building_blocks = cfeature.NaturalEarthFeature(category='physical',name='land_ocean_label_points',scale='10m',facecolor='gray')


geography_regions_points=cfeature.NaturalEarthFeature(
    category='physical',
    name='geography_regions_elevation_points',
    scale='10m',
    facecolor='black')

borders = cfeature.NaturalEarthFeature('cultural', 'admin_0_boundary_lines_land','10m',
            edgecolor='black',facecolor='none')
lakes = cfeature.NaturalEarthFeature(category='physical',name='lakes_europe',scale='10m')
states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='10m',
    facecolor='none')
geoprahic_lines = cfeature.NaturalEarthFeature(
    category='physical',
    name='geographic_lines',
    scale='10m',
    facecolor='black')





SOURCE = 'Natural Earth'
LICENSE = 'public domain'

ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.COASTLINE)

#ax.add_feature(rivers)
ax.add_feature(lakes)
ax.add_feature(states_provinces, edgecolor='none')
ax.add_feature(borders)
#ax.add_feature(geoprahic_lines)#
#ax.add_feature(graticules)
#ax.add_feature(geography_regions_points)

#ax.background_img()
ax.background_img(name='gray-earth', resolution='low')

ax.plot(lon,lat,'^r',transform=ccrs.PlateCarree(),markersize=10,zorder=11)

for i in range(0,len(station)):
    ax.annotate(station[i],(lon[i],lat[i]-0.1),transform=ccrs.PlateCarree(),
        ha='center',va='top',weight='bold')
    
# r = dtlag
# phi = fast_dir
r = lag
phi = fast

for i in range(0,len(phi)):
    u,v = calc_u_v(r[i],phi[i])
    ax.quiver(lon[i], lat[i], u, v,pivot='mid',color='green',width=0.003, zorder=10)    
    
    
ax.quiver(5+0.2,48-0.1,1,0,width=0.003,color='black')
ax.annotate('dt=1 s',(5+0.3,48-0.2),transform=ccrs.PlateCarree(), ha='center',va='top',weight='bold')
#ax.background_img(name='BM', resolution='low')
#ax.background_img()
ext = [5, 11, 45, 48]

############# For Small Plot

# sub_ax = plt.axes([0.55,0.12,0.25,0.25], projection=proj)
# # Add coastlines and background
# sub_ax.coastlines()
# sub_ax.background_img()
# # Plot box with position of main map
# extent_box = sgeom.box(ext[0],ext[2],ext[1],ext[3])

# sub_ax.add_geometries([extent_box], proj, color='none',
#                       edgecolor='red', linewidth=3)

# sub_ax.background_img()
# sub_ax.plot(ev_lon,ev_lat,'*y',transform=proj,markersize=7)
#scale_bar(ax,(0.75,0.05),10)

### plot EQ location and Great circle path
plt.savefig('{0}/Resulting_Map_Barruol.png'.format(figpath),dpi=150)
plt.show()



# In[51]:


## improve Map
## make Topographic one
##### MOVE ALL THE PLOTTING DOWN HERE
### to make a map of study area
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import shapely.geometry as sgeom
#from scalebar import scale_bar

figpath='/media/hein/home2/SplitWave_Results/Project_images'

plt.figure(figsize=(16,9))
plt.rcParams.update({'font.size': 18})

proj = ccrs.PlateCarree()

ax = plt.axes(projection=proj)
#ax = plt.axes([0.2,0.2,0.75,0.75])
ax.set_extent([5, 11, 45, 48],proj)
#ax.set_extent([4, 11, 42, 48],proj)
#ax = fig.add_axes([0.2,0.2,0.75,0.75])    
#ax.set_extent([0, 45, 0, 90],proj)

places =  cfeature.NaturalEarthFeature('cultural','populated_places','10m',facecolor='black')
land = cfeature.NaturalEarthFeature('physical','land','10m',
            edgecolor='k',facecolor='lightgoldenrodyellow',)

rivers = cfeature.NaturalEarthFeature(category='physical',name='rivers_lake_centerlines',scale='110m')

graticules = cfeature.NaturalEarthFeature(category='physical',name='graticules_1',scale='110m',facecolor='gray')
bounding_box = cfeature.NaturalEarthFeature(category='physical',name='wgs84_bounding_box',scale='10m',facecolor='none')
physical_building_blocks = cfeature.NaturalEarthFeature(category='physical',name='land_ocean_label_points',scale='10m',facecolor='gray')


geography_regions_points=cfeature.NaturalEarthFeature(
    category='physical',
    name='geography_regions_elevation_points',
    scale='10m',
    facecolor='black')

borders = cfeature.NaturalEarthFeature('cultural', 'admin_0_boundary_lines_land','10m',
            edgecolor='black',facecolor='none')
lakes = cfeature.NaturalEarthFeature(category='physical',name='lakes_europe',scale='10m')
states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='10m',
    facecolor='none')
geoprahic_lines = cfeature.NaturalEarthFeature(
    category='physical',
    name='geographic_lines',
    scale='10m',
    facecolor='black')

ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(lakes)
ax.add_feature(states_provinces, edgecolor='none')
ax.add_feature(borders)
#ax.add_feature(geoprahic_lines)#
#ax.add_feature(graticules)
#ax.add_feature(geography_regions_points)

## CHANGE BACKROUND HERE
#ax.background_img()
#ax.background_img(name='gray-earth', resolution='low')
#ax.background_img(name='topography', resolution='low')
#ax.background_img(name='NE1_HR', resolution='low')
ax.background_img(name='ne_shaded_real', resolution='low')

#########################################################################################################
### CHANGE COORDINATES HERE
## needed lat,lon,station,fast,dt, xi for ball, error for more quiver
path = '/media/hein/home2/SKS_automatic_splitwave/code_latest/'
station,lat,lon,fast,dfast,lag,dlag,nSKS = load_in_Barruol_table(path)


#########################################################################################################

### PLOT BARRUOL
ax.plot(lon,lat,'^r',transform=ccrs.PlateCarree(),markersize=10,zorder=11)

for i in range(0,len(station)):
    ax.annotate(station[i],(lon[i],lat[i]-0.1),transform=ccrs.PlateCarree(),
        ha='center',va='top',weight='bold')
    
# r = dtlag
# phi = fast_dir
r = dt
phi = fast_dir
for i in range(0,len(phi)):
    u,v = calc_u_v(r[i],phi[i])
    ax.quiver(lon[i], lat[i], u, v,pivot='mid',color='black',width=0.003, zorder=10,headlength=0, headwidth = 1)    

    
ax.quiver(lon[i], lat[i], u, v,pivot='mid',color='black',width=0.003, zorder=10,headlength=0, headwidth = 1,label='Barruol')

ax.quiver(5+0.2,48-0.1,1,0,width=0.003,color='black',headlength=0, headwidth = 1)
ax.annotate('dt=1 s',(5+0.3,48-0.2),transform=ccrs.PlateCarree(), ha='center',va='top',weight='bold')

### READ IN THE OTHER METHODS AND PLOT BEST RESULTS PER STATION, 
save_loc = '/media/hein/home2/SplitWave_Data'
station_list = os.listdir(save_loc)

save_dir = '/media/hein/home2/SplitWave_Results/Splitws/Time_Window_Test/TXT/'
#save_dir = '/home/rud/Dropbox/PhD/code_PhD/SplitWave_Results/Comparison'

for istation in tqdm(station_list):
# for istation in tqdm(station_list[0:8]):       
    method = ['CrossC','TransM','EigM']
    method = method[1]
    print(istation)
    station_meth,st_lat,st_lon,ev_time,ev_depth,ev_mag,ev_lat,ev_lon,fast_dir,dfast_dir,lag,dlag,SNR,window_param_t1,window_param_t2,chi_phi,p_phi,chi_dt,p_dt = read_SKS_methods(save_dir,method,istation)
    print(station_meth[0])
    
    ###

### READ IN THE CHEVROT RESULTS
path = '/media/hein/home2/SplitWave_Results/Splitws/Time_Window_Test/TXT'
filename = 'SKS_Splitting_Chevrot_copy.txt'
station_chev,dt,dtlag,fast_dir,dfast_dir,best_twin,chi_phi,p_phi,chi_dt,p_dt,n_events = read_SKS_files(path,filename)



############# For Smaller Plot
#ax.background_img(name='BM', resolution='low')
#ax.background_img()
#ext = [5, 11, 45, 48]

# sub_ax = plt.axes([0.55,0.12,0.25,0.25], projection=proj)
# # Add coastlines and background
# sub_ax.coastlines()
# sub_ax.background_img()
# # Plot box with position of main map
# extent_box = sgeom.box(ext[0],ext[2],ext[1],ext[3])

# sub_ax.add_geometries([extent_box], proj, color='none',
#                       edgecolor='red', linewidth=3)

# sub_ax.background_img()
# sub_ax.plot(ev_lon,ev_lat,'*y',transform=proj,markersize=7)
#scale_bar(ax,(0.75,0.05),10)

### plot EQ location and Great circle path
handles, labels = ax.get_legend_handles_labels()

ax.legend(handles=handles,labels=labels)

plt.savefig('{0}/Resulting_Map_compare.png'.format(figpath),dpi=150)
plt.show()



# In[58]:


for istation in tqdm(station_list):
# for istation in tqdm(station_list[0:8]):       
    method = ['CrossC','TransM','EigM']
    method = method[0]
    print(istation)
    station_meth,st_lat,st_lon,ev_time,ev_depth,ev_mag,ev_lat,ev_lon,fast_dir,dfast_dir,lag,dlag,SNR,window_param_t1,window_param_t2,chi_phi,p_phi,chi_dt,p_dt = read_SKS_methods(save_dir,method,istation)
    print(station_meth[0])

#print(station[np.argwhere(station=='ZUR')])
## TASKS:
# Plot
## Barruol
## Chevrot
## Others
## error Spread
## 
## dt 
## Fast
## Chi_Suqqra

