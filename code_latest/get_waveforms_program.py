
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

import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit

from tqdm import tqdm
import time
import multiprocessing

import splitwavepy as sw
######################################################
### 
######################################################

### check for maximum value in window to identify real SKS Phase
### write out windows of waveforms..
### Plot all three channels with several incidence times
### Make a proper work flow:
### data exists? Any gaps or spikes?

## Parameters used in Walpols Study: 
## SKS epicentral distance between 95 < delta < 145° 
## distance from wikipedia: 60-141°
## SNR ratio >16
## weighted by SNR 
## Butterworth 0.02-0.1 Hz
## arrival time window between -15,-5 to +15,+30 s
## use eigenvalue method from silver and chan and cross correlation from Andow
## if both disagree by 45° --> Null measurements
## randomly sample and inspect for QC
## lambda2/lambda1 <0.025 (postcorrection), the smaller the fraction, the greater the linear particle motion
## SPOL-BAZ ??

###############################################################
######### CHECK STATIONS AND EVENT CATALOGUE
###############################################################
client = Client("ETH") # ORFEUS
station= '*'
starttime = UTCDateTime("2000-01-01T00:00:00.000")
endtime = UTCDateTime("2018-12-31T00:00:00.000")

inventory = client.get_stations(network="CH", station=station, starttime=starttime, endtime=endtime)

## Total number of 247 available Stations
#print(len(inventory[0]))

stationlist = inventory[0]
station = stationlist[10]
# print(station)
# print(station.code)
# print(station.longitude)
# print(station.latitude)
# print(station.elevation)
# print(station.start_date)
# print(station.end_date)


# 1. get events catalogue
## events after 1999
#cat = obspy.read_events("/home/hein/Dropbox/PhD/code_PhD/qcmt.ndk")
#cat2 = cat.filter("time > 2000-01-01T00:00", "magnitude >= 5.5", "depth <= 10000")
#cat2.write("/home/hein/Dropbox/PhD/code_PhD/qcmt_edit.cmt",format="CMTSOLUTION")

### ADAPT PATH for catalogue and for save_loc
machines = ['local','svrx']
machine=machines[0]

if machine=='local':   
    save_loc = '/media/hein/home2/SplitWave_Data/'
    cat = obspy.read_events("/home/hein/Dropbox/PhD/code_PhD/qcmt.ndk")
elif machine=='svrx':
    save_loc = '/raid/home/srvx7/gerrit/Project_1/Data/SplitWave_Data/'
    cat = obspy.read_events('/raid/home/srvx7/gerrit/Project_1/code_PhD/qcmt.ndk"')

#cat = cat.filter("time > {0}".format(station.start_date),"time < 2018-12-01T00:00:00")
cat = cat.filter("time > 2000-01-01T00:00","time < 2018-12-01T00:00:00")
#print(cat)
###############################################################
cat_m7 = cat.filter("magnitude >= 7.0")
#print(cat_m7)

counter = 0
long_list = []

for ev in cat_m7:    
    event1 = ev
    ### Event parameters
    orig  = event1.origins

    mag = event1.magnitudes[0]
    
    for station in stationlist:        
        ### check if station has recorded before event 
        if (float(station.start_date)<float(orig[0].time)) and (station.end_date == None):


            dist_deg = locations2degrees(orig[0].latitude, orig[0].longitude,
                           station.latitude, station.longitude)
            ## check wheter distance is in the right range            
            if (dist_deg>95) and (dist_deg<145):
                counter +=1

                tmp = [counter, str(station.code), station.start_date, station.latitude,
                                              station.longitude, orig[0].time,
                                    orig[0].latitude, orig[0].longitude, orig[0].depth, mag.mag, dist_deg]
                long_list.append(tmp)

#np.savetxt("/media/hein/home2/SplitWave_Data/m7_stat_list.csv", long_list, delimiter=', ', header='New Data', comments='# ')
# print(stationlist[0])
# print('Station',long_list[0][1])
# print('event',long_list[0][5])
# print('event',long_list[0][8])
# print(len(long_list[:]))
################################################################################
### DOWNLOAD SKRIPT
#################################################################################
FMT = 'SAC'
n_SKS=len(long_list[:])

DO_DOWNLOAD=False
if DO_DOWNLOAD==True:

    for i in range(0,n_SKS):

        try:    
            st = obspy.Stream()
            ## go through EQ time list and download 1h waveforms 
            st = client.get_waveforms("CH", long_list[i][1], "",
                                      "BH?", long_list[i][5],long_list[i][5]+60*60 ,attach_response=True)  
             ## go to folder, save each trace individually        
            path= '{0}/{1}'.format(save_loc,long_list[i][1])

            try:  
                os.mkdir(path)
            except OSError:
                print ("Creation of the directory %s failed" % path)
            else:  
                print ("Successfully created the directory %s " % path)

            pre_filt = (0.001, 0.005, 40.0, 60.0)        
            st.remove_response(pre_filt=pre_filt, plot=False)
            st.decimate(factor=int(st[0].stats.sampling_rate/10), strict_length=False)         ## downsample to 10 Hz
            st.detrend(type='linear')   
            st.filter("bandpass",freqmin=0.01,freqmax=0.5)

            #FORMAT: 2011.164.14.31.26.1450.FR.ASEAF.BHN..SAC
            event_info = UTCDateTime(str(long_list[i][5]))
            for tr in st:
                filename='{0}/{1}/{2}.{3}.{4}.{5}.{6}.{7}.{8}.{9}.{10}..SAC'.format(save_loc,long_list[i][1],event_info.year,
                                                                  event_info.julday,event_info.hour,
                                                                  event_info.minute,event_info.second,
                                                                  event_info.microsecond/100,tr.stats.network,
                                                                  tr.stats.station,tr.stats.channel)        
                tr.write(filename,format=FMT)
                print('saved data for Station: ',long_list[i][1])        
        except:
            print('no data for Station: ',long_list[i][1])
            pass

# # First calculates distance from EQ and Station coordinates.
# # Then calculates the theoretical arrival Times of each Phases from distance and depth of the event.
#################################################################################
######### READ STATION SKRIPT
#################################################################################
def read_station_event_data(istation):
    #### to get all waveforms of 1 station
    st_lat_list = []
    st_lon_list = []        
    ev_lat_list =[]
    ev_lon_list =[]
    ev_time_list = []
    ev_depth_list = []    
    ev_mag_list = []
    ev_dist_list = []
    back_azimut_list =[]    
    t_SKS_list = []
    t_SKKS_list = []
    t_SKS=0     
    t_SKKS=0 
    t_PP=0     
    t_PP_list = []    

    PHASE_LIST = ['PP','SKS','SKKS']
    st_event = obspy.Stream()
    for iSKS in long_list:    
#    for iSKS in long_list[0:25]:
        
        if iSKS[1]==istation:
#            print(iSKS[1])
            filename='{0}/{1}/{2}.{3}.{4}.{5}.{6}.{7}.CH.{8}.BH?..SAC'.format(save_loc,iSKS[1],iSKS[5].year,
                                                              iSKS[5].julday,iSKS[5].hour,
                                                              iSKS[5].minute,iSKS[5].second,
                                                              iSKS[5].microsecond/100,
                                                              iSKS[1])

        ## something which only gets the data for 1 events
#            print(filename)
            st_tmp=obspy.Stream()
            try:
            
        #                print('reading', iSKS[1])
                st_tmp += read(filename)

                if len(st_tmp)>2:

                    if (st_tmp[0].stats.npts > 36000 and st_tmp[1].stats.npts > 36000 and st_tmp[2].stats.npts > 36000):


                        st_lat = iSKS[3]
                        st_lon = iSKS[4]
                        ev_time = UTCDateTime(iSKS[5])     
                        ev_lat = iSKS[6]
                        ev_lon = iSKS[7]
                        ev_depth = iSKS[8]
                        ev_mag = iSKS[9]  
                        ev_dist = iSKS[10]

                        st_lat_list.append(st_lat)
                        st_lon_list.append(st_lon)                    
                        ev_lat_list.append(ev_lat)
                        ev_lon_list.append(ev_lon)        
                        ev_time_list.append(ev_time)
                        ev_depth_list.append(ev_depth)
                        ev_mag_list.append(ev_mag)
                        ev_dist_list.append(ev_dist)

                        dist_deg = locations2degrees(ev_lat,ev_lon,
                                   st_lat,st_lon)

                        arrivals = model.get_travel_times(source_depth_in_km=ev_depth/1000, distance_in_degree=dist_deg,phase_list=PHASE_LIST)

                        geodetics = gps2dist_azimuth(ev_lat, ev_lon,
                               st_lat, st_lon, a=6378137.0, f=0.0033528106647474805)

                        back_azimut = geodetics[2]
                        back_azimut_list.append(back_azimut)

                        for i in range(0,len(arrivals)):
                            if arrivals[i].name=='SKS':
                                t_SKS = arrivals[i].time
                            elif (arrivals[i].name=='SKKS' and t_SKKS==0):
                                t_SKKS = arrivals[i].time     
                            elif arrivals[i].name=='PP':
                                t_PP = arrivals[i].time

                        t_SKS_list.append(t_SKS)
                        t_SKKS_list.append(t_SKKS)                        
                        t_PP_list.append(t_PP)                                                

                        st_event +=st_tmp
                    else:
                        print('Stream has too few samples')
                else:
                    print('Stream has not 3 channels')



            except:
                print('no matching file')
            

    return st_event,st_lat_list,st_lon_list,ev_lat_list,ev_lon_list,ev_time_list,ev_depth_list,ev_mag_list,ev_dist_list,back_azimut_list, t_SKS_list,t_SKKS_list,t_PP_list

### THERE SEEMS TO BE A PROBLEM WITH TAU P NAMING THE PP and S Phase to SKKS
#fig, ax = plt.subplots(figsize=(9, 9))
#ax = plot_travel_times(source_depth=10, phase_list=PHASE_LIST, ax=ax, fig=fig, verbose=True)
#fig.savefig('/media/hein/home2/SplitWave_Results/Project_images/Travel_times_tau-P.png')
#arrivals = model.get_ray_paths(source_depth_in_km=50, distance_in_degree=100, phase_list=PHASE_LIST)

#fig, ax = plt.subplots(figsize=(9, 9))
#fig =plt.figure(figsize=(16,9))
#ax = fig.add_axes([0.2,0.15,0.6,0.7],projection='polar')  
#ax = arrivals.plot_rays(legend=True,fig=fig)
#fig.savefig('/media/hein/home2/SplitWave_Results/Project_images/Ray_path_tau-P.png')

#fig =plt.figure(figsize=(16,9))
#ax = fig.add_axes([0.2,0.15,0.6,0.7])  
#ax = arrivals.plot_rays(legend=True,plot_type="cartesian",fig=fig)
#fig.savefig('/media/hein/home2/SplitWave_Results/Project_images/Ray_path_cartesian_tau-P.png')
#################################################################################
######### automatic Splitting Routine for all methods 
#################################################################################

def automatic_SplitWave_Routine(st_ev,st_lat_l,st_lon_l,ev_lat_l,ev_lon_l,ev_time_l,ev_depth_l,ev_mag_l,ev_dist_l,back_azimut_l, t_SKS_l,t_SKKS_l,t_PP_l):
    
    Az_l = []
    SV_Az_l = []
    for step in tqdm(range(0,(len(st_ev)-3)/3)):
#    for step in tqdm(range(2,5)):

        st_selection = st_ev[step*3:3*step+3]

        ### check if event time outside st_event
        back_az = back_azimut_l[step]
        st_cut,SNR,t_real_SKS = plot_SKS_window(st_selection,t_SKS_l[step],t_SKKS_l[step],ev_time_l[step],ev_mag_l[step],ev_dist_l[step])
        
        #### CHEVROT HERE
        Az,SV_Az = SKS_Intensity_Chevrot(st_selection,ev_time_l[step],float(t_real_SKS)-float(ev_time_l[step]),back_azimut_l[step],plot=True)
        Az_l.append(Az)
        SV_Az_l.append(SV_Az)
        ####        
        
        method= 'TransM'
        fast,dfast,lag,dlag =  Splitwave_TransM(st_cut,back_az,plot_SplitW)
        vals = [str(st_cut[0].stats.station),st_lat_l[step],st_lon_l[step],float(ev_time_l[step]),ev_depth_l[step]/1000,ev_mag_l[step],ev_lat_l[step],ev_lon_l[step], fast,dfast,lag,dlag ,SNR]
    #    print(fast,dfast,lag,dlag)                         
        write_SKS_Results(path,method,vals,st_cut[0].stats.station, header)

        method= 'CrossC'
        fast,dfast,lag,dlag =  Splitwave_CrossC(st_cut,plot_SplitW)
        vals = [str(st_cut[0].stats.station),st_lat_l[step],st_lon_l[step],float(ev_time_l[step]),ev_depth_l[step]/1000,ev_mag_l[step],ev_lat_l[step],ev_lon_l[step], fast,dfast,lag,dlag ,SNR]    
    #    print(fast,dfast,lag,dlag)
        write_SKS_Results(path,method,vals,st_cut[0].stats.station, header)

        method= 'Eig3D'
        fast,dfast,lag,dlag =  Splitwave_Eig3D(st_cut,plot_SplitW)
        vals = [str(st_cut[0].stats.station),st_lat_l[step],st_lon_l[step],float(ev_time_l[step]),ev_depth_l[step]/1000,ev_mag_l[step],ev_lat_l[step],ev_lon_l[step], fast,dfast,lag,dlag ,SNR]
    #    print(fast,dfast,lag,dlag)
        write_SKS_Results(path,method,vals,st_cut[0].stats.station, header)    
        
    ### AFTER CALCULATING ALL INTENSITIES FOR ALL EVENTS, GET BEST PARAMS FOR CHEVROT
    dt,phi,std_dt,std_phi = get_best_dt_and_phi(Az_l,SV_Az_l,st_ev[0].stats.station)
    
    print('Splitting intensity analyis: {0}'.format(st_cut[0].stats.station))    
    print('dt',dt)
    print('phi',phi)
    
    return dt,phi,std_dt,std_phi
#################################################################################
######### Plot the SKS Window 
#################################################################################


def plot_SKS_window(st_event,t_SKS,t_SKKS,ev_time,ev_mag,ev_dist,plot=True):
    
    st_selection = obspy.Stream()
    st_selection = st_event
    twin = 40     ## Time Window 40 s
    t = ev_time
    #### TO MAKE SURE THE TRACES START AND END AT THE SAME TIME AND HAVE THE SAME amount of samples 
    mylist = (float(st_selection[0].stats.starttime),
                           float(st_selection[1].stats.starttime),float(st_selection[2].stats.starttime))
    max_startt = max(mylist)
    id_m = mylist.index(max_startt)
    max_startt = UTCDateTime(max_startt)

    mylist2 = (float(st_selection[0].stats.endtime),
                           float(st_selection[1].stats.endtime),float(st_selection[2].stats.endtime))
    min_endt = min(mylist2)
    id_m = mylist2.index(min_endt)

    min_endt = UTCDateTime(min_endt)
#    print(max_startt,min_endt)
    st_selection = st_selection.slice(starttime=max_startt,endtime=min_endt)

    timevec = np.linspace(float(st_selection[0].stats.starttime),
                          float(st_selection[0].stats.endtime),st_selection[0].stats.npts)

    search_room_N = st_selection[0].data[np.where((timevec>float(t)+t_SKS-twin) & (timevec<float(t)+t_SKS+twin))]    
    search_room_E = st_selection[1].data[np.where((timevec>float(t)+t_SKS-twin) & (timevec<float(t)+t_SKS+twin))]     
    #search_room_Z = st_selection[0].data[np.where((timevec>float(t)+t_SKS-twin) & (timevec<float(t)+t_SKS+twin))]        

    max_trans_vec = np.sqrt((search_room_N)**2+(search_room_E)**2)
    max_ampl_N = max(abs(search_room_N))
    max_ampl_E = max(abs(search_room_E))
    #max_ampl_Z = max(abs(search_room_Z))
    max_trans = max(max_trans_vec)

    max_ampl = max_trans 

    id_x = np.where(max_trans_vec==max_ampl)
    ## theoretical arrival
#    print 'theoretical SKS ',UTCDateTime(float(t)+t_SKS)
    ## pick the real arrival time
    t_SKS_real = timevec[np.where((timevec>float(t)+t_SKS-twin) & (timevec<float(t)+t_SKS+twin))][id_x]

    
    #### CALC Signal to Noise ratio from max amplitude and from absolute average of trace 15 seconds outside the signal
    S = max_ampl
    secs= 15     ## like 15 s outside of max
    N1= np.mean(abs(st_selection[0].data[id_x[0][0]-10*secs*2:id_x[0][0]-10*secs]))
    N2 = np.mean(abs(st_selection[0].data[id_x[0][0]+10*secs:id_x[0][0]+10*secs*2]))
    N = np.mean([N1,N2])
    SNR = S/N
    
    if np.isnan(SNR)==True:
        SNR=0
        
    abs_diff = abs(float(t_SKS_real)-float(t)-t_SKS)    
    st_cut = st_selection.slice(starttime=UTCDateTime(t_SKS_real)-twin,
                                 endtime=UTCDateTime(t_SKS_real)+twin,nearest_sample=True)
    
    ### ONLY PLOTTING BELOW    
    if Splitting_windows==True:    
        fig = plt.figure(figsize=(16,9))
        ax = fig.add_axes([0.15,0.15,0.7,0.7])    

        ax.plot(timevec[0:36000], st_selection[0].data[0:36000], "b-",label=st_selection[0].stats.channel)
        ax.plot(timevec[0:36000], st_selection[1].data[0:36000], "g-",label=st_selection[1].stats.channel)
        ax.plot(timevec[0:36000], st_selection[2].data[0:36000], "k-",label=st_selection[2].stats.channel, linewidth=0.75, alpha=0.5)

        ax.vlines(float(t)+t_SKS, min(st_selection[0].data), max(st_selection[0].data), color='r', linewidth=2,label='SKS-arrival (tau-p)')
        ax.vlines(t_SKS_real,-max_ampl,max_ampl,color='red',linestyle='dashed', linewidth=2,label='SKS-arrival (picked)')
        ax.vlines(float(t)+t_SKKS,min(st_selection[0].data),max(st_selection[0].data),color='chocolate', linewidth=1,label='SKKS-arrival (tau-p)')
        ax.set_title('SKS-window, station: {0} , SNR={1}'.format(st_selection[0].stats.station,int(SNR)),loc='left')

        if float(t_SKS_real)-float(t)>t_SKS:
            ax.text(t_SKS_real-abs_diff/2-1,1.05*-max_ampl,r'$\Delta $'+'t=+{0:.2f} s'.format(abs_diff),fontweight='bold')
        else:
            ax.text(t_SKS_real+abs_diff/2-1,1.05*-max_ampl,r'$\Delta $'+'t=-{0:.2f} s'.format(abs_diff),fontweight='bold')        

        ax.text(float(t)+t_SKS-twin+1,max_ampl*0.95,'Earthquake: \n t={0}, dist={1}$^\circ$, Mw={2}'.format(ev_time.strftime('%Y-%m-%d, %H:%M:%S'),round(ev_dist,2),ev_mag),fontweight='bold')
    ########### SET PROPER TIME AXIS    
        xxticks = np.linspace(timevec[0], timevec[-1],300)
        xxlabels=[]
        for i in range(0,len(xxticks)):
            tmp=UTCDateTime(xxticks[i]).strftime('%H:%M:%S')
            xxlabels.append(tmp)
    ########### SET PROPER TIME AXIS    
        ax.set_xticks(xxticks)
        ax.set_xticklabels(xxlabels)    

        ax.set_xlim(float(t)+t_SKS-twin,float(t)+t_SKS+twin)
        ax.set_ylim(-max_ampl*1.25,max_ampl*1.25)    
        ax.fill_between([float(t_SKS_real)-twin, float(t_SKS_real)+twin], [-max_ampl*1.25, -max_ampl*1.25],
                                 [max_ampl*1.25,max_ampl*1.25],
                                 color='gray', alpha=0.25, label='selected time window $\pm$'+'{0} s'.format(twin))
        ax.set_xlabel('Time [hh:mm:ss]')
        ax.set_ylabel('displacement [m]')    
        ax.grid(alpha=0.5)
        ax.legend(loc=4, bbox_to_anchor=(1.15, 0.8))
        ### SAVE EVERY IMAGE_SOMEWHERE
        try:  
            path_SWindows='{0}/../SplitWave_Results/Splitting_windows/{1}/'.format(save_loc,st_cut[0].stats.station)
            os.mkdir(path_SWindows)
        except:
            pass

        image_name='{0}/../SplitWave_Results/Splitting_windows/{1}/{1}.{2}.png'.format(save_loc,st_cut[0].stats.station,str(t))        
        fig.savefig(image_name)
        plt.close()
            
    return st_cut,SNR,t_SKS_real
#################################################################################
######### Auxiliary Functions to save the output to .txt 
#################################################################################

def write_head(path, method, station, header):
    fnam_out = os.path.join(path, '{1}/SKS_Splitting_{0}_{1}.txt'.format(station,method))
    print('wrote into:', fnam_out)
                            
    with open(fnam_out, 'w') as fid:
        fid.write('# ' + header + '\n')
        
def write_head_CHEV(path, method, header):
    fnam_out = os.path.join(path, '{0}/SKS_Splitting_{0}.txt'.format(method))
    print('wrote into:', fnam_out)
                            
    with open(fnam_out, 'w') as fid:
        fid.write('# ' + header + '\n')        
        
                            
def write_SKS_Results(path, method, vals, station, header):
    fnam_out = os.path.join(path,
                            '{1}/SKS_Splitting_{0}_{1}.txt'.format(station,method))
    fmt = '%s \n'    
    vals_array = np.asarray(vals)

    with open(fnam_out, 'a') as fid:      
        arraystring = np.array2string(vals_array[:], 
                                      precision=6,
                                      max_line_width=255)
        fid.write(fmt % (arraystring[1:-1]))

def write_SKS_Results_CHEV(path, method, vals, header):
    fnam_out = os.path.join(path,
                            '{0}/SKS_Splitting_{0}.txt'.format(method))
#    print(fnam_out)
    fmt = '%s \n'    
    vals_array = np.asarray(vals)
#    print(vals_array)

    with open(fnam_out, 'a') as fid:      
        arraystring = np.array2string(vals_array[:], 
                                      precision=4,
                                      max_line_width=255)
        fid.write(fmt % (arraystring[1:-1]))
        
path = '/media/hein/home2/SplitWave_Results/SKS/'

header='Station.code, ' +         'Station.lat [°],' +         'Station.lon [°],' +         'Event time, ' +         'Event depth [km], ' +         'Event mag, '+         'Event lat [°],' +         'Event lon [°],' +         'fast dir [°],' +         'dfast dir [°],' +        'lag [s],' +         'dlag [s],' +         'SNR dB' 

header2='Station.code, ' +         'dt [s],' +         'dlag [s],' +         'fast dir [°]' +         'dfast dir [°], '


#################################################################################
######### SPLITWAVE ROUTINES 
#################################################################################
## Set The Time Windows t1 and t2
## choose the lag
def Splitwave_Eig3D(st_cut,plot=False):

    tmp = st_cut
    delta = tmp[0].stats.delta
#    print(delta)
    t = sw.Trio(tmp[1].data,tmp[0].data, tmp[2].data,delta=delta)
    #t = sw.Pair(st_slice[1].data,)

    ## write something for time window
    t1 = 10
    t2 = 70
    t.set_window(t1,t2)
#    t.plot()

    b = sw.Eig3dM(t,lags=(3,))
    
    try:  
        path_Methods='{0}/../SplitWave_Results/Methods/Eig3D/{1}/'.format(save_loc,st_cut[0].stats.station)
        os.mkdir(path_Methods)
    except:
        pass
    b.save('{0}/../SplitWave_Results/Methods/Eig3D/{1}/{1}_{2}.eig'.format(save_loc,st_cut[0].stats.station,st_cut[0].stats.starttime.strftime("%Y-%m-%d")))    
    if plot==True:
        b.plot()
    

    return b.fast, b.dfast,round(b.lag,4),round(b.dlag,4)


def Splitwave_EigenM(st_cut,plot=False):
    # get data into Pair object and plot
    tmp = st_cut    
    north = tmp[1].data
    east = tmp[0].data
    sample_interval = tmp[0].stats.delta
    realdata = sw.Pair(north, east, delta=sample_interval)
    ## write something for time window    
    t1 = 10
    t2 = 50
    realdata.set_window(t1,t2)
   # realdata.plot()
    
    measure = sw.EigenM(realdata)

    if plot==True:
        m.plot()

    
    return measure.fast, measure.dfast,round(m.lag,4),round(m.dlag,4)


def Splitwave_TransM(st_cut,back_az,plot=False):
    tmp = st_cut    
    north = tmp[1].data
    east = tmp[0].data
    sample_interval = tmp[0].stats.delta
    realdata = sw.Pair(north, east, delta=sample_interval)
    ## write something for time window    
    t1 = 10
    t2 = 70
    realdata.set_window(t1,t2)
    #realdata.plot()
    
    m = sw.TransM(realdata, pol=back_az, lags=(2,))
    try:  
        path_Methods='{0}/../SplitWave_Results/Methods/TransM/{1}/'.format(save_loc,st_cut[0].stats.station)
        os.mkdir(path_Methods)
    except:
        pass
    
    m.save('{0}/../SplitWave_Results/Methods/TransM/{1}/{1}_{2}.eig'.format(save_loc,st_cut[0].stats.station,st_cut[0].stats.starttime.strftime("%Y-%m-%d")))    
    
    if plot==True:
        m.plot()

    return m.fast, m.dfast,round(m.lag,4),round(m.dlag,4)


def Splitwave_CrossC(st_cut,plot=False):
    tmp = st_cut    
    north = tmp[1].data
    east = tmp[0].data
    sample_interval = tmp[0].stats.delta
    realdata = sw.Pair(north, east, delta=sample_interval)
    ## write something for time window    
    t1 = 10
    t2 = 70
    realdata.set_window(t1,t2)
  #  realdata.plot()

    m = sw.CrossM(realdata, lags=(2,))
    try:  
        path_Methods='{0}/../SplitWave_Results/Methods/CrossC/{1}/'.format(save_loc,st_cut[0].stats.station)
        os.mkdir(path_Methods)
    except:  
        pass
        
    m.save('{0}/../SplitWave_Results/Methods/CrossC/{1}/{1}_{2}.eig'.format(save_loc,st_cut[0].stats.station, st_cut[0].stats.starttime.strftime("%Y-%m-%d")))
    if plot==True:
        m.plot()

    
    return m.fast, m.dfast,round(m.lag,4),round(m.dlag,4)
#################################################################################
######### CHEVROT METHOD 
#################################################################################
## take time window of N and E wave
## get particle direction for max pulse
## rotate to RT
## calc Splitting Vector 
## generalize for EVENTS
## now calc for several events
## make azimuth vs SV amplitude plot
## fit a Least Square sinusoid


def SKS_Intensity_Chevrot(st_ev,ev_time,t_SKS,back_azimut,plot=True):
#    SV_Az = []
#    Az = []
    st_ev = st_ev.sort()

#    for ev_step in range(0,len(ev_time_l)):    
    ### SORT IT AS ZNE
    st_stream = obspy.Stream()
    tmp = st_ev[2]
    st_stream +=tmp
    tmp = st_ev[1]
    st_stream +=tmp
    tmp = st_ev[0]
    st_stream +=tmp

    gridspec.GridSpec(2,3)

    arrival_time = ev_time+t_SKS

    ### USE CORRECT ARRIVAL TIME
    ### Take small time window around arrival time 
    twin = 15
    st_stream = st_stream.slice(arrival_time-twin,arrival_time+twin,nearest_sample=True)

    limits=np.max([abs(st_stream[2].data),abs(st_stream[2].data)])*2*10**6

    #### CALC THE POLARIZATION OF PARTICLE MOTION
    ## only accept the upper half for Signal
    noise_level=st_stream[0].data**2+st_stream[1].data**2+st_stream[2].data**2
    azimuth, incidence, az_error, in_error = particle_motion_odr(st_stream, noise_thres=np.mean([np.max(noise_level), np.min(noise_level)])+np.std([np.max(noise_level), np.min(noise_level)]))
#    print(az_error)
    ### ROTATE THE SYSTEM FROM NE TO RT
    st_rot_RT = rotate_ne_rt(st_stream[1].data,st_stream[2].data,180+azimuth)    
    
    radial = st_rot_RT[0]
    r_dot = np.diff(radial)/st_stream[1].stats.delta
    radial = radial[0:len(r_dot)]
    transverse=st_rot_RT[1][0:len(r_dot)]    

    r_2 = np.sum(r_dot**2)
    ### NORMALIZE SPLITTING VECTOR
    SV_EQ = -np.sum(2*r_dot*transverse)/r_2

    
    ### EVENT AZIMUT IS BACK-AZIMUT +180
    if back_azimut+180>360:
        Az = back_azimut-180
    else:
        Az=back_azimut+180
    SV_Az = SV_EQ 

    if plot==True:
    ### Only Plotting Below     
        fig = plt.figure(figsize=(16,9))
#        plt.subplot2grid((2,3), (0,0), colspan=2, rowspan=1)        
        ax1 = fig.add_axes([0.1,0.5,0.5,0.3])    
        ax2 = fig.add_axes([0.65,0.5,0.25,0.3])        
        ax3 = fig.add_axes([0.1,0.1,0.5,0.3])     
        ax4 = fig.add_axes([0.65,0.1,0.25,0.3])            
        
        timevec = np.linspace(float(st_stream[0].stats.starttime),float(st_stream[0].stats.endtime),st_stream[0].stats.npts)

        xxticks = np.linspace(timevec[0], timevec[-1],10)
        xxlabels=[]
        for i in range(0,len(xxticks)):
            tmp=UTCDateTime(xxticks[i]).strftime('%H:%M:%S')
            xxlabels.append(tmp)
            
    ########### SET PROPER TIME AXIS    


        ax1.plot(timevec,st_stream[1].data*10**6,'g',label='North')
        ax1.plot(timevec,st_stream[2].data*10**6,'b',label='East')
        ax1.vlines(x=float(arrival_time),ymin=1.3*np.min(np.min([st_stream[1].data*10**6,st_stream[2].data*10**6])),ymax=1.3*np.max(np.max([st_stream[1].data*10**6,st_stream[2].data*10**6])),color='k',linewidth=0.5,label='SKS-Phase')
        ax1.set_title('{0}, SKS-arrival at: {1}, Backazimut={2} $^\circ$, SI={3}'.format(st_stream[0].stats.station,arrival_time.strftime('%Y-%m-%d, %H:%M:%S'),round(Az,2),round(SV_Az,2)))
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('displacement [$\mu$m]')  
        ax1.set_xlim(timevec[0],timevec[-1])            
        ax1.set_xticks(xxticks)
        ax1.set_xticklabels(xxlabels)            
        ax1.grid()        
        ax1.legend()

#        plt.subplot2grid((2,3), (0,2))
        ax2.plot(st_stream[2].data*10**6,st_stream[1].data/10**-6,color='black',linestyle='dashed')
        ax2.set_xlabel('East disp. [$\mu$m]')
        ax2.set_ylabel('North disp. [$\mu$m]')
        ax2.axis('equal')
        ax2.set_xlim(-limits, limits)
        ax2.set_ylim(-limits, limits)
        ax2.grid()
        ax2.set_title('Polarization: Azimuth={0}$^\circ$'.format(round(azimuth,2)))
        
#        limits = 1        
#        limits = np.max([np.max(radial),np.max(transverse)])
    
# #        plt.subplot2grid((2,3), (1,0), colspan=2, rowspan=1)
        ax3.plot(timevec[0:-1],radial*10**6,'r',label='Radial')
        ax3.plot(timevec[0:-1],transverse*10**6,'b',label='Transverse') 
        ax3.plot(timevec[0:-1],-0.5*r_dot*(np.max(transverse)/np.max(r_dot))*10**6,color='g',label='radial-derivate',alpha=0.5,linewidth=0.5)
        ax3.vlines(x=float(arrival_time),ymin=1.3*np.min(np.min([st_stream[1].data*10**6,st_stream[2].data*10**6])),ymax=1.3*np.max(np.max([st_stream[1].data*10**6,st_stream[2].data*10**6])),color='k',linewidth=0.5,label='SKS-Phase')        
        ax3.set_xlabel('Time [s]')
        ax3.set_ylabel('displacement [$\mu$m]')                        
        ax3.set_xlim(timevec[0],timevec[-1])            
        ax3.set_xticks(xxticks)
        ax3.set_xticklabels(xxlabels)                    
        ax3.grid()        
        ax3.set_title('rotated System')        
        ax3.legend()

#        plt.subplot2grid((2,3), (1,2))
        ax4.plot(radial*10**6,transverse*10**6,color='black',linestyle='dashed')
        ax4.set_xlabel('Radial disp. [$\mu$m]')
        ax4.set_ylabel('Transverse  disp. [$\mu$m]')
        ax4.axis('equal')        
#        ax4.set_xlim(-limits, limits)
#        ax4.set_ylim(-limits, limits)
        ax4.grid()

        try:  
            path_Methods='{0}/../SplitWave_Results/Splitting_Intensity/{1}/'.format(save_loc,st_stream[0].stats.station)
            os.mkdir(path_Methods)
        except:  
            pass
        plt.savefig('{0}/../SplitWave_Results/Splitting_Intensity/{1}/{1}_{2}'.format(save_loc,st_stream[0].stats.station,arrival_time.strftime('%Y-%m-%d, %H:%M:%S')))
        plt.close()
#        fig.close()
#    plt.show()

    return Az,SV_Az

## SINOSOID FUNCTION TO FIT THROUGH DATA
def func(x, delta_t, phi):
    y = delta_t *np.sin( 2*(np.radians(x)-phi))
    return y


def get_best_dt_and_phi(Az,SV_Az,station, plot=True):
#    print(SV_Az)
    popt = np.zeros(2)
    #try:

    # sort out extrem values
    Az = np.asarray(Az)
    SV_Az = np.asarray(SV_Az)
    xdata = Az
    ydata = SV_Az
    
#    allow_max = 1.3 # lets see how good that works
#    xdata = Az[np.where((SV_Az<=allow_max) & (SV_Az>=-allow_max))]
#    ydata = SV_Az[np.where((SV_Az<=allow_max) & (SV_Az>=-allow_max))]
    
    ### FIT CURVE THROUGH DATA
    popt, pcov = curve_fit(func, xdata, ydata,  bounds=([0, -np.pi], [5, np.pi]))
    azi_theo = np.linspace(0,360)
    perr = np.sqrt(np.diag(pcov))
    if plot==True:
        plt.figure(figsize=(16,8))
        plt.plot(xdata, ydata, 'ko',label='data')

#        plt.plot(xdata, func(xdata, *popt), 'rx',label='prediction')
        
        plt.errorbar(xdata, ydata, yerr=func(xdata, perr[0], perr[1]*180/np.pi)-ydata,color='red',alpha=0.5 ,label='errorbar', fmt='o')
#       plt.plot(azi_theo,func(azi_theo,popt[0]+perr[0],(popt[1]+perr[1])*180/np.pi),color='red',linestyle='dashed',linewidth=0.5,label='1x $\sigma$')
#       plt.plot(azi_theo,func(azi_theo,popt[0]-perr[0],(popt[1]-perr[1])*180/np.pi),color='red',linestyle='dashed',linewidth=0.5)        
        #plt.plot(azi_theo,func(azi_theo,popt[0]+perr[0],(popt[1])*180/np.pi),color='red',linestyle='dashed',linewidth=0.5)    
        plt.plot(azi_theo,func(azi_theo,popt[0],popt[1]),color='black',linestyle='dashed',label='fit')
        plt.xlabel('Azimuth [$^\circ$]')
        plt.ylabel('Amplitude')
        plt.grid()
        plt.xlim(0,360)
#        plt.ylim(-2,2)
#        plt.ylim(-allow_max-1,allow_max+1)        
        plt.title('{4}, best parameters $\Delta$t={0}$\pm${1} s, $\phi$={2}$\pm${3}$^\circ$'.format(round(popt[0],2),round(perr[0],3),round(popt[1]*180/np.pi,2),round(perr[1]*180/np.pi,3),station))
        plt.legend()
        plt.savefig('{0}/../SplitWave_Results/SKS/Chevrot/Splitting_Intensity_fit_{1}.png'.format(save_loc,station))
        plt.close()

#     except:
#         print('some Problem')
#         popt[0]=np.nan
#         popt[1]=np.nan
#         pass
    return popt[0],popt[1]*180/np.pi,perr[0],perr[1]*180/np.pi
#################################################################################
######### MAIN PROGRAM AS MULTIPROG Function 
#################################################################################

def multiproc_SplitWave(station):
    method= 'TransM'
    write_head(path, method, station,header)     
    method= 'CrossC'
    write_head(path, method, station,header)     
    method= 'Eig3D'
    write_head(path, method, station,header)     


    ### Reads in the Station DATE

    st_ev,st_lat_l,st_lon_l,ev_lat_l,ev_lon_l,ev_time_l,ev_depth_l,ev_mag_l,ev_dist_l,back_azimut_l, t_SKS_l,t_SKKS_l,t_PP_l = read_station_event_data(station)


    ### Calls Function for Plotting and Splt
    dt,phi,std_dt,std_phi = automatic_SplitWave_Routine(st_ev,st_lat_l,st_lon_l,ev_lat_l,ev_lon_l,ev_time_l,ev_depth_l,ev_mag_l,ev_dist_l,back_azimut_l, t_SKS_l,t_SKKS_l,t_PP_l)

    method='Chevrot'
    ### Still calculate ERROR Bars and improve Quality of method
    write_SKS_Results_CHEV(path, method, [station, dt, std_dt, phi, std_phi], header2)

    
#print(new_stat_list)


# # MAIN PROGRAM 
# -> Read in Waveforms -> cut small window around arrival -> run SKS SPlitting

# In[16]:


start_time = time.time()
### Choose station from station list
mod = 'ak135' ##mod = 'iasp91'
model = TauPyModel(model=mod)
#save_loc = '/media/hein/home2/SplitWave_Data'

FMT = 'SAC'
Splitting_windows = True
plot_SplitW = False
print(save_loc)
new_stat_list = os.listdir(save_loc) ## Get A list of downloaded Stations


method= 'Chevrot'
write_head_CHEV(path, method,header2)     

run_MAIN=True
do_multiproc=True
######################################################################################################    
if run_MAIN==True:
    if do_multiproc==True:
        print('do Multiprocessing')

        n_cores=4
        p = multiprocessing.Pool(processes=n_cores)
        p.map(multiproc_SplitWave, new_stat_list)
        p.close()       
#####################################################################################################
    else:
        for item in new_stat_list:    
        ## all waveforms for the station
            method= 'TransM'
            write_head(path, method, item,header)     
            method= 'CrossC'
            write_head(path, method, item,header)     
            method= 'Eig3D'
            write_head(path, method, item,header)     
            ### Reads in the Station DATE
            st_ev,st_lat_l,st_lon_l,ev_lat_l,ev_lon_l,ev_time_l,ev_depth_l,ev_mag_l,ev_dist_l,back_azimut_l, t_SKS_l,t_SKKS_l,t_PP_l = read_station_event_data(item)

            ### Calls Function for Plotting and Splt
            dt,phi,std_dt,std_phi = automatic_SplitWave_Routine(st_ev,st_lat_l,st_lon_l,ev_lat_l,ev_lon_l,ev_time_l,ev_depth_l,ev_mag_l,ev_dist_l,back_azimut_l, t_SKS_l,t_SKKS_l,t_PP_l)

            method='Chevrot'
            ### Still calculate ERROR Bars and improve Quality of method
            write_SKS_Results_CHEV(path, method, [item, dt, std_dt, phi, std_phi], header2)


## go through each single event 
print("--- %s seconds ---" % (time.time() - start_time))

# Diagnostics and Evaluation
## check with r_dot and SV_Az
## set to R_dot
## instedad of abs set to only sum

# current Problems
# S/N ratio sometimes nan --> set to zero
## calculation correct?
## 
## CHEVROT ROUTINE
# normalization values sometimes much greater than 1 , threshold at 5?

## Load in The Error Surfaces!

## Load in the eig matrixes FROM METHODS
## read the station files and 
## call function def split_TransM(input_stream,....)
## call function def split_crossM(input_stream,....)
# call function def split_Eig3D(input_stream,....)
### save values and 
## measurements list 
## stat, coord, eventinfos, for TransM(, fast, dfast, lag, dlag,), for crossCorr(, fast, dfast, lag, dlag,)
## read in and weighted average of directions
## histogram
## implement QC, splitting Intensity, SNR

