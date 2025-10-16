import numpy as np
import matplotlib.pyplot as plt
import support_files_car as sfc
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation

import platform
print("Python " + platform.python_version())
import numpy as np
print("Numpy " + np.__version__)
import matplotlib
print("Matplotlib " + matplotlib.__version__)


# Create an object for the support functions.
support=sfc.SupportFilesCar()
constants=support.constants

# Load the constant values needed in the main file
Ts=constants['Ts']
outputs=constants['outputs'] # number of outputs (psi, Y)
hz = constants['hz'] # horizon prediction period
x_dot=constants['x_dot'] # constant longitudinal velocity
time_length=constants['time_length'] # duration of the manoeuvre
track = constants['track']  # Added for 4-wheel visualization

# Generate the refence signals
t=np.arange(0,time_length+Ts,Ts) # time from 0 to 10 seconds, sample time (Ts=0.1 second)
r=constants['r']
f=constants['f']
psi_ref,X_ref,Y_ref=support.trajectory_generator(t,r,f)
sim_length=len(t) # Number of control loop iterations
refSignals=np.zeros(len(X_ref)*outputs)

# Build up the reference signal vector:
# refSignal = [psi_ref_0, Y_ref_0, psi_ref_1, Y_ref_1, psi_ref_2, Y_ref_2, ... etc.]
k=0
for i in range(0,len(refSignals),outputs):
    refSignals[i]=psi_ref[k]
    refSignals[i+1]=Y_ref[k]
    k=k+1

# Load the initial states
# If you want to put numbers here, please make sure that they are float and not
# integers. It means that you should add a point there.
# Example: Please write 0. in stead of 0 (Please add the point to make it float)
y_dot=0.
psi=0.
psi_dot=0.
Y=Y_ref[0]+2.

states=np.array([y_dot,psi,psi_dot,Y])
statesTotal=np.zeros((len(t),len(states))) # It will keep track of all your states during the entire manoeuvre
statesTotal[0][0:len(states)]=states
psi_opt_total=np.zeros((len(t),hz))
Y_opt_total=np.zeros((len(t),hz))

# Load the initial input
U1=0 # Input at t = -1 s (steering wheel angle in rad (delta))
UTotal=np.zeros(len(t)) # To keep track all your inputs over time
UTotal[0]=U1

# To extract psi_opt from predicted x_aug_opt
C_psi_opt=np.zeros((hz,(len(states)+np.size(U1))*hz))
for i in range(1,hz+1):
    C_psi_opt[i-1][i+4*(i-1)]=1

# To extract Y_opt from predicted x_aug_opt
C_Y_opt=np.zeros((hz,(len(states)+np.size(U1))*hz))
for i in range(3,hz+3):
    C_Y_opt[i-3][i+4*(i-3)]=1

# Generate the discrete state space matrices
Ad,Bd,Cd,Dd=support.state_space()

# UPDATE FROM THE VIDEO EXPLANATIONS:
# Generate the compact simplification matrices for the cost function
# The matrices (Hdb,Fdbt,Cdb,Adc) stay mostly constant during the simulation.
# However, in the end of the simulation, the horizon period (hz) will start decreasing.
# That is when the matrices need to be regenerated (done inside the simulation loop)
Hdb,Fdbt,Cdb,Adc=support.mpc_simplification(Ad,Bd,Cd,Dd,hz)

# Initiate the controller - simulation loops
k=0
for i in range(0,sim_length-1):

    # Generate the augmented current state and the reference vector
    x_aug_t=np.transpose([np.concatenate((states,[U1]),axis=0)])

    # From the refSignals vector, only extract the reference values from your [current sample (NOW) + Ts] to [NOW+horizon period (hz)]
    # Example: t_now is 3 seconds, hz = 15 samples, so from refSignals vectors, you move the elements to vector r:
    # r=[psi_ref_3.1, Y_ref_3.1, psi_ref_3.2, Y_ref_3.2, ... , psi_ref_4.5, Y_ref_4.5]
    # With each loop, it all shifts by 0.1 second because Ts=0.1 s
    k=k+outputs
    if k+outputs*hz<=len(refSignals):
        r=refSignals[k:k+outputs*hz]
    else:
        r=refSignals[k:len(refSignals)]
        hz=hz-1

    if hz<constants['hz']: # Check if hz starts decreasing
        # These matrices (Hdb,Fdbt,Cdb,Adc) were created earlier at the beginning of the loop.
        # They constant almost throughout the entire simulation. However,
        # in the end of the simulation, the horizon period (hz) will start decreasing.
        # Therefore, the matrices need to be constantly updated in the end of the simulation.
        Hdb,Fdbt,Cdb,Adc=support.mpc_simplification(Ad,Bd,Cd,Dd,hz)

    ft=np.matmul(np.concatenate((np.transpose(x_aug_t)[0][0:len(x_aug_t)],r),axis=0),Fdbt)
    du=-np.matmul(np.linalg.inv(Hdb),np.transpose([ft]))
    x_aug_opt=np.matmul(Cdb,du)+np.matmul(Adc,x_aug_t)
    psi_opt=np.matmul(C_psi_opt[0:hz,0:(len(states)+np.size(U1))*hz],x_aug_opt)
    Y_opt=np.matmul(C_Y_opt[0:hz,0:(len(states)+np.size(U1))*hz],x_aug_opt)
    # if hz<4:
    #     print(x_aug_opt)
    psi_opt=np.transpose((psi_opt))[0]
    psi_opt_total[i+1][0:hz]=psi_opt
    Y_opt=np.transpose((Y_opt))[0]
    Y_opt_total[i+1][0:hz]=Y_opt

    # exit()

    # Update the real inputs
    U1=U1+du[0][0]

    ######################### PID #############################################
    PID_switch=constants['PID_switch']

    if PID_switch==1:
        if i==0:
            e_int_pid_yaw=0
            e_int_pid_Y=0
        if i>0:
            e_pid_yaw_im1=psi_ref[i-1]-statesTotal[i-1][1]
            e_pid_yaw_i=psi_ref[i]-states[1]
            e_dot_pid_yaw=(e_pid_yaw_i-e_pid_yaw_im1)/Ts
            e_int_pid_yaw=e_int_pid_yaw+(e_pid_yaw_im1+e_pid_yaw_i)/2*Ts
            Kp_yaw=constants['Kp_yaw']
            Kd_yaw=constants['Kd_yaw']
            Ki_yaw=constants['Ki_yaw']
            U1_yaw=Kp_yaw*e_pid_yaw_i+Kd_yaw*e_dot_pid_yaw+Ki_yaw*e_int_pid_yaw

            e_pid_Y_im1=Y_ref[i-1]-statesTotal[i-1][3]
            e_pid_Y_i=Y_ref[i]-states[3]
            e_dot_pid_Y=(e_pid_Y_i-e_pid_Y_im1)/Ts
            e_int_pid_Y=e_int_pid_Y+(e_pid_Y_im1+e_pid_Y_i)/2*Ts
            Kp_Y=constants['Kp_Y']
            Kd_Y=constants['Kd_Y']
            Ki_Y=constants['Ki_Y']
            U1_Y=Kp_Y*e_pid_Y_i+Kd_Y*e_dot_pid_Y+Ki_Y*e_int_pid_Y

            U1=U1_yaw+U1_Y


        old_states=states
    ######################### PID END #########################################

    # Establish the limits for the real inputs (max: pi/6 radians)

    if U1 < -np.pi/6:
        U1=-np.pi/6
    elif U1 > np.pi/6:
        U1=np.pi/6
    else:
        U1=U1

    # Keep track of your inputs as you go from t=0 --> t=7 seconds
    UTotal[i+1]=U1

    # Compute new states in the open loop system (interval: Ts/30)
    states=support.open_loop_new_states(states,U1)
    statesTotal[i+1][0:len(states)]=states
    # print(i)

################################ ANIMATION LOOP ###############################
# print(Y_opt_total)
# print(statesTotal)
# print(X_ref)
frame_amount=int(time_length/Ts)
lf=constants['lf']
lr=constants['lr']
# print(frame_amount)
def update_plot(num):

    hz = constants['hz'] # horizon prediction period

    # Global car body (rectangle)
    cos_psi = np.cos(statesTotal[num,1])
    sin_psi = np.sin(statesTotal[num,1])
    perp_cos = -sin_psi
    perp_sin = cos_psi
    x = X_ref[num]
    y = statesTotal[num,3]
    rear_center_x = x - lr * cos_psi
    rear_center_y = y - lr * sin_psi
    front_center_x = x + lf * cos_psi
    front_center_y = y + lf * sin_psi
    rear_left_x_g = rear_center_x + (track / 2) * perp_cos
    rear_left_y_g = rear_center_y + (track / 2) * perp_sin
    rear_right_x_g = rear_center_x - (track / 2) * perp_cos
    rear_right_y_g = rear_center_y - (track / 2) * perp_sin
    front_left_x_g = front_center_x + (track / 2) * perp_cos
    front_left_y_g = front_center_y + (track / 2) * perp_sin
    front_right_x_g = front_center_x - (track / 2) * perp_cos
    front_right_y_g = front_center_y - (track / 2) * perp_sin
    car_1.set_data([rear_left_x_g, front_left_x_g, front_right_x_g, rear_right_x_g, rear_left_x_g],
                   [rear_left_y_g, front_left_y_g, front_right_y_g, rear_right_y_g, rear_left_y_g])

    # Zoomed view car body and wheels
    rear_center_x = -lr * cos_psi
    rear_center_y = -lr * sin_psi
    front_center_x = lf * cos_psi
    front_center_y = lf * sin_psi
    rear_left_x = rear_center_x + (track / 2) * perp_cos
    rear_left_y = rear_center_y + (track / 2) * perp_sin
    rear_right_x = rear_center_x - (track / 2) * perp_cos
    rear_right_y = rear_center_y - (track / 2) * perp_sin
    front_left_x = front_center_x + (track / 2) * perp_cos
    front_left_y = front_center_y + (track / 2) * perp_sin
    front_right_x = front_center_x - (track / 2) * perp_cos
    front_right_y = front_center_y - (track / 2) * perp_sin

    car_body.set_data([rear_left_x, front_left_x, front_right_x, rear_right_x, rear_left_x],
                      [rear_left_y, front_left_y, front_right_y, rear_right_y, rear_left_y])

    # Added from upper: Body extension in zoomed view
    car_body_extension.set_data([0, (lf + 40) * cos_psi],
                                [0, (lf + 40) * sin_psi])

    half_wheel = 0.5  # Updated to match upper's wheel size
    cos_delta = np.cos(statesTotal[num,1] + UTotal[num])
    sin_delta = np.sin(statesTotal[num,1] + UTotal[num])

    # Rear left wheel
    rl_x1 = rear_left_x - half_wheel * cos_psi
    rl_x2 = rear_left_x + half_wheel * cos_psi
    rl_y1 = rear_left_y - half_wheel * sin_psi
    rl_y2 = rear_left_y + half_wheel * sin_psi
    rear_left_wheel.set_data([rl_x1, rl_x2], [rl_y1, rl_y2])

    # Rear right wheel
    rr_x1 = rear_right_x - half_wheel * cos_psi
    rr_x2 = rear_right_x + half_wheel * cos_psi
    rr_y1 = rear_right_y - half_wheel * sin_psi
    rr_y2 = rear_right_y + half_wheel * sin_psi
    rear_right_wheel.set_data([rr_x1, rr_x2], [rr_y1, rr_y2])

    # Front left wheel
    fl_x1 = front_left_x - half_wheel * cos_delta
    fl_x2 = front_left_x + half_wheel * cos_delta
    fl_y1 = front_left_y - half_wheel * sin_delta
    fl_y2 = front_left_y + half_wheel * sin_delta
    front_left_wheel.set_data([fl_x1, fl_x2], [fl_y1, fl_y2])

    # Front right wheel
    fr_x1 = front_right_x - half_wheel * cos_delta
    fr_x2 = front_right_x + half_wheel * cos_delta
    fr_y1 = front_right_y - half_wheel * sin_delta
    fr_y2 = front_right_y + half_wheel * sin_delta
    front_right_wheel.set_data([fr_x1, fr_x2], [fr_y1, fr_y2])

    # From upper/lower: Front wheel extension
    car_1_front_wheel_extension.set_data([lf * cos_psi, lf * cos_psi + (0.5 + 40) * cos_delta],
                                          [lf * sin_psi, lf * sin_psi + (0.5 + 40) * sin_delta])

    yaw_angle_text.set_text(str(round(statesTotal[num,1],2))+' rad')
    steer_angle.set_text(str(round(UTotal[num],2))+' rad')

    steering_wheel.set_data(t[0:num],UTotal[0:num])
    yaw_angle.set_data(t[0:num],statesTotal[0:num,1])
    Y_position.set_data(t[0:num],statesTotal[0:num,3])

    if num+hz>len(t):
        hz=len(t)-num
    if PID_switch!=1 and num!=0:
        Y_predicted.set_data(t[num:num+hz],Y_opt_total[num][0:hz])
        psi_predicted.set_data(t[num:num+hz],psi_opt_total[num][0:hz])
        car_predicted.set_data(X_ref[num:num+hz],Y_opt_total[num][0:hz])
    car_determined.set_data(X_ref[0:num],statesTotal[0:num,3])

    if PID_switch!=1:
        return car_1, car_body, car_body_extension, rear_left_wheel, rear_right_wheel, front_left_wheel, front_right_wheel, car_1_front_wheel_extension, \
               yaw_angle_text, steer_angle, steering_wheel, \
               yaw_angle, Y_position, car_determined, Y_predicted, psi_predicted, car_predicted
    else:
        return car_1, car_body, car_body_extension, rear_left_wheel, rear_right_wheel, front_left_wheel, front_right_wheel, car_1_front_wheel_extension, \
               yaw_angle_text, steer_angle, steering_wheel,yaw_angle, Y_position, car_determined

# Set up your figure properties
fig_x=16
fig_y=9
fig=plt.figure(figsize=(fig_x,fig_y),dpi=120,facecolor=(0.8,0.8,0.8))
n=3
m=3
gs=gridspec.GridSpec(n,m)

# Car motion

# Create an object for the motorcycle
ax0=fig.add_subplot(gs[0,:],facecolor=(0.9,0.9,0.9))

# Plot the reference trajectory
ref_trajectory=ax0.plot(X_ref,Y_ref,'b',linewidth=1)

# Plot the lanes
lane_width=constants['lane_width']
lane_1,=ax0.plot([X_ref[0],X_ref[frame_amount]],[lane_width/2,lane_width/2],'k',linewidth=0.2)
lane_2,=ax0.plot([X_ref[0],X_ref[frame_amount]],[-lane_width/2,-lane_width/2],'k',linewidth=0.2)

lane_3,=ax0.plot([X_ref[0],X_ref[frame_amount]],[lane_width/2+lane_width,lane_width/2+lane_width],'k',linewidth=0.2)
lane_4,=ax0.plot([X_ref[0],X_ref[frame_amount]],[-lane_width/2-lane_width,-lane_width/2-lane_width],'k',linewidth=0.2)

lane_5,=ax0.plot([X_ref[0],X_ref[frame_amount]],[lane_width/2+2*lane_width,lane_width/2+2*lane_width],'k',linewidth=3)
lane_6,=ax0.plot([X_ref[0],X_ref[frame_amount]],[-lane_width/2-2*lane_width,-lane_width/2-2*lane_width],'k',linewidth=3)

# Draw a motorcycle
car_1,=ax0.plot([],[],'k',linewidth=3)
car_predicted,=ax0.plot([],[],'-m',linewidth=1)
car_determined,=ax0.plot([],[],'-r',linewidth=1)


# Establish the right (x,y) dimensions
plt.xlim(X_ref[0],X_ref[frame_amount])
plt.ylim(-X_ref[frame_amount]/(n*(fig_x/fig_y)*2),X_ref[frame_amount]/(n*(fig_x/fig_y)*2))
plt.ylabel('Y-distance [m]',fontsize=15)


# Create an object for the motorcycle (zoomed)
ax1=fig.add_subplot(gs[1,:],facecolor=(0.9,0.9,0.9))
bbox_props_angle=dict(boxstyle='square',fc=(0.9,0.9,0.9),ec='k',lw=1.0)
bbox_props_steer_angle=dict(boxstyle='square',fc=(0.9,0.9,0.9),ec='r',lw=1.0)

neutral_line=ax1.plot([-50,50],[0,0],'k',linewidth=1)
car_body,=ax1.plot([],[],'k',linewidth=3)
car_body_extension,=ax1.plot([],[],'--k',linewidth=1)
rear_left_wheel,=ax1.plot([],[],'r',linewidth=4)
rear_right_wheel,=ax1.plot([],[],'r',linewidth=4)
front_left_wheel,=ax1.plot([],[],'r',linewidth=4)
front_right_wheel,=ax1.plot([],[],'r',linewidth=4)
car_1_front_wheel_extension,=ax1.plot([],[],'--r',linewidth=1)

n1_start=-5
n1_finish=30
plt.xlim(n1_start,n1_finish)
plt.ylim(-(n1_finish-n1_start)/(n*(fig_x/fig_y)*2),(n1_finish-n1_start)/(n*(fig_x/fig_y)*2))
plt.ylabel('Y-distance [m]',fontsize=15)
yaw_angle_text=ax1.text(25,2,'',size='20',color='k',bbox=bbox_props_angle)
steer_angle=ax1.text(25,-2.5,'',size='20',color='r',bbox=bbox_props_steer_angle)

# Create the function for the steering wheel
ax2=fig.add_subplot(gs[2,0],facecolor=(0.9,0.9,0.9))
steering_wheel,=ax2.plot([],[],'-r',linewidth=1,label='steering angle [rad]')
plt.xlim(0,t[-1])
plt.ylim(np.min(UTotal)-0.1,np.max(UTotal)+0.1)
plt.xlabel('time [s]',fontsize=15)
plt.grid(True)
plt.legend(loc='upper right',fontsize='small')

# Create the function for the yaw angle
ax3=fig.add_subplot(gs[2,1],facecolor=(0.9,0.9,0.9))
yaw_angle_reference=ax3.plot(t,psi_ref,'-b',linewidth=1,label='yaw reference [rad]')
yaw_angle,=ax3.plot([],[],'-r',linewidth=1,label='yaw angle [rad]')
if PID_switch!=1:
    psi_predicted,=ax3.plot([],[],'-m',linewidth=3,label='psi - predicted [rad]')
plt.xlim(0,t[-1])
plt.ylim(np.min(statesTotal[:,1])-0.1,np.max(statesTotal[:,1])+0.1)
plt.xlabel('time [s]',fontsize=15)
plt.grid(True)
plt.legend(loc='upper right',fontsize='small')

# Create the function for the Y-position
ax4=fig.add_subplot(gs[2,2],facecolor=(0.9,0.9,0.9))
Y_position_reference=ax4.plot(t,Y_ref,'-b',linewidth=1,label='Y - reference [m]')
Y_position,=ax4.plot([],[],'-r',linewidth=1,label='Y - position [m]')
if PID_switch!=1:
    Y_predicted,=ax4.plot([],[],'-m',linewidth=3,label='Y - predicted [m]')
plt.xlim(0,t[-1])
plt.ylim(np.min(statesTotal[:,3])-2,np.max(statesTotal[:,3])+2)
plt.xlabel('time [s]',fontsize=15)
plt.grid(True)
plt.legend(loc='upper right',fontsize='small')


car_ani=animation.FuncAnimation(fig, update_plot,
    frames=frame_amount,interval=20,repeat=True,blit=True)
plt.show()

# # Matplotlib 3.3.3 needed - comment out plt.show()
# Writer=animation.writers['ffmpeg']
# writer=Writer(fps=30,metadata={'artist': 'Me'},bitrate=1800)
# car_ani.save('car1.mp4',writer)

##################### END OF THE ANIMATION ############################
# === Plot simulation results ===
if constants['trajectory'] == 4:
    support.plot_simulation_results(t, X_ref, Y_ref, statesTotal, UTotal)
else:
    # fallback default results
    plt.plot(X_ref, Y_ref, 'b', label='Reference')
    plt.plot(X_ref, statesTotal[:, 3], '--r', label='Car Path')
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.legend()
    plt.grid(True)
    plt.show()


# Plot the world
plt.plot(X_ref,Y_ref,'b',linewidth=2,label='The trajectory')
plt.plot(X_ref,statesTotal[:,3],'--r',linewidth=2,label='Car position')
plt.xlabel('x-position [m]',fontsize=15)
plt.ylabel('y-position [m]',fontsize=15)
plt.grid(True)
plt.legend(loc='upper right',fontsize='small')
plt.ylim(-X_ref[-1]/2,X_ref[-1]/2) # Scale roads (x & y sizes should be the same to get a realistic picture of the situation)
plt.show()


# Plot the the input delta(t) and the outputs: psi(t) and Y(t)
plt.subplot(3,1,1)
plt.plot(t,UTotal[:],'r',linewidth=2,label='steering wheel angle')
plt.xlabel('t-time [s]',fontsize=15)
plt.ylabel('steering wheel angle [rad]',fontsize=15)
plt.grid(True)
plt.legend(loc='upper right',fontsize='small')

plt.subplot(3,1,2)
plt.plot(t,psi_ref,'b',linewidth=2,label='Yaw_ref angle')
plt.plot(t,statesTotal[:,1],'--r',linewidth=2,label='Car yaw angle')
plt.xlabel('t-time [s]',fontsize=15)
plt.ylabel('psi_ref-position [rad]',fontsize=15)
plt.grid(True)
plt.legend(loc='center right',fontsize='small')

plt.subplot(3,1,3)
plt.plot(t,Y_ref,'b',linewidth=2,label='Y_ref position')
plt.plot(t,statesTotal[:,3],'--r',linewidth=2,label='Car Y position')
plt.xlabel('t-time [s]',fontsize=15)
plt.ylabel('y-position [m]',fontsize=15)
plt.grid(True)
plt.legend(loc='center right',fontsize='small')
plt.show()

''' Also treat the case when Ki-s are 0'''
if constants['PID_switch'] == 1:
    print("The simulation was done with the PID controller")
else:
    print("The simulation was done with the MPC controller")
    print("The prediction horizon was: " + str(constants['hz']) + " samples")
    print("The simulation is done for: " + str(constants['time_length']) + " seconds")
    print("The sample time is: " + str(constants['Ts']) + " seconds")
    print("The constant longitudinal velocity is: " + str(constants['x_dot']) + " m/s")
    print("The trajectory chosen is number: " + str(constants['trajectory']))
    print("The mass of the car is: " + str(constants['m']) + " kg")
    print("The cornering stiffness of the front tyre is: " + str(constants['Caf']) + " N/rad")
    print("The cornering stiffness of the rear tyre is: " + str(constants['Car']) + " N/rad")
    print("The distance from the center of gravity to the front axle is: " + str(constants['lf']) + " m")
    print("The distance from the center of gravity to the rear axle is: " + str(constants['lr']) + " m")
    print("The moment of inertia of the car is: " + str(constants['Iz']) + " kgm^2")
    print("The lane width is: " + str(constants['lane_width']) + " m")
    print("The amplitude of the road disturbance is: " + str(constants['r']) + " m")
    print("The frequency of the road disturbance is: " + str(constants['f']) + " 1/m")
    print("The weights in the cost function are adaptive based on curvature.")