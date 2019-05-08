
import numpy as np
from numpy import pi
from math import sin, cos
import rospy 
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray
import matplotlib.pyplot as plt
import tf
import time

RunningOdroid = False

numReadings   = 100
lastScan = [0]*numReadings

def scan_cb( msg ):
    global lastScan
    """ Process the scan that comes back from the scanner """
    # NOTE: Scan progresses from least theta to most theta: CCW
    # print "Got a scanner message with" , len( msg.intensities ) , "readings!"
    # ~ print "Scan:" , self.lastScan
    # print "Scan Min:" , min( self.lastScan ) , ", Scan Max:" , max( self.lastScan )

    if RunningOdroid: 
        lastScan = msg.data  #lastScan = [ elem/25.50 for elem in msg.data ] # scale [0,255] to [0,10]
    else: 
        lastScan = msg.intensities 

        
rospy.init_node( 'pose_sherlock' , anonymous = True )
        

if RunningOdroid: 
    rospy.Subscriber( "/filtered_distance" , Float32MultiArray , scan_cb )
else: 
    rospy.Subscriber( "/scan" , LaserScan , scan_cb )

listener = tf.TransformListener()

try:
    ex_count = 0
    last = time.time()
    while ( not rospy.is_shutdown() ):
        try:
            lastScanNP   = np.asarray( lastScan ) # scan data 
            (trans,rot) = listener.lookupTransform('map', 'base_link', rospy.Time(0)) # pose 
            x = trans[0]
            y = trans[1]
            if x >= 0.25: # only record in upper section of the U
                if time.time()-last > 0.1:
                    last = time.time()
                    roll,pitch,yaw = tf.transformations.euler_from_quaternion(rot)
                    if yaw < 0: yaw = -yaw
                    else: yaw = 2*np.pi - yaw
                    yaw += np.pi/2 # basically prevent roll over problems on this map
                    yaw = np.mod(yaw,2*np.pi)
                    ex_count += 1

#                    print(ex_count, ' x:',x,' y:',y,' yaw:',yaw,' scan mean:',lastScanNP.mean())
                    new_row = np.concatenate((np.asarray([x,y,yaw]),lastScanNP),0)
                    new_row = np.reshape(new_row,(1,np.shape(new_row)[0]))
                    try:
                        mini_dataset = np.concatenate((mini_dataset, new_row),0)
                        print(np.shape(mini_dataset))
                    except NameError:
                        mini_dataset = new_row
                    if np.shape(mini_dataset)[0] > 150000: break

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue
    np.save('synth_set.npy',mini_dataset)
except KeyboardInterrupt:
    pass

