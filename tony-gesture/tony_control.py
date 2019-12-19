#!/usr/bin/env python

from dynamixel1_AX import *
import threading
import RPi.GPIO as GPIO
import atexit
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Point

tony_R_Elbow = DY_data(1)
tony_R_sholdX = DY_data(2)
tony_R_sholdY = DY_data(3)
tony_L_Elbow = DY_data(4)
tony_L_sholdX = DY_data(5)
tony_L_sholdY = DY_data(6)
tony_neckX = DY_data(7)
tony_neckY = DY_data(8)

GPIO.setwarnings(False)
L_eye_pin = 18
R_eye_pin = 16
GPIO.setmode(GPIO.BOARD)
GPIO.setup(L_eye_pin, GPIO.OUT)
GPIO.setup(R_eye_pin, GPIO.OUT)
tony_L_eye = GPIO.PWM(L_eye_pin, 50)
tony_R_eye = GPIO.PWM(R_eye_pin, 50)

arm_def_spd =150

def set_init():
	tony_L_eye.start(0)
	tony_R_eye.start(0)

	tony_R_Elbow.torque(1)
	tony_R_sholdX.torque(1)
	tony_R_sholdY.torque(1)
	tony_L_Elbow.torque(1)
	tony_L_sholdX.torque(1)
	tony_L_sholdY.torque(1)
	tony_neckX.torque(1)
	tony_neckY.torque(1)

	tony_R_Elbow.moving_speed(arm_def_spd)
	tony_R_sholdX.moving_speed(arm_def_spd)
	tony_R_sholdY.moving_speed(arm_def_spd)
	tony_L_Elbow.moving_speed(arm_def_spd)
	tony_L_sholdX.moving_speed(arm_def_spd)
	tony_L_sholdY.moving_speed(arm_def_spd)
	tony_neckX.moving_speed(100)
	tony_neckY.moving_speed(50)

	tony_R_Elbow.angle_limit(160,863)
	tony_R_sholdX.angle_limit(164,863)
	tony_R_sholdY.angle_limit(0,1023)
	tony_L_Elbow.angle_limit(160,863)
	tony_L_sholdX.angle_limit(160,860)
	tony_L_sholdY.angle_limit(0,1023)
	tony_neckX.angle_limit(155,868)
	tony_neckY.angle_limit(350,470)


def torque(on):
	tony_R_Elbow.torque(on)
	tony_R_sholdX.torque(on)
	tony_R_sholdY.torque(on)
	tony_L_Elbow.torque(on)
	tony_L_sholdX.torque(on)
	tony_L_sholdY.torque(on)
	tony_neckX.torque(on)
	tony_neckY.torque(on)

def tony_Elbow_goal_position(position, wait=0):
	tony_R_Elbow.goal_position(position)
	tony_L_Elbow.goal_position(1023 - position)
	if wait:
		time.sleep(0.08)

def tony_sholdX_goal_position(position, wait=0):
	tony_R_sholdX.goal_position(position)
	tony_L_sholdX.goal_position(1023 - position)
	if wait:
		time.sleep(0.08)

def tony_sholdY_goal_position(position, wait=0):
	tony_R_sholdY.goal_position(position)
	tony_L_sholdY.goal_position(1023 - position)
	if wait:
		time.sleep(0.08)

def tony_head_XY(X_ang, Y_ang):
	X_ang = 512 + X_ang * 304/90
	Y_ang = 470 - Y_ang * 304/90
	tony_neckX.goal_position(X_ang)
	tony_neckY.goal_position(Y_ang)

def tony_eye(l_eye, r_eye): #0~10, 0 is close, 10 is open
	l_eye = 6.2 + l_eye * 1.8 / 10
	r_eye = 7.3 - r_eye * 1.8 / 10
	tony_L_eye.ChangeDutyCycle(l_eye)
	tony_R_eye.ChangeDutyCycle(r_eye)

def tony_thumup_XY(X_ang, Y_ang, hand=1): #hand=0: left, hand=1: right, hand=2: bothside
	tony_head_XY(X_ang, Y_ang)
	X_ang = 819 + X_ang * 304/90
	Y_ang = 819 + Y_ang * 304/90
	if hand == 0:
		tony_L_Elbow.goal_position(1023 - 207)
		time.sleep(0.7)
		tony_L_sholdX.goal_position(1023 - X_ang)
		tony_L_sholdY.goal_position(1023 - Y_ang)
		time.sleep(1.7)
		tony_L_Elbow.moving_speed(600)
		tony_L_Elbow.goal_position(512)
		time.sleep(2)
		tony_L_Elbow.moving_speed(arm_def_spd)
	elif hand == 1:
		tony_R_Elbow.goal_position(207)
		time.sleep(0.7)
		tony_R_sholdX.goal_position(X_ang)
		tony_R_sholdY.goal_position(Y_ang)
		time.sleep(1.7)
		tony_R_Elbow.moving_speed(600)
		tony_R_Elbow.goal_position(512)
		time.sleep(2)
		tony_R_Elbow.moving_speed(arm_def_spd)
	else:
		tony_Elbow_goal_position(207)
		time.sleep(0.7)

		tony_sholdX_goal_position(X_ang)
		tony_sholdY_goal_position(Y_ang)
		time.sleep(1.7)
		tony_L_Elbow.moving_speed(600)
		tony_R_Elbow.moving_speed(600)

		# add now
		# time.sleep(0.4)
		###################

		tony_Elbow_goal_position(512)

		time.sleep(3)
		tony_L_Elbow.moving_speed(arm_def_spd)
		tony_R_Elbow.moving_speed(arm_def_spd)

def tony_drive_ready_pose(wait=True):
	tony_Elbow_goal_position(185)
	time.sleep(0.7)

	tony_sholdX_goal_position(854)
	tony_sholdY_goal_position(633)
	tony_neckX.goal_position(512)
	tony_neckY.goal_position(470) #350~470

	time.sleep(1)
	# add now
	# time.sleep(1.5)

	tony_Elbow_goal_position(434)
	tony_eye(10,10)
	if wait:
		time.sleep(1)

def tony_swing_head(X_ang):
	X_ang = 512 + X_ang * 304/90
	while True:
		tony_neckX.goal_position(X_ang)
		time.sleep(1.5)
		if got_thumb:
			break
		tony_eye(0,0)
		time.sleep(0.3)
		tony_eye(10,10)
		time.sleep(0.4)
		tony_neckX.goal_position(1023-X_ang)
		time.sleep(1.5)
		if got_thumb:
			break
		tony_eye(0,0)
		time.sleep(0.3)
		tony_eye(10,10)
		time.sleep(0.4)

got_thumb = 0

# add now
callbackFlag = False
semaphore_lock = False
##########

def ThumbUPcallback(data):
	global callbackFlag, semaphore_lock

	if not semaphore_lock:
		print "[ThumbUPcallback] got_thumb"
		semaphore_lock = True
		
		set_init() # no sleep in here

		tony_eye(10,10)
		tony_head_XY(0, 0) # no sleep in here
		##########

		tony_thumup_XY(0, 30, hand=2) # no sleep in here

		# add now
		#################

		tony_drive_ready_pose(wait=True) # no sleep in here
		
		# add now

		semaphore_lock = False
	else:
		print "[ThumbUPcallback] stocked by semaphore"


# add now
def XYZCallback(data):
	global callbackFlag, semaphore_lock
	if not semaphore_lock:
		# if z < 0.5  --> STOP z threshold
		if 0 < data.z < 0.5:
			callbackFlag = True
			semaphore_lock = True

			print "[XYZCallback] person detected!"
			tony_head_XY(0, 0)
			time.sleep(5)

			callbackFlag = False
			semaphore_lock = False
		else:
			print "[XYZCallback] stocked by not person in here!"
	else:
		pass
		# print "[XYZCallback] stocked by semaphore"
##########


### add now ###
stop_here = False

i = 0
def StopHead(data):
	global stop_here, callbackFlag, semaphore_lock, i
	i+=1
	print "hi", i, data.data

	if data.data=="STOP":
		tony_head_XY(0, 0)
		time.sleep(2)
		stop_here = True
	else:
		print "release now!"
		stop_here = False

	

###############

def listener():
	global semaphore_lock, stop_here
	rospy.init_node('Tony_node', anonymous=False)
	rospy.Subscriber('tony_state', String, ThumbUPcallback)
	rospy.Subscriber('stop_motor', String, StopHead,queue_size =1)


	# add now
	# rospy.Subscriber('Targetter', Point, XYZCallback)
	##########

	######### modify now : sep tony_swing_head() #########

	# if not semaphore_lock:
		# if Go == go
	if not stop_here:
		print "help TT"

		tony_neckX.goal_position(512 + 30 * 304/90)
		time.sleep(1.5)
			
		tony_eye(0,0)
		time.sleep(0.3)
			
		tony_eye(10,10)
		time.sleep(0.4)

		tony_neckX.goal_position(1023-(512 + 30 * 304/90))
		time.sleep(1.5)

		tony_eye(0,0)
		time.sleep(0.3)

		tony_eye(10,10)
		time.sleep(0.4)

	### STOP ###
	else:
		print "THUMB", stop_here
		tony_eye(10,10)
		# tony_head_XY(0, 0) # no sleep in here

		tony_thumup_XY(0, 30, hand=2) # no sleep in here

		tony_drive_ready_pose(wait=True) # no sleep in here

	###############

######################

	rospy.spin()

if __name__ == '__main__':
	try:
		set_init()
		atexit.register(tony_L_eye.stop)
		atexit.register(tony_R_eye.stop)
		atexit.register(torque,0)
		tony_drive_ready_pose(1.5)

		# th = threading.Thread(target=tony_swing_head, args=(30,))
		# th.setDaemon(True)
		# th.start()

		# time.sleep(5)
		listener()
	except KeyboardInterrupt:
		tony_L_eye.stop()
		tony_R_eye.stop()
		torque(0)
