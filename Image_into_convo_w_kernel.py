import numpy as np
import math
import cv2
from PIL import Image
imgloc=r"C:\Users\justBaloney\Documents\Final Year Project\MNIST_7.jpg"
img=cv2.imread(imgloc,0)
img=cv2.resize(img, (0,0), fx=0.5, fy=0.5)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
data=np.array(img)
np.set_printoptions(threshold=np.inf)
data_np=np.array([	[4,3,2,13,4,5,1,2,3,4],
				[255,2,3,4,2,4,5,5,1,2],
				[5,170,3,1,2,3,4,5,2,9],
				[5,8,4,2,3,5,7,52,1,1],
				[9,4,3,1,5,3,4,3,4,5],
				[0,5,3,1,3,4,5,12,3,4],
				[1,2,3,4,2,4,5,5,1,2],
				[4,3,2,13,4,5,1,2,3,4],
				[5,3,3,1,2,3,4,5,215,9],
				[4,3,2,13,4,5,1,2,3,4]])

not_use=np.array([[1,2,3,4],
				[5,6,7,8],
				[9,10,11,12],
				[13,14,15,16]])
				
data_shape=np.shape(data)
def convo(data, filter_size, stride, padding='SAME'):
	data_shape=np.shape(data)
	num_column=data_shape[1]
	num_row=data_shape[0]
	filter_x=filter_size[1]
	filter_y=filter_size[0]
	result=[]
	result_temp=[]
	i=0
	j=0
	filter_xcount=0
	filter_ycount=0
	#initializing result array
	for j in range(0, filter_y*(math.ceil((num_row+filter_y-1)/stride))):
		for i in range(0, filter_x*(math.ceil((num_column+filter_x-1)/stride))):
			result_temp.append(0)
		result.append(result_temp)
		result_temp=[]
	result=np.array(result)
	
	num_xcon=0
	num_ycon=0
	for j in range(-(filter_y-1),num_row, stride):
		for i in range(-(filter_x-1), num_column,stride):
			for filter_ycount in range(0,filter_y):
				for filter_xcount in range(0,filter_x):
					filter_xtracking=i+filter_xcount
					filter_ytracking=j+filter_ycount
					filter_xloc=num_xcon*filter_x+filter_xcount
					filter_yloc=num_ycon*filter_y+filter_ycount
					if(filter_xcount==1 and filter_ycount==1):
						kernel=8
					else:
						kernel=-1
					#if((filter_xcount+filter_ycount)%2==1):
					#	kernel=1
					#elif(filter_xcount==1 and filter_ycount==1):
					#	kernel=-4
					#else:
					#	kernel=0
					if( filter_xtracking>(num_column-1) or filter_ytracking>(num_row-1) or filter_xtracking<0 or filter_ytracking<0):
						result[filter_yloc][filter_xloc]=0
					else:
						result[filter_yloc][filter_xloc]=data[filter_ytracking][filter_xtracking]*kernel
			num_xcon+=1
		num_xcon=0
		num_ycon+=1
	return result
filter_size=[3,3]
stride=1
result_conv=np.array(convo(data, filter_size,stride))
print(result_conv)
print(np.shape(result_conv))

def maxpool(conv_result, filter_size, stride):
	num_column=data_shape[1]
	num_row=data_shape[0]
	conv_result_shape=np.shape(conv_result)
	conv_result_row=conv_result_shape[0]
	conv_result_column=conv_result_shape[1]
	filter_x=filter_size[1]
	filter_y=filter_size[0]
	pool_result=[]
	pool_result_temp=[]
	for j in range(0, math.ceil((num_row+filter_y-1)/stride)):
		for i in range(0, math.ceil((num_column+filter_x-1)/stride)):
			pool_result_temp.append(0)
		pool_result.append(pool_result_temp)
		pool_result_temp=[]
	pool_result=np.array(pool_result)
	num_xpool=0
	num_ypool=0
	filter_xcount=0
	filter_ycount=0
	local_max=0
	for j in range(0,conv_result_row, filter_y):
		for i in range(0, conv_result_column,filter_x):
			for filter_ycount in range(0,filter_y):
				for filter_xcount in range(0,filter_x):
					if(local_max<conv_result[j+filter_ycount][i+filter_xcount]):
						local_max=conv_result[j+filter_ycount][i+filter_xcount]
			pool_result[num_ypool][num_xpool]=local_max
			local_max=0
			num_xpool+=1
		num_xpool=0
		num_ypool+=1
	print(pool_result)
	return pool_result
	
def avepool(conv_result, filter_size, stride):
	num_column=data_shape[1]
	num_row=data_shape[0]
	conv_result_shape=np.shape(conv_result)
	conv_result_row=conv_result_shape[0]
	conv_result_column=conv_result_shape[1]
	filter_x=filter_size[1]
	filter_y=filter_size[0]
	pool_result=[]
	pool_result_temp=[]
	for j in range(0, math.ceil((num_row+filter_y-1)/stride)):
		for i in range(0, math.ceil((num_column+filter_x-1)/stride)):
			pool_result_temp.append(0)
		pool_result.append(pool_result_temp)
		pool_result_temp=[]
	pool_result=np.array(pool_result)
	num_xpool=0
	num_ypool=0
	filter_xcount=0
	filter_ycount=0
	local_sum=0
	for j in range(0,conv_result_row, filter_y):
		for i in range(0, conv_result_column,filter_x):
			for filter_ycount in range(0,filter_y):
				for filter_xcount in range(0,filter_x):
					local_sum+=conv_result[j+filter_ycount][i+filter_xcount]
			pool_result[num_ypool][num_xpool]=local_sum
			local_sum=0
			num_xpool+=1
		num_xpool=0
		num_ypool+=1
	print(pool_result)
	return pool_result
result=avepool(result_conv, filter_size, stride)
result=np.array(result)
print(np.shape(result))
#result=np.array(convo(result, filter_size,stride))
#result=np.array(result)
#result=maxpool(result, filter_size, stride)
#result=np.array(result)
#print(np.shape(result))
img = Image.fromarray(result)
img.show()
#result_conv=np.array(convo(result, filter_size,stride))
#data_shape=np.shape(result)
#result=maxpool(result_conv, filter_size, stride)
#img = Image.fromarray(result)
#img.show()
#print(result)