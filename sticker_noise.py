from __future__ import print_function
import tensorflow as tf
import numpy as np
import random
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#set summary dir for tensorflow with FLAGS
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('summary_dir', '/tmp/stickernoise', 'Summaries directory')
#if summary directory exist, delete the previous summaries
if tf.gfile.Exists(FLAGS.summary_dir):
    tf.gfile.DeleteRecursively(FLAGS.summary_dir)
    tf.gfile.MakeDirs(FLAGS.summary_dir)



#Parameters
BatchLength=1  #32 images are in a minibatch
Size=[28, 28, 1] #Input img will be resized to this size
NumIteration=1e2;
ImageNum=123
LearningRate = 1e-1 #learning rate of the algorithm
NumClasses = 2 #number of output classes
EvalFreq=100 #evaluate on every 100th iteration
#select sticker position randomly
StickerColors =[255.0, 255.0,0.0, 0.0]#255 white, 0 Black 
StickerSize=[2,2,1]
StickerPosition=[6,6]

#load data
TrainData= np.load('train_data.npy')
TrainLabels=np.load('train_labels.npy')
TestData= np.load('test_data.npy')
TestLabels=np.load('test_labels.npy')


# Create tensorflow graph
InputLabels = tf.placeholder(tf.int32, [None]) #desired network output
OneHotLabels = tf.one_hot(InputLabels,NumClasses)

#our input si nota  placeholder
ImageToChange=np.zeros([BatchLength]+Size)
ImageToChange[0,:,:,:]=TrainData[ImageNum,:,:,:]
InputData = tf.constant(ImageToChange,dtype=tf.float32)
#InputData=tf.get_variable('Input', initializer=init, dtype=tf.float32, trainable=False)
print(InputData)

def AddSticker(StickerColor, StickerPosition, StickerSize):	
    Noise = tf.get_variable('Noise',[BatchLength]+StickerSize,initializer=tf.constant_initializer(float(StickerColor)), dtype=tf.float32, trainable=True)
    Noise=tf.pad(Noise, [[0,0], [StickerPosition[0],Size[0]-(StickerPosition[0]+StickerSize[0])], [StickerPosition[1],Size[1]-(StickerPosition[1]+StickerSize[1])],[0,0]], "CONSTANT") 
    return Noise
    
#loads the same model, but all variables are frozen in this model
NumKernels = [32,32,32]
def MakeConvNet(Input,Size):
    #add a small noise to the input image
    #in this implementation the values of the sticker can be changed, but position and size are fixed
    #StickerPosition and StickerSize can be optimzied by genetic algorithm
    for i,StC in enumerate(StickerColors):
	StickerPosition=[random.randint(0, Size[0]-StickerSize[0]-1), random.randint(0, Size[1]-StickerSize[1]-1)]
	with tf.variable_scope('noise'+str(i)):
		if i==0:
    			Noise=AddSticker(StC, StickerPosition, StickerSize)
		else:
			NextNoise=AddSticker(StC, StickerPosition, StickerSize)
    			Noise=tf.add(Noise,NextNoise)
    NoisyInput=tf.add(Input,Noise)
    CurrentInput=NoisyInput
    CurrentFilters = Size[2] #the input dim at the first layer is 1, since the input image is grayscale
    for i in range(len(NumKernels)): #number of layers
        with tf.variable_scope('conv'+str(i)):
                NumKernel=NumKernels[i]
                W = tf.get_variable('W',[3,3,CurrentFilters,NumKernel], trainable=False)
                Bias = tf.get_variable('Bias',[NumKernel],initializer=tf.constant_initializer(0.0),trainable=False)
		
                CurrentFilters = NumKernel
                ConvResult = tf.nn.conv2d(CurrentInput,W,strides=[1,1,1,1],padding='VALID') #VALID, SAME
                ConvResult= tf.add(ConvResult, Bias)
                #add batch normalization
                #beta = tf.get_variable('beta',[NumKernel],initializer=tf.constant_initializer(0.0))
                #gamma = tf.get_variable('gamma',[NumKernel],initializer=tf.constant_initializer(1.0))
                #Mean,Variance = tf.nn.moments(ConvResult,[0,1,2])
                #PostNormalized = tf.nn.batch_normalization(ConvResult,Mean,Variance,beta,gamma,1e-10)
	
                ReLU = tf.nn.relu(ConvResult)
                #leaky ReLU
                #alpha=0.01
                #ReLU=tf.maximum(alpha*ConvResult,ConvResult)

                CurrentInput = tf.nn.max_pool(ReLU,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    print(CurrentInput)
    #add fully connected network
    with tf.variable_scope('FC'):
	    CurrentShape=CurrentInput.get_shape()
	    FeatureLength = int(CurrentShape[1]*CurrentShape[2]*CurrentShape[3])
	    FC = tf.reshape(CurrentInput, [-1, FeatureLength])
	    FCInput = FC
	    W = tf.get_variable('W',[FeatureLength,NumClasses], trainable=False)
	    FC = tf.matmul(FC, W)
	    Bias = tf.get_variable('Bias',[NumClasses], trainable=False)
	    FC = tf.add(FC, Bias)	    
    return FC,Noise,NoisyInput,FCInput 

	
# Construct model
PredWeights,Noise,NoisyInput,FCInput  = MakeConvNet(InputData, Size)



# Define loss and optimizer
with tf.name_scope('loss'):
	    Loss = tf.reduce_mean( tf.losses.softmax_cross_entropy(OneHotLabels,PredWeights)  )

with tf.name_scope('optimizer'):    
	    #Use ADAM optimizer this is currently the best performing training algorithm in most cases
	    Optimizer = tf.train.AdamOptimizer(LearningRate).minimize(Loss)
            #Optimizer = tf.train.GradientDescentOptimizer(LearningRate).minimize(Loss)

with tf.name_scope('accuracy'):	  
	    CorrectPredictions = tf.equal(tf.argmax(PredWeights, 1), tf.argmax(OneHotLabels, 1))
	    Accuracy = tf.reduce_mean(tf.cast(CorrectPredictions, tf.float32))
	      


# Initializing the variables
Init = tf.global_variables_initializer()

#restore all varaibles, except the noise:
VariablesToRestore=[]
for v in tf.all_variables():
	if ('Noise' not in v.name) :
		VariablesToRestore.append(v)

#histogram sumamries about the distributio nof the variables
for v in tf.trainable_variables():
    tf.summary.histogram(v.name[:-2], v)

#create image summary from the first 10 images
tf.summary.image('images', TrainData[1:10,:,:,:],  max_outputs=50)

#create scalar summaries for lsos and accuracy
tf.summary.scalar("loss", Loss)
tf.summary.scalar("accuracy", Accuracy)



SummaryOp = tf.summary.merge_all()

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.2



#restore the saved model

# Launch the session with default graph
with tf.Session(config=config) as Sess:
	Sess.run(Init)
	SummaryWriter = tf.summary.FileWriter(FLAGS.summary_dir,tf.get_default_graph())
	saver = tf.train.Saver(VariablesToRestore) # we restore all variables 
	saver.restore(Sess, "./model/mymodel-110")
	Step = 1
	Label=np.zeros([BatchLength])
	Label[0]=TrainLabels[ImageNum]
	print("Original Label was: "+str(Label[0]))
	#lets change this to the other class
	Label[0]=1-Label[0]
	# Keep training until reach max iterations - other stopping criterion could be added
	while Step < NumIteration:
		
		

		#execute teh session
		Summary,_,Pred,N, Nin,InputOfFcLayer  = Sess.run([SummaryOp, Optimizer,PredWeights,Noise,NoisyInput,FCInput ], feed_dict={InputLabels: Label})
		if Step==1:
			print("The Original prediciton was: "+str(Pred))
			OriginalFcInput=InputOfFcLayer 
		if (Step%10)==1:
			print("Prediciton: "+str(Pred))

		
		SummaryWriter.add_summary(Summary,Step)
		Step+=1
	cv2.imwrite('Noise.png',N[0,:,:,:])
	cv2.imwrite('Original.png',ImageToChange[0,:,:,:])
	cv2.imwrite('AddedNoise.png',Nin[0,:,:,:])
	OriginalFcInput=OriginalFcInput[0,:]
	InputOfFcLayer=InputOfFcLayer[0,:]
	print(OriginalFcInput)
	print(InputOfFcLayer)
	plt.plot(range(len(OriginalFcInput)),OriginalFcInput,'b',range(len(InputOfFcLayer)),InputOfFcLayer,'r' )
	plt.savefig('local_nosie.png')


print("Optimization Finished!")
print("Execute tensorboard: tensorboard --logdir="+FLAGS.summary_dir)


