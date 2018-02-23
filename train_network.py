from __future__ import print_function
import tensorflow as tf
import numpy as np
import random


#set summary dir for tensorflow with FLAGS
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('summary_dir', '/tmp/tutorial', 'Summaries directory')
#if summary directory exist, delete the previous summaries
#if tf.gfile.Exists(FLAGS.summary_dir):
#    tf.gfile.DeleteRecursively(FLAGS.summary_dir)
#    tf.gfile.MakeDirs(FLAGS.summary_dir)



#Parameters
BatchLength=32  #32 images are in a minibatch
Size=[28, 28, 1] #Input img will be resized to this size
NumIteration=int(1e2);
LearningRate = 1e-4 #learning rate of the algorithm
NumClasses = 2 #number of output classes
EvalFreq=100 #evaluate on every 100th iteration

#load data
TrainData= np.load('train_data.npy')
TrainLabels=np.load('train_labels.npy')
TestData= np.load('test_data.npy')
TestLabels=np.load('test_labels.npy')


# Create tensorflow graph
InputData = tf.placeholder(tf.float32, [None, Size[0], Size[1], Size[2] ]) #network input
InputLabels = tf.placeholder(tf.int32, [None]) #desired network output
OneHotLabels = tf.one_hot(InputLabels,NumClasses)

NumKernels = [32,32,32]
def MakeConvNet(Input,Size):
    CurrentInput = Input
    CurrentFilters = Size[2] #the input dim at the first layer is 1, since the input image is grayscale
    for i in range(3): #number of layers
        with tf.variable_scope('conv'+str(i)):
                NumKernel=NumKernels[i]
                W = tf.get_variable('W',[3,3,CurrentFilters,NumKernel])
                Bias = tf.get_variable('Bias',[NumKernel],initializer=tf.constant_initializer(0.0))
		
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
    #add fully connected network
    with tf.variable_scope('FC'):
	    CurrentShape=CurrentInput.get_shape()
	    FeatureLength = int(CurrentShape[1]*CurrentShape[2]*CurrentShape[3])
	    FC = tf.reshape(CurrentInput, [-1, FeatureLength])
	    W = tf.get_variable('W',[FeatureLength,NumClasses])
	    FC = tf.matmul(FC, W)
	    Bias = tf.get_variable('Bias',[NumClasses])
	    FC = tf.add(FC, Bias)	    
    return FC

	
# Construct model
PredWeights = MakeConvNet(InputData, Size)



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


#create sumamries, these will be shown on tensorboard

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


# Launch the session with default graph
with tf.Session(config=config) as Sess:
	Sess.run(Init)
	SummaryWriter = tf.summary.FileWriter(FLAGS.summary_dir,tf.get_default_graph())
	Saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
    
	Step = 1
	# Keep training until reach max iterations - other stopping criterion could be added
	while Step < NumIteration:
		
		#create train batch - select random elements for training
		TrainIndices = random.sample(range(TrainData.shape[0]), BatchLength)
		Data=TrainData[TrainIndices,:,:,:]
		Label=TrainLabels[TrainIndices]
		Label=np.reshape(Label,(BatchLength))
		

		#execute teh session
		Summary,_,Acc,L = Sess.run([SummaryOp,Optimizer, Accuracy, Loss], feed_dict={InputData: Data, InputLabels: Label})

		#print loss and accuracy at every 10th iteration
		if (Step%10)==0:
			#train accuracy
			print("Iteration: "+str(Step))
			print("Accuracy:" + str(Acc))
			print("Loss:" + str(L))

		#independent test accuracy
		if (Step%EvalFreq)==0:			
			TotalAcc=0;
			Data=np.zeros([1]+Size)
			for i in range(0,TestData.shape[0]):
				Data[0]=TestData[i]
				Label=TestLabels[i]
				response = Sess.run(PredWeights, feed_dict={InputData: Data})
				if np.argmax(response)==Label:
					TotalAcc+=1
			print("Independent Test set: "+str(float(TotalAcc)/TestData.shape[0]))
		#print("Loss:" + str(L))
		SummaryWriter.add_summary(Summary,Step)
		Step+=1

	print('Saving model...')
	print(Saver.save(Sess, "model/mymodel",Step))

print("Optimization Finished!")
print("Execute tensorboard: tensorboard --logdir="+FLAGS.summary_dir)


