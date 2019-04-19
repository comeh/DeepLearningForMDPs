import numpy as np
import tensorflow as tf
import scipy
from sklearn.preprocessing import StandardScaler
from numpy.core.umath_tests import inner1d
import time

d=2
N=20
T=1
X0=np.full(d,0)
H=T/N
sqrth=np.sqrt(H)
sqrt2=np.sqrt(2)

n_epochs=100
Mbatch=300
n_batches=100
M_training=Mbatch*n_batches*n_epochs # Size of the training set
MBatchValidation=1000 # Size of the Validation set 
M_validation=n_epochs*MBatchValidation

norm2= lambda x:inner1d(x,x)

def g(x):
    return np.log(1/2*(1+norm2(x)))

def g_tf(X):
    return tf.log(1/2*(1+tf.reduce_sum(tf.square(X),axis=1,keepdims=True )))

np.random.seed(1)

NoiseTraining=np.random.normal(0,1,(M_training,d))
NoiseValidation=np.random.normal(0,1,(M_validation,d))

###########################################################################################################
################################## PARTIE NN ##############################################################
###########################################################################################################

n_inputs=d
n_hidden1=d+10
n_hidden2=min(20,10+d/2)
n_outputs_V=1
n_hidden1_A=d+10
n_hidden2_A=d+10
n_outputs_A=d
init_learning_rate_V=.001
learning_rate_A=0.001 #0.001 avant
scale=0.001

nbOuterLearning=n_epochs/10
min_decrease_rateA=0.05
min_decrease_rateV=0.05

# We have a representation of the value function as an expectation of a r.v. (cf paper)
def Vtheo(t,x,n_simu):
    noise=np.random.normal(0,1,(n_simu,d))
    temp=x+sqrt2*np.sqrt(T-t)*noise
    return -np.log(np.mean(np.exp(-g(temp))))

def VtheoVect(t,x,n_simu): # value function at state x using MC estimation and the given representation of the solution as an expectation.
    res=np.zeros((len(x),1))
    for n in range(len(x)):
        res[n]=Vtheo(t,x[n],n_simu)
    return res

def TrainVnnAnnT_(n): # Train the optimal control and value function at time T-1. We do not use the pre-training trick.
    sqrt2sqrt_t_n=sqrt2*np.sqrt(n)*sqrth
    assert(n==N-1)
    V_theo=VtheoVect(H*n,sqrt2sqrt_t_n*NoiseTraining[0:5],10000)
    tf.reset_default_graph()
    Xunsc=tf.placeholder(tf.float64, shape=(None,n_inputs), name="Xunsc") #Le batch unscaled
    Xsc=tf.placeholder(tf.float64, shape=(None,n_inputs), name="Xsc") #Le batch rescaled
    Noise=tf.placeholder(tf.float64, shape=(None,n_inputs), name="Noise") #Bruit gaussien pour le calcul de X au temps n+1
    learning_rate_A=tf.placeholder(tf.float64, name="learning_rate_A")
    learning_rate_V=tf.placeholder(tf.float64, name="learning_rate_V")
    regularizerA=tf.contrib.layers.l2_regularizer(scale)
    regularizerV=tf.contrib.layers.l2_regularizer(scale)
    he_init = tf.contrib.layers.variance_scaling_initializer()
    with tf.name_scope("dnn_A"):
        hidden1_A=tf.layers.dense(Xsc,n_hidden1_A, name="Ahidden1"+str(n), activation=tf.nn.elu, kernel_initializer=he_init,kernel_regularizer=regularizerA)
        hidden2_A=tf.layers.dense(hidden1_A,n_hidden2_A, name="Ahidden2"+str(n), activation=tf.nn.elu, kernel_initializer=he_init,kernel_regularizer=regularizerA)
        #hidden3_A=tf.layers.dense(hidden2_A,n_hidden3_A, name="Ahidden3"+str(n), activation=tf.nn.elu, kernel_initializer=he_init,kernel_regularizer=regularizerA)
        #hidden4_A=tf.layers.dense(hidden3_A,n_hidden4_A, name="hidden4"+str(n), activation=tf.nn.elu,kernel_initializer=he_init,kernel_regularizer=regularizerA)
        controle=tf.layers.dense(hidden2_A, n_outputs_A, name="Aoutput"+str(n), kernel_initializer=he_init,kernel_regularizer=regularizerA)
    Xnext_unsc=Xunsc+2*controle*H+sqrt2*sqrth*Noise
    Vnext=g_tf(Xnext_unsc)
    with tf.name_scope("dnn_V"):
        hidden1_V=tf.layers.dense(Xsc,n_hidden1, name="Vhidden1"+str(n), activation=tf.nn.elu, kernel_initializer=he_init,kernel_regularizer=regularizerV)
        hidden2_V=tf.layers.dense(hidden1_V,n_hidden2, name="Vhidden2"+str(n), activation=tf.nn.elu, kernel_initializer=he_init,kernel_regularizer=regularizerV)
        #hidden3_V=tf.layers.dense(hidden2_V,n_hidden3, name="Vhidden3"+str(n), activation=tf.nn.elu, kernel_initializer=he_init,kernel_regularizer=regularizerV)
        #hidden4_V=tf.layers.dense(hidden3_V,n_hidden4, name="Vhidden4"+str(n), activation=tf.nn.elu, kernel_initializer=he_init,kernel_regularizer=regularizerV)
        output_V=tf.layers.dense(hidden2_V, n_outputs_V, name="Voutput"+str(n), kernel_initializer=he_init,kernel_regularizer=regularizerV)
    with tf.name_scope("loss_A"):
        reglosses_A=tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term_A=tf.contrib.layers.apply_regularization(regularizerA, reglosses_A)
        lossControle=tf.reduce_mean(tf.reduce_sum(tf.square(controle),1,keepdims=True)*H+Vnext)
    with tf.name_scope("train_A"):
        train_vars_A=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Ahidden1"+str(n)+"|Ahidden2"+str(n)+"|Ahidden3"+str(n)+"|Aoutput"+str(n))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_A)
        training_op_A= optimizer.minimize(lossControle+reg_term_A,var_list=train_vars_A)
    with tf.name_scope("loss_V"):
        reglosses_V=tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term_V=tf.contrib.layers.apply_regularization(regularizerV, reglosses_V)
        lossV=tf.reduce_mean(tf.square(tf.reduce_sum(tf.square(controle),1,keepdims=True)*H+Vnext-output_V))
    with tf.name_scope("train_V"):
        train_vars_V=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Vhidden1"+str(n)+"|Vhidden2"+str(n)+"|Vhidden3"+str(n)+"|Voutput"+str(n))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_V)
        training_op_V= optimizer.minimize(lossV+reg_term_V,var_list=train_vars_V)
    init = tf.global_variables_initializer()
    saver=tf.train.Saver()
    with tf.Session() as sess:
        init.run()
        init_learning_rate_A=0.001
        init_learning_rate_V=0.001
        cost_hist=[]
        loss_hist=[]
        for epoch in range(n_epochs):
            SampleNoise=np.random.normal(0,1,(Mbatch*n_batches,d))
            SampleNoiseValidation=np.random.normal(0,1,(MBatchValidation,d))
            val_cost=lossControle.eval(feed_dict={Noise:SampleNoiseValidation,Xunsc: sqrt2sqrt_t_n*NoiseValidation[epoch*MBatchValidation:(epoch+1)*MBatchValidation], Xsc:NoiseValidation[epoch*MBatchValidation:(epoch+1)*MBatchValidation]})
            cost_hist.append(val_cost)
            print("VcostControle: ",val_cost)
            for batch in range(n_batches):
                ind1=n_batches*epoch*Mbatch +batch*Mbatch
                ind2=n_batches*epoch*Mbatch +(batch+1)*Mbatch
                sess.run(training_op_A, feed_dict={learning_rate_A:init_learning_rate_A, Noise:SampleNoise[batch*Mbatch:(batch+1)*Mbatch], Xunsc: sqrt2sqrt_t_n*NoiseTraining[ind1:ind2], Xsc:NoiseTraining[ind1:ind2]})            #
            if epoch%nbOuterLearning==0:
                mean_cost=np.mean(cost_hist)
                if epoch>0:
                    print("mean_cost=",mean_cost)
                    print("last_cost_check",last_cost_check)
                    decrease_rate=(last_cost_check-mean_cost)/last_cost_check
                    print("decrease_rate=",decrease_rate)
                    if decrease_rate<min_decrease_rateA:
                        init_learning_rate_A=np.maximum(1e-6,init_learning_rate_A/2)
                        print("learningRateA decreased to ", init_learning_rate_A)
                last_cost_check=mean_cost
                cost_hist=[]
        for epoch in range(n_epochs):
            SampleNoise=np.random.normal(0,1,(Mbatch*n_batches,d))
            SampleNoiseValidation=np.random.normal(0,1,(MBatchValidation,d))
            val_loss=lossV.eval(feed_dict={Noise:SampleNoiseValidation, Xunsc: sqrt2sqrt_t_n*NoiseValidation[epoch*MBatchValidation:(epoch+1)*MBatchValidation], Xsc:NoiseValidation[epoch*MBatchValidation:(epoch+1)*MBatchValidation]})
            loss_hist.append(val_loss)
            print("VLoss: ", val_loss)
            for batch in range(n_batches):
                ind1=n_batches*epoch*Mbatch +batch*Mbatch
                ind2=n_batches*epoch*Mbatch +(batch+1)*Mbatch
                sess.run(training_op_V, feed_dict={Noise:SampleNoise[batch*Mbatch:(batch+1)*Mbatch], learning_rate_V:init_learning_rate_V, Xunsc: sqrt2sqrt_t_n*NoiseTraining[ind1:ind2], Xsc:NoiseTraining[ind1:ind2]})
            if epoch%nbOuterLearning==0:
                mean_loss=np.mean(loss_hist)
                if epoch>0:
                    print("mean_loss=",mean_loss)
                    print("last_loss_check",last_loss_check)
                    decrease_rate=(last_loss_check-mean_loss)/last_loss_check
                    print("decrease_rate=",decrease_rate)
                    if decrease_rate<min_decrease_rateV:
                        init_learning_rate_V=np.maximum(1e-6,init_learning_rate_V/2)
                        print("learningRateV decreased to ", init_learning_rate_V)
                last_loss_check=mean_loss
                loss_hist=[]
        print("V: ", output_V.eval(feed_dict={Xsc:NoiseTraining[0:5]}))
        print("V_theo: ", V_theo)
        save_path=saver.save(sess,"saver/Vfinal"+str(n)+".ckpt")

TrainVnnAnnT_(N-1)

def TrainVnnAnn(n): # Train the optimal control and value function at time n, for n=T-2,...,0. We use the pre-training trick.
    sqrt2sqrt_t_n=sqrt2*np.sqrt(n)*sqrth
    std_next=np.sqrt(n+1)*sqrth*sqrt2
    V_theo=VtheoVect(H*n,sqrt2sqrt_t_n*NoiseTraining[0:5],10000)
    tf.reset_default_graph()
    Xunsc=tf.placeholder(tf.float64, shape=(None,n_inputs), name="Xunsc") #unscaled batch
    Xsc=tf.placeholder(tf.float64, shape=(None,n_inputs), name="Xsc") #rescaled batch
    Noise=tf.placeholder(tf.float64, shape=(None,n_inputs), name="Noise") #Gaussian noise
    learning_rate_A=tf.placeholder(tf.float64, name="learning_rate_A")
    learning_rate_V=tf.placeholder(tf.float64, name="learning_rate_V")
    regularizerA=tf.contrib.layers.l2_regularizer(scale)
    regularizerV=tf.contrib.layers.l2_regularizer(scale)
    he_init = tf.contrib.layers.variance_scaling_initializer()
    with tf.name_scope("dnn_A_next"):
        hidden1_A_next=tf.layers.dense(Xsc,n_hidden1_A, name="Ahidden1"+str(n+1), activation=tf.nn.elu)
        hidden2_A_next=tf.layers.dense(hidden1_A_next,n_hidden2_A, name="Ahidden2"+str(n+1), activation=tf.nn.elu)
        #hidden3_A=tf.layers.dense(hidden2_A,n_hidden3_A, name="Ahidden3"+str(n), activation=tf.nn.elu, kernel_initializer=he_init,kernel_regularizer=regularizerA)
        #hidden4_A=tf.layers.dense(hidden3_A,n_hidden4_A, name="hidden4"+str(n), activation=tf.nn.elu, kernel_initializer=he_init,kernel_regularizer=regularizerA)
        controle_next=tf.layers.dense(hidden2_A_next, n_outputs_A, name="Aoutput"+str(n+1))
    with tf.name_scope("dnn_A"):
        hidden1_A=tf.layers.dense(Xsc,n_hidden1_A, name="Ahidden1"+str(n), activation=tf.nn.elu, kernel_initializer=he_init,kernel_regularizer=regularizerA)
        hidden2_A=tf.layers.dense(hidden1_A,n_hidden2_A, name="Ahidden2"+str(n), activation=tf.nn.elu, kernel_initializer=he_init,kernel_regularizer=regularizerA)
        #hidden3_A=tf.layers.dense(hidden2_A,n_hidden3_A, name="Ahidden3"+str(n), activation=tf.nn.elu, kernel_initializer=he_init,kernel_regularizer=regularizerA)
        #hidden4_A=tf.layers.dense(hidden3_A,n_hidden4_A, name="hidden4"+str(n), activation=tf.nn.elu, kernel_initializer=he_init,kernel_regularizer=regularizerA)
        controle=tf.layers.dense(hidden2_A, n_outputs_A, name="Aoutput"+str(n), kernel_initializer=he_init, kernel_regularizer=regularizerA)
    update_weights_A = [tf.assign(new, old) for (new, old) in zip(tf.trainable_variables("Ahidden1"+str(n)+"|Ahidden2"+str(n)+"|Aoutput"+str(n)), tf.trainable_variables("Ahidden1"+str(n+1)+"|Ahidden2"+str(n+1)+"|Aoutput"+str(n+1)))]
    Xnext_unsc=Xunsc+2*controle*H+sqrt2*sqrth*Noise
    Xnext_sc=Xnext_unsc/std_next
    with tf.name_scope("dnn_V_next"):
        hidden1_V_next=tf.layers.dense(Xnext_sc,n_hidden1, name="Vhidden1"+str(n+1), activation=tf.nn.elu, kernel_initializer=he_init)
        hidden2_V_next=tf.layers.dense(hidden1_V_next,n_hidden2, name="Vhidden2"+str(n+1), activation=tf.nn.elu, kernel_initializer=he_init)
        #hidden3_V_next=tf.layers.dense(hidden2_V_next,n_hidden3, name="Vhidden3"+str(n+1), activation=tf.nn.elu, kernel_initializer=he_init)
        #hidden4_V_next=tf.layers.dense(hidden3_V_next,n_hidden4, name="Vhidden4"+str(n+1), activation=tf.nn.elu, kernel_initializer=he_init)
        output_V_next=tf.layers.dense(hidden2_V_next, n_outputs_V, name="Voutput"+str(n+1),kernel_initializer=he_init)
    with tf.name_scope("dnn_V"):
        hidden1_V=tf.layers.dense(Xsc,n_hidden1, name="Vhidden1"+str(n), activation=tf.nn.elu, kernel_initializer=he_init,kernel_regularizer=regularizerV)
        hidden2_V=tf.layers.dense(hidden1_V,n_hidden2, name="Vhidden2"+str(n), activation=tf.nn.elu, kernel_initializer=he_init,kernel_regularizer=regularizerV)
        #hidden3_V=tf.layers.dense(hidden2_V,n_hidden3, name="Vhidden3"+str(n), activation=tf.nn.elu, kernel_initializer=he_init,kernel_regularizer=regularizerV)
        #hidden4_V=tf.layers.dense(hidden3_V,n_hidden4, name="Vhidden4"+str(n), activation=tf.nn.elu, kernel_initializer=he_init,kernel_regularizer=regularizerV)
        output_V=tf.layers.dense(hidden2_V, n_outputs_V, name="Voutput"+str(n),kernel_initializer=he_init,kernel_regularizer=regularizerV)
    update_weights_V = [tf.assign(new, old) for (new, old) in zip(tf.trainable_variables("Vhidden1"+str(n)+"|Vhidden2"+str(n)+"|Voutput"+str(n)), tf.trainable_variables("Vhidden1"+str(n+1)+"|Vhidden2"+str(n+1)+"|Voutput"+str(n+1)))]
    with tf.name_scope("loss_A"):
        reglosses_A=tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term_A=tf.contrib.layers.apply_regularization(regularizerA, reglosses_A)
        lossControle=tf.reduce_mean(tf.reduce_sum(tf.square(controle),1,keepdims=True)*H+output_V_next)
    with tf.name_scope("train_A"):
        train_vars_A=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Ahidden1"+str(n)+"|Ahidden2"+str(n)+"|Ahidden3"+str(n)+"|Aoutput"+str(n))
        #train_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="dnn_A")
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_A)
        training_op_A= optimizer.minimize(lossControle +reg_term_A,var_list=train_vars_A)
    with tf.name_scope("loss_V"):
        reglosses_V=tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term_V=tf.contrib.layers.apply_regularization(regularizerV, reglosses_V)
        lossV=tf.reduce_mean(tf.square(tf.reduce_sum(tf.square(controle),1,keepdims=True)*H+output_V_next-output_V))
    with tf.name_scope("train_V"):
        train_vars_V=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Vhidden1"+str(n)+"|Vhidden2"+str(n)+"|Vhidden3"+str(n)+"|Voutput"+str(n))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_V)
        training_op_V= optimizer.minimize(lossV +reg_term_V,var_list=train_vars_V)
    reuse_vars=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="Vhidden1"+str(n+1)+"|Vhidden2"+str(n+1)+"|Voutput"+str(n+1)+"|Ahidden1"+str(n+1)+"|Ahidden2"+str(n+1)+"|Aoutput"+str(n+1))
    reuse_vars_dict=dict([(var.op.name,var) for var in reuse_vars])
    restore_saver=tf.train.Saver(reuse_vars_dict)
    init = tf.global_variables_initializer()
    saver=tf.train.Saver()
    with tf.Session() as sess:
        init.run()
        restore_saver.restore(sess, "saver/Vfinal"+str(n+1)+".ckpt")
        sess.run(update_weights_V)
        sess.run(update_weights_A)
        init_learning_rate_A=0.000005
        init_learning_rate_V=0.00005
        cost_hist=[]
        loss_hist=[]
        for epoch in range(n_epochs):
            SampleNoise=np.random.normal(0,1,(Mbatch*n_batches,d))
            SampleNoiseValidation=np.random.normal(0,1,(MBatchValidation,d))
            val_cost=lossControle.eval(feed_dict={Noise:SampleNoiseValidation,Xunsc: sqrt2sqrt_t_n*NoiseValidation[epoch*MBatchValidation:(epoch+1)*MBatchValidation], Xsc:NoiseValidation[epoch*MBatchValidation:(epoch+1)*MBatchValidation]})
            cost_hist.append(val_cost)
            for batch in range(n_batches):
                ind1=n_batches*epoch*Mbatch +batch*Mbatch
                ind2=n_batches*epoch*Mbatch +(batch+1)*Mbatch
                sess.run(training_op_A, feed_dict={learning_rate_A:init_learning_rate_A, Noise:SampleNoise[batch*Mbatch:(batch+1)*Mbatch], Xunsc: sqrt2sqrt_t_n*NoiseTraining[ind1:ind2], Xsc:NoiseTraining[ind1:ind2]})            #
            if epoch%nbOuterLearning==0:
                mean_cost=np.mean(cost_hist)
                if epoch>0:
                    print("mean_cost=",mean_cost)
                    print("last_cost_check",last_cost_check)
                    decrease_rate=(last_cost_check-mean_cost)/last_cost_check
                    print("decrease_rate=",decrease_rate)
                    if decrease_rate<min_decrease_rateA:
                        init_learning_rate_A=np.maximum(1e-6,init_learning_rate_A/2)
                        print("learningRateA decreased to ", init_learning_rate_A)
                last_cost_check=mean_cost
                cost_hist=[]
        for epoch in range(n_epochs):
            SampleNoise=np.random.normal(0,1,(Mbatch*n_batches,d))
            SampleNoiseValidation=np.random.normal(0,1,(MBatchValidation,d))
            val_loss=lossV.eval(feed_dict={Noise:SampleNoiseValidation, Xunsc: sqrt2sqrt_t_n*NoiseValidation[epoch*MBatchValidation:(epoch+1)*MBatchValidation], Xsc:NoiseValidation[epoch*MBatchValidation:(epoch+1)*MBatchValidation]})
            loss_hist.append(val_loss)
            for batch in range(n_batches):
                ind1=n_batches*epoch*Mbatch +batch*Mbatch
                ind2=n_batches*epoch*Mbatch +(batch+1)*Mbatch
                sess.run(training_op_V, feed_dict={Noise:SampleNoise[batch*Mbatch:(batch+1)*Mbatch], learning_rate_V:init_learning_rate_V, Xunsc: sqrt2sqrt_t_n*NoiseTraining[ind1:ind2], Xsc:NoiseTraining[ind1:ind2]})
            if epoch%nbOuterLearning==0:
                mean_loss=np.mean(loss_hist)
                if epoch>0:
                    print("mean_loss=",mean_loss)
                    print("last_loss_check",last_loss_check)
                    decrease_rate=(last_loss_check-mean_loss)/last_loss_check
                    print("decrease_rate=",decrease_rate)
                    if decrease_rate<min_decrease_rateV:
                        init_learning_rate_V=np.maximum(1e-6,init_learning_rate_V/2)
                        print("learningRateV decreased to ", init_learning_rate_V)
                last_loss_check=mean_loss
                loss_hist=[]
        print("V: ", output_V.eval(feed_dict={Xsc:NoiseTraining[0:5]}))
        print("V_theo: ", V_theo)
        save_path=saver.save(sess,"saver/Vfinal"+str(n)+".ckpt")

start_time=time.time()
for n in range(N-2,-1,-1):
    print("n=",n)
    TrainVnnAnn(n)
elapsed_time=time.time()- start_time
print("elapsed_time: ",elapsed_time)


#### Forward Simulations

def Ann(n,Xarg): # take the time and the state as input, and return the optimal control
    tf.reset_default_graph()
    Xunsc=tf.placeholder(tf.float64, shape=(None,n_inputs), name="Xunsc") #Le batch unscaled
    Xsc=tf.placeholder(tf.float64, shape=(None,n_inputs), name="Xsc") #Le batch rescaled
    regularizerA=tf.contrib.layers.l2_regularizer(scale)
    he_init = tf.contrib.layers.variance_scaling_initializer()
    with tf.name_scope("dnn_A"):
        hidden1_A=tf.layers.dense(Xsc,n_hidden1_A, name="Ahidden1"+str(n), activation=tf.nn.elu, kernel_initializer=he_init,kernel_regularizer=regularizerA)
        hidden2_A=tf.layers.dense(hidden1_A,n_hidden2_A, name="Ahidden2"+str(n), activation=tf.nn.elu, kernel_initializer=he_init,kernel_regularizer=regularizerA)
        #hidden3_A=tf.layers.dense(hidden2_A,n_hidden3_A, name="Ahidden3"+str(n), activation=tf.nn.elu, kernel_initializer=he_init,kernel_regularizer=regularizerA)
        #hidden4_A=tf.layers.dense(hidden3_A,n_hidden4_A, name="hidden4"+str(n), activation=tf.nn.elu, kernel_initializer=he_init,kernel_regularizer=regularizerA)
        controle=tf.layers.dense(hidden2_A, n_outputs_A, name="Aoutput"+str(n), kernel_initializer=he_init, kernel_regularizer=regularizerA)
    reuse_vars=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="Ahidden1"+str(n)+"|Ahidden2"+str(n)+"|Ahidden3"+str(n)+"|Ahidden4"+str(n)+"|Aoutput"+str(n))
    reuse_vars_dict=dict([(var.op.name,var) for var in reuse_vars])
    restore_saver=tf.train.Saver(reuse_vars_dict)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        init.run()
        restore_saver.restore(sess, "saver/Vfinal"+str(n)+".ckpt")
        Xarg_sc=(Xarg)/np.sqrt(n)/sqrth/sqrt2 if n>0 else Xarg
        return controle.eval(feed_dict={Xsc:Xarg_sc})

Nb_simu=10000
Xnn=np.zeros((N+1,Nb_simu,d))
Xbench=np.zeros((N+1,Nb_simu,d))
Jnn=np.zeros((N+1,Nb_simu))
Jbench=np.zeros((N+1,Nb_simu))
for n in range(N):
    noise=np.random.normal(0,1,(Nb_simu,d))
    Controle=Ann(n,Xnn[n])
    Xnn[n+1]=Xnn[n]+2*Controle*H+sqrt2*sqrth*noise
    Xbench[n+1]=Xbench[n]+sqrt2*sqrth*noise
    Jnn[n+1]=Jnn[n]+H*np.sum(Controle**2,1)
Jnn[N]+=g(Xnn[N])
Jbench[N]+=g(Xbench[N])
res_nn=np.mean(Jnn[N])
res_bench=np.mean(Jbench[N])
res_theo=Vtheo(0,np.zeros(d),10000)

from scipy import stats
stats.describe(Jnn[N])

res=np.array([res_nn,res_bench,res_theo])
np.savetxt("res.txt",res)
