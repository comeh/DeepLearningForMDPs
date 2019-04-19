import numpy as np
import tensorflow as tf
import scipy
from sklearn.preprocessing import StandardScaler
import time
import matplotlib.pyplot as plt

N = 30 # N time intervals, N+1 dates
 # number of Monte Carlo simulations
Rbar = 0.1 # mean-reversion level of residual demand
Rho = 0.9 # reversion speed of residual demand
Sigma = 0.2 # volatility of residual demand noise
Cmin = 0.0 # minimum charge
C0   = 0.00 # initial charge
Cmax = 1.0 # maximum charge
K = 2.0 # cost multiplier for diesel generator
Gamma = 2.0 # power of cost of diesel generator
Kappa = 0.2 # on/off switching cost of diesel generator
Qm = 10.0 # multiplicative penalty for negative imbalance
Qp=1000.0
R0 = 0.1 # initial residual demand
Amin = 0.05 # alpha_s minimum
Amax=10
M0=0

### Training and Validation sets generation

n_epochs=100
Mbatch=300
n_batches=100
M_training=Mbatch*n_batches*n_epochs # Size of the training set
MBatchValidation=1000 # Size of the validation set
M_validation=n_epochs*MBatchValidation

np.random.seed(1)
SampleTraining=np.zeros((N+1,M_training,3))
SampleTraining[0,:,0]=C0
SampleTraining[0,:,1]=M0
SampleTraining[0,:,2]=R0
SampleTraining[0][:,2:3]
for n in range(N):
    print("n=",n)
    noise=Sigma*np.random.normal(0,1,(M_training,1))
    next_r=Rbar*(1-Rho)+Rho*SampleTraining[n][:,2:3]+noise
    next_m=np.random.randint(2, size=(M_training,1))
    next_c=np.random.uniform(Cmin,Cmax,size=(M_training,1))
    SampleTraining[n+1]=np.concatenate((next_c,next_m,next_r),1)

plt.plot([SampleTraining[i,8,0] for i in range(N)])

scaler= StandardScaler()
SampleTrainingRescaled=[]

for ind in range(len(SampleTraining)):
    scaler.fit(SampleTraining[ind])
    SampleTrainingRescaled.append(scaler.transform(SampleTraining[ind]))

scaler= StandardScaler()
SampleTrainingRescaled=[]

for ind in range(len(SampleTraining)):
    scaler.fit(SampleTraining[ind])
    SampleTrainingRescaled.append(scaler.transform(SampleTraining[ind]))


## Validation set
SampleValidation=np.zeros((N+1,M_validation,3))
SampleValidation[0,:,0]=C0
SampleValidation[0,:,1]=M0
SampleValidation[0,:,2]=R0
SampleValidation[0][:,2:3]
for n in range(N):
    print("n=",n)
    noise=Sigma*np.random.normal(0,1,(M_validation,1))
    next_r=Rbar*(1-Rho)+Rho*SampleValidation[n][:,2:3]+noise
    next_m=np.random.randint(2, size=(M_validation,1))
    next_c=np.random.uniform(Cmin,Cmax,size=(M_validation,1))
    SampleValidation[n+1]=np.concatenate((next_c,next_m,next_r),1)
plt.plot([SampleValidation[i,40:45,0] for i in range(N)])
plt.hist(SampleTraining[1,:,2])
plt.hist(SampleValidation[1,:,2])
plt.plot([SampleTraining[i,40,0] for i in range(N)])

SampleValidationRescaled=[]

for ind in range(len(SampleValidation)):
    scaler.fit(SampleValidation[ind])
    SampleValidationRescaled.append(scaler.transform(SampleValidation[ind]))

############# Neural Network
Scaler=[]
for n in range(N+1):
    print("n=",n)
    Scaler.append(StandardScaler().fit(SampleTraining[n]))

n_inputs=3
n_hidden1=10
n_hidden2=5
#n_hidden3=10
#n_hidden4=5
n_outputs=1
n_hidden1_A=10
n_hidden2_A=10
#n_hidden3_A=d+5
#n_hidden4_A=d+5
n_outputs_A=3
n_outputs_V=1
init_learning_rate_V=.001
init_learning_rate_A=0.001 #0.001 avant
learning_rate_AV=0.0005
scale=0.001

nbOuterLearning=n_epochs/10
min_decrease_rateA=0.05
min_decrease_rateV=0.05


# Learn the optimal control and value function at time N-1. No pre-training.
def TrainVnnAnnT_(n):
    assert(n==N-1)
    tf.reset_default_graph()
    Amin_t=tf.constant(Amin, dtype="float64")
    Amax_t=tf.constant(Amax, dtype="float64")
    rho_t=tf.constant(Rho,dtype="float64")
    K_t=tf.constant(K,dtype="float64")
    Kappa_t=tf.constant(Kappa,dtype="float64")
    Gamma_t=tf.constant(Gamma,dtype="float64")
    Qm_t=tf.constant(Qm,dtype="float64")
    Qp_t=tf.constant(Qp,dtype="float64")
    Cmax_t=tf.constant(Cmax,dtype="float64")
    Cmin_t=tf.constant(Cmin,dtype="float64")
    learning_rate_A=tf.placeholder(tf.float64, name="learning_rate_A")
    learning_rate_V=tf.placeholder(tf.float64, name="learning_rate_V")
    Xunsc=tf.placeholder(tf.float64, shape=(None,n_inputs), name="Xunsc") #Le batch unscaled
    Xsc=tf.placeholder(tf.float64, shape=(None,n_inputs), name="Xsc") #Le batch rescaled
    Noise=tf.placeholder(tf.float64, shape=(None,n_inputs), name="Noise") #Bruit gaussien pour le calcul de X au temps n+1
    regularizerA=tf.contrib.layers.l2_regularizer(scale)
    regularizerV=tf.contrib.layers.l2_regularizer(scale)
    he_init = tf.contrib.layers.variance_scaling_initializer()
    with tf.name_scope("dnn_A"):
        hidden1_A=tf.layers.dense(Xsc,n_hidden1_A, name="Ahidden1"+str(n), activation=tf.nn.sigmoid, kernel_initializer=he_init,kernel_regularizer=regularizerA)
        hidden2_A=tf.layers.dense(hidden1_A,n_hidden2_A, name="Ahidden2"+str(n), activation=tf.nn.sigmoid, kernel_initializer=he_init,kernel_regularizer=regularizerA)
        #hidden3_A=tf.layers.dense(hidden2_A,n_hidden3_A, name="Ahidden3"+str(n), activation=tf.nn.elu, kernel_initializer=he_init,kernel_regularizer=regularizerA)
        #hidden4_A=tf.layers.dense(hidden3_A,n_hidden4_A, name="hidden4"+str(n), activation=tf.nn.elu,kernel_initializer=he_init,kernel_regularizer=regularizerA)
        controle=tf.layers.dense(hidden2_A, n_outputs_A, name="Aoutput"+str(n),kernel_initializer=he_init,kernel_regularizer=regularizerA)
        zeroUn=tf.nn.softmax(controle[:,:2])
        injection_temp= tf.nn.sigmoid(controle[:,2:])
        injection=Amin_t+tf.multiply(injection_temp,Amax_t-Amin_t)
        I_0_n=tf.minimum(tf.nn.relu(-Xunsc[:,2:3]),Cmax_t-Xunsc[:,0:1])
        I_sup0_n=tf.minimum(tf.nn.relu(injection-Xunsc[:,2:3]),Cmax_t-Xunsc[:,0:1])
        O_0_n=tf.minimum(tf.nn.relu(Xunsc[:,2:3]),Xunsc[:,0:1])
        O_sup0_n=tf.minimum(tf.nn.relu(Xunsc[:,2:3]-injection),Xunsc[:,0:1])
        S_0_n=Xunsc[:,2:3]+I_0_n-O_0_n
        S_sup0_n=Xunsc[:,2:3]-injection+I_sup0_n-O_sup0_n
    with tf.name_scope("dnn_V"):
        hidden1_V=tf.layers.dense(Xsc,n_hidden1, name="Vhidden1"+str(n), activation=tf.nn.elu, kernel_initializer=he_init,kernel_regularizer=regularizerV)
        hidden2_V=tf.layers.dense(hidden1_V,n_hidden2, name="Vhidden2"+str(n), activation=tf.nn.elu, kernel_initializer=he_init,kernel_regularizer=regularizerV)
        #hidden3_V=tf.layers.dense(hidden2_V,n_hidden3, name="Vhidden3"+str(n), activation=tf.nn.elu, kernel_initializer=he_init,kernel_regularizer=regularizerV)
        #hidden4_V=tf.layers.dense(hidden3_V,n_hidden4, name="Vhidden4"+str(n), activation=tf.nn.elu, kernel_initializer=he_init,kernel_regularizer=regularizerV)
        output_V=tf.layers.dense(hidden2_V, n_outputs_V, name="Voutput"+str(n), kernel_initializer=he_init,kernel_regularizer=regularizerV)
    with tf.name_scope("loss_A"):
        reglosses_A=tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term_A=tf.contrib.layers.apply_regularization(regularizerA, reglosses_A)
        running_cost = tf.reduce_mean(zeroUn[:,0:1]*(Kappa_t*tf.cast(tf.equal(Xunsc[:,1:2],1.),tf.float64) + Qm_t*tf.nn.relu(-S_0_n) + Qp_t*tf.nn.relu(Xunsc[:,2:3]-Xunsc[:,0:1])) + zeroUn[:,1:2]*(K_t*tf.pow(injection,Gamma_t)+ Kappa_t*tf.cast(tf.equal(Xunsc[:,1:2],0.),tf.float64) +Qm_t*tf.nn.relu(-S_sup0_n) +Qp_t*tf.nn.relu(Xunsc[:,2:3]-Xunsc[:,0:1] - injection) ) )
    with tf.name_scope("train_A"):
        train_vars_A=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Ahidden1"+str(n)+"|Ahidden2"+str(n)+"|Aoutput"+str(n))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_A)
        training_op_A= optimizer.minimize(running_cost+reg_term_A,var_list=train_vars_A)
    with tf.name_scope("loss_V"):
        reglosses_V=tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term_V=tf.contrib.layers.apply_regularization(regularizerV, reglosses_V)
        pas_active=(K_t*tf.pow(injection,Gamma_t)+ Kappa_t*tf.cast(tf.equal(Xunsc[:,1:2],0.),tf.float64) +Qm_t*tf.nn.relu(-S_sup0_n) +Qp_t*tf.nn.relu(Xunsc[:,2:3]-Xunsc[:,0:1] - injection) )
        active=(K_t*tf.pow(injection,Gamma_t)+ Kappa_t*tf.cast(tf.equal(Xunsc[:,1:2],1.),tf.float64) +Qm_t*tf.nn.relu(-S_sup0_n))
        y_target=zeroUn[:,0:1]*(Kappa_t*tf.cast(tf.equal(Xunsc[:,1:2],0.),tf.float64) +Qm_t*tf.nn.relu(-S_0_n)) + zeroUn[:,1:2]*(K_t*tf.pow(injection,Gamma_t)+ Kappa_t*tf.cast(tf.equal(Xunsc[:,1:2],1.),tf.float64) +Qm_t*tf.nn.relu(-S_sup0_n))
        reg_term_print=reg_term_V
        lossV=tf.reduce_mean(zeroUn[:,0:1]*tf.square(Kappa_t*tf.cast(tf.equal(Xunsc[:,1:2],0.),tf.float64) +Qm_t*tf.nn.relu(-S_0_n) + Qp_t*tf.nn.relu(Xunsc[:,2:3]-Xunsc[:,0:1]) - output_V) + zeroUn[:,1:2]*tf.square(K_t*tf.pow(injection,Gamma_t)+ Kappa_t*tf.cast(tf.equal(Xunsc[:,1:2],1.),tf.float64) +Qm_t*tf.nn.relu(-S_sup0_n) +Qp_t*tf.nn.relu(Xunsc[:,2:3]-Xunsc[:,0:1]- injection) -output_V))
    with tf.name_scope("train_V"):
        train_vars_V=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Vhidden1"+str(n)+"|Vhidden2"+str(n)+"|Voutput"+str(n))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_V)
        training_op_V= optimizer.minimize(lossV+reg_term_V,var_list=train_vars_V)
    init = tf.global_variables_initializer()
    saver=tf.train.Saver()
    with tf.Session() as sess:
        init.run()
        init_learning_rate_A=0.001
        init_learning_rate_V=0.0001
        cost_hist=[]
        loss_hist=[]
        for epoch in range(n_epochs):
            val_cost=running_cost.eval(feed_dict={learning_rate_A:init_learning_rate_A,Xunsc: SampleValidation[n][epoch*MBatchValidation:(epoch+1)*MBatchValidation], Xsc:SampleValidationRescaled[n][epoch*MBatchValidation:(epoch+1)*MBatchValidation]})
            cost_hist.append(val_cost)
            print("VcostControle: ",val_cost)
            for batch in range(n_batches):
                ind1=n_batches*epoch*Mbatch +batch*Mbatch
                ind2=n_batches*epoch*Mbatch +(batch+1)*Mbatch
                sess.run(training_op_A, feed_dict={learning_rate_A:init_learning_rate_A, Xunsc: SampleTraining[n][ind1:ind2], Xsc:SampleTrainingRescaled[n][ind1:ind2]})            #
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
            val_loss=lossV.eval(feed_dict={Xunsc: SampleValidation[n][epoch*MBatchValidation:(epoch+1)*MBatchValidation], Xsc:SampleValidationRescaled[n][epoch*MBatchValidation:(epoch+1)*MBatchValidation]})
            loss_hist.append(val_loss)
            print("VLoss: ", val_loss)
            for batch in range(n_batches):
                ind1=n_batches*epoch*Mbatch +batch*Mbatch
                ind2=n_batches*epoch*Mbatch +(batch+1)*Mbatch
                sess.run(training_op_V, feed_dict={learning_rate_V:init_learning_rate_V, Xunsc: SampleTraining[n][ind1:ind2], Xsc:SampleTrainingRescaled[n][ind1:ind2]})
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
        save_path=saver.save(sess,"saver/Vfinal"+str(n)+".ckpt")

TrainVnnAnnT_(N-1)

def TrainVnnAnn(n): # Learn the optimal control at time n, for n=N-2,...,0. We use the pre-training trick.
    tf.reset_default_graph()
    Amin_t=tf.constant(Amin, dtype="float64")
    Amax_t=tf.constant(Amax, dtype="float64")
    Rbar_t=tf.constant(Rbar, dtype="float64")
    rho_t=tf.constant(Rho,dtype="float64")
    K_t=tf.constant(K,dtype="float64")
    Kappa_t=tf.constant(Kappa,dtype="float64")
    Gamma_t=tf.constant(Gamma,dtype="float64")
    Qm_t=tf.constant(Qm,dtype="float64")
    Qp_t=tf.constant(Qp,dtype="float64")
    Cmax_t=tf.constant(Cmax,dtype="float64")
    Cmin_t=tf.constant(Cmin,dtype="float64")
    Xunsc=tf.placeholder(tf.float64, shape=(None,n_inputs), name="Xunsc") # unscaled batch
    Xsc=tf.placeholder(tf.float64, shape=(None,n_inputs), name="Xsc") # rescaled batch
    Noise=tf.placeholder(tf.float64, shape=(None,1), name="Noise") #Gaussian noise at time n+1
    learning_rate_A=tf.placeholder(tf.float64, name="learning_rate_A")
    learning_rate_V=tf.placeholder(tf.float64, name="learning_rate_V")
    keep_prob = tf.placeholder(tf.float64)
    regularizerA=tf.contrib.layers.l2_regularizer(scale)
    regularizerV=tf.contrib.layers.l2_regularizer(scale)
    he_init = tf.contrib.layers.variance_scaling_initializer()
    with tf.name_scope("dnn_A_next"):
        hidden1_A_next=tf.layers.dense(Xsc,n_hidden1_A, name="Ahidden1"+str(n+1), activation=tf.nn.sigmoid,kernel_regularizer=regularizerA)
        hidden2_A_next=tf.layers.dense(hidden1_A_next,n_hidden2_A, name="Ahidden2"+str(n+1), activation=tf.nn.sigmoid,kernel_regularizer=regularizerA)
        #hidden3_A=tf.layers.dense(hidden2_A,n_hidden3_A, name="Ahidden3"+str(n), activation=tf.nn.elu, kernel_initializer=he_init,kernel_regularizer=regularizerA)
        #hidden4_A=tf.layers.dense(hidden3_A,n_hidden4_A, name="hidden4"+str(n), activation=tf.nn.elu, kernel_initializer=he_init,kernel_regularizer=regularizerA)
        controle_next=tf.layers.dense(hidden2_A_next, n_outputs_A, name="Aoutput"+str(n+1), kernel_regularizer=regularizerA)
    with tf.name_scope("dnn_A"):
        hidden1_A=tf.layers.dense(Xsc,n_hidden1_A, name="Ahidden1"+str(n), activation=tf.nn.sigmoid, kernel_initializer=he_init,kernel_regularizer=regularizerA)
        hidden2_A=tf.layers.dense(hidden1_A,n_hidden2_A, name="Ahidden2"+str(n), activation=tf.nn.sigmoid, kernel_initializer=he_init,kernel_regularizer=regularizerA)
        #hidden3_A=tf.layers.dense(hidden2_A,n_hidden3_A, name="Ahidden3"+str(n), activation=tf.nn.elu, kernel_initializer=he_init,kernel_regularizer=regularizerA)
        #hidden4_A=tf.layers.dense(hidden3_A,n_hidden4_A, name="hidden4"+str(n), activation=tf.nn.elu,kernel_initializer=he_init,kernel_regularizer=regularizerA)
        controle=tf.layers.dense(hidden2_A, n_outputs_A, name="Aoutput"+str(n),kernel_initializer=he_init,kernel_regularizer=regularizerA)
        zeroUn=tf.nn.softmax(controle[:,:2])
        injection_temp= tf.nn.sigmoid(controle[:,2:])
        injection=Amin_t+tf.multiply(injection_temp,Amax_t-Amin_t)
        I_0_n=tf.minimum(tf.nn.relu(-Xunsc[:,2:3]),Cmax_t-Xunsc[:,0:1])
        I_sup0_n=tf.minimum(tf.nn.relu(injection-Xunsc[:,2:3]),Cmax_t-Xunsc[:,0:1])
        O_0_n=tf.minimum(tf.nn.relu(Xunsc[:,2:3]),Xunsc[:,0:1])
        O_sup0_n=tf.minimum(tf.nn.relu(Xunsc[:,2:3]-injection),Xunsc[:,0:1])
        S_0_n=Xunsc[:,2:3]+I_0_n-O_0_n
        S_sup0_n=Xunsc[:,2:3]-injection+I_sup0_n-O_sup0_n
    update_weights_A = [tf.assign(new, old) for (new, old) in zip(tf.trainable_variables("Ahidden1"+str(n)+"|Ahidden2"+str(n)+"|Aoutput"+str(n)), tf.trainable_variables("Ahidden1"+str(n+1)+"|Ahidden2"+str(n+1)+"|Aoutput"+str(n+1)))]
    next_r=Rbar_t*(1-rho_t)+rho_t*Xunsc[:,2:3]+Sigma*Noise
    next_m=tf.cast(tf.greater_equal(zeroUn[:,1:2],.5),tf.float64)
    next_c_0=Xunsc[:,0:1]+I_0_n-O_0_n
    next_c_sup0=Xunsc[:,0:1]+I_sup0_n-O_sup0_n
    Xnext_unsc_0=tf.concat([next_c_0,next_m,next_r],1)
    Xnext_sc_0=(Xnext_unsc_0-Scaler[n+1].mean_)/np.sqrt(Scaler[n+1].var_)
    Xnext_unsc_sup0=tf.concat([next_c_sup0,next_m,next_r],1)
    Xnext_sc_sup0=(Xnext_unsc_sup0-Scaler[n+1].mean_)/np.sqrt(Scaler[n+1].var_)
    with tf.name_scope("dnn_V_next"):
        hidden1_V_next_0=tf.layers.dense(Xnext_sc_0,n_hidden1, name="Vhidden1"+str(n+1), activation=tf.nn.elu, kernel_initializer=he_init)
        hidden2_V_next_0=tf.layers.dense(hidden1_V_next_0,n_hidden2, name="Vhidden2"+str(n+1), activation=tf.nn.elu, kernel_initializer=he_init)
        #hidden3_V_next=tf.layers.dense(hidden2_V_next,n_hidden3, name="Vhidden3"+str(n+1), activation=tf.nn.elu, kernel_initializer=he_init)
        #hidden4_V_next=tf.layers.dense(hidden3_V_next,n_hidden4, name="Vhidden4"+str(n+1), activation=tf.nn.elu, kernel_initializer=he_init)
        output_V_next_0=tf.layers.dense(hidden2_V_next_0, n_outputs_V, name="Voutput"+str(n+1),kernel_initializer=he_init)
        hidden1_V_next_sup0=tf.layers.dense(Xnext_sc_sup0,n_hidden1, name="Vhidden1"+str(n+1), activation=tf.nn.elu, kernel_initializer=he_init,reuse=True)
        hidden2_V_next_sup0=tf.layers.dense(hidden1_V_next_sup0,n_hidden2, name="Vhidden2"+str(n+1), activation=tf.nn.elu, kernel_initializer=he_init, reuse=True)
        #hidden3_V_next=tf.layers.dense(hidden2_V_next,n_hidden3, name="Vhidden3"+str(n+1), activation=tf.nn.elu, kernel_initializer=he_init)
        #hidden4_V_next=tf.layers.dense(hidden3_V_next,n_hidden4, name="Vhidden4"+str(n+1), activation=tf.nn.elu, kernel_initializer=he_init)
        output_V_next_sup0=tf.layers.dense(hidden2_V_next_sup0, n_outputs_V, name="Voutput"+str(n+1),kernel_initializer=he_init,reuse=True)
    with tf.name_scope("dnn_V"):
        hidden1_V=tf.layers.dense(Xsc,n_hidden1, name="Vhidden1"+str(n), activation=tf.nn.elu, kernel_initializer=he_init,kernel_regularizer=regularizerV)
        hidden2_V=tf.layers.dense(hidden1_V,n_hidden2, name="Vhidden2"+str(n), activation=tf.nn.elu, kernel_initializer=he_init,kernel_regularizer=regularizerV)
        output_V=tf.layers.dense(hidden2_V, n_outputs_V, name="Voutput"+str(n),kernel_initializer=he_init,kernel_regularizer=regularizerV)
    update_weights_V = [tf.assign(new, old) for (new, old) in zip(tf.trainable_variables("Vhidden1"+str(n)+"|Vhidden2"+str(n)+"|Voutput"+str(n)), tf.trainable_variables("Vhidden1"+str(n+1)+"|Vhidden2"+str(n+1)+"|Voutput"+str(n+1)))]
    with tf.name_scope("loss_A"):
        reglosses_A=tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term_A=tf.contrib.layers.apply_regularization(regularizerA, reglosses_A)
        running_cost = tf.reduce_mean(zeroUn[:,0:1]*(Kappa_t*tf.cast(tf.equal(Xunsc[:,1:2],1.),tf.float64) + Qm_t*tf.nn.relu(-S_0_n) + Qp_t*tf.nn.relu(Xunsc[:,2:3]-Xunsc[:,0:1]) + output_V_next_0) + zeroUn[:,1:2]*(K_t*tf.pow(injection,Gamma_t)+ Kappa_t*tf.cast(tf.equal(Xunsc[:,1:2],0.),tf.float64) +Qm_t*tf.nn.relu(-S_sup0_n) +Qp_t*tf.nn.relu(Xunsc[:,2:3]-Xunsc[:,0:1] - injection) +output_V_next_sup0) )
    with tf.name_scope("train_A"):
        train_vars_A=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Ahidden1"+str(n)+"|Ahidden2"+str(n)+"|Ahidden3"+str(n)+"|Aoutput"+str(n))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_A)
        training_op_A= optimizer.minimize(running_cost+reg_term_A,var_list=train_vars_A)
    with tf.name_scope("loss_V"):
        reglosses_V=tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term_V=tf.contrib.layers.apply_regularization(regularizerV, reglosses_V)
        lossV=tf.reduce_mean(zeroUn[:,0:1]*tf.square(Kappa_t*tf.cast(tf.equal(Xunsc[:,1:2],1.),tf.float64) + Qm_t*tf.nn.relu(-S_0_n) + Qp_t*tf.nn.relu(Xunsc[:,2:3]-Xunsc[:,0:1]) + output_V_next_0-output_V) + zeroUn[:,1:2]*tf.square(K_t*tf.pow(injection,Gamma_t)+ Kappa_t*tf.cast(tf.equal(Xunsc[:,1:2],0.),tf.float64) +Qm_t*tf.nn.relu(-S_sup0_n) +Qp_t*tf.nn.relu(Xunsc[:,2:3]-Xunsc[:,0:1] - injection) +output_V_next_sup0-output_V))
    with tf.name_scope("train_V"):
        train_vars_V=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Vhidden1"+str(n)+"|Vhidden2"+str(n)+"|Voutput"+str(n))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_V)
        training_op_V= optimizer.minimize(lossV+reg_term_V,var_list=train_vars_V)
    reuse_vars=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="Vhidden1"+str(n+1)+"|Vhidden2"+str(n+1)+"|Voutput"+str(n+1)+"|Ahidden1"+str(n+1)+"|Ahidden2"+str(n+1)+"|Aoutput"+str(n+1))  # pre-training
    reuse_vars_dict=dict([(var.op.name,var) for var in reuse_vars]) # pre-training
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
            SampleNoise=np.random.normal(0,1,(Mbatch*n_batches,1))
            SampleNoiseValidation=np.random.normal(0,1,(MBatchValidation,1))
            val_cost=running_cost.eval(feed_dict={learning_rate_A:init_learning_rate_A,Noise:SampleNoiseValidation, Xunsc: SampleValidation[n][epoch*MBatchValidation:(epoch+1)*MBatchValidation], Xsc:SampleValidationRescaled[n][epoch*MBatchValidation:(epoch+1)*MBatchValidation]})
            cost_hist.append(val_cost)
            print("VcostControle: ",val_cost)
            for batch in range(n_batches):
                ind1=n_batches*epoch*Mbatch +batch*Mbatch
                ind2=n_batches*epoch*Mbatch +(batch+1)*Mbatch
                sess.run(training_op_A, feed_dict={learning_rate_A:init_learning_rate_A, Noise:SampleNoise[batch*Mbatch:(batch+1)*Mbatch], Xunsc: SampleTraining[n][ind1:ind2], Xsc:SampleTrainingRescaled[n][ind1:ind2]})            #
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
            SampleNoise=np.random.normal(0,1,(Mbatch*n_batches,1))
            SampleNoiseValidation=np.random.normal(0,1,(M_validation,1))
            val_loss=lossV.eval(feed_dict={learning_rate_A:init_learning_rate_A,Noise:SampleNoiseValidation,Xunsc: SampleValidation[n], Xsc:SampleValidationRescaled[n]})
            loss_hist.append(val_loss)
            print("VLoss: ",val_loss)
            for batch in range(n_batches):
                ind1=n_batches*epoch*Mbatch +batch*Mbatch
                ind2=n_batches*epoch*Mbatch +(batch+1)*Mbatch
                sess.run(training_op_V, feed_dict={learning_rate_V:init_learning_rate_V,Noise:SampleNoise[batch*Mbatch:(batch+1)*Mbatch],Xunsc: SampleTraining[n][ind1:ind2], Xsc:SampleTrainingRescaled[n][ind1:ind2]})
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
        save_path=saver.save(sess,"saver/Vfinal"+str(n)+".ckpt")

start_time=time.time()
for n in range(N-2,-1,-1):
    print("n=",n)
    TrainVnnAnn(n)
elapsed_time=time.time()- start_time
print("elapsed_time: ",elapsed_time)

start_time=time.time()
for n in range(1,-1,-1):
    print("n=",n)
    TrainVnnAnn(n)
elapsed_time=time.time()- start_time
print("elapsed_time: ",elapsed_time)

#### Forward Simulations

def Vnn(n,Xarg): # Take the time and the state as input, and return the corresponding value function.
    tf.reset_default_graph()
    Amin_t=tf.constant(Amin, dtype="float64")
    Amax_t=tf.constant(Amax, dtype="float64")
    Rbar_t=tf.constant(Rbar, dtype="float64")
    rho_t=tf.constant(Rho,dtype="float64")
    K_t=tf.constant(K,dtype="float64")
    Kappa_t=tf.constant(Kappa,dtype="float64")
    Gamma_t=tf.constant(Gamma,dtype="float64")
    Qm_t=tf.constant(Qm,dtype="float64")
    Qp_t=tf.constant(Qp,dtype="float64")
    Cmax_t=tf.constant(Cmax,dtype="float64")
    Cmin_t=tf.constant(Cmin,dtype="float64")
    Xunsc=tf.placeholder(tf.float64, shape=(None,n_inputs), name="Xunsc") #unscaled batch
    Xsc=tf.placeholder(tf.float64, shape=(None,n_inputs), name="Xsc") #rescaled batch
    he_init = tf.contrib.layers.variance_scaling_initializer()
    with tf.name_scope("dnn_V"):
        hidden1_V=tf.layers.dense(Xsc,n_hidden1, name="Vhidden1"+str(n), activation=tf.nn.elu, kernel_initializer=he_init)#,kernel_regularizer=regularizerV)
        hidden2_V=tf.layers.dense(hidden1_V,n_hidden2, name="Vhidden2"+str(n), activation=tf.nn.elu, kernel_initializer=he_init)#,kernel_regularizer=regularizerV)
        #hidden3_V=tf.layers.dense(hidden2_V,n_hidden3, name="Vhidden3"+str(n), activation=tf.nn.elu, kernel_initializer=he_init)#,kernel_regularizer=regularizerV)
        #hidden4_V=tf.layers.dense(hidden3_V,n_hidden4, name="Vhidden4"+str(n), activation=tf.nn.elu, kernel_initializer=he_init,kernel_regularizer=regularizerV)
        output_V=tf.layers.dense(hidden2_V, n_outputs_V, name="Voutput"+str(n),kernel_initializer=he_init)#,kernel_regularizer=regularizerV)
    reuse_vars=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="Vhidden1"+str(n)+"|Vhidden2"+str(n)+"|Vhidden3"+str(n)+"|Vhidden4"+str(n)+"|Voutput"+str(n))
    reuse_vars_dict=dict([(var.op.name,var) for var in reuse_vars])
    restore_saver=tf.train.Saver(reuse_vars_dict)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        init.run()
        restore_saver.restore(sess, "saver/Vfinal"+str(n)+".ckpt")
        Xarg_sc=(Xarg-Scaler[n].mean_)/np.sqrt(Scaler[n].var_) if n>0 else Xarg
        return output_V.eval(feed_dict={Xsc:Xarg_sc})


def Ann(n,Xarg): # Take time and state as input, and return the corresponding optimal control
    tf.reset_default_graph()
    Amin_t=tf.constant(Amin, dtype="float64")
    Amax_t=tf.constant(Amax, dtype="float64")
    rho_t=tf.constant(Rho,dtype="float64")
    K_t=tf.constant(K,dtype="float64")
    Kappa_t=tf.constant(Kappa,dtype="float64")
    Gamma_t=tf.constant(Gamma,dtype="float64")
    Qm_t=tf.constant(Qm,dtype="float64")
    Cmax_t=tf.constant(Cmax,dtype="float64")
    Cmin_t=tf.constant(Cmin,dtype="float64")
    Gamma_t=tf.constant(Gamma,dtype="float64")
    learning_rate_A=tf.placeholder(tf.float64, name="learning_rate_A")
    learning_rate_V=tf.placeholder(tf.float64, name="learning_rate_V")
    Xunsc=tf.placeholder(tf.float64, shape=(None,n_inputs), name="Xunsc") #Le batch unscaled
    Xsc=tf.placeholder(tf.float64, shape=(None,n_inputs), name="Xsc") #Le batch rescaled
    Noise=tf.placeholder(tf.float64, shape=(None,n_inputs), name="Noise") #Bruit gaussien pour le calcul de X au temps n+1
    he_init = tf.contrib.layers.variance_scaling_initializer()
    with tf.name_scope("dnn_A"):
        hidden1_A=tf.layers.dense(Xsc,n_hidden1_A, name="Ahidden1"+str(n), activation=tf.nn.sigmoid, kernel_initializer=he_init)
        hidden2_A=tf.layers.dense(hidden1_A,n_hidden2_A, name="Ahidden2"+str(n), activation=tf.nn.sigmoid, kernel_initializer=he_init)
        #hidden3_A=tf.layers.dense(hidden2_A,n_hidden3_A, name="Ahidden3"+str(n), activation=tf.nn.elu, kernel_initializer=he_init,kernel_regularizer=regularizerA)
        #hidden4_A=tf.layers.dense(hidden3_A,n_hidden4_A, name="hidden4"+str(n), activation=tf.nn.elu,kernel_initializer=he_init,kernel_regularizer=regularizerA)
        controle=tf.layers.dense(hidden2_A, n_outputs_A, name="Aoutput"+str(n),kernel_initializer=he_init)
        zeroUn=tf.nn.softmax(controle[:,:2])
        injection_temp= tf.nn.sigmoid(controle[:,2:])
        injection=Amin_t+tf.multiply(injection_temp,Amax_t-Amin_t)
        decision=tf.cast(tf.greater(zeroUn[:,1:2],.5),tf.float64)*injection
    reuse_vars=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="Ahidden1"+str(n)+"|Ahidden2"+str(n)+"|Aoutput"+str(n))
    reuse_vars_dict=dict([(var.op.name,var) for var in reuse_vars])
    restore_saver=tf.train.Saver(reuse_vars_dict)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        init.run()
        restore_saver.restore(sess, "saver/Vfinal"+str(n)+".ckpt")
        Xarg_sc=(Xarg-Scaler[n].mean_)/np.sqrt(Scaler[n].var_) if n>0 else Xarg
        return decision.eval(feed_dict={Xsc:Xarg_sc})

### Plots and forward simulations

Nb_simu=1000
Nb_times=10
res_nn=np.zeros(Nb_times)
for test in range(Nb_times):
    Xnn=np.zeros((N+1,Nb_simu,3))
    Xnn[0,:,0]=C0
    Xnn[0,:,1]=M0
    Xnn[0,:,2]=R0
    Jnn=np.zeros((N+1,Nb_simu))
    for n in range(N):
        print("time: ", n)
        noise=Sigma*np.random.normal(0,1,(Nb_simu,1))
        next_r=Rbar*(1-Rbar)+Rbar*Xnn[n,:,2:3]+noise
        controle=Ann(n,Xnn[n])
        next_m=1.*np.not_equal(controle,0)
        next_c=Xnn[n,:,0:1] + np.minimum(np.maximum(controle-Xnn[n,:,2:3],0.),Cmax-Xnn[n,:,0:1])-np.minimum(np.maximum(Xnn[n,:,2:3]-controle,0.),Xnn[n,:,0:1])
        I_n=np.minimum(np.maximum(controle-Xnn[n,:,2:3],0.),Cmax-Xnn[n,:,0:1])
        O_n=np.minimum(np.maximum(Xnn[n,:,2:3]-controle,0.),Xnn[n,:,0:1])
        S=Xnn[n,:,2:3]-controle+I_n-O_n
        running_cost=K*controle**Gamma + Kappa*np.not_equal(Xnn[n,:,1:2],controle) + Qm*np.maximum(-S,0) + Qp*np.maximum(Xnn[n,:,2:3]-Xnn[n,:,0:1]-controle,0.)
        Xnn[n+1]=np.concatenate((next_c,next_m,next_r),1)
        Jnn[n+1]=Jnn[n]+running_cost[:,0]
    res_nn[test]=np.mean(Jnn[N])
np.mean(res_nn)
np.std(res_nn)

Nb_simu=3
plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
for i in range(Nb_simu):
    plt.step([i for i in range(N+1)],Xnn[:,i,0])
    plt.xlabel("n")
    plt.ylabel("P")
    plt.grid(True)
plt.subplot(2,2,2)
for i in range(Nb_simu):
    plt.step([i for i in range(N+1)],Xnn[:,i,1])
    plt.xlabel("n")
    plt.ylabel("W")
    plt.grid(True)
plt.subplot(2,2,3)
for i in range(Nb_simu):
    plt.step([i for i in range(N+1)],Xnn[:,i,2])
    plt.xlabel("n")
    plt.ylabel("C_1")
    plt.grid(True)
plt.subplot(2,2,4)
for i in range(Nb_simu):
    plt.step([i for i in range(N+1)],Xnn[:,i,3])
    plt.xlabel("n")
    plt.ylabel("C_2")
plt.subplots_adjust(wspace=0.2)
plt.grid(True)
plt.title("")
plt.show()

for n in range(N):
    test=np.copy(SampleValidation[n,:2000])
    test_0=np.copy(test)
    test_0[:,1]=0
    test_1=np.copy(test)
    test_1[:,1]=1
    plt.figure(figsize=(12,8))
    plt.subplot(2,1,1)
    vf_0=Vnn(n,test_0)
    plt.scatter(test_0[:,0],test_0[:,2],c=vf_0[:,0])
    plt.title('Value function at time n='+str(n)+ ' for m=0')
    plt.xlabel("C")
    plt.ylabel("R")
    plt.colorbar()
    plt.subplot(2,1,2)
    vf_1=Vnn(n,test_1)
    plt.scatter(test_1[:,0],test_1[:,2],c=vf_1[:,0])
    plt.title('Value function at time n='+str(n)+' for m=1')
    plt.xlabel("C")
    plt.ylabel("R")
    plt.colorbar()
    plt.subplots_adjust(hspace=0.5)
    #plt.show()
    plt.savefig("7VFresults"+str(n)+'.pdf')
    plt.close()

for n in range(N):
    test=SampleValidation[n,:4000]
    test_0=np.copy(test)
    test_0[:,1]=0.
    test_1=np.copy(test)
    test_1[:,1]=1.
    plt.figure(figsize=(15,6))
    ax1=plt.subplot(1,2,1)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    decision_0=Ann(n,test_0)
    plt.scatter(test_0[:,0],test_0[:,2],c=decision_0[:,0],cmap=plt.cm.Reds)
    plt.title('Decisions at time n='+str(n)+ ' for m=0')
    plt.xlabel("C")
    plt.ylabel("R")
    plt.colorbar()
    ax2=plt.subplot(1,2,2)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    decision_1=Ann(n,test_1)
    plt.scatter(test_1[:,0],test_1[:,2],c=decision_1[:,0],cmap=plt.cm.Reds)
    plt.title('Decisions at time n='+str(n)+' for m=1')
    plt.xlabel("C")
    plt.ylabel("R")
    plt.colorbar()
    plt.subplots_adjust(hspace=0.5)
    plt.savefig("2Decisions"+str(n)+'.pdf')
    plt.close()
