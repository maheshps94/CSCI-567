import hw_utils as utils
from timeit import default_timer


X_tr_old,y_tr_old,X_te_old,y_te_old = utils.loaddata("MiniBooNE_PID.txt")
X_tr_norm,X_te_norm = utils.normalize(X_tr_old,X_te_old)
X_tr = X_tr_norm
y_tr = y_tr_old
X_te = X_te_norm
y_te = y_te_old
d_in = 50
d_out =2

#Linear Activation
print "\n-----------------------------------------------------------------------------"	
print "\nLinear Activation, Architecture I\n"
archs = [[d_in,d_out],[d_in,50,d_out],[d_in,50,50,d_out],[d_in,50,50,50,d_out]]
startTime = default_timer()
utils.testmodels(X_tr, y_tr, X_te, y_te, archs, actfn='linear', last_act='softmax', reg_coeffs=[0.0], 
				num_epoch=30, batch_size=1000, sgd_lr=0.001, sgd_decays=[0.0], sgd_moms=[0.0], 
					sgd_Nesterov=False, EStop=False, verbose=0)
print "The training time for Linear Activation, Architecture I " + str((default_timer() - startTime)) + " seconds"

print "\n-----------------------------------------------------------------------------"	
print "\nLinear Activation, Architecture II\n"
archs = [[d_in,50,d_out],[d_in,500,d_out],[d_in,500,300,d_out],[d_in,800,500,300,d_out],[d_in,800,800,500,300,d_out]]
startTime = default_timer()
utils.testmodels(X_tr, y_tr, X_te, y_te, archs, actfn='linear', last_act='softmax', reg_coeffs=[0.0], 
				num_epoch=30, batch_size=1000, sgd_lr=0.001, sgd_decays=[0.0], sgd_moms=[0.0], 
					sgd_Nesterov=False, EStop=False, verbose=0)
print "The training time for Linear Activation, Architecture II " + str((default_timer() - startTime)) + " seconds"

#Sigmoid Activation
print "\n-----------------------------------------------------------------------------"	
print "\nSigmoid Activation\n"
archs = [[d_in,50,d_out],[d_in,500,d_out],[d_in,500,300,d_out],[d_in,800,500,300,d_out],[d_in,800,800,500,300,d_out]]
startTime = default_timer()
utils.testmodels(X_tr, y_tr, X_te, y_te, archs, actfn='sigmoid', last_act='softmax', reg_coeffs=[0.0], 
				num_epoch=30, batch_size=1000, sgd_lr=0.001, sgd_decays=[0.0], sgd_moms=[0.0], 
					sgd_Nesterov=False, EStop=False, verbose=0)
print "The training time for Sigmoid Activation " + str((default_timer() - startTime)) + " seconds"

#ReLu Activation
print "\n-----------------------------------------------------------------------------"	
print "\nReLu Activation\n"
archs = [[d_in,50,d_out],[d_in,500,d_out],[d_in,500,300,d_out],[d_in,800,500,300,d_out],[d_in,800,800,500,300,d_out]]
startTime = default_timer()
utils.testmodels(X_tr, y_tr, X_te, y_te, archs, actfn='relu', last_act='softmax', reg_coeffs=[0.0], 
				num_epoch=30, batch_size=1000, sgd_lr=5e-4, sgd_decays=[0.0], sgd_moms=[0.0], 
					sgd_Nesterov=False, EStop=False, verbose=0)
print "The training time for ReLu Activation " + str((default_timer() - startTime)) + " seconds"

#L-2 Regularization
print "\n-----------------------------------------------------------------------------"	
print "\nL-2 Regularization\n"
archs = [[d_in,800,500,300,d_out]]
l2_params = [1e-7,5e-7,1e-6,5e-6,1e-5]
startTime = default_timer()
best_config_l2 = utils.testmodels(X_tr, y_tr, X_te, y_te, archs, actfn='relu', last_act='softmax', reg_coeffs = l2_params, 
				num_epoch=30, batch_size=1000, sgd_lr=5e-4, sgd_decays=[0.0], sgd_moms=[0.0], 
					sgd_Nesterov=False, EStop=False, verbose=0)
print "The training time for L-2 Regularization " + str((default_timer() - startTime)) + " seconds"

#Early Stopping and L2-regularization
print "\n-----------------------------------------------------------------------------"	
print "\nEarly Stopping and L2-regularization\n"
archs = [[d_in,800,500,300,d_out]]
l2_params = [1e-7,5e-7,1e-6,5e-6,1e-5]
startTime = default_timer()
best_config_l2_es = utils.testmodels(X_tr, y_tr, X_te, y_te, archs, actfn='relu', last_act='softmax', reg_coeffs = l2_params, 
				num_epoch=30, batch_size=1000, sgd_lr=5e-4, sgd_decays=[0.0], sgd_moms=[0.0], 
					sgd_Nesterov=False, EStop=True, verbose=0)
print "The training time for Early Stopping and L2-regularization " + str((default_timer() - startTime)) + " seconds"

if best_config_l2[5] > best_config_l2_es[5]:
	best_config_l2_reg =best_config_l2[1]
else:
	best_config_l2_reg = best_config_l2_es[1]

#SGD with weight decay
print "\n-----------------------------------------------------------------------------"	
print "\nSGD with weight decay\n"
archs = [[d_in,800,500,300,d_out]]
decays= [1e-5,5e-5,1e-4,3e-4,7e-4,1e-3]
startTime = default_timer()
best_config_decay = utils.testmodels(X_tr, y_tr, X_te, y_te, archs, actfn='relu', last_act='softmax', reg_coeffs = [5e-7], 
				num_epoch=100, batch_size=1000, sgd_lr=1e-5, sgd_decays=decays, sgd_moms=[0.0], 
					sgd_Nesterov=False, EStop=False, verbose=0)
print "The training time for SGD with weight decay " + str((default_timer() - startTime)) + " seconds"

#Momentum
print "\n-----------------------------------------------------------------------------"	
print "\nMomentum\n"
archs = [[d_in,800,500,300,d_out]]
momentum = [0.99, 0.98, 0.95, 0.9,0.85]
startTime = default_timer()
best_config_momentum = utils.testmodels(X_tr, y_tr, X_te, y_te, archs, actfn='relu', last_act='softmax', reg_coeffs = [0.0], 
				num_epoch=50, batch_size=1000, sgd_lr=1e-5, sgd_decays=[best_config_decay[2]], sgd_moms=momentum, 
					sgd_Nesterov=True, EStop=False, verbose=0)
print "The training time for Momentum " + str((default_timer() - startTime)) + " seconds"

#Combination of different coefficients
print "\n-----------------------------------------------------------------------------"	
print "\nCombination of different coefficents\n"
archs = [[d_in,800,500,300,d_out]]
startTime = default_timer()
utils.testmodels(X_tr, y_tr, X_te, y_te, archs, actfn='relu', last_act='softmax', reg_coeffs = [best_config_l2_reg], 
				num_epoch=100, batch_size=1000, sgd_lr=1e-5, sgd_decays=[best_config_decay[2]], sgd_moms=[best_config_momentum[3]], 
					sgd_Nesterov=True, EStop=True, verbose=0)
print "The training time for Combination of different coefficients " + str((default_timer() - startTime)) + " seconds"

#Grid search with cross-validation
print "\n-----------------------------------------------------------------------------"	
print "\nGrid search with cross-validation\n"
archs = [[d_in,50,d_out],[d_in,500,d_out],[d_in,500,300,d_out],[d_in,800,500,300,d_out],[d_in,800,800,500,300,d_out]]
l2_params = [1e-7,5e-7,1e-6,5e-6,1e-5]
decays= [1e-5,5e-5,1e-4]
startTime = default_timer()
utils.testmodels(X_tr, y_tr, X_te, y_te, archs, actfn='relu', last_act='softmax', reg_coeffs = l2_params, 
				num_epoch=100, batch_size=1000, sgd_lr=1e-5, sgd_decays=decays, sgd_moms=[0.99], 
					sgd_Nesterov=True, EStop=True, verbose=0)
print "The training time for Grid search with cross-validation " + str((default_timer() - startTime)) + " seconds"


#python test.py | tee output.txt
#use "top" command to see CPU usage