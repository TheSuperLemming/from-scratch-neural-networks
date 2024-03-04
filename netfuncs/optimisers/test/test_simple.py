from netfuncs.optimisers.simple import *
import numpy as np

print('TEST CLASS: GradDescent')
print('-----------------------')

""" Test parameter update """
print('TEST: optimiser.update_parameters')
alpha = 0.1
params = {'A': np.random.rand(4, 5),
          'B': np.random.rand(3, 2),
          'C': np.random.rand(5, 1)}
grads = {'dA': 0.2*np.random.rand(params['A'].shape[0], params['A'].shape[1]),
         'dB': 0.2*np.random.rand(params['B'].shape[0], params['B'].shape[1]),
         'dC': 0.2*np.random.rand(params['C'].shape[0], params['C'].shape[1])}
optimiser = GradDescent(alpha=alpha)
new_params = optimiser.update_parameters(params, grads)
passA = (new_params['A'] == params['A']-alpha*grads['dA']).all()
passB = (new_params['B'] == params['B']-alpha*grads['dB']).all()
passC = (new_params['C'] == params['C']-alpha*grads['dC']).all()
print('> check A params >> test passed={}'.format(passA))
print('> check B params >> test passed={}'.format(passB))
print('> check C params >> test passed={}'.format(passC))
