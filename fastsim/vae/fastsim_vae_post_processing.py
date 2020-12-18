####################################### POST-PROCESSING #######################################
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import energyflow as ef
import energyflow.utils as ut
import numpy as np
import time
import skhep.math as hep
from functools import reduce
from matplotlib.colors import LogNorm
import mplhep as mhep
plt.style.use(mhep.style.CMS)
from sklearn.preprocessing import MinMaxScaler
from pickle import load

print('############# Post-processing #############')

# Plots' colors based on colorbrewer schemes
cb_red = (228/255, 26/255, 28/255)
cb_blue = (55/255, 126/255, 184/255)
cb_green = (77/255, 175/255, 74/255) 

# Starting time
start_time = time.time()

num_particles = 100
batch_size = 100
select_portion_of_unseen_data = 80000

# Specify model name
model_name = 'pi_loss_fastsim_beta1_'+ str(num_particles) + 'p'

# Model reco input 
px_reco = torch.from_numpy(torch.load('px_reco_input_test_std_pi_loss_reco_fastsim_beta0001_alpha1_lr0001100p_jetpt_jetmass.pt'))[:select_portion_of_unseen_data, :]
py_reco = torch.from_numpy(torch.load('py_reco_input_test_std_pi_loss_reco_fastsim_beta0001_alpha1_lr0001100p_jetpt_jetmass.pt'))[:select_portion_of_unseen_data, :]
pz_reco = torch.from_numpy(torch.load('pz_reco_input_test_std_pi_loss_reco_fastsim_beta0001_alpha1_lr0001100p_jetpt_jetmass.pt'))[:select_portion_of_unseen_data, :]

# Model output 
px_gen_output = torch.from_numpy(torch.load('px_gen_output_test_std_pi_loss_reco_fastsim_beta0001_alpha1_lr0001100p_jetpt_jetmass.pt'))[:select_portion_of_unseen_data, :]
py_gen_output = torch.from_numpy(torch.load('py_gen_output_test_std_pi_loss_reco_fastsim_beta0001_alpha1_lr0001100p_jetpt_jetmass.pt'))[:select_portion_of_unseen_data, :]
pz_gen_output = torch.from_numpy(torch.load('pz_gen_output_test_std_pi_loss_reco_fastsim_beta0001_alpha1_lr0001100p_jetpt_jetmass.pt'))[:select_portion_of_unseen_data, :]

# load the scaler
gen_scaler = load(open('gen_data_scaler_train_pxpypz_wDijets_100p.pkl', 'rb'))
reco_scaler = load(open('reco_data_scaler_train_pxpypz_wDijets_100p.pkl', 'rb'))

####################################### INVERSE-STANDARDIZATION #######################################
print('############# INVERSE-STANDARDIZATION #############')

def reshape_data(px, py, pz):
    data = torch.stack([px, py, pz], dim=1) # [N-jets, 3, 30]
    data = data[:, None, :, :] # [N-jets, 1, 3, 30]
    return data

gen_data_out_normed = reshape_data(px_gen_output, py_gen_output, pz_gen_output)
reco_data_in_normed = reshape_data(px_reco, py_reco, pz_reco)

def inverse_minmax_normalization(data, scaler): # data of shape [n_jets, 1, 3_features, n_particles]
    data = data[:,0,:,:].permute(0,2,1)
    data = data.reshape(-1, 3)
    # INVERSE TRANSFORM
    inverse_norm_data = scaler.inverse_transform(data.numpy())
    inverse_norm_data = torch.from_numpy(inverse_norm_data).reshape(-1, 100, 3)
    px_inverse = inverse_norm_data[:, :, 0]
    py_inverse = inverse_norm_data[:, :, 1]
    pz_inverse = inverse_norm_data[:, :, 2]
    return px_inverse, py_inverse, pz_inverse

px_gen_r_output, py_gen_r_output, pz_gen_r_output = inverse_minmax_normalization(gen_data_out_normed, gen_scaler)
px_reco_input, py_reco_input, pz_reco_input = inverse_minmax_normalization(reco_data_in_normed, reco_scaler)

# Stack data
gen_data_out_r = reshape_data(px_gen_r_output, py_gen_r_output, pz_gen_r_output)
reco_data_in_r = reshape_data(px_reco_input, py_reco_input, pz_reco_input)
######################################################################################
n_jets = px_gen_output.shape[0]
print(n_jets, '(unseen) jets in the dataset.')

# Model gen input
test_data_gen_input = torch.load('test_data_gen_pxpypz_wDijets_100p.pt')[:n_jets,:,:,:]

px_gen_input = test_data_gen_input[:, 0, 0, :]
py_gen_input = test_data_gen_input[:, 0, 1, :]
pz_gen_input = test_data_gen_input[:, 0, 2, :]

#######################################################################################
# Masking for input & output constraints
####################################### MASKING #######################################
'''
print('############# Remove zero-padding for reco and gen input data #############')
#That will result to having (plots with) reco & gen data of unequal size.

# Input constraints
def remove_zero_padding(input_data): #input data shape [n_jets, 1, 3, n_particles]
    print('input_data data shape', input_data.shape)
    # Remove zero-padded particles. 
    px = input_data[:,0,0,:]
    py = input_data[:,0,1,:]
    pz = input_data[:,0,2,:]
    masked_px = px[px!=0]
    masked_py = py[py!=0]
    masked_pz = pz[pz!=0]
    print(masked_px.shape)
    print(masked_py.shape)
    print(masked_pz.shape)
    return masked_px, masked_py, masked_pz

px_reco_input_cut, py_reco_input_cut, pz_reco_input_cut = remove_zero_padding(reco_data_in_r)
px_gen_input_cut, py_gen_input_cut, pz_gen_input_cut = remove_zero_padding(test_data_gen_input)

# Output constraints
def mask_min_pt(output_data):
    print('output data shape', output_data.shape) # ([124100, 3, 30])
    # Mask output for min-pt
    min_pt_cut = 0.25
    mask =  output_data[:,0,0,:] * output_data[:,0,0,:] + output_data[:,0,1,:] * output_data[:,0,1,:] > min_pt_cut**2
    print(mask.shape)
    # Expand over the features' dimension
    mask = mask.unsqueeze(1)
    print(mask.shape)
    # Then, you can apply the mask
    data_masked = mask * output_data
    print(data_masked.shape) # Now, values that correspond to the min-pt should be zeroed. Check zeros again.
    return data_masked

# Test data
masked_outputs_reco = mask_min_pt(gen_data_out_r) # Now, values that correspond to the min-pt should be zeroed.
'''
######################################################################################
print('############# PLOT PARTICLES DISTRIBUTIONS #############')
# Px
plt.figure(figsize=(8, 6))
plt.hist(px_reco_input.cpu().detach().numpy().flatten(), bins=150,  range=[-1500, 1500], histtype = 'step', density=False, label= 'Reconstruction', color = cb_red, linewidth=1.5)
plt.hist(px_gen_input.cpu().detach().numpy().flatten(), bins=150,  range=[-1500, 1500], histtype = 'step', density=False, label= 'Generation', color = cb_green, linewidth=1.5)
plt.hist(px_gen_r_output.cpu().detach().numpy().flatten(), bins=150,  range=[-1500, 1500], histtype = 'step', density=False, label= 'DL Prediction', color = cb_blue, linewidth=1.5)
plt.ylabel('Particles')
plt.xlabel('particles $p_X$ (GeV)')
plt.title('Chamfer Loss')
plt.yscale('log')
plt.legend(loc='upper right', prop={'size': 16})
plt.savefig('px_gen_vs_reco_log_distribution_'+ model_name + str(num_particles) +'p.png', dpi=250, bbox_inches='tight')
plt.clf()

# Py
plt.figure(figsize=(8, 6))
plt.hist(py_reco_input.cpu().detach().numpy().flatten(), bins=150, range=[-1500, 1500], histtype = 'step', density=False, label= 'Reconstruction', color = cb_red, linewidth=1.5)
plt.hist(py_gen_input.cpu().detach().numpy().flatten(), bins=150,  range=[-1500, 1500], histtype = 'step', density=False, label= 'Generation', color = cb_green, linewidth=1.5)
plt.hist(py_gen_r_output.cpu().detach().numpy().flatten(), bins=150,  range=[-1500, 1500], histtype = 'step', density=False, label= 'DL Prediction', color = cb_blue, linewidth=1.5)
plt.ylabel('Particles')
plt.xlabel('particles $p_Y$ (GeV)')
plt.title('Chamfer Loss')
plt.yscale('log')
plt.legend(loc='upper right', prop={'size': 16})
plt.savefig('py_gen_vs_reco_log_distribution_'+ model_name + str(num_particles) +'p.png', dpi=250, bbox_inches='tight')
plt.clf()

# Pz
plt.figure(figsize=(8, 6))
plt.hist(pz_reco_input.cpu().detach().numpy().flatten(), bins=150, range=[-1500, 1500], histtype = 'step', density=False, label= 'Reconstruction', color = cb_red, linewidth=1.5)
plt.hist(pz_gen_input.cpu().detach().numpy().flatten(), bins=150,  range=[-1500, 1500], histtype = 'step', density=False, label= 'Generation', color = cb_green, linewidth=1.5)
plt.hist(pz_gen_r_output.cpu().detach().numpy().flatten(), bins=150,  range=[-1500, 1500], histtype = 'step', density=False, label= 'DL Prediction', color = cb_blue, linewidth=1.5)
plt.ylabel('Particles')
plt.xlabel('particles $p_Z$ (GeV)')
plt.title('Chamfer Loss')
plt.yscale('log')
plt.legend(loc='upper right', prop={'size': 16})
plt.savefig('pz_gen_vs_reco_log_distribution_'+ model_name+ str(num_particles) +'p.png', dpi=250, bbox_inches='tight')
plt.clf()

# Px
plt.figure(figsize=(8, 6))
plt.hist(px_reco_input.cpu().detach().numpy().flatten(), bins=250, range=[-500, 500],  histtype = 'step', density=False, label= 'Reconstruction', color = cb_red, linewidth=1.5)
plt.hist(px_gen_input.cpu().detach().numpy().flatten(), bins=250, range=[-500, 500],  histtype = 'step', density=False, label= 'Generation', color = cb_green, linewidth=1.5)
plt.hist(px_gen_r_output.cpu().detach().numpy().flatten(), bins=250, range=[-500, 500],  histtype = 'step', density=False, label= 'DL Prediction', color = cb_blue, linewidth=1.5)
plt.ylabel('Particles')
plt.xlabel('particles $p_X$ (GeV)')
plt.xlim(-100,100)
plt.title('Chamfer Loss')
plt.yscale('linear')
plt.legend(loc='upper right', prop={'size': 16})
plt.savefig('px_gen_vs_reco_llinear_distribution_'+ model_name + str(num_particles) +'p.png', dpi=250, bbox_inches='tight')
plt.clf()

# Py
plt.figure(figsize=(8, 6))
plt.hist(py_reco_input.cpu().detach().numpy().flatten(), bins=250, range=[-500, 500], histtype = 'step', density=False, label= 'Reconstruction', color = cb_red, linewidth=1.5)
plt.hist(py_gen_input.cpu().detach().numpy().flatten(), bins=250,  range=[-500, 500], histtype = 'step', density=False, label= 'Generation', color = cb_green, linewidth=1.5)
plt.hist(py_gen_r_output.cpu().detach().numpy().flatten(), bins=250,  range=[-500, 500], histtype = 'step', density=False, label= 'DL Prediction', color = cb_blue, linewidth=1.5)
plt.ylabel('Particles')
plt.xlabel('particles $p_Y$ (GeV)')
plt.xlim(-100,100)
plt.title('Chamfer Loss')
plt.yscale('linear')
plt.legend(loc='upper right', prop={'size': 16})
plt.savefig('py_gen_vs_reco_linear_distribution_'+ model_name + str(num_particles) +'p.png', dpi=250, bbox_inches='tight')
plt.clf()

# Pz
plt.figure(figsize=(8, 6))
plt.hist(pz_reco_input.cpu().detach().numpy().flatten(), bins=250, range=[-500, 500], histtype = 'step', density=False, label= 'Reconstruction', color = cb_red, linewidth=1.5)
plt.hist(pz_gen_input.cpu().detach().numpy().flatten(), bins=250,  range=[-500, 500], histtype = 'step', density=False, label= 'Generation', color = cb_green, linewidth=1.5)
plt.hist(pz_gen_r_output.cpu().detach().numpy().flatten(), bins=250,  range=[-500, 500], histtype = 'step', density=False, label= 'DL Prediction', color = cb_blue, linewidth=1.5)
plt.ylabel('Particles')
plt.xlabel('particles $p_Z$ (GeV)')
plt.xlim(-200,200)
plt.title('Chamfer Loss')
plt.yscale('linear')
plt.legend(loc='upper right', prop={'size': 16})
plt.savefig('pz_gen_vs_reco_linear_distribution_'+ model_name+ str(num_particles) +'p.png', dpi=250, bbox_inches='tight')
plt.clf()

######################################################################################
####################################### JET OBSERVABLES #######################################
print('############# JET OBSERVABLES #############')

# LVs from jets
# Create a four Lorentz Vector from px,py,pz for each jet, set mass to zero
def jet_from_pxpypzm(one_particle):
    px, py, pz, m = one_particle 
    jet = hep.vectors.LorentzVector()
    jet.setpxpypzm(px, py, pz, m)
    return jet 

def jet_samples_from_particle_samples(jet_constituents_data):
    '''
    :param particles: N x 100 x 3 ( N .. number of events, 100 particles, 3 features (px, py, pz) )
    :return: N x 1 ( N events, each consisting of 1 jet )
    '''
    event_jets = []
    for jet in jet_constituents_data:
        particle_jets = [jet_from_pxpypzm(particle) for particle in jet] #j1p1-30
        event_jets.append(reduce(lambda x,y: x+y, particle_jets)) # sum all particle-jets to get event-jet
    return event_jets

#px_r_masked = reco_inputs[:,0,:].detach().cpu().numpy()
#py_r_masked = reco_inputs[:,1,:].detach().cpu().numpy()
#pz_r_masked = reco_inputs[:,2,:].detach().cpu().numpy()

mass_reco_input = np.zeros((pz_reco_input.shape[0], num_particles))
mass_gen_input = np.zeros((pz_gen_input.shape[0], num_particles))
mass_gen_output = np.zeros((pz_gen_r_output.shape[0], num_particles))

reco_input_data = np.stack((px_reco_input, py_reco_input, pz_reco_input, mass_reco_input), axis=2)
gen_input_data = np.stack((px_gen_input, py_gen_input, pz_gen_input, mass_gen_input), axis=2)
gen_output_data = np.stack((px_gen_r_output, py_gen_r_output, pz_gen_r_output, mass_gen_output), axis=2)

#px_gen_r_reco_masked = gen_outputs[:,0,:].detach().cpu().numpy()
#py_gen_r_reco_masked = gen_outputs[:,1,:].detach().cpu().numpy()
#pz_gen_r_reco_masked = gen_outputs[:,2,:].detach().cpu().numpy()

print('Reco input data shape:', reco_input_data.shape)
print('Gen input data shape:', gen_input_data.shape)
print('Gen VAE output data shape:', gen_output_data.shape)

lvjets_reco_in = jet_samples_from_particle_samples(reco_input_data)
lvjets_gen_in = jet_samples_from_particle_samples(gen_input_data)
lvjets_gen_out = jet_samples_from_particle_samples(gen_output_data)

###########################################################################################################
########################## JET MASS #########################
# Reco 
m_j = [lvjet.mass for lvjet in lvjets_reco_in] 
mass_jet_reco = np.array(m_j)
# Gen input
m_j_gen_in = [lvjet.mass for lvjet in lvjets_gen_in] 
mass_jet_gen_in = np.array(m_j_gen_in)
# Gen output
m_j_gen_out = [lvjet.mass for lvjet in lvjets_gen_out] 
mass_jet_gen_out = np.array(m_j_gen_out)

########################## JET PT ###########################
# Reco 
pt_j = [lvjet.pt for lvjet in lvjets_reco_in]
pt_jet_reco = np.array(pt_j)
# Gen input
pt_j_gen_in = [lvjet.pt for lvjet in lvjets_gen_in]
pt_jet_gen_in = np.array(pt_j_gen_in)
# Gen output
pt_j_gen_out = [lvjet.pt for lvjet in lvjets_gen_out]
pt_jet_gen_out = np.array(pt_j_gen_out)

########################## JET ENERGY #######################
# Reco 
e_j = [lvjet.e for lvjet in lvjets_reco_in]
energy_jet_reco = np.array(e_j)
# Gen input
e_j_gen_in = [lvjet.e for lvjet in lvjets_gen_in]
energy_jet_gen_in = np.array(e_j_gen_in)
# Gen output
e_j_gen_out = [lvjet.e for lvjet in lvjets_gen_out]
energy_jet_gen_out = np.array(e_j_gen_out)

########################## JET ETA ##########################
# Reco 
eta_j = [lvjet.eta for lvjet in lvjets_reco_in]
eta_jet_reco = np.array(eta_j)
# Gen input
eta_j_gen_in = [lvjet.eta for lvjet in lvjets_gen_in]
eta_jet_gen_in = np.array(eta_j_gen_in)
# Gen output
eta_j_gen_out = [lvjet.eta for lvjet in lvjets_gen_out]
eta_jet_gen_out = np.array(eta_j_gen_out) 

########################## JET PHI ##########################
# Reco 
phi_j = [lvjet.phi() for lvjet in lvjets_reco_in]
phi_jet_reco = np.array(phi_j)
# Gen input
phi_j_gen_in = [lvjet.phi() for lvjet in lvjets_gen_in]
phi_jet_gen_in = np.array(phi_j_gen_in)
# Gen output
phi_j_gen_out = [lvjet.phi() for lvjet in lvjets_gen_out]
phi_jet_gen_out = np.array(phi_j_gen_out)

# Mass
'''
plt.figure(figsize=(8, 6))
plt.hist(mass_jet_reco, bins=50,  range = [-1.0, 500], histtype = 'step', density=False, label= 'Reconstruction', color = cb_red, linewidth=1.5)
plt.hist(mass_jet_gen_in, bins=50,  range = [-1.0, 500], histtype = 'step', density=False, label= 'Generation', color = cb_green, linewidth=1.5)
plt.hist(mass_jet_gen_out, bins=50,  range = [-1.0, 500], histtype = 'step', density=False, label= 'DL Prediction', color = cb_blue, linewidth=1.5)
plt.ylabel('Jets')
plt.xlabel('jet mass (GeV)')
plt.yscale('log')
plt.legend(loc='upper right', prop={'size': 16})
plt.savefig('lv_jet_mass_gen_vs_reco_log_distribution_'+ str(num_particles) +'p.png', dpi=250, bbox_inches='tight')
plt.clf()
'''

# Mass
plt.figure(figsize=(8, 6))
plt.hist(mass_jet_reco, bins=150,  range = [-1.0, 5000], histtype = 'step', density=False, label= 'Reconstruction', color = cb_red, linewidth=1.5)
plt.hist(mass_jet_gen_in, bins=150,  range = [-1.0, 5000], histtype = 'step', density=False, label= 'Generation', color = cb_green, linewidth=1.5)
plt.hist(mass_jet_gen_out, bins=150,  range = [-1.0, 5000], histtype = 'step', density=False, label= 'DL Prediction', color = cb_blue, linewidth=1.5)
plt.ylabel('Jets')
plt.xlabel('jet mass (GeV)')
plt.title('Chamfer Loss')
plt.yscale('linear')
plt.legend(loc='upper right', prop={'size': 16})
plt.savefig('lv_jet_mass_gen_vs_reco_linear_distribution_'+ model_name+ str(num_particles) +'p.png', dpi=250, bbox_inches='tight')
plt.clf()

# Pt    
plt.figure(figsize=(8, 6))
plt.hist(pt_jet_reco, bins=150,  range = [0.0, 8000], histtype = 'step', density=False, label= 'Reconstruction', color = cb_red, linewidth=1.5)
plt.hist(pt_jet_gen_in, bins=150,  range = [0.0, 8000], histtype = 'step', density=False, label= 'Generation', color = cb_green, linewidth=1.5)
plt.hist(pt_jet_gen_out, bins=150,  range = [0.0, 8000], histtype = 'step', density=False, label= 'DL Prediction', color = cb_blue, linewidth=1.5)
plt.ylabel('Jets')
plt.xlabel('jet $p_T$ (GeV)')
plt.title('Chamfer Loss')
plt.yscale('linear')
plt.legend(loc='upper right', prop={'size': 16})
plt.savefig('lv_jet_pt_gen_vs_reco_linear_distribution_'+ model_name + str(num_particles) +'p.png', dpi=250, bbox_inches='tight')
plt.clf()

# Energy   
plt.figure(figsize=(8, 6))
plt.hist(energy_jet_reco, bins=150,  range = [0.0, 15000], histtype = 'step', density=False, label= 'Reconstruction', color = cb_red, linewidth=1.5)
plt.hist(energy_jet_gen_in, bins=150,  range = [0.0, 15000], histtype = 'step', density=False, label= 'Generation', color = cb_green, linewidth=1.5)
plt.hist(energy_jet_gen_out, bins=150,  range = [0.0, 15000], histtype = 'step', density=False, label= 'DL Prediction', color = cb_blue, linewidth=1.5)
plt.ylabel('Jets')
plt.xlabel('jet energy (GeV)')
plt.title('Chamfer Loss')
plt.yscale('linear')
plt.legend(loc='upper right', prop={'size': 16})
plt.savefig('lv_jet_energy_gen_vs_reco_linear_distribution_'+ model_name + str(num_particles) +'p.png', dpi=250, bbox_inches='tight')
plt.clf()

# Eta   
plt.figure(figsize=(8, 6))
plt.hist(eta_jet_reco, bins=50, range=[-3.3, 3.3], histtype = 'step', density=False, label= 'Reconstruction', color = cb_red, linewidth=1.5)
plt.hist(eta_jet_gen_in, bins=50, range=[-3.3, 3.3], histtype = 'step', density=False, label= 'Generation', color = cb_green, linewidth=1.5)
plt.hist(eta_jet_gen_out, bins=50, range=[-3.3, 3.3], histtype = 'step', density=False, label= 'DL Prediction', color = cb_blue, linewidth=1.5)
plt.ylabel('Jets')
plt.xlabel('jet $\eta$')
plt.title('Chamfer Loss')
plt.yscale('linear')
plt.legend(loc='lower right', prop={'size': 16})
plt.savefig('lv_jet_eta_gen_vs_reco_linear_distribution_'+ model_name + str(num_particles) +'p.png', dpi=250, bbox_inches='tight')
plt.clf()

# Phi   
plt.figure(figsize=(8, 6))
plt.hist(phi_jet_reco, bins=50, range=[-3.14, 3.14], histtype = 'step', density=False, label= 'Reconstruction', color = cb_red, linewidth=1.5)
plt.hist(phi_jet_gen_in, bins=50, range=[-3.14, 3.14], histtype = 'step', density=False, label= 'Generation', color = cb_green, linewidth=1.5)
plt.hist(phi_jet_gen_out, bins=50, range=[-3.14, 3.14], histtype = 'step', density=False, label= 'DL Prediction', color = cb_blue, linewidth=1.5)
plt.ylabel('Jets')
plt.xlabel('jet $\phi$')
plt.title('Chamfer Loss')
plt.yscale('linear')
plt.legend(loc='lower right', prop={'size': 16})
plt.savefig('lv_jet_phi_gen_vs_reco_linear_distribution_'+ model_name + str(num_particles) +'p.png', dpi=250, bbox_inches='tight')
plt.clf()

################################################### RESIDUALS ###################################################
# Residuals
def prediction_error(target, gen):
    pred_error = (target - gen)/target
    return pred_error

def residual_error(target, pred):
    rel_resid = (target - pred)/target
    return rel_resid

def inp_out_error(gen, pred):
    vae_resid = (gen - pred)/gen
    return vae_resid

# JET KINEMATICS RESIDUALS
mass_res_error = residual_error(mass_jet_reco, mass_jet_gen_out)
pt_res_error = residual_error(pt_jet_reco, pt_jet_gen_out)
energy_res_error = residual_error(energy_jet_reco, energy_jet_gen_out)
eta_res_error = residual_error(eta_jet_reco, eta_jet_gen_out)
phi_res_error = residual_error(phi_jet_reco, phi_jet_gen_out)

mass_pred_error = prediction_error(mass_jet_reco, mass_jet_gen_in)
pt_pred_error = prediction_error(pt_jet_reco, pt_jet_gen_in)
energy_pred_error = prediction_error(energy_jet_reco, energy_jet_gen_in)
eta_pred_error = prediction_error(eta_jet_reco, eta_jet_gen_in)
phi_pred_error = prediction_error(phi_jet_reco, phi_jet_gen_in)

mass_vae_error = inp_out_error(mass_jet_gen_in, mass_jet_gen_out)
pt_vae_error = inp_out_error(pt_jet_gen_in, pt_jet_gen_out)
energy_vae_error = inp_out_error(energy_jet_gen_in, energy_jet_gen_out)
eta_vae_error = inp_out_error(eta_jet_gen_in, eta_jet_gen_out)
phi_vae_error = inp_out_error(phi_jet_gen_in, phi_jet_gen_out)

################################################### RESIDUALS ###################################################
# Mass residual
plt.figure(figsize=(8, 6))
plt.hist(mass_res_error, bins=150, range=[-300, 300],  histtype = 'step', density=False, label= '(RECO - DL)/RECO', color = cb_red, linewidth=1.5)
plt.hist(mass_pred_error, bins=150, range=[-300, 300], histtype = 'step', density=False, label= '(RECO - GEN)/RECO', color = cb_green, linewidth=1.5)
plt.hist(mass_vae_error, bins=150, range=[-300, 300], histtype = 'step', density=False, label= '(GEN - DL)/GEN', color = cb_blue, linewidth=1.5)
plt.ylabel('Jets')
plt.xlabel('jet mass (GeV)')
plt.title('Chamfer Loss')
plt.yscale('linear')
plt.legend(loc='upper right', prop={'size': 16})
plt.savefig('jet_mass_relative_error_'+ model_name + str(num_particles) +'p.png', dpi=250, bbox_inches='tight')
plt.clf()

# Pt residual
plt.figure(figsize=(8, 6))
plt.hist(pt_res_error, bins=150, range=[-300, 300], histtype = 'step', density=False, label= '(RECO - DL)/RECO', color = cb_red, linewidth=1.5)
plt.hist(pt_pred_error, bins=150, range=[-300, 300], histtype = 'step', density=False, label= '(RECO - GEN)/RECO', color = cb_green, linewidth=1.5)
plt.hist(pt_vae_error, bins=150, range=[-300, 300], histtype = 'step', density=False, label= '(GEN - DL)/GEN', color = cb_blue, linewidth=1.5)
plt.ylabel('Jets')
plt.xlabel('jet $p_T$ (GeV)')
plt.title('Chamfer Loss')
plt.yscale('linear')
plt.legend(loc='upper right', prop={'size': 16})
plt.savefig('jet_pt_relative_error_'+ model_name + str(num_particles) +'p.png', dpi=250, bbox_inches='tight')
plt.clf()

# Energy residual
plt.figure(figsize=(8, 6))
plt.hist(energy_res_error, bins=150,range=[-300, 300],  histtype = 'step', density=False, label= '(RECO - DL)/RECO', color = cb_red, linewidth=1.5)
plt.hist(energy_pred_error, bins=150, range=[-300, 300], histtype = 'step', density=False, label= '(RECO - GEN)/RECO', color = cb_green, linewidth=1.5)
plt.hist(energy_vae_error, bins=150, range=[-300, 300], histtype = 'step', density=False, label= '(GEN - DL)/GEN', color = cb_blue, linewidth=1.5)
plt.ylabel('Jets')
plt.xlabel('jet energy (GeV)')
plt.title('Chamfer Loss')
plt.yscale('linear')
plt.legend(loc='upper right', prop={'size': 16})
plt.savefig('jet_energy_relative_error_'+ model_name + str(num_particles) +'p.png', dpi=250, bbox_inches='tight')
plt.clf()

# Eta residual
plt.figure(figsize=(8, 6))
plt.hist(eta_res_error, bins=150,  range=[-300, 300],histtype = 'step', density=False, label= '(RECO - DL)/RECO', color = cb_red, linewidth=1.5)
plt.hist(eta_pred_error, bins=150,range=[-300, 300],  histtype = 'step', density=False, label= '(RECO - GEN)/RECO', color = cb_green, linewidth=1.5)
plt.hist(eta_vae_error, bins=150, range=[-300, 300], histtype = 'step', density=False, label= '(GEN - DL)/GEN', color = cb_blue, linewidth=1.5)
plt.ylabel('Jets')
plt.xlabel('jet $\eta$')
plt.title('Chamfer Loss')
plt.yscale('linear')
plt.legend(loc='upper right', prop={'size': 16})
plt.savefig('jet_eta_relative_error_'+ model_name + str(num_particles) +'p.png', dpi=250, bbox_inches='tight')
plt.clf()

# Phi residual
plt.figure(figsize=(8, 6))
plt.hist(phi_res_error, bins=150, range=[-300, 300], histtype = 'step', density=False, label= '(RECO - DL)/RECO', color = cb_red, linewidth=1.5)
plt.hist(phi_pred_error, bins=150, range=[-300, 300], histtype = 'step', density=False, label= '(RECO - GEN)/RECO', color = cb_green, linewidth=1.5)
plt.hist(phi_vae_error, bins=150, range=[-300, 300], histtype = 'step', density=False, label= '(GEN - DL)/GEN', color = cb_blue, linewidth=1.5)
plt.ylabel('Jets')
plt.xlabel('jet $\phi$')
plt.title('Chamfer Loss')
plt.yscale('linear')
plt.legend(loc='upper right', prop={'size': 16})
plt.savefig('jet_phi_relative_error_'+ model_name + str(num_particles) +'p.png', dpi=250, bbox_inches='tight')
plt.clf()

####################################### PT FLOWS #######################################
print('############# PT FLOWS #############')

##########################################################################################################################
# Get arrays/tensors of masked data
reco_inputs = np.stack([px_reco_input, py_reco_input, pz_reco_input], axis=1)
gen_inputs = np.stack([px_gen_input, py_gen_input, pz_gen_input], axis=1)
gen_outputs = np.stack([px_gen_r_output, py_gen_r_output, pz_gen_r_output], axis=1)

##########################################################################################################################
def particles_Energy (particles_momentum): # input should be of shape [N-events, features, M-particles]
    E_particles = np.sqrt(np.sum(particles_momentum*particles_momentum, axis=1)) # E per particle shape: [N-events, M-particles]
    return E_particles

E_reco = particles_Energy(reco_inputs)
PX_reco = reco_inputs[:,0,:]
PY_reco = reco_inputs[:,1,:]
PZ_reco = reco_inputs[:,2,:]

E_gen_in = particles_Energy(gen_inputs)
PX_gen_in = gen_inputs[:,0,:]
PY_gen_in = gen_inputs[:,1,:]
PZ_gen_in = gen_inputs[:,2,:]

E_gen_out = particles_Energy(gen_outputs)
PX_gen_out = gen_outputs[:,0,:]
PY_gen_out = gen_outputs[:,1,:]
PZ_gen_out = gen_outputs[:,2,:]

################################################################################################
###################################### Compute Pt,Eta,Phi ######################################
# -> particles pt, eta, phi
def compute_handr_coords (e, px, py, pz):
    pt = np.sqrt(px**2 + py**2)
    try:
        eta = 0.5*np.log((e+pz)/(e-pz))
    except (RuntimeError, TypeError, NameError):
        pass
    y = np.nan_to_num(eta)
    phi = np.arctan2(py, px)
    return pt, y, phi

pt_reco, eta_reco, phi_reco = compute_handr_coords(E_reco, PX_reco, PY_reco, PZ_reco)
pt_gen_in, eta_gen_in, phi_gen_in = compute_handr_coords(E_gen_in, PX_gen_in, PY_gen_in, PZ_gen_in)
pt_gen_out, eta_gen_out, phi_gen_out = compute_handr_coords(E_gen_out, PX_gen_out, PY_gen_out, PZ_gen_out)

p_ptetaphi_reco = torch.from_numpy(np.stack([pt_reco, eta_reco, phi_reco], axis=1)).float().cuda()[:select_portion_of_unseen_data, :,:]
p_ptetaphi_gen_in = torch.from_numpy(np.stack([pt_gen_in, eta_gen_in, phi_gen_in], axis=1)).float().cuda()[:select_portion_of_unseen_data, :,:]
p_ptetaphi_gen_out = torch.from_numpy(np.stack([pt_gen_out, eta_gen_out, phi_gen_out], axis=1)).float().cuda()[:select_portion_of_unseen_data, :,:]

jet_ptetaphi_reco = torch.from_numpy((np.stack([pt_jet_reco, eta_jet_reco, phi_jet_reco], axis=1))).float().cuda()
jet_ptetaphi_gen_in = torch.from_numpy((np.stack([pt_jet_gen_in, eta_jet_gen_in, phi_jet_gen_in], axis=1))).float().cuda()
jet_ptetaphi_gen_out = torch.from_numpy((np.stack([pt_jet_gen_out, eta_jet_gen_out, phi_jet_gen_out], axis=1))).float().cuda()

#torch.save(jet_ptetaphi_reco, 'jet_ptetaphi_reco' + model_name + '.pt')
#torch.save(jet_ptetaphi_gen_in, 'jet_ptetaphi_gen_in' + model_name + '.pt')
#torch.save(jet_ptetaphi_gen_out, 'jet_ptetaphi_gen_out' + model_name + '.pt')

################################################################################################
n_rings = 4
DR_0 = 0.0
DR_1 = 0.1
DR_2 = 0.15
DR_3 = 0.3
DR_4 = 0.8

colors = [cb_red, cb_green, cb_blue]

def heaviside_negative_n_to_zero(data): # equivalent to torch.heaviside(tensor, 1)
    # Sets numbers <0 to 0 and everything else to 1.
    binary_tensor = torch.where(data < torch.zeros_like(data), torch.zeros_like(data), torch.ones_like(data))
    return binary_tensor

def heaviside_negativeOrzero_n_to_zero(data): # equivalent to torch.heaviside(tensor, 0)
    # Sets numbers <=0 to 0 and everything else to 1.
    binary_tensor = torch.where(data <= torch.zeros_like(data), torch.zeros_like(data), torch.ones_like(data))
    return binary_tensor

def compute_pt_flow (particle_ptetaphi, jet_ptetaphi, DR_lower_bound, DR_upper_bound):
    DR = torch.sqrt((particle_ptetaphi[:, 2, :] - jet_ptetaphi[:, 2].unsqueeze(1))*(particle_ptetaphi[:, 2, :] - jet_ptetaphi[:, 2].unsqueeze(1)) + 
        (particle_ptetaphi[:, 1, :] - jet_ptetaphi[:, 1].unsqueeze(1))*(particle_ptetaphi[:, 1, :] - jet_ptetaphi[:, 1].unsqueeze(1)))
    pt_flow = torch.sum(particle_ptetaphi[:, 0, :]*heaviside_negative_n_to_zero(DR - DR_lower_bound)*(1-heaviside_negativeOrzero_n_to_zero(DR_upper_bound - DR)), dim=1)/jet_ptetaphi[:,0]
    return pt_flow


pt_flow_reco_in_1 = compute_pt_flow(p_ptetaphi_reco, jet_ptetaphi_reco, DR_0, DR_1).cpu().detach().numpy()
pt_flow_reco_in_2 = compute_pt_flow(p_ptetaphi_reco, jet_ptetaphi_reco, DR_1, DR_2).cpu().detach().numpy()
pt_flow_reco_in_3 = compute_pt_flow(p_ptetaphi_reco, jet_ptetaphi_reco, DR_2, DR_3).cpu().detach().numpy()
pt_flow_reco_in_4 = compute_pt_flow(p_ptetaphi_reco, jet_ptetaphi_reco, DR_3, DR_4).cpu().detach().numpy()

pt_flow_gen_in_1 = compute_pt_flow(p_ptetaphi_gen_in, jet_ptetaphi_gen_in, DR_0, DR_1).cpu().detach().numpy()
pt_flow_gen_in_2 = compute_pt_flow(p_ptetaphi_gen_in, jet_ptetaphi_gen_in, DR_1, DR_2).cpu().detach().numpy()
pt_flow_gen_in_3 = compute_pt_flow(p_ptetaphi_gen_in, jet_ptetaphi_gen_in, DR_2, DR_3).cpu().detach().numpy()
pt_flow_gen_in_4 = compute_pt_flow(p_ptetaphi_gen_in, jet_ptetaphi_gen_in, DR_3, DR_4).cpu().detach().numpy()

pt_flow_gen_out_1 = compute_pt_flow(p_ptetaphi_gen_out, jet_ptetaphi_gen_out, DR_0, DR_1).cpu().detach().numpy()
pt_flow_gen_out_2 = compute_pt_flow(p_ptetaphi_gen_out, jet_ptetaphi_gen_out, DR_1, DR_2).cpu().detach().numpy()
pt_flow_gen_out_3 = compute_pt_flow(p_ptetaphi_gen_out, jet_ptetaphi_gen_out, DR_2, DR_3).cpu().detach().numpy()
pt_flow_gen_out_4 = compute_pt_flow(p_ptetaphi_gen_out, jet_ptetaphi_gen_out, DR_3, DR_4).cpu().detach().numpy()

#pt_flow_in = np.sum(np.stack([pt_flow_in_1, pt_flow_in_2, pt_flow_in_3, pt_flow_in_4], axis=1), axis=1)
#pt_flow_reco_reco = np.sum(np.stack([pt_flow_reco_reco_gen_out, pt_flow_reco_reco_2, pt_flow_reco_reco_3, pt_flow_reco_reco_4], axis=1), axis=1)

################################################## Plot pt-flows #############################
plt.figure(figsize=(8, 6))
plt.hist(pt_flow_reco_in_1, bins=200, range=[0.00, 2.0], density=True, label='Reco input jet $p_T$ flow (Ring:1)', histtype='step', fill=False, linewidth=1.5, color=colors[0])
plt.hist(pt_flow_gen_in_1, bins=200, range=[0.00, 2.0], density=True, label='Gen input jet $p_T$ flow (Ring:1)', histtype='step', fill=False, linewidth=1.5, color=colors[1])
plt.hist(pt_flow_gen_out_1, bins=200, range=[0.00, 2.0], density=True, label='Gen VAE output jet $p_T$ flow (Ring:1)', histtype='step', fill=False, linewidth=1.5, color=colors[2])
plt.xlabel('$p_T$-Flow')
plt.ylabel("Probability (a.u.)")
plt.semilogy()
plt.title('ΔR < 0.1')
plt.legend(loc='upper right', prop={'size': 16})
plt.savefig('pt_flow_1_'+str(model_name)+'.png', dpi=250, bbox_inches='tight')
plt.clf()

plt.figure(figsize=(8, 6))
plt.hist(pt_flow_reco_in_2, bins=200,  range=[0.00, 2.0], density=True, label='Reco input jet $p_T$ flow (Ring:2)', histtype='step', fill=False, linewidth=1.5, color=colors[0])
plt.hist(pt_flow_gen_in_2, bins=200,  range=[0.00, 2.0], density=True, label='Gen input jet $p_T$ flow (Ring:2)', histtype='step', fill=False, linewidth=1.5, color=colors[1])
plt.hist(pt_flow_gen_out_2, bins=200,  range=[0.00, 2.0], density=True, label='Gen VAE output jet $p_T$ flow (Ring:2)', histtype='step', fill=False, linewidth=1.5, color=colors[2])
plt.xlabel('$p_T$-Flow')
plt.ylabel("Probability (a.u.)")
plt.semilogy()
plt.title('0.1 < ΔR < 0.15')
plt.legend(loc='upper right', prop={'size': 16})
plt.savefig('pt_flow_2_'+str(model_name)+'.png', dpi=250, bbox_inches='tight')
plt.clf()

plt.figure(figsize=(8, 6))
plt.hist(pt_flow_reco_in_3, bins=200, range=[0.00, 2.0], density=True, label='Reco input jet $p_T$ flow (Ring:3)', histtype='step', fill=False, linewidth=1.5, color=colors[0])
plt.hist(pt_flow_gen_in_3, bins=200, range=[0.00, 2.0], density=True, label='Gen input jet $p_T$ flow (Ring:3)', histtype='step', fill=False, linewidth=1.5, color=colors[1])
plt.hist(pt_flow_gen_out_3, bins=200, range=[0.00, 2.0], density=True, label='Gen VAE output jet $p_T$ flow (Ring:3)', histtype='step', fill=False, linewidth=1.5, color=colors[2])
plt.xlabel('$p_T$-Flow')
plt.ylabel("Probability (a.u.)")
plt.semilogy()
plt.title('0.15 < ΔR < 0.3')
plt.legend(loc='upper right', prop={'size': 16})
plt.savefig('pt_flow_3_'+str(model_name)+'.png', dpi=250, bbox_inches='tight')
plt.clf()

plt.figure(figsize=(8, 6))
plt.hist(pt_flow_reco_in_4, bins=200, range=[0.00, 2.0], density=True, label='Reco input jet $p_T$ flow (Ring:4)', histtype='step', fill=False, linewidth=1.5, color=colors[0])
plt.hist(pt_flow_gen_in_4, bins=200, range=[0.00, 2.0], density=True, label='Gen input jet $p_T$ flow (Ring:4)', histtype='step', fill=False, linewidth=1.5, color=colors[1])
plt.hist(pt_flow_gen_out_4, bins=200, range=[0.00, 2.0], density=True, label='Gen VAE output jet $p_T$ flow (Ring:4)', histtype='step', fill=False, linewidth=1.5, color=colors[2])
plt.xlabel('$p_T$-Flow')
plt.ylabel("Probability (a.u.)")
plt.semilogy()
plt.title('0.3 < ΔR < 0.8')
plt.legend(loc='upper right', prop={'size': 16})
plt.savefig('pt_flow_4_'+str(model_name)+'.png', dpi=250, bbox_inches='tight')
plt.clf()

####################################### ENERGY FLOW POLYNOMIALS #######################################
print('############# ENERGY FLOW POLYNOMIALS #############')

####################################### HADRONIC COORDS #######################################
'''
# Measures also deal with converting between different representations of particle momenta, e.g. Cartesian [E,px,py,pz] or hadronic [pt,y,phi,m]
# Input
ptetaphim_reco = np.stack([pt_reco, eta_reco, phi_reco, mass], axis=2)
# Output
ptetaphim_gen_out = np.stack([pt_gen_out, eta_gen_out, phi_gen_out, mass_gen_out], axis=2)

#epxpypzs_reco = np.stack([E_reco, PX_reco, PY_reco, PZ_reco], axis=2)
#epxpypzs_reco.shape # (N, M, features)

#epxpypzs_gen_out = np.stack([E_gen_out, PX_gen_out, PY_gen_out, PZ_gen_out], axis=2)
#epxpypzs_gen_out.shape

# events = epxpypzs # shape should be [events, M-particles, N-features]
events_reco = torch.Tensor(ptetaphim_reco)
events_gen_out = torch.Tensor(ptetaphim_gen_out)

# Create iterable data loader Initializing Data Set Iterators for easy and batched access
test_loader_reco = DataLoader(dataset=events_reco, batch_size=batch_size, shuffle=False)
test_loader_gen_out = DataLoader(dataset=events_gen_out, batch_size=batch_size, shuffle=False)

# get all EFPs with n<=4, d<=4
efpset = ef.EFPSet(('n==',4), ('d==',4),  ('p==',1),  measure='hadr', beta=1, normed=None, check_input=True, verbose=True)
# efpset = ef.EFPSet(('n==',4), ('d==',4),  ('p==',1),  measure='hadr', beta=1, normed=None, coords='ptyphim', check_input=True, verbose=True)

results_reco = np.empty(shape=(0, 5),  dtype=np.float32)
results_gen_out = np.empty(shape=(0, 5),  dtype=np.float32)

for w, (jets) in enumerate(test_loader_reco):
    if w == (len(test_loader_reco)-1):
        break
    jets = (jets.float()).cpu().detach().numpy()
    batch_results = efpset.batch_compute(jets).astype('float32')
    results_reco = np.concatenate((results_reco, batch_results), axis=0)

for w, (jets) in enumerate(test_loader_gen_out):
    if w == (len(test_loader_gen_out)-1):
        break
    jets = (jets.float()).cpu().detach().numpy()
    batch_results = efpset.batch_compute(jets).astype('float32')
    results_gen_out = np.concatenate((results_gen_out, batch_results), axis=0)

print('Results shape')
print(results_reco.shape)
print(results_reco.dtype)
print(results_gen_out.shape)
print(results_gen_out.dtype)

labels = ['EFP-1', 'EFP-2', 'EFP-3', 'EFP-4', 'EFP-5']
colors = ['red', cb_green, 'orange']

results = np.stack([results_reco, results_gen_out], axis=2)

plt.figure(figsize=(8, 6))
# Plot each EFP separately 
#box = dict(pad=5, alpha=0.8)
for i in range(len(labels)):
    graph = efpset.graphs(i)
    n, _, d, v, _, c, p, _ = efpset.specs[i]
    plt.hist(results[:, i,0], bins=100, label='Reco level jets', density=False, range=(results[:, i,1].min(), results[:, i,1].max()), histtype='step', fill=False, linewidth=1.5, color=colors[0])
    plt.hist(results[:, i,1], bins=100, label='Gen level jets', density=False, range=(results[:, i,1].min(), results[:, i,1].max()), histtype='step', fill=False, linewidth=1.5, color=colors[1])
    #plt.hist(results[:, i,2], bins=100, label=('MSE EFP graph:'+' '.join(map(str, graph))), density=False, range=(results[:, i,2].min(), results[:, i,2].max()), histtype='step', fill=False, linewidth=1.5, color=colors[2])
    plt.xlabel('EFP' + str(i+1), horizontalalignment='center')
    plt.ylabel('Jets', horizontalalignment='center')
    #plt.set_ylabel('Jets', bbox=box)
    #plt.title('MSE')
    plt.legend(loc='upper right')
    #plt.yscale('log')
    plt.yscale('linear')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    # plt.semilogx()
    plt.savefig('in_vs_out_efp_'+ str(i+1) +str(model_name)+'.png', dpi=250, bbox_inches='tight')
    plt.clf()

'''
####################################### CARTESIAN COORDS #######################################
'''
# Measures also deal with converting between different representations of particle momenta, e.g. Cartesian [E,px,py,pz] or hadronic [pt,y,phi,m]
# Input
epxpypzs_reco = np.stack([E_reco, PX_reco, PY_reco, PZ_reco], axis=2)
epxpypzs_reco.shape # (N, M, features)
# Output
epxpypzs_gen_out = np.stack([E_gen_out, PX_gen_out, PY_gen_out, PZ_gen_out], axis=2)
epxpypzs_gen_out.shape

# events = epxpypzs # shape should be [events, M-particles, N-features]
events_reco = torch.Tensor(epxpypzs_reco)
events_gen_out = torch.Tensor(epxpypzs_gen_out)

# Create iterable data loader Initializing Data Set Iterators for easy and batched access
test_loader_reco = DataLoader(dataset=events_reco, batch_size=batch_size, shuffle=False)
test_loader_gen_out = DataLoader(dataset=events_gen_out, batch_size=batch_size, shuffle=False)

# get all EFPs with n<=4, d<=4
#efpset = ef.EFPSet(('n==',4), ('d==',4),  ('p==',1),  measure='ee', beta=1, normed=None, coords='epxpypz', check_input=True, verbose=True)
#efpset = ef.EFPSet(('n==',4), ('d==',4),  ('p==',1),  measure='ee', beta=2, normed=None, coords='epxpypz', check_input=True, verbose=True)
efpset = ef.EFPSet(('n==',4), ('d==',4),  ('p==',1),  measure='hadrdot', beta=1, normed=None, coords='epxpypz', check_input=True, verbose=True)
#efpset = ef.EFPSet(('n==',4), ('d==',4),  ('p==',1),  normed=None, coords='epxpypz', check_input=True, verbose=True)

results_reco = np.empty(shape=(0, 5),  dtype=np.float32)
results_gen_out = np.empty(shape=(0, 5),  dtype=np.float32)

for w, (jets) in enumerate(test_loader_reco):
    if w == (len(test_loader_reco)-1):
        break
    jets = (jets.float()).cpu().detach().numpy()
    batch_results = efpset.batch_compute(jets).astype('float32').cuda()
    results_reco = np.concatenate((results_reco, batch_results), axis=0).cpu().detach()

print('1st loop done.')

for w, (jets) in enumerate(test_loader_gen_out):
    if w == (len(test_loader_gen_out)-1):
        break
    jets = (jets.float()).cpu().detach().numpy()
    batch_results = efpset.batch_compute(jets).astype('float32')
    results_gen_out = np.concatenate((results_gen_out, batch_results), axis=0)

print('2nd loop done.')

print('Results shape')
print(results_reco.shape)
print(results_reco.dtype)
print(results_gen_out.shape)
print(results_gen_out.dtype)

labels = ['EFP-1', 'EFP-2', 'EFP-3', 'EFP-4', 'EFP-5']
colors = ['red', cb_green, 'orange']

results = np.stack([results_reco, results_gen_out], axis=2)

plt.figure(figsize=(8, 6))
# Plot each EFP separately 
#box = dict(pad=5, alpha=0.8)
for i in range(len(labels)):
    graph = efpset.graphs(i)
    n, _, d, v, _, c, p, _ = efpset.specs[i]
    plt.hist(results[:, i,0], bins=100, label='Reco level jets', density=False, histtype='step', fill=False, linewidth=1.5, color=colors[0])
    plt.hist(results[:, i,1], bins=100, label='Gen level jets', density=False, histtype='step', fill=False, linewidth=1.5, color=colors[1])
    #plt.hist(results[:, i,2], bins=100, label=('MSE EFP graph:'+' '.join(map(str, graph))), density=False, range=(results[:, i,2].min(), results[:, i,2].max()), histtype='step', fill=False, linewidth=1.5, color=colors[2])
    plt.xlabel('EFP' + str(i+1), horizontalalignment='center')
    plt.ylabel('Jets', horizontalalignment='center')
    #plt.set_ylabel('Jets', bbox=box)
    #plt.title('MSE')
    plt.legend(loc='upper right')
    #plt.yscale('log')
    plt.yscale('linear')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    # plt.semilogx()
    plt.savefig('in_vs_out_efp_'+ str(i+1) +str(model_name)+'.png', dpi=250, bbox_inches='tight')
    plt.clf()
'''
###########################################################################################################
sum = 0
end_time = time.time()
print("The total time is ",((end_time-start_time)/60.0)," minutes.")
