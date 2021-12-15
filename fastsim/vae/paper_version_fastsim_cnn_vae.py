"""
============================================================
VAE for fast simulation of jets 
@authors: Breno Orzari, Maurizio Pierini, Maria Touranakou
@CERN, 2021 
============================================================
"""
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import time
import numpy as np 
import mplhep as mhep
plt.style.use(mhep.style.CMS)
#from sklearn.preprocessing import MinMaxScaler
from pickle import load
import copy
import awkward as ak
from coffea.nanoevents.methods import vector
ak.behavior.update(vector.behavior)
from scipy.stats import wasserstein_distance

####################################### SETTING PARAMS #######################################
#gpu_ids = [0,2] # or run with CUDA_VISIBLE_DEVICES=0,2 python3 train.py

# Hyperparameters
# Input data specific params
num_particles = 50

# Training params
N_epochs = 500
batch_size = 100
learning_rate = 0.0001

# Model params
model_name = '_fastsim_bvae_paperVersion_dropout03_conv_'+ str(num_particles) + 'p_'
#num_classes = 1
latent_dim = 20
beta_KL = 0.3
# Jet features loss weighting
beta_pT = 0.1
beta_mass = 1.
beta_p = 0.015
#beta_loss = 1/1000.
beta_loss = 1.

step_epoch = 2
step = 2 # over how many epochs to save/calculate EMD data
emd_zscore = []
emd =[]
epochs = []

emd_sum_all = []
emd_zscore_sum_all = []

EMD_mass = []
EMD_pt = []

# Starting time
start_time = time.time()

# Plots' colors based on colorbrewer schemes
cb_red = (228/255, 26/255, 28/255)
cb_blue = (55/255, 126/255, 184/255)
cb_green = (77/255, 175/255, 74/255) 
cb_purple = (201/255, 148/255, 199/255)
cb_must = (254/255, 178/255, 76/255)

# Probability to keep a node in the dropout layer
drop_prob_fc = 0.3
drop_prob_conv = 0.05

# Set patience for Early Stopping
patience = 80000
n_jets = 348900

####################################### DATA LOADING #######################################
print('Loading data')
# Load already scaled data.
train_dataset_gen = torch.load('train_data_gen_normed_pxpypz_w_'+ str(num_particles) +'pCut.pt').cuda()
valid_dataset_gen = torch.load('valid_data_gen_normed_pxpypz_w_'+ str(num_particles) +'pCut.pt').cuda()
test_dataset_gen = torch.load('test_data_gen_normed_pxpypz_w_'+ str(num_particles) +'pCut.pt').cuda()

train_dataset_reco = torch.load('train_data_reco_normed_pxpypz_w_'+ str(num_particles) +'pCut.pt').cuda()
valid_dataset_reco = torch.load('valid_data_reco_normed_pxpypz_w_'+ str(num_particles) +'pCut.pt').cuda()
test_dataset_reco = torch.load('test_data_reco_normed_pxpypz_w_'+ str(num_particles) +'pCut.pt').cuda()

################################################# Data for EMD #################################################
# Model gen input
print('Loading gen input')
test_data_gen_input = torch.load('test_data_gen_pxpypz_wDijets_50pCut.pt').cuda()
test_data_gen_input = test_data_gen_input[:n_jets,:,:,:].detach().cpu()
print('Done')

# Model reco input
print('Loading reco input')
test_data_reco = torch.load('test_data_reco_pxpypz_wDijets_50pCut.pt').cuda()
test_data_reco = test_data_reco[:n_jets,:,:,:].detach().cpu()
print('Done')

print(test_data_gen_input.shape)
print(test_data_reco.shape)
# Tensors to numpy arrays
test_data_gen_input = np.array(test_data_gen_input)
print(test_data_gen_input.shape)
test_data_gen_input = np.reshape(test_data_gen_input,(test_data_gen_input.shape[0], test_data_gen_input.shape[2],test_data_gen_input.shape[3]))

test_data_reco = np.array(test_data_reco)
print(test_data_reco.shape)
test_data_reco = np.reshape(test_data_reco,(test_data_reco.shape[0], test_data_reco.shape[2],test_data_reco.shape[3]))

################################################################################################
# Load the mean and std per feature
mean_v = load(open('mean_pxpypz_wDijets_50pCut.pkl', 'rb'))
sigma_v = load(open('sigma_pxpypz_wDijets_50pCut.pkl', 'rb'))
print(mean_v)
print(sigma_v)
print("DONE")

####################################### DATA LOADERS #######################################
train_dataset_gen_reco = TensorDataset(train_dataset_gen, train_dataset_reco)
train_loader = DataLoader(train_dataset_gen_reco, batch_size=batch_size, shuffle=True)
valid_dataset_gen_reco = TensorDataset(valid_dataset_gen, valid_dataset_reco)
valid_loader = DataLoader(valid_dataset_gen_reco, batch_size=batch_size, shuffle=False)
test_dataset_gen_reco = TensorDataset(test_dataset_gen, test_dataset_reco)
test_loader = DataLoader(test_dataset_gen_reco, batch_size=batch_size, shuffle=False)

mean_array = torch.cat((mean_v[0]*torch.ones(batch_size,1,num_particles).cuda(),
                        mean_v[1]*torch.ones(batch_size,1,num_particles).cuda(),
                        mean_v[2]*torch.ones(batch_size,1,num_particles).cuda()),1).cuda()
sigma_array = torch.cat((sigma_v[0]*torch.ones(batch_size,1,num_particles).cuda(),
                         sigma_v[1]*torch.ones(batch_size,1,num_particles).cuda(),
                         sigma_v[2]*torch.ones(batch_size,1,num_particles).cuda()),1).cuda()
print('mean array:', mean_array.shape)
print(sigma_array.shape)

mean_array_emd = torch.cat((mean_v[0]*torch.ones(n_jets,1,num_particles),
                        mean_v[1]*torch.ones(n_jets,1,num_particles),
                        mean_v[2]*torch.ones(n_jets,1,num_particles)),1)
sigma_array_emd = torch.cat((sigma_v[0]*torch.ones(n_jets,1,num_particles),
                         sigma_v[1]*torch.ones(n_jets,1,num_particles),
                         sigma_v[2]*torch.ones(n_jets,1,num_particles)),1)

####################################### DEFINE MODEL ####################################

# # Define models' architecture & helper functions
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,5), stride=(1), padding=(0))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(1,5), stride=(1), padding=(0))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(1,5), stride=(1), padding=(0))
        #
        self.fc1 = nn.Linear(1 * int(num_particles - 12) * 128, 1500)
        self.fc2 = nn.Linear(1500, 2 * latent_dim)
        #
        self.fc3 = nn.Linear(latent_dim, 1500)
        self.fc4 = nn.Linear(1500, 1 * int(num_particles - 12) * 128)
        self.conv4 = nn.ConvTranspose2d(128, 64, kernel_size=(1,5), stride=(1), padding=(0))
        self.conv5 = nn.ConvTranspose2d(64, 32, kernel_size=(1,5), stride=(1), padding=(0))
        self.conv6 = nn.ConvTranspose2d(32, 1, kernel_size=(3,5), stride=(1), padding=(0))
        # Define proportion or neurons to dropout
        self.dropout_fc = nn.Dropout(drop_prob_fc)
        self.dropout_conv = nn.Dropout(drop_prob_conv)

    def encode(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        # Apply dropout
        out = self.dropout_conv(out)
        out = self.conv2(out)
        out = F.relu(out)
        # Apply dropout
        out = self.dropout_conv(out)
        out = self.conv3(out)
        out = F.relu(out)
        # Apply dropout
        out = self.dropout_conv(out)
        out = out.view(out.size(0), -1) # flattening
        out = self.fc1(out)
        out = F.relu(out)
        # Apply dropout
        out = self.dropout_fc(out)
        out = self.fc2(out)
        mean = out[:,:latent_dim]
        logvar = 1e-6 + (out[:,latent_dim:])
        return mean, logvar

    def decode(self, z):
        out = self.fc3(z)
        out = F.relu(out)
        # Apply dropout
        out = self.dropout_fc(out)
        out = self.fc4(out)
        out = F.relu(out)
        # Apply dropout
        out = self.dropout_fc(out)
        out = out.view(batch_size, 128, 1, int(num_particles - 12)) # reshaping
        out = self.conv4(out)
        out = F.relu(out)
        # Apply dropout
        out = self.dropout_conv(out)
        out = self.conv5(out)
        out = F.relu(out)
        # Apply dropout
        out = self.dropout_conv(out)
        out = self.conv6(out)
        return out

    #def reparameterize(self, mean, logvar):
        #z = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
        #return z

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mean + eps*std
        return z

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        out = self.decode(z)
        return out

# Jet observables manual calculation
def jet_p(p_part): # input should be of shape[batch_size, features, n_particles]
    pjet = torch.sum(p_part, dim=2).cuda() # [batch, features (px, py, pz)]
    return pjet

def jet_Energy (p_part): # input should be of shape [batch_size, features, n_particles]
    E_particles = torch.sqrt(torch.sum(p_part*p_part, dim=1)) # E per particle shape: [100, 30]
    E_jet = torch.sum(E_particles, dim=1).cuda() # Energy per jet [100]
    return E_jet

def jet_mass (p_part):
    jet_e = jet_Energy(p_part)
    P_jet = jet_p(p_part)
    m_jet = torch.sqrt(jet_e*jet_e - (P_jet[:,0]*P_jet[:,0]) - (P_jet[:,1]*P_jet[:,1]) - (P_jet[:,2]*P_jet[:,2])).cuda()
    return m_jet # mass per jet [100]

def jet_pT(p_part):# input should be of shape[batch_size, features, n_particles]
    p_jet = jet_p(p_part) # [100, 3]
    jet_px = p_jet[:, 0]  # [100]
    jet_py = p_jet[:, 1]  # [100]
    jet_pt = torch.sqrt(jet_px*jet_px + jet_py*jet_py)
    return jet_pt

def compute_loss(model, x_gen, x_reco, mean_array, sigma_array):
    # VAE
    mean, logvar = model.encode(x_gen)
    z = model.reparameterize(mean, logvar)
    x_gen_decoded = model.decode(z)
    
    pdist = nn.PairwiseDistance(p=2) # Euclidean distance

    # input and output
    x_target = torch.zeros(batch_size,3,num_particles).cuda()
    x_target = x_reco[:,0,:,:] # [100, 3, 100]
    x_gen_output = torch.zeros(batch_size,3,num_particles).cuda()
    x_gen_output = x_gen_decoded[:,0,:,:]
    
    # compute input and output in GeV
    # particle features in GeV (input and output)
    x_target = (x_target*sigma_array + mean_array).cuda()
    x_gen_output = (x_gen_output*sigma_array + mean_array).cuda()

    # jet features in GeV (input and output)
    jets_mass_target = jet_mass(x_target).unsqueeze(1).cuda()
    jets_mass_output = jet_mass(x_gen_output).unsqueeze(1).cuda()

    # jet features in GeV (input and output)
    jets_pT_target = jet_pT(x_target).unsqueeze(1).cuda()
    jets_pT_output = jet_pT(x_gen_output).unsqueeze(1).cuda()

    # Reshape for loss computation
    x_target = x_target.view(batch_size, 3, 1, num_particles)
    x_gen_output = x_gen_output.view(batch_size, 3, num_particles, 1)
    x_gen_output = torch.repeat_interleave(x_gen_output, num_particles, -1)

    # Permutation-invariant Chamfer Loss / NND / 3D Sparse Loss
    dist = torch.pow(pdist(x_target, x_gen_output),2)
    #jet_mass_dist = torch.pow(pdist(jets_mass_target, jets_mass_output),2) 
    #jet_pT_dist = torch.pow(pdist(jets_pT_target, jets_pT_output),2) 

    mse_jet_mass = nn.functional.mse_loss(jets_mass_target, jets_mass_output, reduction='mean')
    mse_jet_pt = nn.functional.mse_loss(jets_pT_target, jets_pT_output, reduction='mean')

    # For every output value, find its closest input value; for every input value, find its closest output value.
    min_dist_xy = torch.min(dist, dim = 1)  # Get min distance per row - Find the closest input to the output
    min_dist_yx = torch.min(dist, dim = 2)  # Get min distance per column - Find the closest output to the input

    # Symmetrical euclidean distances
    eucl = min_dist_xy.values + min_dist_yx.values # [100, n_particles]

    # Loss components. Average losses (mean)
    loss_rec_p = beta_p * (torch.sum(eucl)/batch_size)
    #nnd_jet_mass = (torch.sum(jet_mass_dist) / batch_size)
    #nnd_jet_pT = (torch.sum(jet_pT_dist) / batch_size)

    loss_rec = (1 - beta_KL)*((loss_rec_p) + (beta_mass * mse_jet_mass) + (beta_pT * mse_jet_pt))

    # KLD
    KL_divergence = beta_KL * (0.5 * torch.sum(torch.pow(mean, 2) + torch.exp(logvar) - logvar - 1)/batch_size) 
    #KL_divergence = beta_KL * (0.5 * torch.mean(torch.pow(mean, 2) + torch.exp(logvar) - logvar - 1)) 
    #KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    #total_loss = (loss_rec_p + beta_KL*KL_divergence)/(1 + beta_KL)

    reconstruction_loss = - loss_rec
    ELBO = reconstruction_loss - (KL_divergence)
    total_loss = - ELBO

    total_loss = beta_loss * total_loss
    
    return total_loss, KL_divergence, loss_rec_p, (beta_mass * mse_jet_mass), (beta_pT * mse_jet_pt)

##### Training function per batch #####
def train(model, batch_data_train_gen, batch_data_train_reco, optimizer):

    input_train_gen = batch_data_train_gen.cuda()
    input_train_reco = batch_data_train_reco.cuda()
    output_train_gen = model(input_train_gen)
    # loss per batch
    train_loss, train_KLD_loss, train_reco_loss, train_jet_mass, train_jet_pT  = compute_loss(model, input_train_gen, input_train_reco, mean_array, sigma_array)
    # Backprop and perform Adam optimisation
    # Backpropagation
    optimizer.zero_grad()
    train_loss.backward()
    # Adam optimization using the gradients from backprop
    optimizer.step()

    return input_train_reco, output_train_gen, train_loss, train_KLD_loss, train_reco_loss, train_jet_mass, train_jet_pT

##### Validation function per batch #####
def validate(model, batch_data_valid_gen, batch_data_valid_reco):
    model.eval()
    with torch.no_grad():
        input_valid_gen = batch_data_valid_gen.cuda()
        input_valid_reco = batch_data_valid_reco.cuda()

        valid_loss, valid_KLD_loss, valid_reco_loss, valid_jet_mass, valid_jet_pT = compute_loss(model, input_valid_gen, input_valid_reco, mean_array, sigma_array)
        return valid_loss, valid_KLD_loss, valid_reco_loss, valid_jet_mass, valid_jet_pT
       
##### Test function #####
def test_unseed_data(model, batch_data_test_gen, batch_data_test_reco):
    model.eval()
    with torch.no_grad():
        input_test_gen = batch_data_test_gen.cuda() 
        input_test_reco = batch_data_test_reco.cuda()

        output_test_gen = model(input_test_gen)
        test_loss, test_KLD_loss, test_reco_loss, test_jet_mass, test_jet_pT = compute_loss(model, input_test_gen, input_test_reco, mean_array, sigma_array)

    return input_test_reco, output_test_gen, test_loss, test_KLD_loss, test_reco_loss

####################################### TRAINING #######################################
# Initialize model and load it on GPU
model = ConvNet()
model = model.cuda()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Gen level data
all_output_gen = np.empty(shape=(0, 1, 3, num_particles))
# Reco level data
all_input_reco = np.empty(shape=(0, 1, 3, num_particles))

x_graph = []
y_graph = []

tr_y_rec = []
tr_y_kl = []
tr_y_loss = []

# Individual loss components
tr_y_loss_pt = []
tr_y_loss_mass = []

val_y_rec = []
val_y_kl = []
val_y_loss = []
val_y_loss_pt = []
val_y_loss_mass = []

min_loss, stale_epochs = 999999.0, 0
min_emd_mass_loss = 999999.0
min_emd_pt_loss = 999999.0

emd_sum_per_epoch = []

print('Training started...')
for epoch in range(N_epochs):

    x_graph.append(epoch)
    y_graph.append(epoch)

    tr_loss_aux = 0.0
    tr_kl_aux = 0.0
    tr_rec_aux = 0.0

    val_loss_aux = 0.0
    val_kl_aux = 0.0
    val_rec_aux = 0.0

    tr_rec_mass_aux = 0.0
    val_rec_mass_aux = 0.0

    tr_rec_pT_aux = 0.0
    val_rec_pT_aux = 0.0

    # for index, (xb1, xb2) in enumerate(dataloader):
    for y, (jets_train_gen, jets_train_reco) in enumerate(train_loader):
        #print(jets_train_gen)
        #print(jets_train_reco)

        if y == (len(train_loader) - 1): # guarantees batches of fixed/equal size 
            break

        # Run train function on batch data
        tr_inputs_reco, tr_outputs_gen, tr_loss, tr_kl, tr_eucl, tr_mass, tr_pT  = train(model, jets_train_gen, jets_train_reco, optimizer)

        tr_loss_aux += tr_loss
        tr_kl_aux += tr_kl
        tr_rec_aux += tr_eucl
        tr_rec_mass_aux += tr_mass
        tr_rec_pT_aux += tr_pT

        #if epoch == (N_epochs-1):
        #if stale_epochs > patience:
            # Concat input and output per batch
            #batch_input_reco = tr_inputs_reco.cpu().detach().numpy()
            #batch_output_gen = tr_outputs_gen.cpu().detach().numpy()
            #all_input_reco = np.concatenate((all_input_reco, batch_input_reco), axis=0)
            #all_output_gen = np.concatenate((all_output_gen, batch_output_gen), axis=0)
            #return input_train_reco, output_train_gen, train_loss, train_KLD_loss, train_reco_loss
    
    # Training losses        
    tr_y_loss.append(tr_loss_aux.cpu().detach().item()/(len(train_loader) - 1))
    tr_y_kl.append(tr_kl_aux.cpu().detach().item()/(len(train_loader) - 1))
    tr_y_rec.append(tr_rec_aux.cpu().detach().item()/(len(train_loader) - 1))

    tr_y_loss_mass.append(tr_rec_mass_aux.cpu().detach().item()/(len(train_loader) - 1))
    tr_y_loss_pt.append(tr_rec_pT_aux.cpu().detach().item()/(len(train_loader) - 1))


    for w, (jets_valid_gen, jets_valid_reco) in enumerate(valid_loader):
        if w == (len(valid_loader) - 1):
            break

        # Run validate function on batch data
        val_loss, val_kl, val_eucl, val_mass, val_pT = validate(model, jets_valid_gen, jets_valid_reco)

        val_loss_aux += val_loss
        val_kl_aux += val_kl
        val_rec_aux += val_eucl
        val_rec_mass_aux += val_mass
        val_rec_pT_aux += val_pT


    # Validation losses        
    val_y_loss.append(val_loss_aux.cpu().detach().item()/(len(valid_loader) - 1))
    val_y_kl.append(val_kl_aux.cpu().detach().item()/(len(valid_loader) - 1))
    val_y_rec.append(val_rec_aux.cpu().detach().item()/(len(valid_loader) - 1))
    # Loss jet features components
    val_y_loss_mass.append(val_rec_mass_aux.cpu().detach().item()/(len(valid_loader) - 1))
    val_y_loss_pt.append(val_rec_pT_aux.cpu().detach().item()/(len(valid_loader) - 1))


    if epoch == (N_epochs-1):
    #if stale_epochs > patience:
        print("Training ended.")
        break

    if stale_epochs > patience:
        print("Early stopped")
        break

    '''    
    if val_loss_aux.cpu().detach().item()/(len(valid_loader) - 1) < min_loss:
        min_loss = val_loss_aux.cpu().detach().item()/(len(valid_loader) - 1)
        # Keep as best model that with the min validation loss
        best_model = copy.deepcopy(model) # apply a deep copy on the parameters
        stale_epochs = 0
    else:
        stale_epochs += 1
        print('stale_epochs:', stale_epochs)
    '''

    print('Epoch: {} -- Train loss: {} -- Validation loss: {}'.format(epoch, tr_loss_aux.cpu().detach().item()/(len(train_loader)-1), val_loss_aux.cpu().detach().item()/(len(valid_loader)-1)))

    print("TRAINING:")
    print("Total: %f" %tr_y_loss[-1])
    print("Reco:  %f" %tr_y_rec[-1])
    print("KL:    %f" %tr_y_kl[-1])
    print("mass:  %f" %tr_y_loss_mass[-1])
    print("pT:    %f" %tr_y_loss_pt[-1])

    print("VALIDATION")
    print("Total: %f" %val_y_loss[-1])
    print("Reco:  %f" %val_y_rec[-1])
    print("KL:    %f" %val_y_kl[-1])
    print("mass:  %f" %val_y_loss_mass[-1])
    print("pT:    %f" %val_y_loss_pt[-1])


    if((epoch+1)%step==0): # THE FOLLOWING CALCULATIONS ARE BEING PERFORMED AT EVERY N EPOCHS ONLY
    #if((epoch+1)%saving_epoch==0 or stale_epochs>patience):

        # Keep as best model that with the min validation loss
        epoch_model = copy.deepcopy(model) # apply a deep copy on the parameters
        epoch_optimizer = copy.deepcopy(optimizer)

        #torch.save(epoch_model.state_dict(),  'model_epoch_'+ str(epoch+1) + str(model_name) + '.pt')
        #torch.save(epoch_optimizer.state_dict(),  'epoch_optimizer_'+ str(epoch+1) + str(model_name) + '.pt')

        print('Saved model.')
        ####################################### EVALUATION #######################################
        print('Evaluation Epoch '+ str(epoch+1))

        all_input_reco_test = np.empty(shape=(0, 1, 3, num_particles))
        all_output_gen_test = np.empty(shape=(0, 1, 3, num_particles))

        for i, (jets_gen, jets_reco) in enumerate(test_loader):
            if i == (len(test_loader)-1):
                break
            # run test function on batch data for testing
            test_inputs_reco, test_outputs_gen, ts_loss, ts_kl, ts_eucl = test_unseed_data(epoch_model, jets_gen, jets_reco)
            batch_input_reco_ts = test_inputs_reco.cpu().detach().numpy()
            batch_output_gen_ts = test_outputs_gen.cpu().detach().numpy()
            all_input_reco_test = np.concatenate((all_input_reco_test, batch_input_reco_ts), axis=0)
            all_output_gen_test = np.concatenate((all_output_gen_test, batch_output_gen_ts), axis=0)

        ######################## TEST DATA ########################
        print('input all_input_reco_test shape: ', all_input_reco_test.shape)
        print('output all_output_gen_test shape:', all_output_gen_test.shape)

        #################### EVERY N EPOCHS DATA ####################
        px_gen_output = all_output_gen_test[:,0,0,:]
        py_gen_output = all_output_gen_test[:,0,1,:]
        pz_gen_output = all_output_gen_test[:,0,2,:]

        print(px_gen_output.shape)
        print(py_gen_output.shape)
        print(pz_gen_output.shape)

        select_portion_of_unseen_data = px_gen_output.shape[0]
        print(select_portion_of_unseen_data)

        ################### HELPER DATA ###################
        gen_data_normed = np.stack((px_gen_output, py_gen_output, pz_gen_output), axis=1)
        print(gen_data_normed.shape)
        gen_data_normed = torch.from_numpy(gen_data_normed)

        print('############# INVERSE-STANDARDIZATION #############') 
        gen_data_out_r = (gen_data_normed*sigma_array_emd + mean_array_emd)
        print(gen_data_out_r.shape) # (372400, 3, 100)
        print(gen_data_out_r.min(), gen_data_out_r.max())

        test_data_gen_output = np.array(gen_data_out_r)
        print(test_data_gen_input.shape)
        print(test_data_reco.shape)
        print(test_data_gen_output.shape)

        ############################################### INPUT DATA ZEROS ###############################################
        # All 3 features have zeros at the same positions (same particles do not exist). So, I just need to find the non_zeros in the dim [jets, particles].
        # Zeros at reco data
        test_data_reco2d = np.sum(test_data_reco, axis=1)
        print(test_data_reco2d.shape)
        non_zeros_reco = (test_data_reco2d != 0)
        zeros_reco = (test_data_reco2d == 0)
        print(non_zeros_reco.shape)
        print(zeros_reco.shape)
        #print(non_zeros_reco[0,20:])

        # All 3 features have zeros at the same positions (same particles do not exist). So, I just need to find the non_zeros in the dim [jets, particles].
        # Zeros at gen data
        test_data_gen_2d = np.sum(test_data_gen_input, axis=1)
        print(test_data_gen_2d.shape)
        non_zeros_gen = (test_data_gen_2d != 0)
        zeros_gen = (test_data_gen_2d == 0)
        print(non_zeros_gen.shape)
        print(zeros_gen.shape)

        ############################################### MASK OUTPUT DATA ###############################################
        # Perform cut on min-pT on DL output
        # Mask output for min-pt
        min_pt_cut = 0.25
        non_zeros_dl =  gen_data_out_r[:,0,:] * gen_data_out_r[:,0,:] + gen_data_out_r[:,1,:] * gen_data_out_r[:,1,:] > min_pt_cut**2
        print(non_zeros_dl.shape)
        # Expand over the features dimension mask
        non_zeros_dl = non_zeros_dl.unsqueeze(1)
        print('min-pT mask', non_zeros_dl.shape)
        # Then you can apply the mask
        gen_data_out_masked = (non_zeros_dl * gen_data_out_r).numpy()
        print('DL masked/zeroed.', gen_data_out_masked.shape) # Now, values that correspond to the min-pt should be zeroed. Check zeros again.
        #print(gen_data_out_masked)

        test_data_DL_2d = np.sum(gen_data_out_masked, axis=1)
        non_zeros_dl = (test_data_DL_2d != 0)
        print(non_zeros_dl.shape)

        ############################################### ZEROS ###############################################
        non_zeros_reco = np.reshape(non_zeros_reco, (non_zeros_reco.shape[0]*non_zeros_reco.shape[1]))
        print(non_zeros_reco.shape)
        #print(non_zeros_reco)
        #print(np.count_nonzero(non_zeros_reco))

        non_zeros_gen = np.reshape(non_zeros_gen, (non_zeros_gen.shape[0]*non_zeros_gen.shape[1]))
        print(non_zeros_gen.shape)
        #print(non_zeros_gen)
        #print(np.count_nonzero(non_zeros_gen))

        non_zeros_dl = np.reshape(non_zeros_dl, (non_zeros_dl.shape[0]*non_zeros_dl.shape[1]))
        print(non_zeros_dl.shape)
        #print(non_zeros_dl)
        #print(np.count_nonzero(non_zeros_dl))

        ############################### JET FEATURES COMPUTATION ###############################
        ############################### ETA & PHI COMPUTATION ###############################
        # Compute eta & phi.
        def compute_eta(pz, pt):
            eta = np.nan_to_num(np.arcsinh(pz/pt))
            return eta

        def compute_phi(px, py):
            phi = np.arctan2(py, px)
            return phi

        def particle_pT(p_part): # input of shape [n_jets, 3_features, n_particles]
            p_px = p_part[:, 0, :]  
            p_py = p_part[:, 1, :]  
            p_pt = np.sqrt(p_px*p_px + p_py*p_py)
            return p_pt

        def ptetaphim_particles(data):
            part_pt = particle_pT(data)
            part_eta = compute_eta(data[:,2,:], part_pt)
            part_phi = compute_phi(data[:,0,:], data[:,1,:])
            #part_mass = data[:,:,3]
            part_mass = np.zeros((part_pt.shape[0], num_particles))
            print(part_pt.shape)
            print(part_eta.shape)
            print(part_phi.shape)
            print(part_mass.shape)
            return np.stack((part_pt, part_eta, part_phi, part_mass), axis=2)

        # Compute hadronic coords in order to compute the jet features awkward arrays
        hadr_gen_data = ptetaphim_particles(test_data_gen_input)
        hadr_reco_data = ptetaphim_particles(test_data_reco)
        hadr_dl_data = ptetaphim_particles(gen_data_out_masked)

        print(hadr_gen_data.shape)
        print(hadr_reco_data.shape)
        print(hadr_dl_data.shape)

        def jet_features(jets, mask_bool=False, mask=None):
            vecs = ak.zip({
                    "pt": jets[:, :, 0],
                    "eta": jets[:, :, 1],
                    "phi": jets[:, :, 2],
                    "mass": jets[:, :, 3],
                    }, with_name="PtEtaPhiMLorentzVector")

            sum_vecs = vecs.sum(axis=1)
            jf = np.stack((ak.to_numpy(sum_vecs.mass), ak.to_numpy(sum_vecs.pt), ak.to_numpy(sum_vecs.energy), ak.to_numpy(sum_vecs.eta), ak.to_numpy(sum_vecs.phi)), axis=1)

            return ak.to_numpy(jf)

        ############ JET FEATURES COMPUTED WITH AWKWARD ARRAYS & COFFEA
        jets_gen_data = jet_features(hadr_gen_data)
        jets_reco_data = jet_features(hadr_reco_data)
        jets_dl_data = jet_features(hadr_dl_data)

        print(jets_gen_data.shape, 'jet features: mass, pt, energy, eta, phi.') # [n_jets, 5_features]
        # Get a slice to access each jet feature, e.g. jets_gen_data[:, i]
        print(jets_gen_data[:,0].shape)
        print(jets_gen_data[:,1].shape)
        print(jets_gen_data[:,2].shape)
        print(jets_gen_data[:,3].shape)
        print(jets_gen_data[:,4].shape)

        ################################## JET FEATURES ##################################
        labels = ["Jet Mass z-score", "Jet $p_{T}$ z-score", "Jet Energy z-score", "jet $\eta$ z-score", "jet $\phi$ z-score"]
        names = ["jet_mass_zscore", "jet_pt_zscore", "jet_energy_zscore", "jet_eta_zscore", "jet_phi_zscore"]
        #x_min = [0., 0., 0., -3.0, -3.0]
        #x_max = [300., 750., 3000., +3.0, +3.0]

        #x_min = [0., 150., 100., -2.5, -3.0]
        #x_max = [120., 500., 1800., +2.5, +3.0]

        x_min = [0., 150., 100., -2.5, -3.0]
        x_max = [140., 500., 1600., +2.5, +3.0]

        emd_array_zscore = []

        for i in range(0,5):
            plt.figure(figsize=(8, 6))
            # Compute the histograms
            zscore1 = ((jets_reco_data[:, i]-jets_gen_data[:, i])/jets_gen_data[:, i])
            n_x1, bin_edges_x1, patches_x1 = plt.hist(zscore1, range = (-1, 1), bins=100, density=True, label="z-score Reco-Gen", color = cb_blue, histtype='step', fill=False, linewidth=1.5)
            zscore = ((jets_dl_data[:, i] - jets_reco_data[:, i])/jets_reco_data[:, i])
            n_x2, bin_edges_x2, patches_x2 = plt.hist(zscore, range = (-1, 1), bins=100, density=True, label="z-score DL-Reco", color = cb_red, histtype='step', fill=False, linewidth=1.5)
            zscore2 = ((jets_dl_data[:, i]-jets_gen_data[:, i])/jets_gen_data[:, i])
            n_x3, bin_edges_x3, patches_x3 = plt.hist(zscore2, range = (-1, 1), bins=100, density=True, label="z-score DL-Gen", color = cb_green, histtype='step', fill=False, linewidth=1.5)
            plt.xlabel(labels[i], fontsize=20)
            plt.ylabel("Entries", fontsize=20)
            axes = plt.gca()
            y_min, y_max = axes.get_ylim()
            # Get the normalized count per bin & add epsilon to sum to 1 (numerical trick)
            x1 = (n_x1/n_x1.sum()) + 0.000000000001
            x3 = (n_x3/n_x3.sum()) + 0.000000000001
            # Compute their emd_zscore distance for Reco-Gen & DL-Gen
            blue_green_hist_dist = wasserstein_distance(x1, x3)
            plt.text(-1, 0.9*(y_max-y_min),"emd_zscore Reco-DL = %.6f" %(blue_green_hist_dist), fontsize=12)
            plt.text(-1, 0.85*(y_max-y_min),"Epoch: %s" %(epoch+1), fontsize=12)
            #plt.text(-7, y_min+0.5*(y_max-y_min),"mean Reco-Gen = %f" %np.nanmean(zscore1), fontsize=12)
            #plt.text(-7, y_min+0.3*(y_max-y_min),"std Reco-Gen = %f" %np.nanstd(zscore1), fontsize=12)
            #plt.text(-7, y_min+0.2*(y_max-y_min),"mean Reco-DL = %f" %np.nanmean(zscore), fontsize=12)
            #plt.text(-7, y_min+0.12*(y_max-y_min),"std Reco-DL = %f" %np.nanstd(zscore), fontsize=12)
            plt.legend(loc='upper right', prop={'size': 16})
            plt.draw()
            # Compute their emd_zscore distance
            emd_array_zscore.append(blue_green_hist_dist)
            #plt.savefig( '%s' %names[i] + '_epoch_'+str(step_epoch) +'.png', dpi=250, bbox_inches='tight')
            plt.close()


        labels = ["Jet Mass (GeV)", "Jet $p_{T}$ (GeV)", "Jet Energy (GeV)", "jet $\eta$", "jet $\phi$"]
        names = ["jet_mass_linear", "jet_pt_linear", "jet_energy_linear", "jet_eta_linear", "jet_phi_linear"]

        emd_array = []

        for i in range(0,5):
            plt.figure(figsize=(8, 6))
            n_x1, bin_edges_x1, patches_x1 = plt.hist(jets_reco_data[:, i], range = [x_min[i], x_max[i]], bins=100, label="Reco", density=True, color = cb_red, histtype='step', fill=False, linewidth=1.5)
            n_x2, bin_edges_x2, patches_x2 = plt.hist(jets_dl_data[:, i], range = [x_min[i], x_max[i]], bins=100, label="DL", density=True, color = cb_green, histtype='step', fill=False, linewidth=1.5)
            plt.xlabel(labels[i], fontsize=20)
            plt.ylabel("Entries", fontsize=20)
            plt.yscale('linear')
            axes = plt.gca()
            y_min, y_max = axes.get_ylim()
            # Get the normalized count per bin & add epsilon to sum to 1 (numerical trick)
            x = (n_x1/n_x1.sum()) + 0.000000000001
            x2 = (n_x2/n_x2.sum()) + 0.000000000001
            # Compute their emd_zscore distance
            reco_dl_hist_dist = wasserstein_distance(x, x2)
            plt.text(0.5*x_max[i], 0.9*(y_max-y_min),"EMD Reco-DL = %.6f" %(reco_dl_hist_dist), fontsize=12)
            plt.text(0.5*x_max[i], 0.85*(y_max-y_min),"Epoch: %s" %(epoch+1), fontsize=12)
            plt.draw()
            plt.legend(loc="upper right", prop={'size': 16})
            # Compute their emd_zscore distance
            emd_array.append(reco_dl_hist_dist)
            #plt.savefig( '%s' %names[i] + '_epoch_'+str(step_epoch) +'.png', dpi=250, bbox_inches='tight')
            plt.close()

        # transformed to a numpy array
        emd_array_zscore = np.array(emd_array_zscore) 
        print('model ' + str(step_epoch) + ' emd_zscore of 5 jet features (mass, pt, energy, eta, phi): ', emd_array_zscore)
        #torch.save(emd_array_zscore, 'emd_zscore_jetFeatures_epoch_' + str(step_epoch) +'.pt')
        emd_array_zscore = emd_array_zscore[None, :]

        emd_array = np.array(emd_array) 
        print('model ' + str(step_epoch) + ' EMD of 5 jet features (mass, pt, energy, eta, phi): ', emd_array)
        #torch.save(emd_array, 'emd_jetFeatures_epoch_' + str(step_epoch) +'.pt')
        emd_array = emd_array[None, :]

        # keep EMD metrics per jet feature for each checkpoint
        emd_mass = np.round((emd_array[0, 0]), 7)
        emd_pt = np.round((emd_array[0, 1]), 7)
        emd_energy = np.round((emd_array[0, 2]), 7)
        emd_eta = np.round((emd_array[0, 3]), 7)
        emd_phi = np.round((emd_array[0, 4]), 7)

        # EMD list for mass and pt 
        EMD_mass.append(emd_mass)
        EMD_pt.append(emd_pt)

        ###################################
        if step_epoch==2:
            emd_zscore = emd_array_zscore
            print(emd_zscore.shape) 
            emd = emd_array
            print(emd.shape)        
        else:
            print(emd_array.shape)        
            print(emd_array_zscore.shape) 
            emd_zscore = np.concatenate([emd_zscore, emd_array_zscore], axis=0)
            print(emd_zscore.shape) 
            emd = np.concatenate([emd, emd_array], axis=0)
            print(emd.shape)


# Print and save EMD metrics
#print(emd_zscore.shape)
#torch.save(emd_zscore, 'emd_zscore_jetFeatures_all.pt')
#print(emd.shape)
#torch.save(emd, 'emd_jetFeatures_all.pt') # [n_epochs, n_features]

        epochs.append(epoch+1)

        # In each checkpoint, calculate total EMD and keep it to get to the min.
        emd_sum = np.round((np.sum(emd_array, axis=1)), 6)
        emd_zscore_sum = np.round((np.sum(emd_array_zscore, axis=1)), 6)

        emd_sum_all.append(emd_sum)
        emd_zscore_sum_all.append(emd_zscore_sum)


# EMD sums
#emd_sum_all = np.round((np.sum(emd, axis=1)), 6)
#emd_zscore_sum_all = np.round((np.sum(emd_zscore, axis=1)), 6)

        if ((emd_mass < min_emd_mass_loss) and (emd_pt < min_emd_pt_loss)):
            # Keep min EMD loss per feature
            min_emd_mass_loss = emd_mass
            min_emd_pt_loss = emd_pt

            # Keep as best model that with the min EMD sum loss
            best_pt_mass_model = copy.deepcopy(model) # apply a deep copy on the parameters
            stale_epochs = 0
            # Save checkpoints only for min EMD sums
            ###### SAVE THIS FOR BEST EPOCHS ONLY  - MIN EMD SUM 
            torch.save(px_gen_output, 'px_gen_output_test_min_EMD_mass_and_pt_std'+ str(epoch+1) + str(model_name) + '.pt')
            torch.save(py_gen_output, 'py_gen_output_test_min_EMD_mass_and_pt_std'+ str(epoch+1) + str(model_name) + '.pt')
            torch.save(pz_gen_output, 'pz_gen_output_test_min_EMD_mass_and_pt_std'+ str(epoch+1) + str(model_name) + '.pt')
            torch.save(model.state_dict(), 'best_model_min_EMD_mass_and_pt_loss_'+ str(epoch+1) + str(model_name) + '.pt')


        if (emd_mass < min_emd_mass_loss):
            # Keep min EMD loss per feature
            min_emd_mass_loss = emd_mass
            # Keep as best model that with the min EMD sum loss
            best_mass_model = copy.deepcopy(model) # apply a deep copy on the parameters
            stale_epochs = 0
            # Save checkpoints only for min EMD sums
            ###### SAVE THIS FOR BEST EPOCHS ONLY  - MIN EMD SUM 
            torch.save(px_gen_output, 'px_gen_output_test_min_emd_mass_std'+ str(epoch+1) + str(model_name) + '.pt')
            torch.save(py_gen_output, 'py_gen_output_test_min_emd_mass_std'+ str(epoch+1) + str(model_name) + '.pt')
            torch.save(pz_gen_output, 'pz_gen_output_test_min_emd_mass_std'+ str(epoch+1) + str(model_name) + '.pt')
            torch.save(model.state_dict(), 'best_model_min_emd_mass_loss_'+ str(epoch+1) + str(model_name) + '.pt')

        # MAYBE TAKE EMD AVERAGE
        if emd_sum < min_loss:
            # Keep min EMD sum loss
            min_loss = emd_sum
            # Keep as best model that with the min EMD sum loss
            best_model = copy.deepcopy(model) # apply a deep copy on the parameters
            stale_epochs = 0
            # Save checkpoints only for min EMD sums
            ###### SAVE THIS FOR BEST EPOCHS ONLY  - MIN EMD SUM 
            torch.save(px_gen_output, 'px_gen_output_test_emd_sum_std'+ str(epoch+1) + str(model_name) + '.pt')
            torch.save(py_gen_output,'py_gen_output_test_emd_sum_std'+ str(epoch+1) + str(model_name) + '.pt')
            torch.save(pz_gen_output, 'pz_gen_output_test_emd_sum_std'+ str(epoch+1) + str(model_name) + '.pt')
            # To resume training, you can restore your model and optimizer's state dict.
            torch.save(model.state_dict(), 'one_best_model_min_EMD_sum_loss_5jf_'+ str(epoch+1) + str(model_name) + '.pt')
            torch.save(optimizer.state_dict(),  'optimizer_best_model_min_EMD_sum_loss_5jf_'+ str(epoch+1) + str(model_name) + '.pt')

        else:
            stale_epochs += 1
            print('stale_epochs:', stale_epochs)

        #### to plot WITH EMD SUM PER EPOCH
        emd_sum_per_epoch.append(emd_sum)
            
        ##########
        # Plot each component of the loss function
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, emd_sum_per_epoch, label = "Total EMD", linewidth=1.5)
        plt.yscale('linear')
        plt.xlabel('Epoch', fontsize=20)
        plt.ylabel('EMD', fontsize=20)
        plt.ylim([0.0, 0.02])
        plt.legend(loc='upper left', prop={'size': 16})
        plt.savefig('emd_5jf_sum_per_checkpoint_'+ str(epoch+1) + str(model_name) + '.png', dpi=250, bbox_inches='tight')
        plt.close()

        # Plot each component of the loss function
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, EMD_mass, label = "EMD jet mass", linewidth=1.5, color = cb_green)
        plt.plot(epochs, EMD_pt, label = "EMD jet pt", linewidth=1.5, color = cb_blue)
        plt.yscale('linear')
        plt.xlabel('Epoch', fontsize=20)
        plt.ylabel('EMD', fontsize=20)
        #plt.ylim([0.0, 0.02])
        plt.legend(loc='upper left', prop={'size': 16})
        plt.savefig('emd_mass_vs_pt_per_checkpoint_'+ str(epoch+1) + str(model_name) + '.png', dpi=250, bbox_inches='tight')
        plt.close()


        # These will get overwritten eventually, but I keep them in case training dies.
        # Plot each component of the loss function
        plt.figure(figsize=(8, 6))
        plt.plot(x_graph, tr_y_kl,'--', label = "Train KL Divergence", linewidth=1.5, color = cb_purple)
        plt.plot(x_graph, tr_y_rec, '--', label = 'Train Reconstruction Loss', linewidth=1.5, color = cb_red)
        plt.plot(x_graph, tr_y_loss, '--', label = 'Train Total Loss', linewidth=1.5, color = cb_must)
        plt.plot(x_graph, val_y_kl, label = "Validation KL Divergence", linewidth=1.5, color = cb_purple)
        plt.plot(x_graph, val_y_rec, label = 'Validation Reconstruction Loss', linewidth=1.5, color = cb_red)
        plt.plot(x_graph, val_y_loss, label = 'Validation Total Loss', linewidth=1.5, color = cb_must)
        plt.yscale('log', nonpositive='clip')
        plt.xlabel('Epoch')
        plt.ylabel('A. U.')
        plt.title('Loss Function Components')
        plt.legend()
        plt.savefig('pxpypz_loss_comp_log_' + str(epoch+1) + str(model_name) + '.png', dpi=250, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.plot(x_graph, tr_y_kl, '--', label = "Train KL Divergence", linewidth=1.5, color = cb_purple)
        plt.plot(x_graph, tr_y_rec, '--', label = 'Train Reconstruction Loss', linewidth=1.5, color = cb_red)
        plt.plot(x_graph, tr_y_loss_pt, '--', label = 'Train Jet $p_T$ Loss', linewidth=1.5, color = cb_blue)
        plt.plot(x_graph, tr_y_loss_mass, '--', label = 'Train Jet Mass Loss', linewidth=1.5, color = cb_green)
        plt.plot(x_graph, val_y_kl, label = "Validation KL Divergence", linewidth=1.5, color = cb_purple)
        plt.plot(x_graph, val_y_rec, label = 'Validation Reconstruction Loss', linewidth=1.5, color = cb_red)
        plt.plot(x_graph, val_y_loss_pt, label = 'Validation Jet $p_T$ Loss', linewidth=1.5, color = cb_blue)
        plt.plot(x_graph, val_y_loss_mass, label = 'Validation Jet Mass Loss', linewidth=1.5, color = cb_green)
        plt.yscale('log', nonpositive='clip')
        plt.xlabel('Epoch')
        plt.ylabel('A. U.')
        plt.title('Loss Function Components')
        plt.legend()
        plt.savefig('paper_loss_comp_log_' + str(epoch+1) + str(model_name) + '.png', dpi=250, bbox_inches='tight')
        plt.close()


        plt.figure(figsize=(8, 6))
        plt.plot(x_graph, tr_y_kl, '--', label = "Train KL Divergence", linewidth=1.5, color = cb_purple)
        plt.plot(x_graph, tr_y_rec, '--', label = 'Train Reconstruction Loss', linewidth=1.5, color = cb_red)
        plt.plot(x_graph, tr_y_loss_pt, '--', label = 'Train Jet $p_T$ Loss', linewidth=1.5, color = cb_blue)
        plt.plot(x_graph, tr_y_loss_mass, '--', label = 'Train Jet Mass Loss', linewidth=1.5, color = cb_green)
        plt.plot(x_graph, val_y_kl, label = "Validation KL Divergence", linewidth=1.5, color = cb_purple)
        plt.plot(x_graph, val_y_rec, label = 'Validation Reconstruction Loss', linewidth=1.5, color = cb_red)
        plt.plot(x_graph, val_y_loss_pt, label = 'Validation Jet $p_T$ Loss', linewidth=1.5, color = cb_blue)
        plt.plot(x_graph, val_y_loss_mass, label = 'Validation Jet Mass Loss', linewidth=1.5, color = cb_green)
        plt.yscale('linear')
        plt.xlabel('Epoch')
        plt.ylabel('A. U.')
        plt.title('Loss Function Components')
        plt.legend()
        plt.savefig('paper_loss_comp_linear_' + str(epoch+1) + str(model_name) + '.png', dpi=250, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.plot(x_graph, tr_y_kl, '--', label = "Train KL Divergence", linewidth=1.5, color = cb_purple)
        plt.plot(x_graph, tr_y_rec, '--', label = 'Train Reconstruction Loss', linewidth=1.5, color = cb_red)
        plt.plot(x_graph, tr_y_loss, '--', label = 'Train Total Loss', linewidth=1.5, color = cb_must)
        plt.plot(x_graph, val_y_kl, label = "Validation KL Divergence", linewidth=1.5, color = cb_purple)
        plt.plot(x_graph, val_y_rec, label = 'Validation Reconstruction Loss', linewidth=1.5, color = cb_red)
        plt.plot(x_graph, val_y_loss, label = 'Validation Total Loss', linewidth=1.5, color = cb_must)
        plt.yscale('linear')
        plt.xlabel('Epoch')
        plt.ylabel('A. U.')
        plt.title('Loss Function Components')
        plt.legend()
        plt.savefig('pxpypz_loss_comp_linear_' + str(epoch+1) + str(model_name) + '.png', dpi=250, bbox_inches='tight')
        plt.close()

        # Plot each depedent component of the loss function 
        plt.figure(figsize=(8, 6))
        plt.plot(y_graph, tr_y_rec, '--', label = 'Train Reco - Particles Loss', color = cb_red)
        plt.plot(y_graph, tr_y_loss_pt, '--', label = 'Train Reco - Jets $p_T$', color = cb_blue)
        plt.plot(y_graph, tr_y_loss_mass, '--', label = 'Train Reco - Jets Mass', color = cb_green)
        plt.plot(y_graph, val_y_rec, label = 'Validation Reco - Particles Loss', color = cb_red)
        plt.plot(y_graph, val_y_loss_pt, label = 'Validation Reco - Jets $p_T$', color = cb_blue)
        plt.plot(y_graph, val_y_loss_mass, label = 'Validation Reco - Jets Mass', color = cb_green)
        plt.yscale('log', nonpositive='clip')
        plt.xlabel('Epoch')
        #plt.ylabel('A. U.')
        plt.title('Dependent Components - NND')
        plt.legend()
        plt.savefig('pxpypz_loss_individual_comp_log_train_vs_valid_' + str(epoch+1) + str(model_name) + '.png', dpi=250, bbox_inches='tight')
        plt.close()    

        #torch.save(tr_y_kl,  'train_KLD_loss' + str(epoch+1) + str(model_name) + '.pt')
        #torch.save(tr_y_rec,  'train_reco_loss' + str(epoch+1) + str(model_name) + '.pt')
        #torch.save(tr_y_loss,  'train_total_loss' + str(epoch+1) + str(model_name) + '.pt')

        #torch.save(val_y_kl,  'valid_KLD_loss' + str(epoch+1) + str(model_name) + '.pt')
        #torch.save(val_y_rec,  'valid_reco_loss' + str(epoch+1) + str(model_name) + '.pt')
        #torch.save(val_y_loss,  'valid_total_loss' + str(epoch+1) + str(model_name) + '.pt')

        #torch.save(tr_y_loss_pt,  'train_jetpT_loss' + str(epoch+1) + str(model_name) + '.pt')
        #torch.save(tr_y_loss_mass,  'train_jetmass_loss' + str(epoch+1) + str(model_name) + '.pt')

    if((epoch+1)%10==0): # THE FOLLOWING CALCULATIONS ARE BEING PERFORMED AT EVERY 10 EPOCHS ONLY
        # Loss ratio for debugging
        loss_reco_ratio = [x/y for x, y in zip(tr_y_rec, val_y_rec)]
        loss_kld_ratio = [x/y for x, y in zip(tr_y_kl, val_y_kl)]
        loss_pt_ratio = [x/y for x, y in zip(tr_y_loss_pt, val_y_loss_pt)]
        loss_mass_ratio = [x/y for x, y in zip(tr_y_loss_mass, val_y_loss_mass)]

        # Loss ratio for debugging
        plt.figure(figsize=(8, 6))
        plt.plot(x_graph, loss_reco_ratio, label = 'Reco Loss Ratio (Tr/Val)', linewidth=1.5)
        plt.plot(x_graph, loss_kld_ratio, label = "KLD Loss Ratio (Tr/Val)", linewidth=1.5)
        plt.plot(x_graph, loss_pt_ratio, label = "Jet Pt Loss Ratio (Tr/Val)", linewidth=1.5)
        plt.plot(x_graph, loss_mass_ratio, label = "Jet Mass Loss Ratio (Tr/Val)", linewidth=1.5)
        plt.yscale('log', nonpositive='clip')
        plt.xlabel('Epoch')
        plt.ylabel('A. U.')
        plt.title('Loss Function Components')
        plt.legend()
        plt.savefig('paper_loss_comp_log_ratio_' + str(epoch+1) + str(model_name) + '.png', dpi=250, bbox_inches='tight')
        plt.close()

        # Loss ratio for debugging
        plt.figure(figsize=(8, 6))
        plt.plot(x_graph, loss_reco_ratio, label = 'Reco Loss Ratio (Tr/Val)', linewidth=1.5)
        plt.plot(x_graph, loss_kld_ratio, label = "KLD Loss Ratio (Tr/Val)", linewidth=1.5)
        plt.plot(x_graph, loss_pt_ratio, label = "Jet Pt Loss Ratio (Tr/Val)", linewidth=1.5)
        plt.plot(x_graph, loss_mass_ratio, label = "Jet Mass Loss Ratio (Tr/Val)", linewidth=1.5)
        plt.yscale('linear')
        plt.xlabel('Epoch')
        plt.ylabel('A. U.')
        plt.title('Loss Function Components')
        plt.legend()
        plt.savefig('paper_loss_comp_linear_ratio_' + str(epoch+1) + str(model_name) + '.png', dpi=250, bbox_inches='tight')
        plt.close()

        # Save loss components for best model (bast model the one with min EMD sum overall).
        torch.save(tr_y_kl, 'train_kld_loss' + str(model_name) +  '.pt')
        torch.save(tr_y_rec, 'train_reco_loss' + str(model_name) +  '.pt')
        torch.save(tr_y_loss, 'train_total_loss' + str(model_name) +  '.pt')

        torch.save(val_y_kl, 'valid_kld_loss' + str(model_name) +  '.pt')
        torch.save(val_y_rec, 'valid_reco_loss' + str(model_name) +  '.pt')
        torch.save(val_y_loss, 'valid_total_loss' + str(model_name) +  '.pt')

        torch.save(tr_y_loss_pt, 'train_jetpT_loss' + str(model_name) +  '.pt')
        torch.save(tr_y_loss_mass, 'train_jetmass_loss' + str(model_name) + '.pt')

        torch.save(val_y_loss_pt, 'valid_jetpT_loss' + str(model_name) +  '.pt')
        torch.save(val_y_loss_mass, 'valid_jetmass_loss' + str(model_name) + '.pt')


#######################################################################################################
# Print and save EMD metrics
#print(emd_zscore.shape)
torch.save(emd_zscore_sum_all, 'emd_zscore_jetFeatures_all.pt')
#print(emd.shape)
torch.save(emd_sum_all, 'emd_jetFeatures_all.pt') # [n_epochs, n_features]

# Save the model
torch.save(best_model.state_dict(), 'best_model_min_emdSum_model_pxpypz_standardized_'+ str(epoch+1) + str(model_name) + '.pt')

int_time = time.time()
print('The time to run the network is:', (int_time - start_time)/60.0, 'minutes')

# Plot each component of the loss function
plt.figure(figsize=(8, 6))
plt.plot(x_graph, tr_y_kl, '--', label = "Train KL Divergence", linewidth=1.5, color = cb_purple)
plt.plot(x_graph, tr_y_rec, '--', label = 'Train Reconstruction Loss', linewidth=1.5, color = cb_red)
plt.plot(x_graph, tr_y_loss, '--', label = 'Train Total Loss', linewidth=1.5, color = cb_must)
plt.plot(x_graph, val_y_kl, label = "Validation KL Divergence", linewidth=1.5, color = cb_purple)
plt.plot(x_graph, val_y_rec, label = 'Validation Reconstruction Loss', linewidth=1.5, color = cb_red)
plt.plot(x_graph, val_y_loss, label = 'Validation Total Loss', linewidth=1.5, color = cb_must)
plt.yscale('log', nonpositive='clip')
plt.xlabel('Epoch')
plt.ylabel('A. U.')
plt.title('Loss Function Components')
plt.legend()
plt.savefig('pxpypz_loss_comp_log_' + str(model_name) + '.png', dpi=250, bbox_inches='tight')
plt.close()

plt.figure(figsize=(8, 6))
plt.plot(x_graph, tr_y_kl, '--', label = "Train KL Divergence", linewidth=1.5, color = cb_purple)
plt.plot(x_graph, tr_y_rec, '--', label = 'Train Reconstruction Loss', linewidth=1.5, color = cb_red)
plt.plot(x_graph, tr_y_loss, '--', label = 'Train Total Loss', linewidth=1.5, color = cb_must)
plt.plot(x_graph, val_y_kl, label = "Validation KL Divergence", linewidth=1.5, color = cb_purple)
plt.plot(x_graph, val_y_rec, label = 'Validation Reconstruction Loss', linewidth=1.5, color = cb_red)
plt.plot(x_graph, val_y_loss, label = 'Validation Total Loss', linewidth=1.5, color = cb_must)
plt.yscale('linear')
plt.xlabel('Epoch')
plt.ylabel('A. U.')
plt.title('Loss Function Components')
plt.legend()
plt.savefig('pxpypz_loss_comp_linear_' + str(model_name) + '.png', dpi=250, bbox_inches='tight')
plt.close()

# Plot each depedent component of the loss function 
plt.figure()
plt.plot(y_graph, tr_y_rec, label = 'Train Reco - Particles Loss', color = cb_red)
plt.plot(y_graph, tr_y_loss_pt, label = 'Train Reco - Jets $p_T$', color = cb_blue)
plt.plot(y_graph, tr_y_loss_mass, label = 'Train Reco - Jets Mass', color = cb_green)
plt.yscale('log')
plt.xlabel('Epoch')
#plt.ylabel('A. U.')
plt.title('Dependent Components - NND')
plt.legend()
plt.savefig('pxpypz_loss_individual_comp_linear_' + str(model_name) + '.png', dpi=250, bbox_inches='tight')
plt.close()    


# Save loss components for best model (bast model the one with min EMD sum overall).
torch.save(tr_y_kl,  'train_kld_loss' + str(model_name) +  '.pt')
torch.save(tr_y_rec,  'train_reco_loss' + str(model_name) +  '.pt')
torch.save(tr_y_loss,  'train_total_loss' + str(model_name) +  '.pt')

torch.save(val_y_kl,  'valid_kld_loss' + str(model_name) +  '.pt')
torch.save(val_y_rec,  'valid_reco_loss' + str(model_name) +  '.pt')
torch.save(val_y_loss,  'valid_total_loss' + str(model_name) +  '.pt')

torch.save(tr_y_loss_pt,  'train_jetpT_loss' + str(model_name) +  '.pt')
torch.save(tr_y_loss_mass,  'train_jetmass_loss' + str(model_name) + '.pt')

torch.save(val_y_loss_pt,  'valid_jetpT_loss' + str(model_name) +  '.pt')
torch.save(val_y_loss_mass,  'valid_jetmass_loss' + str(model_name) + '.pt')

torch.save(EMD_mass, 'EMD_mass_all.pt') # [n_epochs, n_features]
torch.save(EMD_pt, 'EMD_pt_all.pt') # [n_epochs, n_features]

##############################################################################
sum = 0
end_time = time.time()

print("The total time is ",((end_time-start_time)/60.0)," minutes.")
