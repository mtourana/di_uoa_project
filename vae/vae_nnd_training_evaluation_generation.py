#########################################################################################
# VAE 3D Sparse loss - trained on JEDI-net gluons' dataset
#########################################################################################
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import numpy as np
import mplhep as mhep
plt.style.use(mhep.style.CMS)
import skhep.math as hep
from functools import reduce
from matplotlib.colors import LogNorm

# Hyperparameters
# Input data specific params
num_particles = 100
jet_type = 'g'

# Training params
N_epochs = 800
batch_size = 100
learning_rate = 0.001

# Model params
model_name = '_sparse_nnd_beta01_train_evaluate'+ str(num_particles) + 'p_jetpt_jetmass_'
num_classes = 1
latent_dim = 20
beta = 0.1
 
# Regularizer for loss penalty
# Jet features loss weighting
gamma = 1.0
gamma_1 = 1.0
gamma_2 = 1.0

# Particle features loss weighting
alpha = 1.0

# Starting time
start_time = time.time()

# Plots' colors
spdred = (177/255, 4/255, 14/255)
spdblue = (0/255, 124/255, 146/255)

# Probability to keep a node in the dropout layer
drop_prob = 0.0

# Set patience for Early Stopping
patience = 2

####################################### LOAD DATA #######################################
train_dataset = torch.load('train_data_scaled_pxpypz_g_'+ str(num_particles) +'p.pt')
valid_dataset = torch.load('valid_data_scaled_pxpypz_g_'+ str(num_particles) +'p.pt')
test_dataset = torch.load('test_data_scaled_pxpypz_g_'+ str(num_particles) +'p.pt')
gen_dataset = torch.zeros([test_dataset.shape[0], 1, 3, num_particles])

# Create iterable data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
gen_loader = DataLoader(dataset=gen_dataset, batch_size=batch_size, shuffle=False)

####################################### DEFINE MODEL ####################################
# # Define models' architecture & helper functions
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3,5), stride=(1), padding=(0))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(1,5), stride=(1), padding=(0))
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(1,5), stride=(1), padding=(0))
        #
        self.fc1 = nn.Linear(1 * int(num_particles - 12) * 64, 1500)
        self.fc2 = nn.Linear(1500, 2 * latent_dim)
        #
        self.fc3 = nn.Linear(latent_dim, 1500)
        self.fc4 = nn.Linear(1500, 1 * int(num_particles - 12) * 64)
        self.conv4 = nn.ConvTranspose2d(64, 32, kernel_size=(1,5), stride=(1), padding=(0))
        self.conv5 = nn.ConvTranspose2d(32, 16, kernel_size=(1,5), stride=(1), padding=(0))
        self.conv6 = nn.ConvTranspose2d(16, 1, kernel_size=(3,5), stride=(1), padding=(0))
        #
        self.drop = nn.Dropout(drop_prob)

    def encode(self, x):
        out = self.conv1(x)
        out = torch.relu(out)
        out = self.conv2(out)
        out = torch.relu(out)
        out = self.conv3(out)
        out = torch.relu(out)
        out = out.view(out.size(0), -1) # flattening
        out = self.fc1(out)
        out = torch.relu(out)
        out = self.fc2(out)
        mean = out[:,:latent_dim]
        logvar = 1e-6 + (out[:,latent_dim:])
        return mean, logvar

    def decode(self, z):
        out = self.fc3(z)
        out = torch.relu(out)
        out = self.fc4(out)
        out = torch.relu(out)
        out = out.view(batch_size, 64, 1, int(num_particles - 12)) # reshaping
        out = self.conv4(out)
        out = torch.relu(out)
        out = self.conv5(out)
        out = torch.relu(out)
        out = self.conv6(out)
        return out

    def reparameterize(self, mean, logvar):
        z = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
        return z

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        out = self.decode(z)
        return out

# Jet observables manual calculation
def jet_p(p_part): # input should be of shape[batch_size, features, Nparticles]
    pjet = torch.sum(p_part, dim=2).cuda() # [batch, features (px, py, pz)]
    return pjet

def jet_Energy (p_part): # input should be of shape [batch_size, features, Nparticles]
    E_particles = torch.sqrt(torch.sum(p_part*p_part, dim=1)) # E per particle shape: [100, 30]
    E_jet = torch.sum(E_particles, dim=1).cuda() # Energy per jet [100]
    return E_jet

def jet_mass (p_part):
    jet_e = jet_Energy(p_part)
    P_jet = jet_p(p_part)
    m_jet = torch.sqrt(jet_e*jet_e - (P_jet[:,0]*P_jet[:,0]) - (P_jet[:,1]*P_jet[:,1]) - (P_jet[:,2]*P_jet[:,2])).cuda()
    return m_jet # mass per jet [100]

def jet_pT(p_part):# input should be of shape[batch_size, features, Nparticles]
    p_jet = jet_p(p_part) # [100, 3]
    jet_px = p_jet[:, 0]  # [100]
    jet_py = p_jet[:, 1]  # [100]
    jet_pt = torch.sqrt(jet_px*jet_px + jet_py*jet_py)
    return jet_pt

# Custom loss function VAE
def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_decoded = model.decode(z)
    pdist = nn.PairwiseDistance(p=2) # Euclidean distance
    x_pos = torch.zeros(batch_size,3,num_particles).cuda()
    x_pos = x[:,0,:,:] # [100, 3, 30]
    jets_pt = jet_pT(x_pos).unsqueeze(1).cuda() # [100, 1]
    jets_mass = jet_mass(x_pos).unsqueeze(1).cuda()
    x_pos = x_pos.view(batch_size, 3, 1, num_particles)
    x_decoded_pos = torch.zeros(batch_size,3,num_particles).cuda()
    x_decoded_pos = x_decoded[:,0,:,:]
    jets_pt_reco = jet_pT(x_decoded_pos).unsqueeze(1).cuda() # [100, 1]
    jets_mass_reco = jet_mass(x_decoded_pos).unsqueeze(1).cuda()
    x_decoded_pos = x_decoded_pos.view(batch_size, 3, num_particles, 1)
    x_decoded_pos = torch.repeat_interleave(x_decoded_pos, num_particles, -1)
    # Permutation-invariant Loss / NND / 3D Sparse Loss
    dist = torch.pow(pdist(x_pos, x_decoded_pos),2)

    # NND original version
    jet_pt_dist = torch.pow(pdist(jets_pt, jets_pt_reco),2)
    jet_mass_dist = torch.pow(pdist(jets_mass, jets_mass_reco),2) 
    # Relative NND loss
    # jet_pt_dist = torch.pow(pdist(jets_pt, jets_pt_reco),2)/(jets_pt*jets_pt) # [100] pt MSE on inp-outp
    # jet_mass_dist = torch.pow(pdist(jets_mass, jets_mass_reco),2)/(jets_mass*jets_mass)  # [100] jet mass MSE on inp-outp

    # For every output value, find its closest input value; for every input value, find its closest output value.
    ieo = torch.min(dist, dim = 1)  # Get min distance per row - Find the closest input to the output
    oei = torch.min(dist, dim = 2)  # Get min distance per column - Find the closest output to the input
    # Symmetrical euclidean distances
    eucl = ieo.values + oei.values # [100, 30]

    # Loss per jet (batch size)   
    loss_rec_p = alpha*(torch.sum(eucl, dim=1))
    loss_rec_j = gamma*(gamma_1*(jet_pt_dist) + gamma_2*(jet_mass_dist)) 
    eucl = loss_rec_p + loss_rec_j  # [100]

    # Loss individual components
    loss_rec_p = torch.sum(loss_rec_p)
    loss_rec_j = torch.sum(loss_rec_j)
    jet_pt_dist = torch.sum(jet_pt_dist)
    jet_mass_dist = torch.sum(jet_mass_dist)

    # Average symmetrical euclidean distance per image
    eucl = (torch.sum(eucl) / batch_size)
    reconstruction_loss = - eucl

    # Separate particles' loss components to plot them 
    #eucl_ieo = ieo.values                
    #eucl_oei = oei.values
    #eucl_in = torch.sum(eucl_ieo) / batch_size 
    #eucl_out = torch.sum(eucl_oei) / batch_size 
    KL_divergence = beta * (0.5 * torch.sum(torch.pow(mean, 2) + torch.exp(logvar) - logvar - 1.0).sum() / batch_size) 
    ELBO = reconstruction_loss - KL_divergence
    loss = - ELBO

    return loss, KL_divergence, eucl, loss_rec_p, loss_rec_j, jet_pt_dist, jet_mass_dist

##### Training function per batch #####
def train(model, batch_data_train, optimizer):
    """train_loss = 0.0
    train_KLD_loss = 0.0
    train_reco_loss = 0.0"""
    input_train = batch_data_train[:, :, :].cuda()
    output_train = model(input_train)
    # loss per batch
    train_loss, train_KLD_loss, train_reco_loss, train_reco_loss_p, train_reco_loss_j, train_reco_loss_pt, train_reco_loss_mass  = compute_loss(model, input_train)

    # Backprop and perform Adam optimisation
    # Backpropagation
    optimizer.zero_grad()
    train_loss.backward()
    # Adam optimization using the gradients from backprop
    optimizer.step()

    return input_train, output_train, train_loss, train_KLD_loss, train_reco_loss, train_reco_loss_p, train_reco_loss_j, train_reco_loss_pt, train_reco_loss_mass

##### Validation function per batch #####
def validate(model, batch_data_valid):
    """valid_loss = 0
    valid_KLD_loss = 0
    valid_reco_loss = 0"""

    model.eval()
    with torch.no_grad():
        input_valid = batch_data_valid.cuda()
        # loss per batch
        valid_loss, valid_KLD_loss, valid_reco_loss, valid_reco_loss_p, valid_reco_loss_j, valid_reco_loss_pt, valid_reco_loss_mass = compute_loss(model, input_valid)

        return valid_loss, valid_KLD_loss, valid_reco_loss

##### Test function #####
def test_unseed_data(model, batch_data_test):
    model.eval()
    with torch.no_grad():
        input_test = batch_data_test.cuda()
        output_test = model(input_test)
        test_loss, test_KLD_loss, test_reco_loss, loss_particle, loss_jet, jet_pt_loss, jet_mass_loss = compute_loss(model, input_test)
    return input_test, output_test, test_loss, test_KLD_loss, test_reco_loss

####################################### TRAINING #######################################
# Initialize model and load it on GPU
model = ConvNet()
model = model.cuda()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

all_input = np.empty(shape=(0, 1, 3, num_particles))
all_output = np.empty(shape=(0, 1, 3, num_particles))

x_graph = []
y_graph = []

tr_y_rec = []
tr_y_kl = []
tr_y_loss = []

# Individual loss components
tr_y_loss_p = []
tr_y_loss_j = []
tr_y_loss_pt = []
tr_y_loss_mass = []

val_y_rec = []
val_y_kl = []
val_y_loss = []

min_loss, stale_epochs = 999999.0, 0

for epoch in range(N_epochs):

    x_graph.append(epoch)
    y_graph.append(epoch)

    tr_loss_aux = 0.0
    tr_kl_aux = 0.0
    tr_rec_aux = 0.0
    # Individual loss components
    tr_rec_p_aux = 0.0
    tr_rec_j_aux = 0.0
    tr_rec_pt_aux = 0.0
    tr_rec_mass_aux = 0.0

    val_loss_aux = 0.0
    val_kl_aux = 0.0
    val_rec_aux = 0.0

    for y, (jets_train) in enumerate(train_loader):
        if y == (len(train_loader) - 1):
            break

        # Run train function on batch data
        tr_inputs, tr_outputs, tr_loss, tr_kl, tr_eucl, tr_reco_p, tr_reco_j, tr_reco_pt, tr_rec_mass  = train(model, jets_train, optimizer)
        tr_loss_aux += tr_loss
        tr_kl_aux += tr_kl
        tr_rec_aux += tr_eucl

        # Individual loss components
        tr_rec_p_aux += tr_reco_p
        tr_rec_j_aux += tr_reco_j
        tr_rec_pt_aux += tr_reco_pt
        tr_rec_mass_aux += tr_rec_mass

        if stale_epochs > patience:
            # Concat input and output per batch
            batch_input = tr_inputs.cpu().detach().numpy()
            batch_output = tr_outputs.cpu().detach().numpy()
            all_input = np.concatenate((all_input, batch_input), axis=0)
            all_output = np.concatenate((all_output, batch_output), axis=0)

    for w, (jets_valid) in enumerate(valid_loader):
        if w == (len(valid_loader) - 1):
            break

        # Run validate function on batch data
        val_loss, val_kl, val_eucl = validate(model, jets_valid)
        val_loss_aux += val_loss
        val_kl_aux += val_kl
        val_rec_aux += val_eucl

    tr_y_loss.append(tr_loss_aux.cpu().detach().item()/(len(train_loader) - 1))
    tr_y_kl.append(tr_kl_aux.cpu().detach().item()/(len(train_loader) - 1))
    tr_y_rec.append(tr_rec_aux.cpu().detach().item()/(len(train_loader) - 1))

    # Individual loss components
    tr_y_loss_p.append(tr_rec_p_aux.cpu().detach().item()/(len(train_loader) - 1))
    tr_y_loss_j.append(tr_rec_j_aux.cpu().detach().item()/(len(train_loader) - 1))
    tr_y_loss_pt.append(tr_rec_pt_aux.cpu().detach().item()/(len(train_loader) - 1))
    tr_y_loss_mass.append(tr_rec_mass_aux.cpu().detach().item()/(len(train_loader) - 1))

    val_y_loss.append(val_loss_aux.cpu().detach().item()/(len(valid_loader) - 1))
    val_y_kl.append(val_kl_aux.cpu().detach().item()/(len(valid_loader) - 1))
    val_y_rec.append(val_rec_aux.cpu().detach().item()/(len(valid_loader) - 1))

    if stale_epochs > patience:
        print("Early stopped")
        break

    if val_loss_aux.cpu().detach().item()/(len(valid_loader) - 1) < min_loss:
        min_loss = val_loss_aux.cpu().detach().item()/(len(valid_loader) - 1)
        stale_epochs = 0
    else:
        stale_epochs += 1
        print('stale_epochs:', stale_epochs)

    print('Epoch: {} -- Train loss: {} -- Validation loss: {}'.format(epoch, tr_loss_aux.cpu().detach().item()/(len(train_loader)-1), val_loss_aux.cpu().detach().item()/(len(valid_loader)-1)))

#######################################################################################################
int_time = time.time()
print('The time to run the network is:', (int_time - start_time)/60.0, 'minutes')

######################## Training data ########################
px_train = all_input[:,0,0,:]
py_train = all_input[:,0,1,:]
pz_train = all_input[:,0,2,:]

px_reco_train = all_output[:,0,0,:]
py_reco_train = all_output[:,0,1,:]
pz_reco_train = all_output[:,0,2,:]

print(px_train.shape)
print(py_train.shape)
print(pz_train.shape)
print(px_reco_train.shape)
print(py_reco_train.shape)
print(pz_reco_train.shape)

torch.save(px_train, 'px_beta01_latent20_std'+ str(model_name) + '.pt')
torch.save(py_train, 'py_beta01_latent20_std'+ str(model_name) + '.pt')
torch.save(pz_train, 'pz_beta01_latent20_std'+ str(model_name) + '.pt')

torch.save(px_reco_train, 'px_reco_beta01_latent20_std'+ str(model_name) + '.pt')
torch.save(py_reco_train, 'py_reco_beta01_latent20_std'+ str(model_name) + '.pt')
torch.save(pz_reco_train, 'pz_reco_beta01_latent20_std'+ str(model_name) + '.pt')

print('input shape: ', all_input.shape)
print('output shape', all_output.shape)

torch.save(all_input, 'all_input_beta01_latent20_std'+ str(model_name) + '.pt')
torch.save(all_output, 'all_output_beta01_latent20_std'+ str(model_name) + '.pt')

# Plot each component of the loss function
plt.figure()
plt.plot(x_graph, tr_y_kl, label = "Train KL Divergence")
plt.plot(x_graph, tr_y_rec, label = 'Train Reconstruction Loss')
plt.plot(x_graph, tr_y_loss, label = 'Train Total Loss')
plt.plot(x_graph, val_y_kl, label = "Validation KL Divergence")
plt.plot(x_graph, val_y_rec, label = 'Validation Reconstruction Loss')
plt.plot(x_graph, val_y_loss, label = 'Validation Total Loss')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('A. U.')
plt.title('Loss Function Components')
plt.legend()
plt.savefig('pxpypz_standardized_beta01_latent20' + str(model_name) + '.png')
plt.clf()

torch.save(tr_y_kl, 'train_KLD_loss' + str(model_name) + '.pt')
torch.save(tr_y_rec, 'train_reco_loss' + str(model_name) + '.pt')
torch.save(tr_y_loss, 'train_total_loss' + str(model_name) + '.pt')

# Individual loss components
torch.save(tr_y_loss_p, 'tr_reco_loss_particles' + str(model_name) + '.pt')
torch.save(tr_y_loss_j, 'tr_reco_loss_jet' + str(model_name) + '.pt')
torch.save(tr_y_loss_pt, 'tr_reco_loss_jet_pt' + str(model_name) + '.pt')
torch.save(tr_y_loss_mass, 'tr_reco_loss_jet_mass' + str(model_name) + '.pt')

# Plot each depedent component of the loss function 
plt.figure()
plt.plot(y_graph, tr_y_loss_p, label = 'Train Reco - Particles Loss')
plt.plot(y_graph, tr_y_loss_j, label = 'Train Reco - Jets Loss (a_Penalty)')
plt.plot(y_graph, tr_y_loss_pt, label = 'Train Reco - Jets $p_T$')
plt.plot(y_graph, tr_y_loss_mass, label = 'Train Reco - Jets Mass')
plt.yscale('log')
plt.xlabel('Epoch')
#plt.ylabel('A. U.')
plt.title('Dependent Components - NND')
plt.legend()
plt.savefig('pxpypz_standardized_loss_components_latent20' + str(model_name) + '.png')
plt.clf()    

# Save the model
torch.save(model.state_dict(), 'model_pxpypz_standardized_3DLoss_beta01_latent20'+ str(model_name) + '.pt')

####################################### EVALUATION #######################################
print('############# Evaluation mode #############')

all_input_test = np.empty(shape=(0, 1, 3, num_particles))
all_output_test = np.empty(shape=(0, 1, 3, num_particles))

for i, (jets) in enumerate(test_loader):
    if i == (len(test_loader)-1):
        break
    # run test function on batch data for testing
    test_inputs, test_outputs, ts_loss, ts_kl, ts_eucl = test_unseed_data(model, jets)
    batch_input_ts = test_inputs.cpu().detach().numpy()
    batch_output_ts = test_outputs.cpu().detach().numpy()
    all_input_test = np.concatenate((all_input_test, batch_input_ts), axis=0)
    all_output_test = np.concatenate((all_output_test, batch_output_ts), axis=0)

######################## Test data ########################
print('input test shape: ', all_input_test.shape)
print('output test shape', all_output_test.shape)

px_test = all_input_test[:,0,0,:]
py_test = all_input_test[:,0,1,:]
pz_test = all_input_test[:,0,2,:] 

px_reco_test = all_output_test[:,0,0,:]
py_reco_test = all_output_test[:,0,1,:]
pz_reco_test = all_output_test[:,0,2,:]

print(px_test.shape)
print(py_test.shape)
print(pz_test.shape)
print(px_reco_test.shape)
print(py_reco_test.shape)
print(pz_reco_test.shape)

torch.save(px_test, 'px_test_latent20_std'+ str(model_name) + '.pt')
torch.save(py_test, 'py_test_latent20_std'+ str(model_name) + '.pt')
torch.save(pz_test, 'pz_test_latent20_std'+ str(model_name) + '.pt')

torch.save(px_reco_test, 'px_reco_test_latent20_std'+ str(model_name) + '.pt')
torch.save(py_reco_test, 'py_reco_test_latent20_std'+ str(model_name) + '.pt')
torch.save(pz_reco_test, 'pz_reco_test_latent20_std'+ str(model_name) + '.pt')

####################################### GENERATION #######################################
print('############# Generation #############')

gen_output = np.empty(shape=(0, 1, 3, num_particles))

for g, (jets) in enumerate(gen_loader):
    if g == (len(gen_loader) - 1):
        break
    # generation
    z = torch.randn(batch_size, latent_dim).cuda()
    generated_output = model.decode(z)
    batch_gen_output = generated_output.cpu().detach().numpy()
    gen_output = np.concatenate((gen_output, batch_gen_output), axis=0)

print('generated output:', gen_output.shape)   
torch.save(gen_output, 'generated_data_GeV'+ str(model_name) +'.pt')

# Check arrays expected size
px_gen = gen_output[:,0,0,:]
py_gen = gen_output[:,0,1,:]
pz_gen = gen_output[:,0,2,:]

print('px gen:', px_gen.shape)
print('py gen:', py_gen.shape)
print('pz gen:', pz_gen.shape)

# Save tensors for post-processing
torch.save(px_gen, 'px_gen_std'+ str(model_name) + '.pt')
torch.save(py_gen, 'py_gen_std'+ str(model_name) + '.pt')
torch.save(pz_gen, 'pz_gen_std'+ str(model_name) + '.pt')

############################################ Compute ############################################
# Read data (input & output scaled). 
px = torch.from_numpy(px_test)
py = torch.from_numpy(py_test)
pz = torch.from_numpy(pz_test)

# Model output
px_reco_0 = torch.from_numpy(px_reco_test)
py_reco_0 = torch.from_numpy(py_reco_test)
pz_reco_0 = torch.from_numpy(pz_reco_test)

# Model generation 
px_gen_0 = torch.from_numpy(px_gen)
py_gen_0 = torch.from_numpy(py_gen)
pz_gen_0 = torch.from_numpy(pz_gen)

# Read features' scalers
px_scaler = torch.load('px_scaler_fixed_gluons'+str(num_particles)+'p.pt')
py_scaler = torch.load('py_scaler_fixed_gluons'+str(num_particles)+'p.pt')
pz_scaler = torch.load('pz_scaler_fixed_gluons'+str(num_particles)+'p.pt')

# Inverse standardize data. Values in GeV'+str(model_name)+'
def inverse_standardize(X, scaler): 
    mean = scaler[0]
    std = scaler[1] 
    original_X = ((X * std) + mean)
    return original_X

px_r = inverse_standardize(px, px_scaler)
py_r = inverse_standardize(py, py_scaler)
pz_r = inverse_standardize(pz, pz_scaler)

# Test data 
px_reco_r_0 = inverse_standardize(px_reco_0, px_scaler)
py_reco_r_0 = inverse_standardize(py_reco_0, py_scaler)
pz_reco_r_0 = inverse_standardize(pz_reco_0, pz_scaler)

# Gen data 
px_gen_r_0 = inverse_standardize(px_gen_0, px_scaler)
py_gen_r_0 = inverse_standardize(py_gen_0, py_scaler)
pz_gen_r_0 = inverse_standardize(pz_gen_0, pz_scaler)

n_jets = px_r.shape[0]
######################################################################################
# Masking for input & output constraints

# Input constraints
def mask_zero_padding(input_data):
    print('input_data data shape', input_data.shape)
    # Mask input for zero-padded particles. Set to zero values between -10^-8 and 10^-8
    px = input_data[:,0,:]
    py = input_data[:,1,:]
    pz = input_data[:,2,:]
    mask_px = ((px <= -0.00000001) | (px >= 0.00000001)) 
    mask_py = ((py <= -0.00000001) | (py >= 0.00000001))
    mask_pz = ((pz <= -0.00000001) | (pz >= 0.00000001))
    masked_px = px * mask_px
    masked_py = py * mask_py
    masked_pz = pz * mask_pz
    # Count zeros
    print('px zeros', torch.nonzero(masked_px==0).shape[0])
    print('py zeros', torch.nonzero(masked_py==0).shape[0])
    print('pz zeros', torch.nonzero(masked_pz==0).shape[0])
    data = torch.stack([masked_px, masked_py, masked_pz], dim=1)
    print(data.shape)
    return data

inputs = torch.stack([px_r, py_r, pz_r], dim=1)
masked_inputs = mask_zero_padding(inputs)

print('inputs shape:', inputs.shape)
print('masked inputs shape:', masked_inputs.shape)
######################################################################################
# Output constraints
def mask_min_pt(output_data):
    print('output data shape', output_data.shape) # ([124100, 3, 30])
    # Mask output for min-pt
    min_pt_cut = 0.25
    mask =  output_data[:,0,:] * output_data[:,0,:] + output_data[:,1,:] * output_data[:,1,:] > min_pt_cut**2
    print(mask.shape)
    # Expand over the features' dimension
    mask = mask.unsqueeze(1)
    print(mask.shape)
    # Then, you can apply the mask
    data_masked = mask * output_data
    print(data_masked.shape) # Now, values that correspond to the min-pt should be zeroed. Check zeros again.
    return data_masked

# Test data
outputs_0 = torch.stack([px_reco_r_0, py_reco_r_0, pz_reco_r_0], dim=1)
masked_outputs_0 = mask_min_pt(outputs_0) # Now, values that correspond to the min-pt should be zeroed.

# Gen data
gen_outputs_0 = torch.stack([px_gen_r_0, py_gen_r_0, pz_gen_r_0], dim=1)
masked_gen_outputs_0 = mask_min_pt(gen_outputs_0)
######################################################################################
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

px_r_masked = masked_inputs[:,0,:].detach().cpu().numpy()
py_r_masked = masked_inputs[:,1,:].detach().cpu().numpy()
pz_r_masked = masked_inputs[:,2,:].detach().cpu().numpy()
mass = np.zeros((pz_r_masked.shape[0], num_particles))

input_data = np.stack((px_r_masked, py_r_masked, pz_r_masked, mass), axis=2)

# Test data
px_reco_r_0_masked = masked_outputs_0[:,0,:].detach().cpu().numpy()
py_reco_r_0_masked = masked_outputs_0[:,1,:].detach().cpu().numpy()
pz_reco_r_0_masked = masked_outputs_0[:,2,:].detach().cpu().numpy()
mass_reco_0 = np.zeros((pz_reco_r_0_masked.shape[0], num_particles))

# Test data
output_data_0 = np.stack((px_reco_r_0_masked, py_reco_r_0_masked, pz_reco_r_0_masked, mass_reco_0), axis=2)

# Gen data
px_gen_r_0_masked = masked_gen_outputs_0[:,0,:].detach().cpu().numpy()
py_gen_r_0_masked = masked_gen_outputs_0[:,1,:].detach().cpu().numpy()
pz_gen_r_0_masked = masked_gen_outputs_0[:,2,:].detach().cpu().numpy()
mass_gen_0 = np.zeros((pz_gen_r_0_masked.shape[0], num_particles))

# Gen data
gen_output_data_0 = np.stack((px_gen_r_0_masked, py_gen_r_0_masked, pz_gen_r_0_masked, mass_gen_0), axis=2)

print('input data shape:', input_data.shape)
print('output data shape:', output_data_0.shape)
print('generated data shape:', gen_output_data_0.shape)

lvjets_in = jet_samples_from_particle_samples(input_data)

# Output test
lvjets_out_0 = jet_samples_from_particle_samples(output_data_0)

# Output gen
lvjets_gen_0 = jet_samples_from_particle_samples(gen_output_data_0)
###########################################################################################################
# Jet mass
m_j = [lvjet.mass for lvjet in lvjets_in] 
mass_jet = np.array(m_j)
# Output Test
m_j_reco_0 = [lvjet.mass for lvjet in lvjets_out_0] 
mass_jet_reco_0 = np.array(m_j_reco_0)

# Output Gen
m_j_gen_0 = [lvjet.mass for lvjet in lvjets_gen_0] 
mass_jet_gen_0 = np.array(m_j_gen_0)

_, bins, _ = plt.hist(mass_jet, bins=100, range = [0, 400], histtype = 'step', density=False, label='Input Test', color = spdred,linewidth=1.5)
_ = plt.hist(mass_jet_reco_0, bins=100, range = [0, 400], histtype = 'step', density=False, label='Output Test VAE-NND + Penalty (pt,mass)', color = 'black',linewidth=1.5)
plt.ylabel("Probability (a.u.)")
plt.xlabel('jet mass (GeV)')
plt.yscale('linear')
plt.legend(loc='lower right', prop={'size': 16})
plt.savefig('jet_mass_GeV'+str(model_name)+'.png', dpi=250, bbox_inches='tight')
plt.clf()

_, bins, _ = plt.hist(mass_jet, bins=100, range = [0, 400], histtype = 'step', density=False, label='Input Test', color = spdred,linewidth=1.5)
_ = plt.hist(mass_jet_gen_0, bins=100, range = [0, 400], histtype = 'step', density=False, label='Randomly Generated VAE-NND + Penalty (pt,mass)', color = 'black',linewidth=1.5)
plt.ylabel("Probability (a.u.)")
plt.xlabel('jet mass (GeV)')
plt.yscale('linear')
plt.legend(loc='lower right', prop={'size': 16})
plt.savefig('jet_mass_GeV_gaussian'+str(model_name)+'.png', dpi=250, bbox_inches='tight')
plt.clf()

# Jet pt 
pt_j = [lvjet.pt for lvjet in lvjets_in]
pt_jet = np.array(pt_j)
# Output Test
pt_j_reco_0 = [lvjet.pt for lvjet in lvjets_out_0]
pt_jet_reco_0 = np.array(pt_j_reco_0)

# Output Gen
pt_j_gen_0 = [lvjet.pt for lvjet in lvjets_gen_0]
pt_jet_gen_0 = np.array(pt_j_gen_0)

_, bins, _ = plt.hist(pt_jet, bins=100, range=[0, 3000], histtype = 'step', density=False, label='Input Test', color = spdred, linewidth=1.5)
_ = plt.hist(pt_jet_reco_0, bins=100, range=[0, 3000], histtype = 'step', density=False, label='Output Test VAE-NND + Penalty (pt,mass)', color = 'black', linewidth=1.5)
plt.ylabel("Probability (a.u.)")
plt.xlabel('jet $p_T$ (GeV)')
plt.yscale('linear')
#plt.xlim(-50.00, 50.00)
plt.legend(loc='lower right', prop={'size': 16})
plt.savefig('jet_pt_GeV'+str(model_name)+'.png', dpi=250, bbox_inches='tight')
plt.clf()

_, bins, _ = plt.hist(pt_jet, bins=100, range=[0, 3000], histtype = 'step', density=False, label='Input Test', color = spdred, linewidth=1.5)
_ = plt.hist(pt_jet_gen_0, bins=100, range=[0, 3000], histtype = 'step', density=False, label='Randomly Generated VAE-NND + Penalty (pt,mass)', color = 'black', linewidth=1.5)
plt.ylabel("Probability (a.u.)")
plt.xlabel('jet $p_T$ (GeV)')
plt.yscale('linear')
#plt.xlim(-50.00, 50.00)
plt.legend(loc='lower right', prop={'size': 16})
plt.savefig('jet_pt_GeV_gaussian'+str(model_name)+'.png', dpi=250, bbox_inches='tight')
plt.clf()

# Jet energy
e_j = [lvjet.e for lvjet in lvjets_in]
e_jet = np.array(e_j)
# Output Test
e_j_reco_0 = [lvjet.e for lvjet in lvjets_out_0]
e_jet_reco_0 = np.array(e_j_reco_0)

# Output Test
e_j_gen_0 = [lvjet.e for lvjet in lvjets_gen_0]
e_jet_gen_0 = np.array(e_j_gen_0)

_, bins, _ = plt.hist(e_jet, bins=100, range = [200,4000], histtype = 'step', density=False, label='Input Test', color = spdred, linewidth=1.5)
_ = plt.hist(e_jet_reco_0, bins=100, range = [200,4000], histtype = 'step', density=False, label='Output Test VAE-NND + Penalty (pt,mass)', color = 'black', linewidth=1.5)
plt.ylabel("Probability (a.u.)")
plt.xlabel('$jet energy$ (GeV)')
plt.yscale('linear')
#plt.xlim(-50.00, 50.00)
plt.legend(loc='lower right', prop={'size': 16})
plt.savefig('jet_energy_GeV'+str(model_name)+'.png', dpi=250, bbox_inches='tight')
plt.clf()

_, bins, _ = plt.hist(e_jet, bins=100, range = [200,4000], histtype = 'step', density=False, label='Input Test', color = spdred, linewidth=1.5)
_ = plt.hist(e_jet_gen_0, bins=100, range = [200,4000], histtype = 'step', density=False, label='Randomly Generated VAE-NND + Penalty (pt,mass)', color = 'black', linewidth=1.5)
plt.ylabel("Probability (a.u.)")
plt.xlabel('$jet energy$ (GeV)')
plt.yscale('linear')
#plt.xlim(-50.00, 50.00)
plt.legend(loc='lower right', prop={'size': 16})
plt.savefig('jet_energy_GeV_gaussian'+str(model_name)+'.png', dpi=250, bbox_inches='tight')
plt.clf()

# Jet eta
eta_j = [lvjet.eta for lvjet in lvjets_in]
eta_jet = np.array(eta_j)
# Output Test
eta_j_reco_0 = [lvjet.eta for lvjet in lvjets_out_0]
eta_jet_reco_0 = np.array(eta_j_reco_0)

# Output Test
eta_j_gen_0 = [lvjet.eta for lvjet in lvjets_gen_0]
eta_jet_gen_0 = np.array(eta_j_gen_0)

_, bins, _ = plt.hist(eta_jet, bins=80, range = [-3,3], histtype = 'step', density=False, label='Input Test', color = spdred, linewidth=1.5)
_ = plt.hist(eta_jet_reco_0, bins=80, range = [-3,3], histtype = 'step', density=False, label='Output Test VAE-NND + Penalty (pt,mass)', color = 'black', linewidth=1.5)
plt.ylabel("Probability (a.u.)")
plt.xlabel('jet $\eta$')
plt.yscale('linear')
#plt.xlim(-50.00, 50.00)
plt.legend(loc='lower right', prop={'size': 16})
plt.savefig('jet_eta_GeV'+str(model_name)+'.png', dpi=250, bbox_inches='tight')
plt.clf()

_, bins, _ = plt.hist(eta_jet, bins=80, range = [-3,3], histtype = 'step', density=False, label='Input Test', color = spdred, linewidth=1.5)
_ = plt.hist(eta_jet_gen_0, bins=80, range = [-3,3], histtype = 'step', density=False, label='Randomly Generated VAE-NND + Penalty (pt,mass)', color = 'black', linewidth=1.5)
plt.ylabel("Probability (a.u.)")
plt.xlabel('jet $\eta$')
plt.yscale('linear')
#plt.xlim(-50.00, 50.00)
plt.legend(loc='lower right', prop={'size': 16})
plt.savefig('jet_eta_GeV_gaussian'+str(model_name)+'.png', dpi=250, bbox_inches='tight')
plt.clf()

# Jet phi
phi_j = [lvjet.phi() for lvjet in lvjets_in]
phi_jet = np.array(phi_j)
# Output Test
phi_j_reco_0 = [lvjet.phi() for lvjet in lvjets_out_0]
phi_jet_reco_0 = np.array(phi_j_reco_0)

# Output Gen
phi_j_gen_0 = [lvjet.phi() for lvjet in lvjets_gen_0]
phi_jet_gen_0 = np.array(phi_j_gen_0)

_, bins, _ = plt.hist(phi_jet, bins=80, range=[-3,3], histtype = 'step', density=False, label='Input Test', color = spdred, linewidth=1.5)
_ = plt.hist(phi_jet_reco_0, bins=80, range=[-3,3], histtype = 'step', density=False, label='Output Test VAE-NND + Penalty (pt,mass)', color = 'black', linewidth=1.5)
plt.ylabel("Probability (a.u.)")
plt.xlabel('jet $\phi$')
plt.yscale('linear')
#plt.xlim(-50.00, 50.00)
plt.legend(loc='lower right', prop={'size': 16})
plt.savefig('jet_phi_GeV'+str(model_name)+'.png', dpi=250, bbox_inches='tight')
plt.clf()

_, bins, _ = plt.hist(phi_jet, bins=80, range=[-3,3], histtype = 'step', density=False, label='Input Test', color = spdred, linewidth=1.5)
_ = plt.hist(phi_jet_gen_0, bins=80, range=[-3,3], histtype = 'step', density=False, label='Randomly Generated VAE-NND + Penalty (pt,mass)', color = 'black', linewidth=1.5)
plt.ylabel("Probability (a.u.)")
plt.xlabel('jet $\phi$')
plt.yscale('linear')
#plt.xlim(-50.00, 50.00)
plt.legend(loc='lower right', prop={'size': 16})
plt.savefig('jet_phi_GeV_gaussian'+str(model_name)+'.png', dpi=250, bbox_inches='tight')
plt.clf()


sum = 0
end_time = time.time()
print("The total time is ",((end_time-start_time)/60.0)," minutes.")
