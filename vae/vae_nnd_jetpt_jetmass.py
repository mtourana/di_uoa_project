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

# Hyperparameters
# Input data specific params
num_particles = 100
jet_type = 'g'

# Training params
N_epochs = 800
batch_size = 100
learning_rate = 0.001

# Model params
model_name = '_sparse_nnd_'+ str(num_particles) + 'p_jetpt_jetmass_'
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
patience = 50

####################################### LOAD DATA #######################################
train_dataset = torch.load('train_data_scaled_pxpypz_g_'+ str(num_particles) +'p.pt')
valid_dataset = torch.load('valid_data_scaled_pxpypz_g_'+ str(num_particles) +'p.pt')

# Create iterable data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)

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
def validate(model, batch_data_test):
    """valid_loss = 0
    valid_KLD_loss = 0
    valid_reco_loss = 0"""

    model.eval()
    with torch.no_grad():
        input_valid = batch_data_test.cuda()
        # loss per batch
        valid_loss, valid_KLD_loss, valid_reco_loss, valid_reco_loss_p, valid_reco_loss_j, valid_reco_loss_pt, valid_reco_loss_mass = compute_loss(model, input_valid)

        return valid_loss, valid_KLD_loss, valid_reco_loss

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

px = all_input[:,0,0,:]
py = all_input[:,0,1,:]
pz = all_input[:,0,2,:]

px_reco = all_output[:,0,0,:]
py_reco = all_output[:,0,1,:]
pz_reco = all_output[:,0,2,:]

print(px.shape)
print(py.shape)
print(pz.shape)
print(px_reco.shape)
print(py_reco.shape)
print(pz_reco.shape)

torch.save(px, 'px_beta01_latent20_std'+ str(model_name) + '.pt')
torch.save(py, 'py_beta01_latent20_std'+ str(model_name) + '.pt')
torch.save(pz, 'pz_beta01_latent20_std'+ str(model_name) + '.pt')

torch.save(px_reco, 'px_reco_beta01_latent20_std'+ str(model_name) + '.pt')
torch.save(py_reco, 'py_reco_beta01_latent20_std'+ str(model_name) + '.pt')
torch.save(pz_reco, 'pz_reco_beta01_latent20_std'+ str(model_name) + '.pt')

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

sum = 0
end_time = time.time()
print("The total time is ",((end_time-start_time)/60.0)," minutes.")
