#########################################################################################
# VAE FastSim Dataset
#########################################################################################
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, TensorDataset
import matplotlib.pyplot as plt
import time
import numpy as np
import mplhep as mhep
plt.style.use(mhep.style.CMS)
from sklearn.preprocessing import MinMaxScaler
from pickle import dump

####################################### PRE-PROCESSING #######################################
# LOAD DATA
train_data_gen = torch.load('train_data_gen_pxpypz_wDijets_100p.pt')
valid_data_gen = torch.load('valid_data_gen_pxpypz_wDijets_100p.pt')
test_data_gen = torch.load('test_data_gen_pxpypz_wDijets_100p.pt')

train_data_reco = torch.load('train_data_reco_pxpypz_wDijets_100p.pt')
valid_data_reco = torch.load('valid_data_reco_pxpypz_wDijets_100p.pt')
test_data_reco = torch.load('test_data_reco_pxpypz_wDijets_100p.pt')

# RESHAPE DATA TO BE IN FORMAT [N, features]
def reshape_data_for_normalization(data):
    return data.reshape(-1, 3)

def minmax_normalization(train_data, valid_data, test_data):
    # SCALE TRAIN DATA & SAVE SCALERS
    # CREATE SCALER. You can explicitly set the range with the feature_range attribute.
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    # RESHAPE DATA TO BE IN FORMAT [N, 3]
    train_data_reshape = reshape_data_for_normalization(train_data)
    valid_data_reshape = reshape_data_for_normalization(valid_data)
    test_data_reshape = reshape_data_for_normalization(test_data)
    # FIT AND TRANSFORM IN ONE STEP FOR TRAIN DATA
    normed_train_data = torch.from_numpy(scaler.fit_transform(train_data_reshape.numpy())).reshape(-1, 1, 100, 3)
    normed_train_data = normed_train_data.permute(0,1,3,2)
    # TRANSFORM IN ONE STEP FOR VALID/TEST DATA
    normed_valid_data = torch.from_numpy(scaler.transform(valid_data_reshape.numpy())).reshape(-1, 1, 100, 3)
    normed_valid_data = normed_valid_data.permute(0,1,3,2)
    normed_test_data = torch.from_numpy(scaler.transform(test_data_reshape.numpy())).reshape(-1, 1, 100, 3)
    normed_test_data = normed_test_data.permute(0,1,3,2)
    return normed_train_data, normed_valid_data, normed_test_data, scaler

# GEN DATA NORMALIZATION
train_data_gen_normed, valid_data_gen_normed, test_data_gen_normed, gen_data_scaler = minmax_normalization(train_data_gen, valid_data_gen, test_data_gen)
# RECO DATA NORMALIZATION
train_data_reco_normed, valid_data_reco_normed, test_data_reco_normed, reco_data_scaler = minmax_normalization(train_data_reco, valid_data_reco, test_data_reco)

# Save the scalers
dump(gen_data_scaler, open('gen_data_scaler_train_pxpypz_wDijets_100p.pkl', 'wb'))
dump(reco_data_scaler, open('reco_data_scaler_train_pxpypz_wDijets_100p.pkl', 'wb'))

####################################### SETTING PARAMS #######################################
# Hyperparameters
# Input data specific params
num_particles = 100
jet_type = 'g'

# Training params
N_epochs = 1800
batch_size = 100
learning_rate = 0.001

# Model params
model_name = 'pi_loss_fastsim_beta1_'+ str(num_particles) + 'p_'
num_classes = 1
latent_dim = 20
beta = 1
epsilon = 0.01

# Regularizer for loss penalty
# Jet features loss weighting
gamma = 1.0
gamma_1 = 1.0
gamma_2 = 1.0

# Particle features loss weighting
alpha = 1

# Starting time
start_time = time.time()

# Plots' colors
spdred = (177/255, 4/255, 14/255)
spdblue = (0/255, 124/255, 146/255)

# Probability to keep a node in the dropout layer
drop_prob = 0.0

# Set patience for Early Stopping
patience = 25

####################################### SCALED DATA #######################################
train_dataset_gen = train_data_gen_normed
valid_dataset_gen = valid_data_gen_normed
test_dataset_gen = test_data_gen_normed

train_dataset_reco = train_data_reco_normed
valid_dataset_reco = valid_data_reco_normed
test_dataset_reco = test_data_reco_normed

# Create iterable data loaders
'''train_loader_gen = DataLoader(dataset=train_dataset_gen, batch_size=batch_size, shuffle=True)
valid_loader_gen = DataLoader(dataset=valid_dataset_gen, batch_size=batch_size, shuffle=False)
test_loader_gen = DataLoader(dataset=test_dataset_gen, batch_size=batch_size, shuffle=False)

train_loader_reco = DataLoader(dataset=train_dataset_reco, batch_size=batch_size, shuffle=True)
valid_loader_reco = DataLoader(dataset=valid_dataset_reco, batch_size=batch_size, shuffle=False)
test_loader_reco = DataLoader(dataset=test_dataset_reco, batch_size=batch_size, shuffle=False)'''

#inps = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
#tgts = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)

train_dataset_gen_reco = TensorDataset(train_dataset_gen, train_dataset_reco)
train_loader = DataLoader(train_dataset_gen_reco, batch_size=batch_size, shuffle=True)

valid_dataset_gen_reco = TensorDataset(valid_dataset_gen, valid_dataset_reco)
valid_loader = DataLoader(valid_dataset_gen_reco, batch_size=batch_size, shuffle=False)

test_dataset_gen_reco = TensorDataset(test_dataset_gen, test_dataset_reco)
test_loader = DataLoader(test_dataset_gen_reco, batch_size=batch_size, shuffle=False)

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
        #out = torch.sigmoid(out)
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

# Custom loss function VAE
def compute_loss(model, x_gen, x_reco):
    mean, logvar = model.encode(x_gen)
    z = model.reparameterize(mean, logvar)
    x_gen_decoded = model.decode(z)

    pdist = nn.PairwiseDistance(p=2) # Euclidean distance

    x_target = torch.zeros(batch_size,3,num_particles).cuda()
    # INPUT
    x_target = x_reco[:,0,:,:] # [100, 3, 100]

    #jets_pt_reco = jet_pT(x_target).unsqueeze(1).cuda() # [100, 1]
    #jets_mass_reco = jet_mass(x_target).unsqueeze(1).cuda()

    x_target = x_target.view(batch_size, 3, 1, num_particles)

    # Set reco level particles as the target
    # OUTPUT/TARGET
    x_gen_output = torch.zeros(batch_size,3,num_particles).cuda()
    x_gen_output = x_gen_decoded[:,0,:,:]

    #jets_pt_gen = jet_pT(x_gen_output).unsqueeze(1).cuda() # [100, 1]
    #jets_mass_gen = jet_mass(x_gen_output).unsqueeze(1).cuda()

    x_gen_output = x_gen_output.view(batch_size, 3, num_particles, 1)
    x_gen_output = torch.repeat_interleave(x_gen_output, num_particles, -1)

    # Permutation-invariant Chamfer Loss / NND / 3D Sparse Loss
    dist = torch.pow(pdist(x_target, x_gen_output),2)

    # NND Jet kinematics losses
    #jet_pt_dist = torch.pow(pdist(jets_pt_reco, jets_pt_gen),2)
    #jet_mass_dist = torch.pow(pdist(jets_mass_reco, jets_mass_gen),2) 

    # Relative NND 
    # dist = torch.pow(pdist(x_target, x_gen_output), 2)/(x_target*x_target)
    # jet_pt_dist = torch.pow(pdist(jets_pt_reco, jets_pt_gen),2)/(jets_pt_reco*jets_pt_reco) # [100] pt on inp-outp
    # jet_mass_dist = torch.pow(pdist(jets_mass_reco, jets_mass_gen),2)/(jets_mass_reco*jets_mass_reco)  # [100] jet mass on inp-outp

    # MAPE
    #loss_rec_p = (torch.sum(torch.abs((x_target - x_gen_output)/(x_target + epsilon))/batch_size)) * 100
    #jet_pt_dist = (torch.sum(torch.abs((jets_pt_reco - jets_pt_gen)/jets_pt_reco+ epsilon))/batch_size) * 100
    #jet_mass_dist = (torch.sum(torch.abs((jets_mass_reco - jets_mass_gen)/jets_mass_reco+ epsilon))/batch_size) * 100

    # For every output value, find its closest input value; for every input value, find its closest output value.
    ieo = torch.min(dist, dim = 1)  # Get min distance per row - Find the closest input to the output
    oei = torch.min(dist, dim = 2)  # Get min distance per column - Find the closest output to the input
    # Symmetrical euclidean distances
    eucl = ieo.values + oei.values # [100, 30]

    # loss per jet (batch size)   
    loss_rec_p = (torch.sum(eucl, dim=1))
    #loss_rec_j = gamma*((jet_pt_dist) + (jet_mass_dist)) 
    #eucl = alpha*loss_rec_p + loss_rec_j  # [100]
    eucl = alpha*loss_rec_p  # [100]

    # Average symmetrical euclidean distance per image
    eucl = (torch.sum(eucl) / batch_size)
    loss_rec = - eucl

    # Loss individual components
    #loss_rec_p = torch.sum(loss_rec_p)
    #loss_rec_j = torch.sum(loss_rec_j)
    #jet_pt_dist = torch.sum(jet_pt_dist)
    #jet_mass_dist = torch.sum(jet_mass_dist)

    # Separate particles' loss components to plot them 
    #eucl_ieo = ieo.values                
    #eucl_oei = oei.values
    #eucl_in = torch.sum(eucl_ieo) / batch_size 
    #eucl_out = torch.sum(eucl_oei) / batch_size 
    #loss_KLD = beta * (0.5 * torch.sum(torch.pow(mean, 2) + torch.exp(logvar) - logvar - 1.0).sum() / batch_size) 
    loss_KLD = beta * (0.5 * torch.sum(torch.pow(mean, 2) + torch.exp(logvar) - logvar - 1.0).sum() / batch_size) 

    ELBO = loss_rec - loss_KLD
    total_loss = - ELBO

    #return total_loss, KL_divergence, loss_rec_p
    return total_loss, loss_KLD, eucl

##### Training function per batch #####
def train(model, batch_data_train_gen, batch_data_train_reco, optimizer):
    input_train_gen = batch_data_train_gen.cuda()
    input_train_reco = batch_data_train_reco.cuda()
    output_train_gen = model(input_train_gen)
    # loss per batch
    #train_loss, train_KLD_loss, train_reco_loss, train_reco_loss_p, train_reco_loss_j, train_reco_loss_pt, train_reco_loss_mass  = compute_loss(model, input_train_gen, input_train_reco)
    train_loss, train_KLD_loss, train_reco_loss  = compute_loss(model, input_train_gen, input_train_reco)

    # Backprop and perform Adam optimisation
    # Backpropagation
    optimizer.zero_grad()
    train_loss.backward()
    # Adam optimization using the gradients from backprop
    optimizer.step()

    return input_train_reco, output_train_gen, train_loss, train_KLD_loss, train_reco_loss

##### Validation function per batch #####
def validate(model, batch_data_train_gen, batch_data_train_reco):
    """valid_loss = 0
    valid_KLD_loss = 0
    valid_reco_loss = 0"""

    model.eval()
    with torch.no_grad():
        input_valid_gen = batch_data_train_gen.cuda()
        input_valid_reco = batch_data_train_reco.cuda()
        valid_loss, valid_KLD_loss, valid_reco_loss = compute_loss(model, input_valid_gen, input_valid_reco)
        return valid_loss, valid_KLD_loss, valid_reco_loss
        # loss per batch
        #valid_loss, valid_KLD_loss, valid_reco_loss, valid_reco_loss_p, valid_reco_loss_j, valid_reco_loss_pt, valid_reco_loss_mass = compute_loss(model, input_valid)
    
##### Test function #####
def test_unseed_data(model, batch_data_test_gen, batch_data_test_reco):
    model.eval()
    with torch.no_grad():
        input_test_gen = batch_data_test_gen.cuda() ##### FIXXX
        input_test_reco = batch_data_test_reco.cuda()
        output_test_gen = model(input_test_gen)
        test_loss, test_KLD_loss, test_reco_loss = compute_loss(model, input_test_gen, input_test_reco)
        # test_loss, test_KLD_loss, test_reco_loss, loss_particle, loss_jet, jet_pt_loss, jet_mass_loss = compute_loss(model, input_test)

    return input_test_reco, output_test_gen, test_loss, test_KLD_loss, test_reco_loss

####################################### TRAINING #######################################
# Initialize model and load it on GPU
model = ConvNet()
model = model.cuda()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Gen level data
#all_input_gen = np.empty(shape=(0, 1, 3, num_particles))
all_output_gen = np.empty(shape=(0, 1, 3, num_particles))

# Reco level data
all_input_reco = np.empty(shape=(0, 1, 3, num_particles))
#all_output_reco = np.empty(shape=(0, 1, 3, num_particles))

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

    # for index, (xb1, xb2) in enumerate(dataloader):
    for y, (jets_train_gen, jets_train_reco) in enumerate(train_loader):
        if y == (len(train_loader) - 1):
            break

        # Run train function on batch data
        tr_inputs_reco, tr_outputs_gen, tr_loss, tr_kl, tr_eucl  = train(model, jets_train_gen, jets_train_reco, optimizer)
        tr_loss_aux += tr_loss
        tr_kl_aux += tr_kl
        tr_rec_aux += tr_eucl

        # Individual loss components
        '''tr_rec_p_aux += tr_reco_p
        tr_rec_j_aux += tr_reco_j
        tr_rec_pt_aux += tr_reco_pt
        tr_rec_mass_aux += tr_rec_mass'''

        if stale_epochs > patience:
            # Concat input and output per batch
            batch_input_reco = tr_inputs_reco.cpu().detach().numpy()
            batch_output_gen = tr_outputs_gen.cpu().detach().numpy()
            all_input_reco = np.concatenate((all_input_reco, batch_input_reco), axis=0)
            all_output_gen = np.concatenate((all_output_gen, batch_output_gen), axis=0)
            #return input_train_reco, output_train_gen, train_loss, train_KLD_loss, train_reco_loss

    for w, (jets_valid_gen, jets_valid_reco) in enumerate(valid_loader):
        if w == (len(valid_loader) - 1):
            break

        # Run validate function on batch data
        val_loss, val_kl, val_eucl = validate(model, jets_valid_gen, jets_valid_reco)
        val_loss_aux += val_loss
        val_kl_aux += val_kl
        val_rec_aux += val_eucl

    tr_y_loss.append(tr_loss_aux.cpu().detach().item()/(len(train_loader) - 1))
    tr_y_kl.append(tr_kl_aux.cpu().detach().item()/(len(train_loader) - 1))
    tr_y_rec.append(tr_rec_aux.cpu().detach().item()/(len(train_loader) - 1))

    # Individual loss components
    ''' tr_y_loss_p.append(tr_rec_p_aux.cpu().detach().item()/(len(train_loader) - 1))
    tr_y_loss_j.append(tr_rec_j_aux.cpu().detach().item()/(len(train_loader) - 1))
    tr_y_loss_pt.append(tr_rec_pt_aux.cpu().detach().item()/(len(train_loader) - 1))
    tr_y_loss_mass.append(tr_rec_mass_aux.cpu().detach().item()/(len(train_loader) - 1))'''

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

'''px = all_input[:,0,0,:]
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

torch.save(px, 'px_std'+ str(model_name) + '.pt')
torch.save(py, 'py_std'+ str(model_name) + '.pt')
torch.save(pz, 'pz_std'+ str(model_name) + '.pt')

torch.save(px_reco, 'px_reco_std'+ str(model_name) + '.pt')
torch.save(py_reco, 'py_reco_std'+ str(model_name) + '.pt')
torch.save(pz_reco, 'pz_reco_std'+ str(model_name) + '.pt')'''

print('all_input_reco shape: ', all_input_reco.shape)
print('all_output_gen shape', all_output_gen.shape)

#torch.save(all_input_reco, 'all_input_reco_std'+ str(model_name) + '.pt')
#torch.save(all_output_gen, 'all_output_gen_std'+ str(model_name) + '.pt')

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
plt.savefig('pxpypz_standardized_loss_comp' + str(model_name) + '.png', dpi=250, bbox_inches='tight')
plt.clf()

torch.save(tr_y_kl, 'train_KLD_loss' + str(model_name) + '.pt')
torch.save(tr_y_rec, 'train_reco_loss' + str(model_name) + '.pt')
torch.save(tr_y_loss, 'train_total_loss' + str(model_name) + '.pt')

# Individual loss components
'''
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
plt.savefig('pxpypz_standardized_loss_components_' + str(model_name) + '.png')
plt.clf()    
'''
# Save the model
torch.save(model.state_dict(), 'model_pxpypz_standardized_'+ str(model_name) + '.pt')

####################################### EVALUATION #######################################
print('############# Evaluation mode #############')

all_input_reco_test = np.empty(shape=(0, 1, 3, num_particles))
all_output_gen_test = np.empty(shape=(0, 1, 3, num_particles))

for i, (jets_gen, jets_reco) in enumerate(test_loader):
    if i == (len(test_loader)-1):
        break
    # run test function on batch data for testing
    test_inputs_reco, test_outputs_gen, ts_loss, ts_kl, ts_eucl = test_unseed_data(model, jets_gen, jets_reco)
    batch_input_reco_ts = test_inputs_reco.cpu().detach().numpy()
    batch_output_gen_ts = test_outputs_gen.cpu().detach().numpy()
    all_input_reco_test = np.concatenate((all_input_reco_test, batch_input_reco_ts), axis=0)
    all_output_gen_test = np.concatenate((all_output_gen_test, batch_output_gen_ts), axis=0)

######################## Test data ########################
print('input all_input_reco_test shape: ', all_input_reco_test.shape)
print('output all_output_gen_test shape', all_output_gen_test.shape)

px_reco_input_test = all_input_reco_test[:,0,0,:]
py_reco_input_test = all_input_reco_test[:,0,1,:]
pz_reco_input_test = all_input_reco_test[:,0,2,:] 

px_gen_output_test = all_output_gen_test[:,0,0,:]
py_gen_output_test = all_output_gen_test[:,0,1,:]
pz_gen_output_test = all_output_gen_test[:,0,2,:]

print(px_reco_input_test.shape)
print(py_reco_input_test.shape)
print(pz_reco_input_test.shape)

print(px_gen_output_test.shape)
print(py_gen_output_test.shape)
print(pz_gen_output_test.shape)

torch.save(px_reco_input_test, 'px_reco_input_test_std'+ str(model_name) + '.pt')
torch.save(py_reco_input_test, 'py_reco_input_test_std'+ str(model_name) + '.pt')
torch.save(pz_reco_input_test, 'pz_reco_input_test_std'+ str(model_name) + '.pt')

torch.save(px_gen_output_test, 'px_gen_output_test_std'+ str(model_name) + '.pt')
torch.save(py_gen_output_test, 'py_gen_output_test_std'+ str(model_name) + '.pt')
torch.save(pz_gen_output_test, 'pz_gen_output_test_std'+ str(model_name) + '.pt')

##############################################################################
sum = 0
end_time = time.time()

print("The total time is ",((end_time-start_time)/60.0)," minutes.")

