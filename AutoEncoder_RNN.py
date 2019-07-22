# Imports
import os.path
import cv2
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
from Renderer.model import FCN
from AutoEncoder import AutoEncoder
import matplotlib
matplotlib.use('Agg') # Need this to print on the cluster
import matplotlib.pyplot as plt


######################
# FUNCTIONS ##########
######################
def decode(x, canvas):  # b * (10 + 3)
    x = x.view(-1, 10 + 3)
    stroke = 1 - renderer(x[:, :10])
    stroke = stroke.view(-1, width, width, 1)
    color_stroke = stroke * x[:, -3:].view(-1, 1, 1, 3)
    stroke = stroke.permute(0, 3, 1, 2)
    color_stroke = color_stroke.permute(0, 3, 1, 2)
    num_strokes = len(stroke)
    stroke = stroke.view(-1, num_strokes, 1, width, width)
    color_stroke = color_stroke.view(-1, num_strokes, 3, width, width)
    res = []

    for i in range(num_strokes):
        canvas = canvas * (1 - stroke[:, i]) + color_stroke[:, i]
        res.append(canvas)
    return canvas, res

def small2large(x):
    # (d * d, width, width) -> (d * width, d * width)
    x = x.reshape(args.divide, args.divide, width, width, -1)
    x = np.transpose(x, (0, 2, 1, 3, 4))
    x = x.reshape(args.divide * width, args.divide * width, -1)
    return x

#def large2small(x):
#    # (d * width, d * width) -> (d * d, width, width)
#    x = x.reshape(args.divide, width, args.divide, width, 3)
#    x = np.transpose(x, (0, 2, 1, 3, 4))
#    x = x.reshape(canvas_cnt, width, width, 3)
#    return x

def smooth(img):
    def smooth_pix(img, tx, ty):
        if tx == args.divide * width - 1 or ty == args.divide * width - 1 or tx == 0 or ty == 0:
            return img
        img[tx, ty] = (img[tx, ty] + img[tx + 1, ty] + img[tx, ty + 1] + img[tx - 1, ty] + img[tx, ty - 1] + img[tx + 1, ty - 1] + img[tx - 1, ty + 1] + img[tx - 1, ty - 1] + img[tx + 1, ty + 1]) / 9
        return img

    for p in range(args.divide):
        for q in range(args.divide):
            x = p * width
            y = q * width
            for k in range(width):
                img = smooth_pix(img, x + k, y + width - 1)
                if q != args.divide - 1:
                    img = smooth_pix(img, x + k, y + width)
            for k in range(width):
                img = smooth_pix(img, x + width - 1, y + k)
                if p != args.divide - 1:
                    img = smooth_pix(img, x + width, y + k)
    return img

def save_img(res, imgid, divide=False):
    global output
    output = res.detach().cpu().numpy()  # d * d, 3, width, width
    output = np.transpose(output, (0, 2, 3, 1))
    if divide:
        output = small2large(output)
        output = smooth(output)
    else:
        output = output[0]
    output = (output * 255).astype('uint8')
    output = cv2.resize(output, (128,128))
    cv2.imwrite('output/generated' + str(imgid) + '.png', output)
#####################################################################################


##########################
### SETTINGS #############
##########################
parser = argparse.ArgumentParser(description='Learning to Paint')
parser.add_argument('--actor', default='./actor.pkl', type=str, help='Actor model')
parser.add_argument('--renderer', default='./renderer.pkl', type=str, help='renderer model')
parser.add_argument('--divide', default=4, type=int, help='divide the target image to get better resolution')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
learning_rate = 0.0005
num_epochs = 2000
batch_size = 1
width = 128

# Transforms for our input images
img_transform = transforms.Compose([
    transforms.Resize(size=(width, width)),
    transforms.ToTensor()
])
    
# Loading Dataset of celebrity face images
#dataset = ImageFolder('C:\\Users\\Shawn\\Desktop\\NYU\\LTP_SVG\\one_image', transform=img_transform) # laptop
dataset = ImageFolder('/home/so1463/LearningToPaint/baseline/one_image', transform=img_transform) # cluster
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# AutoEncoder model
model = AutoEncoder().to(device)
#if os.path.exists('./AutoEncoder.pth'):
#    model.load_state_dict(torch.load('/home/so1463/LearningToPaint/baseline/AutoEncoder.pth'))





###############################
# or use the with no_grad block in our loop when we call the save_image function

# Freeze weights of the renderer
renderer = FCN().to(device)
renderer.load_state_dict(torch.load(args.renderer))
renderer = renderer.to(device).eval()
for p in renderer.parameters():
    p.requires_grad = True

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate)             
criterion = nn.MSELoss(reduction='sum')

loss_plot = []
###############################################################################


#################################
# Training ######################
#################################
imgid = 1
for epoch in range(num_epochs): # each training epoch

    for data in dataloader: # for each image
        optimizer.zero_grad() # zeros the gradient, must do every iteration
        image, _=data
        
        # Change this to the non-deprecated version, get rid of Variable!
        img = Variable(image, requires_grad=True).to(device) 
        
        stroke_matrix = model(img)
        
        canvas = torch.zeros([1, 3, width, width]).to(device)
        # Calling our neural renderer
        canvas, res = decode(stroke_matrix, canvas) 
        
        save_img(canvas, imgid)
        imgid += 1
                
        loss = criterion(canvas, img)
        print('loss', loss)
        loss_plot.append(loss)
        
        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        
        # Calling the step function on an Optimizer makes an update to its parameters
        optimizer.step()
       
        
#        for j in range(len(res)):
#            save_img(res[j], imgid)
#            imgid += 1

    print('Epoch {} Finished ############################'.format(epoch))

print('Training Complete')
###############################################################################


#########################
# Saving the model#######
#########################
#torch.save(model, './AutoEncoder.pth')
torch.save(model.state_dict(), './AutoEncoder.pth')

#########################
# Plotting loss #########
#########################
plt.plot(loss_plot)
plt.ylabel('loss')
plt.savefig('./loss_graph.png')

