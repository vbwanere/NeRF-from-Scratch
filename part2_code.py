import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import time
from part1_code import *

def get_rays(height, width, intrinsics, Rcw, Tcw):
    
    """
    Compute the origin and direction of rays passing through all pixels of an image (one ray per pixel).

    Args:
    height: the height of an image.
    width: the width of an image.
    intrinsics: camera intrinsics matrix of shape (3, 3).
    Rcw: Rotation matrix of shape (3,3) from camera to world coordinates.
    Tcw: Translation vector of shape (3,1) that transforms

    Returns:
    ray_origins (torch.Tensor): A tensor of shape (height, width, 3) denoting the centers of
      each ray. Note that desipte that all ray share the same origin, here we ask you to return 
      the ray origin for each ray as (height, width, 3).
    ray_directions (torch.Tensor): A tensor of shape (height, width, 3) denoting the
      direction of each ray.
    """

    device = intrinsics.device
    ray_directions = torch.zeros((height, width, 3), device=device)  # placeholder
    ray_origins = torch.zeros((height, width, 3), device=device)  # placeholder
    
    #############################  TODO 2.1 BEGIN  ##########################  

    u = torch.arange(width) # this is 100
    v = torch.arange(height)
    v_grid, u_grid = torch.meshgrid(u, v)
    
    ones = torch.ones((height, width, 1))
    coords_tensor = torch.cat([u_grid[..., None], v_grid[..., None], ones], dim=-1)
    pixels = coords_tensor.reshape(height*width, 3).T
    K_inv = torch.inverse(intrinsics)
    rays_camera = torch.matmul(K_inv, pixels)
    
    ray_directions = torch.matmul(Rcw, rays_camera).T.reshape(height, width, 3)  

    ray_origins = torch.tile(Tcw.reshape(1, 3), (height, height, 1))

    #############################  TODO 2.1 END  ############################
    return ray_origins, ray_directions

def stratified_sampling(ray_origins, ray_directions, near, far, samples):

    """
    Sample 3D points on the given rays. The near and far variables indicate the bounds of sampling range.

    Args:
    ray_origins: Origin of each ray in the "bundle" as returned by the
      get_rays() function. Shape: (height, width, 3).
    ray_directions: Direction of each ray in the "bundle" as returned by the
      get_rays() function. Shape: (height, width, 3).
    near: The 'near' extent of the bounding volume.
    far:  The 'far' extent of the bounding volume.
    samples: Number of samples to be drawn along each ray.
  
    Returns:
    ray_points: Query 3D points along each ray. Shape: (height, width, samples, 3).
    depth_points: Sampled depth values along each ray. Shape: (height, width, samples).
    """

    #############################  TODO 2.2 BEGIN  ############################
    H, W = ray_origins.shape[:2]
    N = samples
    # Generate stratified samples
    t = (torch.rand(H, W, samples, device=ray_origins.device)\
                 + torch.arange(samples, device=ray_origins.device)) / (samples + 1e-6)
    t = t.expand(H, W, samples)
    
    depth_points = near*(1.0-t) + far*t

    ray_points = ray_origins[..., None, :] + depth_points[..., None] * ray_directions[..., None, :]

    #############################  TODO 2.2 END  ############################
    return ray_points, depth_points
    
class nerf_model(nn.Module):
    
    """
    Define a NeRF model comprising eight fully connected layers and following the
    architecture described in the NeRF paper. 
    """

    def __init__(self, filter_size=256, num_x_frequencies=6, num_d_frequencies=3):
        super().__init__()

        #############################  TODO 2.3 BEGIN  ############################
        n_gamma_x = 3 + 2*3*num_x_frequencies #(2DL: D= dimension of the input, L= no of frequency)
        n_gamma_d = 3 + 2*3*num_d_frequencies

        # Regular layers:
        self.layer_1 = nn.Linear(n_gamma_x, filter_size) # 1st linear layer
        self.layer_2 = nn.Linear(filter_size, filter_size) # 2nd linear layer
        self.layer_3 = nn.Linear(filter_size, filter_size) # 3rd linear layer
        self.layer_4 = nn.Linear(filter_size, filter_size) # 4th linear layer
        self.layer_5 = nn.Linear(filter_size, filter_size) # 5th linear layer

        # Concatenate the input pos with fifth layer and pass it to sixth layer. ?
        self.layer_6 = nn.Linear(filter_size + n_gamma_x, filter_size) # 6th linear layer
        self.layer_7 = nn.Linear(filter_size, filter_size) # 7th linear layer
        self.layer_8 = nn.Linear(filter_size, filter_size) # 8th linear layer

        # Get density: sigma (a scalar value) as the output of 9th layer
        self.layer_9 = nn.Linear(filter_size, 1) # 9th linear layer

        # Regular layers
        self.layer_10 = nn.Linear(filter_size, filter_size) # 10th linear layer

        # Concatenate input direction with tenth layer and pass it to eleventh layer. ?
        self.layer_11 = nn.Linear(filter_size + n_gamma_d, 128) # 11th linear layer
        self.layer_12 = nn.Linear(128, 3)
        #############################  TODO 2.3 END  ############################


    def forward(self, x, d):
        #############################  TODO 2.3 BEGIN  ############################

        # Regular layers:
        l_1 = F.relu(self.layer_1(x))
        l_2 = F.relu(self.layer_2(l_1))
        l_3 = F.relu(self.layer_3(l_2))
        l_4 = F.relu(self.layer_4(l_3))
        l_5 = F.relu(self.layer_5(l_4))

        # Concatenate l5 with input again:
        l_5_cat = torch.cat([l_5, x], dim=-1)

        #Regular layers:
        l_6 = F.relu(self.layer_6(l_5_cat))
        l_7 = F.relu(self.layer_7(l_6))
        l_8 = F.relu(self.layer_8(l_7))

        # Get sigma from eighth layer:
        sigma = self.layer_9(l_8)

        # Regular layers: No activation!
        l_10 = self.layer_10(l_8)

        # Concatenate input direction with tenth layer and pass it to eleventh layer.
        l_10_cat = torch.cat([l_10, d], dim=-1)
        l_11 = F.relu(self.layer_11(l_10_cat))

        # Get rgb:
        rgb = torch.sigmoid(self.layer_12(l_11))

        #############################  TODO 2.3 END  ############################
        return rgb, sigma

def get_batches(ray_points, ray_directions, num_x_frequencies, num_d_frequencies):
    
    def get_chunks(inputs, chunksize = 2**15):
        return [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]
    
    """
    This function returns chunks of the ray points and directions to avoid memory errors with the
    neural network. It also applies positional encoding to the input points and directions before 
    dividing them into chunks, as well as normalizing and populating the directions.
    """
    #############################  TODO 2.3 BEGIN  ############################
  
    # normalize directions
    ray_directions_norm = ray_directions / torch.norm(ray_directions, dim=-1, keepdim=True)

    # repeat directions for every point
    ray_directions_pop = ray_directions_norm.unsqueeze(-2).repeat(1, 1, ray_points.shape[2], 1)

    # flatten vectors
    ray_points_flat = ray_points.view(-1, 3) #(H * W * N of samples, 3)
    ray_directions_flat = ray_directions_pop.view(-1, 3) #(H * W * N of samples, 3)

    # Apply positional encoding:
    en_ray_directions_flat = positional_encoding(ray_directions_flat, num_d_frequencies)
    en_ray_points_flat = positional_encoding(ray_points_flat, num_x_frequencies)

    # Call the get_chunks function:
    ray_points_batches = get_chunks(en_ray_points_flat)
    ray_directions_batches = get_chunks(en_ray_directions_flat)

    #############################  TODO 2.3 END  ############################

    return ray_points_batches, ray_directions_batches

def volumetric_rendering(rgb, s, depth_points):

    """
    Differentiably renders a radiance field, given the origin of each ray in the
    "bundle", and the sampled depth values along them.

    Args:
    rgb: RGB color at each query location (X, Y, Z). Shape: (height, width, samples, 3).
    sigma: Volume density at each query location (X, Y, Z). Shape: (height, width, samples).
    depth_points: Sampled depth values along each ray. Shape: (height, width, samples).
  
    Returns:
    rec_image: The reconstructed image after applying the volumetric rendering to every pixel.
    Shape: (height, width, 3)
    """
    
    #############################  TODO 2.4 BEGIN  ############################
    device = rgb.device

    # Transmittance:
    delta = torch.ones_like(depth_points).to(device) * (1e09)
    delta[...,:-1] = torch.diff(depth_points, dim=-1)
    sigma_dot_delta = - F.relu(s) * delta.reshape_as(s)
    T = torch.cumprod(torch.exp(sigma_dot_delta), dim = -1)
    T = torch.roll(T,1,dims=-1)

    # Apply the volumetric rendering equation to each ray:
    C = ((T * (1 - torch.exp(sigma_dot_delta)))[..., None]) * rgb
    rec_image = torch.sum(C, dim=-2)

    #############################  TODO 2.4 END  ############################

    return rec_image

def one_forward_pass(height, width, intrinsics, pose, near, far, samples, model, num_x_frequencies, num_d_frequencies):
    
    #############################  TODO 2.5 BEGIN  ############################

    #compute all the rays from the image


    #sample the points from the rays


    #divide data into batches to avoid memory errors


    #forward pass the batches and concatenate the outputs at the end
    





    # Apply volumetric rendering to obtain the reconstructed image


    #############################  TODO 2.5 END  ############################

    return rec_image