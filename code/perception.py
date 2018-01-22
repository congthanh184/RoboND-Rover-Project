import numpy as np
import cv2

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
                            
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result  
    return xpix_rotated, ypix_rotated

def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result  
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

def color_in_range(img, rgb_min=(135, 115, 0), rgb_max=(200, 155, 30)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be in the ranges established by threshold max and min values in RGB
    # in_thresh will now contain a boolean array with "True"
    # where threshold was met
    in_thresh = (img[:,:,0] > rgb_min[0]) & (img[:,:,0] < rgb_max[0]) \
                & (img[:,:,1] > rgb_min[1]) & (img[:,:,1] < rgb_max[1]) \
                & (img[:,:,2] > rgb_min[2]) & (img[:,:,2] < rgb_max[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[in_thresh] = 1
    # Return the binary image
    return color_select

def get_world_coords_from_binary_map(threshed, rover_xpos, rover_ypos, rover_yaw, worldmap, scale):
    xpix, ypix = rover_coords(threshed)
    x_world, y_world = pix_to_world(xpix, ypix, rover_xpos, 
        rover_ypos, rover_yaw, 
        worldmap.shape[0], scale)
    return x_world, y_world

# Determine if the current situation is suitable to update the world map
# by checking the pitch and roll angle of rover
def is_motion_stable(pitch, roll, threshold=(0.3, 0.3)):
    return (abs(pitch) < threshold[0]) & (abs(roll) < threshold[1])

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    
    return warped


# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # TODO: 
    # NOTE: camera image is coming to you in Rover.img
    # 1) Define source and destination points for perspective transform
    dst_size = 5
    bottom_offset = 6
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[Rover.img.shape[1]/2 - dst_size, Rover.img.shape[0] - bottom_offset],
                  [Rover.img.shape[1]/2 + dst_size, Rover.img.shape[0] - bottom_offset],
                  [Rover.img.shape[1]/2 + dst_size, Rover.img.shape[0] - 2*dst_size - bottom_offset], 
                  [Rover.img.shape[1]/2 - dst_size, Rover.img.shape[0] - 2*dst_size - bottom_offset],
                  ])
    # 2) Apply perspective transform
    warped = perspect_transform(Rover.img, source, destination)
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    navigable_threshed = color_thresh(warped)
    obstacles_threshed = 1 - navigable_threshed
    rock_threshed = color_in_range(warped)
    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
        # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
        #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
        #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image
    Rover.vision_image[:,:,0] = obstacles_threshed * 150
    Rover.vision_image[:,:,1] = rock_threshed * 255
    Rover.vision_image[:,:,2] = navigable_threshed * 255
    # 5) Convert map image pixel values to rover-centric coords
    xpix, ypix = rover_coords(navigable_threshed)
    dist, angles = to_polar_coords(xpix, ypix)
    mean_dir = np.mean(angles)
    # 6) Convert rover-centric pixel values to world coordinates
    rover_xpos, rover_ypos = Rover.pos 
    rover_yaw = Rover.yaw
    worldmap = Rover.worldmap
    scale = 10
    navigable_x_world, navigable_y_world = get_world_coords_from_binary_map(navigable_threshed, rover_xpos, 
        rover_ypos, rover_yaw, 
        worldmap, scale)
    obstacle_x_world, obstacle_y_world = get_world_coords_from_binary_map(obstacles_threshed, rover_xpos, 
        rover_ypos, rover_yaw, 
        worldmap, scale)    
    rock_x_world, rock_y_world = get_world_coords_from_binary_map(rock_threshed, rover_xpos, 
        rover_ypos, rover_yaw, 
        worldmap, scale)
    # 7) Update Rover worldmap (to be displayed on right side of screen)
        # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
        #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1
    if (is_motion_stable(Rover.pitch, Rover.roll)):
        Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1
    Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
    # 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
        # Rover.nav_dists = rover_centric_pixel_distances
        # Rover.nav_angles = rover_centric_angles
    Rover.nav_dists = dist
    Rover.nav_angles = angles
    
    
    return Rover