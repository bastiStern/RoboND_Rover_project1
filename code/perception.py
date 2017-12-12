import numpy as np
import cv2
import operator

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, mode, rgb_thresh=(160, 160, 160)):
    # create a dict of he possible modes
    modes ={'>':operator.gt, '<':operator.lt}
    #  make sure mode is valid
    assert mode == '>' or mode == '<'
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = modes[mode](img[:,:,0], rgb_thresh[0])\
                 & modes[mode](img[:,:,1], rgb_thresh[1])\
                 & modes[mode](img[:,:,2], rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

def find_color_object(img, lower_bound, upper_bound):
    # Convert BGR to HSV
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # define limits
    low = np.array(lower_bound)
    up = np.array(upper_bound)
    # create a mask
    result = cv2.inRange(hsv_img, low, up)
    return result

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

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    mask = cv2.warpPerspective(np.ones_like(img[:,:,0]), M, (img.shape[1], img.shape[0]))
    return warped, mask

def custom_mask(origin_mask, axis, mode, limit ):
    # mode can be: 
    # low (lowpass of the axis)
    # high (highpass of the axis)
    # band (bandpass of the axis, uses limit as width of band)
    modes = ['low', 'high', 'band']
    mode = mode.lower()
    assert mode in modes
    
    # check axis
    axis = axis.lower()
    if axis == 'y':
        # apply mask
        mask = np.zeros_like(origin_mask[:,:])
        if mode == 'low':
            width = round((origin_mask.shape[0] / 100) * limit)
            mask[width:,:] = 1
            pass
        elif mode == 'high':
            width = round((origin_mask.shape[0] / 100) * (100 - limit))
            mask[0:width,:] = 1
        elif mode == 'band':
            width = round((origin_mask.shape[0] / 100) * limit)
            start = round(origin_mask.shape[0]/2 - width/2)
            end = round(origin_mask.shape[0]/2 + width/2)
            mask[start:end,:] = 1
    elif axis == 'x':
        # apply mask
        mask = np.zeros_like(origin_mask[:,:])
        if mode == 'low':
            width = round((origin_mask.shape[1] / 100) * limit)
            mask[:,width:] = 1
            pass
        elif mode == 'high':
            width = round((origin_mask.shape[1] / 100) * (100 - limit))
            mask[:,0:width] = 1
        elif mode == 'band':
            width = round((origin_mask.shape[1] / 100) * limit)
            start = round(origin_mask.shape[1]/2 - width/2)
            end = round(origin_mask.shape[1]/2 + width/2)
            mask[:,start:end] = 1       
    else:
        print('Not a valid axis specification must be x,y')
        raise
    return mask  

# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # TODO: 
    # NOTE: camera image is coming to you in Rover.img
    image = Rover.img
    # Define source and destination points for perspective transform
    dst_size = 5 
    bottom_offset = 6
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[image.shape[1]/2 - dst_size, image.shape[0] - bottom_offset],
                      [image.shape[1]/2 + dst_size, image.shape[0] - bottom_offset],
                      [image.shape[1]/2 + dst_size, image.shape[0] - 2*dst_size - bottom_offset], 
                      [image.shape[1]/2 - dst_size, image.shape[0] - 2*dst_size - bottom_offset],
                      ])
 
    # Apply perspective transform
    warped, mask = perspect_transform(image, source, destination) 
    
    # Apply color threshold to identify navigable terrain/obstacles/rock samples
    navigable = color_thresh(warped, '>')
    obstacles = color_thresh(warped, '<')
    rocks = find_color_object(warped,[5, 80, 80], [100, 255, 255])
    # make some copies for the rover navigation 
    rover_nav = np.copy(navigable)
    rover_obs = np.copy(obstacles)
    rover_obs_mask = np.copy(mask)
    # limit the field of view for mapping purposes depending on actual pitch and roll
    pitch = abs(Rover.pitch)
    roll = abs(Rover.roll)
    
    pitch_error = False
    if not ((pitch >= 359 and pitch <= 360) or (pitch >= 0 and pitch <= 1)):
        print('pitch error: ',pitch)
        pitch_error = True
       
    roll_error = False
    if not ((roll >= 359 and roll <= 360) or (roll >= 0 and roll <= 1)):
        print('roll error: ',roll)
        roll_error = True
     
    if pitch_error and not roll_error:
        map_mask_1 = custom_mask(mask, 'y', 'low', 95)
        map_mask_2 = custom_mask(mask, 'x', 'band', 10)
    elif roll_error and not pitch_error:    
        map_mask_1 = custom_mask(mask, 'y', 'low', 60)
        map_mask_2 = custom_mask(mask, 'x', 'band', 5)
    elif roll_error and pitch_error:    
        map_mask_1 = custom_mask(mask, 'y', 'low', 95)
        map_mask_2 = custom_mask(mask, 'x', 'band', 8)
    else:
        map_mask_1 = custom_mask(mask, 'y', 'low', 60)
        map_mask_2 = custom_mask(mask, 'x', 'band', 20)
     
    navigable *= map_mask_1
    navigable *= map_mask_2
    obstacles *= map_mask_1
    obstacles *= map_mask_2
    
    # limit the field of view for navigation purposes
    nav_mask_1 = custom_mask(mask, 'y', 'low', 50)
    rover_nav *= nav_mask_1
    
    # adapt a blindness on the left "eye" if no obstacles are straight ahead
    obs_mask_1 = custom_mask(mask, 'y', 'low', 75)
    obs_mask_2 = custom_mask(mask, 'x', 'band', 2)
    obs_mask_3 = custom_mask(mask, 'y', 'high', 10)
    rover_obs *= obs_mask_1
    rover_obs *= obs_mask_2
    rover_obs *= obs_mask_3
    rover_obs_mask *= obs_mask_1
    rover_obs_mask *= obs_mask_2
    rover_obs_mask *= obs_mask_3
        
    if rover_obs.any():
        print('obstacle ahead no blindness')
    else:
        vel = Rover.vel
        if vel < 0:
            vel = 0 
            
        blindness = vel * 40
        if blindness > 45:
            blindness = 45
            
        nav_mask_2 = custom_mask(mask, 'x', 'low', blindness)
        rover_nav *= nav_mask_2
        print('blindness: ', blindness)

    # Convert map image pixel values to rover-centric coords
    x_nav_map, y_nav_map = rover_coords(navigable)
    x_obs_map, y_obs_map = rover_coords(obstacles)
    x_nav_rover, y_nav_rover = rover_coords(rover_nav)
    x_obs_rover, y_obs_rover = rover_coords(rover_obs)
    
    # Convert rover-centric pixel values to world coordinates
    x_nav_world, y_nav_world = pix_to_world(x_nav_map, y_nav_map,
                                            Rover.pos[0],
                                            Rover.pos[1],
                                            Rover.yaw,
                                            Rover.worldmap.shape[0],
                                            dst_size*2)
    
    x_obs_world, y_obs_world = pix_to_world(x_obs_map, y_obs_map,
                                            Rover.pos[0],
                                            Rover.pos[1],
                                            Rover.yaw,
                                            Rover.worldmap.shape[0],
                                            dst_size*2)
                                 
    # Update Rover worldmap (to be displayed on right side of screen)
    Rover.worldmap[y_obs_world, x_obs_world, 0] += 1
    Rover.worldmap[y_nav_world, x_nav_world, 2] += 10 
    
    # check for rocks
    if rocks.any:
        x_rocks_map, y_rocks_map = rover_coords(rocks)
        
        x_rocks_world, y_rocks_world = pix_to_world(x_obs_map, y_obs_map,
                                          Rover.pos[0],
                                          Rover.pos[1],
                                          Rover.yaw,
                                          Rover.worldmap.shape[0],
                                          dst_size*2)
        
        Rover.worldmap[y_rocks_world, x_rocks_world, 1] = 255  
        Rover.vision_image[:,:,1] = rocks * 255
        
    # convert rover-centric pixel positions to polar coordinates
    xpix, ypix = rover_coords(rover_nav)
    dist, angles = to_polar_coords(xpix, ypix)
    # Update Rover pixel distances and angles
    Rover.nav_dists = dist
    Rover.nav_angles = angles
    
    # Update Rover.vision_image (this will be displayed on left side of screen)
    Rover.vision_image[:,:,2] = rover_nav * 255
    Rover.vision_image[:,:,0] = rover_obs_mask * 255 
    Rover.vision_image[:,:,1] = rover_obs * 255 
  
    return Rover