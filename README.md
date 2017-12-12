[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)
# Search and Sample Return Project

[//]: # (Image References)
[image1]: ./pics/nav_obs.png
[image2]: ./pics/rocks.png
[image3]: ./pics/filter.png
[image4]: ./pics/blind.png

### Data Analysis
In the first step i created and customized some existing functions within the jupyter notebook `Rover_Project_Test_Notebook.ipynb`. Mainly to detect obstacles and rocks in the path of the rover. Furthermore i've written a function to create a custom mask to restrict the field of view.

#### 1.Detection of obstacles
The detection of obstacles is based on the function `color_thresh()` i introduced a new parameter to select between pixels above ('>') or below ('<') the threshold the result looks like this
![alt text][image1]

#### 2.Detection of rocks
For the detection of the colored rocks i created a new function `find_color_object()` it uses the Open CV inRange function to create a mask of those pixels which are in the specified range between the lower and the upper limit.
![alt text][image2]

#### 3.Custom masking
The function `custom_mask()` is used to adapt the rovers field of view even further. It takes four parameters: 
1. origin_mask is used to create a duplicate mask with the original shape
2. axis defines the image axis you want to modify
3. mode can be low, band, or high 
4. limit is the number responsible for the size of the filter

using two filters on the result with the navigable terrain get us the following result

```python
threshed = color_thresh(warped, '>')
test_1 = np.copy(threshed)
mask_1 = custom_mask(mask, 'y', 'band', 50)
mask_2 = custom_mask(mask, 'x', 'band', 20)
test_1 *= mask_1
test_1 *= mask_2
```
![alt text][image3]

### Autonomous Navigation and Mapping
I divided the autonomous driving in two simple tasks
1. Effectively navigate through the map without visiting areas twice
2. Minimize errors within the mapping process to reach a high accuracy
In order to work on both tasks independently i used a dataset for mapping purposes and one for navigation
```python
# Apply color threshold to identify navigable 
# terrain/obstacles/rock samples
navigable = color_thresh(warped, '>')
obstacles = color_thresh(warped, '<')
rocks = find_color_object(warped,[5, 80, 80], [100, 255, 255])

# make some copies for the rover navigation 
rover_nav = np.copy(navigable)
rover_obs = np.copy(obstacles)
```
#### 1.Effecitive navigation
To achieve a effective way of traveling across the map i decided to limit the 
field of view of the rover on the left side. The "left eye blindness" is achieved 
with the `custom_mask()` function mentioned above. In order to correctly navigate when 
switching from stop to forward mode the bindness is dependent on the actual velocity of the rover.
This code is from my `perception_step()` inside `perception.py`.
```python
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
```
To prevent the rover from crashing into obstacles on the left side i created a small zone in 
front of the rover to check for obstacles. I called it obstacle detector. 
```python
obs_mask_1 = custom_mask(mask, 'y', 'low', 75)
obs_mask_2 = custom_mask(mask, 'x', 'band', 2)
obs_mask_3 = custom_mask(mask, 'y', 'high', 10)
rover_obs *= obs_mask_1
rover_obs *= obs_mask_2
rover_obs *= obs_mask_3
```
These are the results of the blindness mask and the masked zone to check for obstacels. 
![alt text][image4]
Check the [autonomous_driving](https://www.youtube.com/watch?v=Ui_xZ8EhY9w) video for the adaptive blindness in action.
With this simple method the rover maps over 95% of the map without visiting any areas twice.

#### 2.Minimize mapping errors
In order to minimize the errors whilst mapping the terrain i mainly used the `custom_mask()` function again. 
The most error occured when the pitch and roll of the rover where diverging from zero.
So first i checked the actual pitch and role of the rover.
```python
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
```
After that is was easy to adapt the mappable terrain with some masking
```python
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
```
As you can clearly see i made the field of view shorter if the pitch was out of range and 
more like a tunnel if the roll was out of range. With this simple adjustment i was able to 
raise the fidelity from ~60% to above 80%.

### Improvements
In some cases the rover overlooks the small junction in the southern part of the map due the 
obstacle detector in front of it. In order to prevent this it could help to bend the mask 
which defines the small area in front of the rover dependent on the current steering angle of the rover.










  



