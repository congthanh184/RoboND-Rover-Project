## Project: Search and Sample Return

---

**The goals / steps of this project are the following:**  

**Training / Calibration**  

* Download the simulator and take data in "Training Mode"
* Test out the functions in the Jupyter Notebook provided
* Add functions to detect obstacles and samples of interest (golden rocks)
* Fill in the `process_image()` function with the appropriate image processing steps (perspective transform, color threshold etc.) to get from raw images to a map.  The `output_image` you create in this step should demonstrate that your mapping pipeline works.
* Use `moviepy` to process the images in your saved dataset with the `process_image()` function.  Include the video you produce as part of your submission.

**Autonomous Navigation / Mapping**

* Fill in the `perception_step()` function within the `perception.py` script with the appropriate image processing functions to create a map and update `Rover()` data (similar to what you did with `process_image()` in the notebook). 
* Fill in the `decision_step()` function within the `decision.py` script with conditional statements that take into consideration the outputs of the `perception_step()` in deciding how to issue throttle, brake and steering commands. 
* Iterate on your perception and decision function until your rover does a reasonable (need to define metric) job of navigating and mapping.  

[//]: # (Image References)

[image1]: ./misc/rover_image.jpg
[image2]: ./calibration_images/example_grid1.jpg
[image3]: ./calibration_images/example_rock1.jpg 
[img-classify-obs]: ./classify-obs-navi.png
[img-classify-rock]: ./classify-rock.png
[img-result]: ./result-notebook.png
[img-settings]: ./simulation-settings.png
[img-nav-angle]: ./nav-angle.png
[img-sim-result]: ./roversim-result.png

## [Rubric](https://review.udacity.com/#!/rubrics/916/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Notebook Analysis
#### 1. Run the functions provided in the notebook on test images (first with the test data provided, next on data you have recorded). Add/modify functions to allow for color selection of obstacles and rock samples.

* A rgb color threshold with value `(150, 150, 160)` can help to select the navigable terrain. Next, an inverted image of navigable terrain will yield the obstalces. Python has many ways to invert a binary image, and one way is subtracting the binary image by one.

```python
# Find the navigable_threshed
navigable_threshed = color_thresh(warped)
# Invert the color from navigable threshed
obstacles_threshed = 1 - navigable_threshed
```
![alt text][img-classify-obs]

* To select a rock, one way to do is to select pixels which has rgb colors stay in certain ranges. It can be easier and more accuracy to perform color threshold on HSV image.

```python
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
```
![alt text][img-classify-rock]


#### 2. Populate the `process_image()` function with the appropriate analysis steps to map pixels identifying navigable terrain, obstacles and rock samples into a worldmap.  Run `process_image()` on your test data using the `moviepy` functions provided to create video output of your result. 

Below is the step-by-step to implement `process_image()`

* It starts with applying the perpective transform on the image taken from rover's camera. warped picture provides the map in rover's centric pixel coordinates

```python
warped = perspect_transform(img, source, destination)
```

* Color threshold can be applied to select navigable area, obstacles areas ans rocks.

```python
navigable_threshed = color_thresh(warped)
obstacles_threshed = 1 - navigable_threshed
rock_threshed = color_in_range(warped)
```

* The rover-centric pixel value can be converted to world coordinates using the current position of rover and its yaw angle. The convert function will first convert the threshed image pixels to rover-centric coordinates, then turn them to world coordinate.

```python
# Define a helper function to retrieve world_coords from the binary_map
def get_world_coords_from_binary_map(threshed, rover_xpos, rover_ypos, rover_yaw, worldmap, scale):
    # Input
    # threshed: the binary map
    # rover_xpos, rover_ypos, rover_yaw: the current position and yaw angle of rover
    # worldmap, scale: world map and scale we use in settings
    # Ouput 
    # x_world, y_world: world coords of rover after process
    
    # Convert thresholded image pixel values to rover-centric coords
    xpix, ypix = rover_coords(threshed)
    x_world, y_world = pix_to_world(xpix, ypix, rover_xpos, 
        rover_ypos, rover_yaw, 
        worldmap.shape[0], scale)
    return x_world, y_world
```

* With the coordinates, world map can be updated. Each time a pixel is classified as navigable, or a rock, or an obstacle, its according belief state will be increased. The higher the state, the more confident the type of that position is. 

```python
data.worldmap[rock_y_world, rock_x_world, 1] += 1
data.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
data.worldmap[navigable_y_world, navigable_x_world, 2] += 1
``` 

* Finally, all vital images are put together in a mosaic image for testing and tuning.

![alt text][img-result]

### Autonomous Navigation and Mapping

#### 1. Fill in the `perception_step()` (at the bottom of the `perception.py` script) and `decision_step()` (in `decision.py`) functions in the autonomous mapping scripts and an explanation is provided in the writeup of how and why these functions were modified as they were.

##### `perception_step()`

The `perception_step()` is modified according to the 'process_image()' in notebook. However, there are some updates.

* A set of constants is declared to transform and convert the camera's image

```python
# The destination box will be 2*dst_size on each side
dst_size = 5
# Set a bottom offset to account for the fact that the bottom of the image 
# is not the position of the rover but a bit in front of it
bottom_offset = 6
source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
destination = np.float32([[Rover.img.shape[1]/2 - dst_size, Rover.img.shape[0] - bottom_offset],
              [Rover.img.shape[1]/2 + dst_size, Rover.img.shape[0] - bottom_offset],
              [Rover.img.shape[1]/2 + dst_size, Rover.img.shape[0] - 2*dst_size - bottom_offset], 
              [Rover.img.shape[1]/2 - dst_size, Rover.img.shape[0] - 2*dst_size - bottom_offset],
              ])

```

```python
# set scale of map
scale = 10
```

* It also updates the vision image which is displayed on left side of screen. Note that as the threshed image are binary image, it requires to be scaled in order to clearly visualize the color in the vision image

```python
Rover.vision_image[:,:,0] = obstacles_threshed * 150
Rover.vision_image[:,:,1] = rock_threshed * 255
Rover.vision_image[:,:,2] = navigable_threshed * 255
```

* Navigable angles and distances also calculated from the navigable threshed image. The angles will help the `decision_step()`

```python
# First calculate pixel x,y of rover-centric coords
xpix, ypix = rover_coords(navigable_threshed)
# Then calculate angles and distances to all positive pixel
dist, angles = to_polar_coords(xpix, ypix)
```

![alt text][img-nav-angle]

* To improve the fidelity, the world map will only update when the rover's motion is considered as stable. The stable condition is determined when the pitch and roll of rover are under thresholds. 

```python
def is_motion_stable(pitch, roll, threshold=(0.3, 0.3)):
    return (abs(pitch) < threshold[0]) & (abs(roll) < threshold[1])
```

* Update the world map

```python
if (is_motion_stable(Rover.pitch, Rover.roll)):
    Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
    Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1
Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
```

#### `decision_step()` and `drive_rover.py`

* To smooth the rover's motion and prevent it to moving around an area forever, a modification can be made to stop rover when its navigable angle become too large. Also, when `nav_angles` is not significant, it can be ignored so rover will not make small adjustment all the time 

```python
# when the steering angle becomes too large
# it better to be stopped and rotates before 
# back to moving forward
if abs(nav_angles) > over_steering_angle:
    Rover.brake = Rover.brake_set
    Rover.mode = 'stop'
# add a filter to smooth the motion
elif abs(nav_angles) > smoothing_angle:
    # Set steering to average angle clipped to the range +/- 15
    Rover.steer = np.clip(nav_angles, -15, 15)
```

* With the change above, it requires another change in condition to switch back `stop` mode to `forward` mode

```python
# If we're stopped but see sufficient navigable terrain in front and the nav_angles not too steep then go!
if ((len(Rover.nav_angles) >= Rover.go_forward) and (abs(nav_angles) < over_steering_angle)): 
    # Set throttle back to stored value
    Rover.throttle = Rover.throttle_set
    # Release the brake
    Rover.brake = 0
    # Set steer to mean angle
    Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
    Rover.mode = 'forward'
```

* The threshold for initiate stopping and return to forward also be modified to improve the movement

```python
self.stop_forward = 500 # Threshold to initiate stopping
self.go_forward = 1700 # Threshold to go forward again
```

#### 2. Launching in autonomous mode your rover can navigate and map autonomously.  Explain your results and how you might improve them in your writeup.  

##### Simulator settings

* The simulation is run with **1024 x 768** resolution amd **Good** Graphic Quality, and the FPS is about **34 ~ 38** 

![alt text][img-settings]

##### Result

* Rover can map successfully more than **40%** with **fidelity > 80%**
* Rover can locate the position of rocks and occasionally pick it up
* Rover can miss small roads and repeatedly visit a road it already passed
* Rover is unable to pick up rock at this moment

![alt text][img-sim-result]

##### Improvement

* Convert images to HSV format to enhance the classify process
* Implement following wall mode to improve exploring speed and pick up rock
* Currently, the world map only be updated by positive increase value. It can be improved by implementing a reduce when there is a conflict between current result and the map. For example, beside increase the confidence of obstacle and nagivable area, the obstacle map also reduces values of pixel which is considered as navigable and vice versa.

```python
# Increase confidence at obstacle pixels and navigable pixels
Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1
# Decrease confidence from obstacle map at navigable pixels and 
# vice versa
Rover.worldmap[obstacle_y_world, obstacle_x_world, 2] -= 1
Rover.worldmap[navigable_y_world, navigable_x_world, 0] -= 1
```




