# watermark
An attempt to remove repeated watermark from a set of pictures with opencv.

# Preprocessing:
<ol>
<li>Average all pictures with watermarks and matching sizes to intensify
watermarks.</li>
<li>Laplace operator to extract regions with "intensity jumps" -> watermarks</li>
<li>Otsu for emphasizing watermarks</li>

</ol>

# Checked methods
Finding a watermark with a unified (same values) background allowed me to test below methods:

(parameters for images blending were chosen arbitrarily - alpha = 0.85, gamma = 0)
## Gradient descent
Assuming alpha and having a fixed background (R=G=B 244) find a `src2` array which after blending with the bacground will give the original sample.
- Selecting learning rate as 0.02
- [momentum](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Momentum) reduced number of iterations from > 1500 to 67 (sic!)
- Using [SSIM](https://scikit-image.org/docs/dev/auto_examples/transform/plot_ssim.html) as a stopping criteria increaded the execution time 3 times (!!!) -> helped to choose a proper error limit (stop when error < eps=0.0001) 
## Reversing addWeighted
- [addWeighted](https://docs.opencv.org/3.4/d5/dc4/tutorial_adding_images.html) function from OpenCV is shown in Google as the most popular answer about a method for blending images
- `dst=α⋅src1+β⋅src2` -> `src2 = (dst - α⋅src1)/(1-α)`
- max pixel of src2 (watermark) is 317.3 0_o
  - using `cv2.convertScaleAbs` isn't the best option

