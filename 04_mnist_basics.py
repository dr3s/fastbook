

# %%
#hide
from fastai.vision.all import *
from fastbook import *

matplotlib.rc('image', cmap='Greys')


path = untar_data(URLs.MNIST_SAMPLE)

# %%
#hide
Path.BASE_PATH = path

# %% [markdown]
# The MNIST dataset follows a common layout for machine learning datasets: separate folders for the training set and the validation set (and/or test set). Let's see what's inside the training set:

# %%
(path/'train').ls()

# %% [markdown]
# There's a folder of 3s, and a folder of 7s. In machine learning parlance, we say that "3" and "7" are the *labels* (or targets) in this dataset. Let's take a look in one of these folders (using `sorted` to ensure we all get the same order of files):

# %%
threes = (path/'train'/'3').ls().sorted()
sevens = (path/'train'/'7').ls().sorted()
threes

# %% [markdown]
# As we might expect, it's full of image files. Let’s take a look at one now. Here’s an image of a handwritten number 3, taken from the famous MNIST dataset of handwritten numbers:

# %%
im3_path = threes[1]
im3 = Image.open(im3_path)
im3


# %% [markdown]
# The `4:10` indicates we requested the rows from index 4 (included) to 10 (not included) and the same for the columns. NumPy indexes from top to bottom and left to right, so this section is located in the top-left corner of the image. Here's the same thing as a PyTorch tensor:

# %%
tensor(im3)[4:10,4:10]

# %% [markdown]
# We can slice the array to pick just the part with the top of the digit in it, and then use a Pandas DataFrame to color-code the values using a gradient, which shows us clearly how the image is created from the pixel values:

# %%
#hide_output
im3_t = tensor(im3)
df = pd.DataFrame(im3_t[4:15,4:22])
df.style.set_properties(**{'font-size':'6pt'}).background_gradient('Greys')


# %% [markdown]
# Step one for our simple model is to get the average of pixel values for each of our two groups. In the process of doing this, we will learn a lot of neat Python numeric programming tricks!

# %%
seven_tensors = [tensor(Image.open(o)) for o in sevens]
three_tensors = [tensor(Image.open(o)) for o in threes]
len(three_tensors),len(seven_tensors)

show_image(three_tensors[1]);

# %% [markdown]
# For every pixel position, we want to compute the average over all the images of the intensity of that pixel. To do this we first combine all the images in this list into a single three-dimensional tensor. The most common way to describe such a tensor is to call it a *rank-3 tensor*. We often need to stack up individual tensors in a collection into a single tensor. Unsurprisingly, PyTorch comes with a function called `stack` that we can use for this purpose.
# 
# Some operations in PyTorch, such as taking a mean, require us to *cast* our integer types to float types. Since we'll be needing this later, we'll also cast our stacked tensor to `float` now. Casting in PyTorch is as simple as typing the name of the type you wish to cast to, and treating it as a method.
# 
# Generally when images are floats, the pixel values are expected to be between 0 and 1, so we will also divide by 255 here:

# %%
stacked_sevens = torch.stack(seven_tensors).float()/255
stacked_threes = torch.stack(three_tensors).float()/255
stacked_threes.shape

# %% [markdown]
# Perhaps the most important attribute of a tensor is its *shape*. This tells you the length of each axis. In this case, we can see that we have 6,131 images, each of size 28×28 pixels. There is nothing specifically about this tensor that says that the first axis is the number of images, the second is the height, and the third is the width—the semantics of a tensor are entirely up to us, and how we construct it. As far as PyTorch is concerned, it is just a bunch of numbers in memory.
# 
# The *length* of a tensor's shape is its rank:

# %%
len(stacked_threes.shape)

# %% [markdown]
# It is really important for you to commit to memory and practice these bits of tensor jargon: _rank_ is the number of axes or dimensions in a tensor; _shape_ is the size of each axis of a tensor.
# 

# %% [markdown]
# We can also get a tensor's rank directly with `ndim`:

# %%
stacked_threes.ndim

# %% [markdown]
# Finally, we can compute what the ideal 3 looks like. We calculate the mean of all the image tensors by taking the mean along dimension 0 of our stacked, rank-3 tensor. This is the dimension that indexes over all the images.
# 
# In other words, for every pixel position, this will compute the average of that pixel over all images. The result will be one value for every pixel position, or a single image. Here it is:

# %%
mean3 = stacked_threes.mean(0)
show_image(mean3);

# %% [markdown]
# According to this dataset, this is the ideal number 3! (You may not like it, but this is what peak number 3 performance looks like.) You can see how it's very dark where all the images agree it should be dark, but it becomes wispy and blurry where the images disagree. 
# 
# Let's do the same thing for the 7s, but put all the steps together at once to save some time:

# %%
mean7 = stacked_sevens.mean(0)
show_image(mean7);

# %%
a_3 = stacked_threes[1]
show_image(a_3);

# %% [markdown]
# Let's try both of these now:

# %%
dist_3_abs = (a_3 - mean3).abs().mean()
dist_3_sqr = ((a_3 - mean3)**2).mean().sqrt()
dist_3_abs,dist_3_sqr

# %%
dist_7_abs = (a_3 - mean7).abs().mean()
dist_7_sqr = ((a_3 - mean7)**2).mean().sqrt()
dist_7_abs,dist_7_sqr

# %% [markdown]
# In both cases, the distance between our 3 and the "ideal" 3 is less than the distance to the ideal 7. So our simple model will give the right prediction in this case.

# %% [markdown]
# PyTorch already provides both of these as *loss functions*. You'll find these inside `torch.nn.functional`, which the PyTorch team recommends importing as `F` (and is available by default under that name in fastai):

# %%
F.l1_loss(a_3.float(),mean7), F.mse_loss(a_3,mean7).sqrt()


# %% [markdown]
# To create an array or tensor, pass a list (or list of lists, or list of lists of lists, etc.) to `array()` or `tensor()`:

# %%
data = [[1,2,3],[4,5,6]]
arr = array (data)
tns = tensor(data)

# %%
arr  # numpy

# %%
tns  # pytorch

# %% [markdown]
# All the operations that follow are shown on tensors, but the syntax and results for NumPy arrays is identical.
# 
# You can select a row (note that, like lists in Python, tensors are 0-indexed so 1 refers to the second row/column):

# %%
tns[1]

# %% [markdown]
# or a column, by using `:` to indicate *all of the first axis* (we sometimes refer to the dimensions of tensors/arrays as *axes*):

# %%
tns[:,1]

# %% [markdown]
# You can combine these with Python slice syntax (`[start:end]` with `end` being excluded) to select part of a row or column:

# %%
tns[1,1:3]

# %% [markdown]
# And you can use the standard operators such as `+`, `-`, `*`, `/`:

# %%
tns+1

# %% [markdown]
# Tensors have a type:

# %%
tns.type()

# %% [markdown]
# And will automatically change type as needed, for example from `int` to `float`:

# %%
tns*1.5

# %% [markdown]
# So, is our baseline model any good? To quantify this, we must define a metric.

# %% [markdown]
# ## Computing Metrics Using Broadcasting

# %% [markdown]
# So to start with, let's create tensors for our 3s and 7s from that directory. These are the tensors we will use to calculate a metric measuring the quality of our first-try model, which measures distance from an ideal image:

# %%
valid_3_tens = torch.stack([tensor(Image.open(o)) 
                            for o in (path/'valid'/'3').ls()])
valid_3_tens = valid_3_tens.float()/255
valid_7_tens = torch.stack([tensor(Image.open(o)) 
                            for o in (path/'valid'/'7').ls()])
valid_7_tens = valid_7_tens.float()/255
valid_3_tens.shape,valid_7_tens.shape

# %% [markdown]
# 
# We can write a simple function that calculates the mean absolute error using an expression very similar to the one we wrote in the last section:

# %%
def mnist_distance(a,b): return (a-b).abs().mean((-1,-2))
mnist_distance(a_3, mean3)

# %% [markdown]
# This is the same value we previously calculated for the distance between these two images, the ideal 3 `mean3` and the arbitrary sample 3 `a_3`, which are both single-image tensors with a shape of `[28,28]`.
#
# Something very interesting happens when we take this exact same distance function, designed for comparing two single images, but pass in as an argument `valid_3_tens`, the tensor that represents the 3s validation set:

# %%
valid_3_dist = mnist_distance(valid_3_tens, mean3)
valid_3_dist, valid_3_dist.shape

# %% [markdown]
# Instead of complaining about shapes not matching, it returned the distance for every single image as a vector (i.e., a rank-1 tensor) of length 1,010 (the number of 3s in our validation set). How did that happen?
#
# After broadcasting so the two argument tensors have the same rank, PyTorch applies its usual logic for two tensors of the same rank: it performs the operation on each corresponding element of the two tensors, and returns the tensor result. For instance:

# %%
tensor([1,2,3]) + tensor(1)

# %% [markdown]
# So in this case, PyTorch treats `mean3`, a rank-2 tensor representing a single image, as if it were 1,010 copies of the same image, and then subtracts each of those copies from each 3 in our validation set. What shape would you expect this tensor to have? Try to figure it out yourself before you look at the answer below:

# %%
(valid_3_tens-mean3).shape

# %%
def is_3(x): return mnist_distance(x,mean3) < mnist_distance(x,mean7)

# %% [markdown]
# Let's test it on our example case:

# %%
is_3(a_3), is_3(a_3).float()

# %% [markdown]
# Note that when we convert the Boolean response to a float, we get `1.0` for `True` and `0.0` for `False`. Thanks to broadcasting, we can also test it on the full validation set of 3s:

# %%
is_3(valid_3_tens)

# %% [markdown]
# Now we can calculate the accuracy for each of the 3s and 7s by taking the average of that function for all 3s and its inverse for all 7s:

# %%
accuracy_3s =      is_3(valid_3_tens).float() .mean()
accuracy_7s = (1 - is_3(valid_7_tens).float()).mean()

accuracy_3s,accuracy_7s,(accuracy_3s+accuracy_7s)/2


# %%
#id gradient_descent
#caption The gradient descent process
#alt Graph showing the steps for Gradient Descent
gv('''
init->predict->loss->gradient->step->stop
step->predict[label=repeat]
''')


# %% [markdown]
# Before applying these steps to our image classification problem, let's illustrate what they look like in a simpler case. First we will define a very simple function, the quadratic—let's pretend that this is our loss function, and `x` is a weight parameter of the function:

# %%
def f(x): return x**2

# %% [markdown]
# Here is a graph of that function:

# %%
plot_function(f, 'x', 'x**2')

# %% [markdown]
# The sequence of steps we described earlier starts by picking some random value for a parameter, and calculating the value of the loss:

# %%
plot_function(f, 'x', 'x**2')
plt.scatter(-1.5, f(-1.5), color='red');

# %%
xt = tensor(3.).requires_grad_()


# %%
yt = f(xt)
yt

# %% [markdown]
# Finally, we tell PyTorch to calculate the gradients for us:

# %%
yt.backward()

# %% [markdown]
# We can now view the gradients by checking the `grad` attribute of our tensor:

# %%
xt.grad

# %% [markdown]
# If you remember your high school calculus rules, the derivative of `x**2` is `2*x`, and we have `x=3`, so the gradients should be `2*3=6`, which is what PyTorch calculated for us!
# 
# Now we'll repeat the preceding steps, but with a vector argument for our function:

# %%
xt = tensor([3.,4.,10.]).requires_grad_()
xt

# %% [markdown]
# And we'll add `sum` to our function so it can take a vector (i.e., a rank-1 tensor), and return a scalar (i.e., a rank-0 tensor):

# %%
def f(x): return (x**2).sum()

yt = f(xt)
yt

# %% [markdown]
# Our gradients are `2*xt`, as we'd expect!

# %%
yt.backward()
xt.grad

# %% [markdown]
# We've seen how to use gradients to find a minimum. Now it's time to look at an SGD example and see how finding a minimum can be used to train a model to fit data better.
# 
# Let's start with a simple, synthetic, example model. Imagine you were measuring the speed of a roller coaster as it went over the top of a hump. It would start fast, and then get slower as it went up the hill; it would be slowest at the top, and it would then speed up again as it went downhill. You want to build a model of how the speed changes over time. If you were measuring the speed manually every second for 20 seconds, it might look something like this:

# %%
time = torch.arange(0,20).float(); time

# %%
speed = torch.randn(20)*3 + 0.75*(time-9.5)**2 + 1
plt.scatter(time,speed);

# %% [markdown]
# We've added a bit of random noise, since measuring things manually isn't precise. This means it's not that easy to answer the question: what was the roller coaster's speed? Using SGD we can try to find a function that matches our observations. We can't consider every possible function, so let's use a guess that it will be quadratic; i.e., a function of the form `a*(time**2)+(b*time)+c`.
# 
# We want to distinguish clearly between the function's input (the time when we are measuring the coaster's speed) and its parameters (the values that define *which* quadratic we're trying). So, let's collect the parameters in one argument and thus separate the input, `t`, and the parameters, `params`, in the function's signature: 

# %%
def f(t, params):
    a,b,c = params
    return a*(t**2) + (b*t) + c

# %% [markdown]
# In other words, we've restricted the problem of finding the best imaginable function that fits the data, to finding the best *quadratic* function. This greatly simplifies the problem, since every quadratic function is fully defined by the three parameters `a`, `b`, and `c`. Thus, to find the best quadratic function, we only need to find the best values for `a`, `b`, and `c`.
# 
# If we can solve this problem for the three parameters of a quadratic function, we'll be able to apply the same approach for other, more complex functions with more parameters—such as a neural net. Let's find the parameters for `f` first, and then we'll come back and do the same thing for the MNIST dataset with a neural net.
# 
# We need to define first what we mean by "best." We define this precisely by choosing a *loss function*, which will return a value based on a prediction and a target, where lower values of the function correspond to "better" predictions. It is important for loss functions to return _lower_ values when predictions are more accurate, as the SGD procedure we defined earlier will try to _minimize_ this loss. For continuous data, it's common to use *mean squared error*:

# %%
def mse(preds, targets): return ((preds-targets)**2).mean()

# %% [markdown]
# Now, let's work through our 7 step process.

# %% [markdown]
# #### Step 1: Initialize the parameters

# %% [markdown]
# First, we initialize the parameters to random values, and tell PyTorch that we want to track their gradients, using `requires_grad_`:

# %%
params = torch.randn(3).requires_grad_()

# %%
#hide
orig_params = params.clone()

# %% [markdown]
# #### Step 2: Calculate the predictions

# %% [markdown]
# Next, we calculate the predictions:

# %%
preds = f(time, params)

# %% [markdown]
# Let's create a little function to see how close our predictions are to our targets, and take a look:

# %%
def show_preds(preds, ax=None):
    if ax is None: ax=plt.subplots()[1]
    ax.scatter(time, speed)
    ax.scatter(time, to_np(preds), color='red')
    ax.set_ylim(-300,100)

# %%
show_preds(preds)

# %% [markdown]
# This doesn't look very close—our random parameters suggest that the roller coaster will end up going backwards, since we have negative speeds!

# %% [markdown]
# #### Step 3: Calculate the loss

# %% [markdown]
# We calculate the loss as follows:

# %%
loss = mse(preds, speed)
loss

# %% [markdown]
# Our goal is now to improve this. To do that, we'll need to know the gradients.

# %% [markdown]
# #### Step 4: Calculate the gradients

# %% [markdown]
# The next step is to calculate the gradients. In other words, calculate an approximation of how the parameters need to change:

# %%
loss.backward()
params.grad

# %%
params.grad * 1e-5

# %% [markdown]
# We can use these gradients to improve our parameters. We'll need to pick a learning rate (we'll discuss how to do that in practice in the next chapter; for now we'll just use 1e-5, or 0.00001):

# %%
params

# %% [markdown]
# #### Step 5: Step the weights. 

# %% [markdown]
# Now we need to update the parameters based on the gradients we just calculated:

# %%
lr = 1e-5
params.data -= lr * params.grad.data
params.grad = None

# %% [markdown]
# > a: Understanding this bit depends on remembering recent history. To calculate the gradients we call `backward` on the `loss`. But this `loss` was itself calculated by `mse`, which in turn took `preds` as an input, which was calculated using `f` taking as an input `params`, which was the object on which we originally called `requires_grad_`—which is the original call that now allows us to call `backward` on `loss`. This chain of function calls represents the mathematical composition of functions, which enables PyTorch to use calculus's chain rule under the hood to calculate these gradients.

# %% [markdown]
# Let's see if the loss has improved:

# %%
preds = f(time,params)
mse(preds, speed)

# %% [markdown]
# And take a look at the plot:

# %%
show_preds(preds)

# %% [markdown]
# We need to repeat this a few times, so we'll create a function to apply one step:

# %%
def apply_step(params, prn=True):
    preds = f(time, params)
    loss = mse(preds, speed)
    loss.backward()
    params.data -= lr * params.grad.data
    params.grad = None
    if prn: print(loss.item())
    return preds

# %% [markdown]
# #### Step 6: Repeat the process 

# %% [markdown]
# Now we iterate. By looping and performing many improvements, we hope to reach a good result:

# %%
for i in range(10): apply_step(params)

# %%
#hide
params = orig_params.detach().requires_grad_()

# %%
_,axs = plt.subplots(1,4,figsize=(12,3))
for ax in axs: show_preds(apply_step(params, False), ax)
plt.tight_layout()

# %% [markdown]
# #### Step 7: stop

# %% [markdown]
# We just decided to stop after 10 epochs arbitrarily. In practice, we would watch the training and validation losses and our metrics to decide when to stop, as we've discussed.

# %% [markdown]
# ### Summarizing Gradient Descent

# %%
#hide_input
#id gradient_descent
#caption The gradient descent process
#alt Graph showing the steps for Gradient Descent
gv('''
init->predict->loss->gradient->step->stop
step->predict[label=repeat]
''')

# %% [markdown]
# ## The MNIST Loss Function

# %% [markdown]
# We already have our independent variables `x`—these are the images themselves. We'll concatenate them all into a single tensor, and also change them from a list of matrices (a rank-3 tensor) to a list of vectors (a rank-2 tensor). We can do this using `view`, which is a PyTorch method that changes the shape of a tensor without changing its contents. `-1` is a special parameter to `view` that means "make this axis as big as necessary to fit all the data":

# %%
train_x = torch.cat([stacked_threes, stacked_sevens]).view(-1, 28*28)

# %% [markdown]
# We need a label for each image. We'll use `1` for 3s and `0` for 7s:

# %%
train_y = tensor([1]*len(threes) + [0]*len(sevens)).unsqueeze(1)
train_x.shape,train_y.shape

# %% [markdown]
# A `Dataset` in PyTorch is required to return a tuple of `(x,y)` when indexed. Python provides a `zip` function which, when combined with `list`, provides a simple way to get this functionality:

# %%
dset = list(zip(train_x,train_y))
x,y = dset[0]
x.shape,y

# %%
valid_x = torch.cat([valid_3_tens, valid_7_tens]).view(-1, 28*28)
valid_y = tensor([1]*len(valid_3_tens) + [0]*len(valid_7_tens)).unsqueeze(1)
valid_dset = list(zip(valid_x,valid_y))

# %% [markdown]
# Now we need an (initially random) weight for every pixel (this is the *initialize* step in our seven-step process):

# %%
def init_params(size, std=1.0): return (torch.randn(size)*std).requires_grad_()

# %%
weights = init_params((28*28,1))

# %% [markdown]
# The function `weights*pixels` won't be flexible enough—it is always equal to 0 when the pixels are equal to 0 (i.e., its *intercept* is 0). You might remember from high school math that the formula for a line is `y=w*x+b`; we still need the `b`. We'll initialize it to a random number too:

# %%
bias = init_params(1)

# %% [markdown]
# In neural networks, the `w` in the equation `y=w*x+b` is called the *weights*, and the `b` is called the *bias*. Together, the weights and bias make up the *parameters*.

# %% [markdown]
# > jargon: Parameters: The _weights_ and _biases_ of a model. The weights are the `w` in the equation `w*x+b`, and the biases are the `b` in that equation.

# %% [markdown]
# We can now calculate a prediction for one image:

# %%
(train_x[0]*weights.T).sum() + bias

# %% [markdown]
# While we could use a Python `for` loop to calculate the prediction for each image, that would be very slow. Because Python loops don't run on the GPU, and because Python is a slow language for loops in general, we need to represent as much of the computation in a model as possible using higher-level functions.
# 
# In this case, there's an extremely convenient mathematical operation that calculates `w*x` for every row of a matrix—it's called *matrix multiplication*. <<matmul>> shows what matrix multiplication looks like.

# %% [markdown]
# <img alt="Matrix multiplication" width="400" caption="Matrix multiplication" src="images/matmul2.svg" id="matmul"/>

# %% [markdown]
# This image shows two matrices, `A` and `B`, being multiplied together. Each item of the result, which we'll call `AB`, contains each item of its corresponding row of `A` multiplied by each item of its corresponding column of `B`, added together. For instance, row 1, column 2 (the yellow dot with a red border) is calculated as $a_{1,1} * b_{1,2} + a_{1,2} * b_{2,2}$. If you need a refresher on matrix multiplication, we suggest you take a look at the [Intro to Matrix Multiplication](https://youtu.be/kT4Mp9EdVqs) on *Khan Academy*, since this is the most important mathematical operation in deep learning.
# 
# In Python, matrix multiplication is represented with the `@` operator. Let's try it:

# %%
def linear1(xb): return xb@weights + bias
preds = linear1(train_x)
preds

# %% [markdown]
# The first element is the same as we calculated before, as we'd expect. This equation, `batch@weights + bias`, is one of the two fundamental equations of any neural network (the other one is the *activation function*, which we'll see in a moment).

# %% [markdown]
# Let's check our accuracy. To decide if an output represents a 3 or a 7, we can just check whether it's greater than 0.0, so our accuracy for each item can be calculated (using broadcasting, so no loops!) with:

# %%
corrects = (preds>0.0).float() == train_y
corrects

# %%
corrects.float().mean().item()

# %% [markdown]
# Now let's see what the change in accuracy is for a small change in one of the weights (note that we have to ask PyTorch not to calculate gradients as we do this, which is what `with torch.no_grad()` is doing here):

# %%
with torch.no_grad(): weights[0] *= 1.0001

# %%
preds = linear1(train_x)
((preds>0.0).float() == train_y).float().mean().item()

# %% [markdown]
# As we've seen, we need gradients in order to improve our model using SGD, and in order to calculate gradients we need some *loss function* that represents how good our model is. That is because the gradients are a measure of how that loss function changes with small tweaks to the weights.
# 
# So, we need to choose a loss function. The obvious approach would be to use accuracy, which is our metric, as our loss function as well. In this case, we would calculate our prediction for each image, collect these values to calculate an overall accuracy, and then calculate the gradients of each weight with respect to that overall accuracy.
# 
# Unfortunately, we have a significant technical problem here. The gradient of a function is its *slope*, or its steepness, which can be defined as *rise over run*—that is, how much the value of the function goes up or down, divided by how much we changed the input. We can write this in mathematically as: `(y_new - y_old) / (x_new - x_old)`. This gives us a good approximation of the gradient when `x_new` is very similar to `x_old`, meaning that their difference is very small. But accuracy only changes at all when a prediction changes from a 3 to a 7, or vice versa. The problem is that a small change in weights from `x_old` to `x_new` isn't likely to cause any prediction to change, so `(y_new - y_old)` will almost always be 0. In other words, the gradient is 0 almost everywhere.

# %% [markdown]
# A very small change in the value of a weight will often not actually change the accuracy at all. This means it is not useful to use accuracy as a loss function—if we do, most of the time our gradients will actually be 0, and the model will not be able to learn from that number.
# 
# > S: In mathematical terms, accuracy is a function that is constant almost everywhere (except at the threshold, 0.5), so its derivative is nil almost everywhere (and infinity at the threshold). This then gives gradients that are 0 or infinite, which are useless for updating the model.
# 
# Instead, we need a loss function which, when our weights result in slightly better predictions, gives us a slightly better loss. So what does a "slightly better prediction" look like, exactly? Well, in this case, it means that if the correct answer is a 3 the score is a little higher, or if the correct answer is a 7 the score is a little lower.
# 
# Let's write such a function now. What form does it take?
# 
# The loss function receives not the images themselves, but the predictions from the model. Let's make one argument, `prds`, of values between 0 and 1, where each value is the prediction that an image is a 3. It is a vector (i.e., a rank-1 tensor), indexed over the images.
# 
# The purpose of the loss function is to measure the difference between predicted values and the true values — that is, the targets (aka labels). Let's make another argument, `trgts`, with values of 0 or 1 which tells whether an image actually is a 3 or not. It is also a vector (i.e., another rank-1 tensor), indexed over the images.
# 
# So, for instance, suppose we had three images which we knew were a 3, a 7, and a 3. And suppose our model predicted with high confidence (`0.9`) that the first was a 3, with slight confidence (`0.4`) that the second was a 7, and with fair confidence (`0.2`), but incorrectly, that the last was a 7. This would mean our loss function would receive these values as its inputs:

# %%
trgts  = tensor([1,0,1])
prds   = tensor([0.9, 0.4, 0.2])

# %% [markdown]
# Here's a first try at a loss function that measures the distance between `predictions` and `targets`:

# %%
def mnist_loss(predictions, targets):
    return torch.where(targets==1, 1-predictions, predictions).mean()

# %% [markdown]
# We're using a new function, `torch.where(a,b,c)`. This is the same as running the list comprehension `[b[i] if a[i] else c[i] for i in range(len(a))]`, except it works on tensors, at C/CUDA speed. In plain English, this function will measure how distant each prediction is from 1 if it should be 1, and how distant it is from 0 if it should be 0, and then it will take the mean of all those distances.
# 
# > note: Read the Docs: It's important to learn about PyTorch functions like this, because looping over tensors in Python performs at Python speed, not C/CUDA speed! Try running `help(torch.where)` now to read the docs for this function, or, better still, look it up on the PyTorch documentation site.

# %% [markdown]
# Let's try it on our `prds` and `trgts`:

# %%
torch.where(trgts==1, 1-prds, prds)

# %% [markdown]
# You can see that this function returns a lower number when predictions are more accurate, when accurate predictions are more confident (higher absolute values), and when inaccurate predictions are less confident. In PyTorch, we always assume that a lower value of a loss function is better. Since we need a scalar for the final loss, `mnist_loss` takes the mean of the previous tensor:

# %%
mnist_loss(prds,trgts)

# %% [markdown]
# For instance, if we change our prediction for the one "false" target from `0.2` to `0.8` the loss will go down, indicating that this is a better prediction:

# %%
mnist_loss(tensor([0.9, 0.4, 0.8]),trgts)

# %% [markdown]
# One problem with `mnist_loss` as currently defined is that it assumes that predictions are always between 0 and 1. We need to ensure, then, that this is actually the case! As it happens, there is a function that does exactly that—let's take a look.

# %% [markdown]
# ### Sigmoid

# %% [markdown]
# The `sigmoid` function always outputs a number between 0 and 1. It's defined as follows:

# %%
def sigmoid(x): return 1/(1+torch.exp(-x))

# %% [markdown]
# Pytorch defines an accelerated version for us, so we don’t really need our own. This is an important function in deep learning, since we often want to ensure values are between 0 and 1. This is what it looks like:

# %%
plot_function(torch.sigmoid, title='Sigmoid', min=-4, max=4)

# %% [markdown]
# As you can see, it takes any input value, positive or negative, and smooshes it onto an output value between 0 and 1. It's also a smooth curve that only goes up, which makes it easier for SGD to find meaningful gradients. 
# 
# Let's update `mnist_loss` to first apply `sigmoid` to the inputs:

# %%
def mnist_loss(predictions, targets):
    predictions = predictions.sigmoid()
    return torch.where(targets==1, 1-predictions, predictions).mean()


# %%
coll = range(15)
dl = DataLoader(coll, batch_size=5, shuffle=True)
list(dl)

# %% [markdown]
# For training a model, we don't just want any Python collection, but a collection containing independent and dependent variables (that is, the inputs and targets of the model). A collection that contains tuples of independent and dependent variables is known in PyTorch as a `Dataset`. Here's an example of an extremely simple `Dataset`:

# %%
ds = L(enumerate(string.ascii_lowercase))
ds

# %% [markdown]
# When we pass a `Dataset` to a `DataLoader` we will get back mini-batches which are themselves tuples of tensors representing batches of independent and dependent variables:

# %%
dl = DataLoader(ds, batch_size=6, shuffle=True)
list(dl)

# %% [markdown]
# We are now ready to write our first training loop for a model using SGD!

# %% [markdown]
# ## Putting It All Together

# %% [markdown]
# It's time to implement the process we saw in <<gradient_descent>>. In code, our process will be implemented something like this for each epoch:
# 
# ```python
# for x,y in dl:
#     pred = model(x)
#     loss = loss_func(pred, y)
#     loss.backward()
#     parameters -= parameters.grad * lr
# ```

# %% [markdown]
# First, let's re-initialize our parameters:

# %%
weights = init_params((28*28,1))
bias = init_params(1)

# %% [markdown]
# A `DataLoader` can be created from a `Dataset`:

# %%
dl = DataLoader(dset, batch_size=256)
xb,yb = first(dl)
xb.shape,yb.shape

# %% [markdown]
# We'll do the same for the validation set:

# %%
valid_dl = DataLoader(valid_dset, batch_size=256)

# %% [markdown]
# Let's create a mini-batch of size 4 for testing:

# %%
batch = train_x[:4]
batch.shape

# %%
preds = linear1(batch)
preds

# %%
loss = mnist_loss(preds, train_y[:4])
loss

# %% [markdown]
# Now we can calculate the gradients:

# %%
loss.backward()
weights.grad.shape,weights.grad.mean(),bias.grad

# %% [markdown]
# Let's put that all in a function:

# %%
def calc_grad(xb, yb, model):
    preds = model(xb)
    loss = mnist_loss(preds, yb)
    loss.backward()

# %% [markdown]
# and test it:

# %%
calc_grad(batch, train_y[:4], linear1)
weights.grad.mean(),bias.grad

# %% [markdown]
# But look what happens if we call it twice:

# %%
calc_grad(batch, train_y[:4], linear1)
weights.grad.mean(),bias.grad

# %% [markdown]
# The gradients have changed! The reason for this is that `loss.backward` actually *adds* the gradients of `loss` to any gradients that are currently stored. So, we have to set the current gradients to 0 first:

# %%
weights.grad.zero_()
bias.grad.zero_();

# %% [markdown]
# > note: Inplace Operations: Methods in PyTorch whose names end in an underscore modify their objects _in place_. For instance, `bias.zero_()` sets all elements of the tensor `bias` to 0.

# %% [markdown]
# Our only remaining step is to update the weights and biases based on the gradient and learning rate. When we do so, we have to tell PyTorch not to take the gradient of this step too—otherwise things will get very confusing when we try to compute the derivative at the next batch! If we assign to the `data` attribute of a tensor then PyTorch will not take the gradient of that step. Here's our basic training loop for an epoch:

# %%
def train_epoch(model, lr, params):
    for xb,yb in dl:
        calc_grad(xb, yb, model)
        for p in params:
            p.data -= p.grad*lr
            p.grad.zero_()

# %% [markdown]
# We also want to check how we're doing, by looking at the accuracy of the validation set. To decide if an output represents a 3 or a 7, we can just check whether it's greater than 0. So our accuracy for each item can be calculated (using broadcasting, so no loops!) with:

# %%
(preds>0.0).float() == train_y[:4]

# %% [markdown]
# That gives us this function to calculate our validation accuracy:

# %%
def batch_accuracy(xb, yb):
    preds = xb.sigmoid()
    correct = (preds>0.5) == yb
    return correct.float().mean()

# %% [markdown]
# We can check it works:

# %%
batch_accuracy(linear1(batch), train_y[:4])

# %% [markdown]
# and then put the batches together:

# %%
def validate_epoch(model):
    accs = [batch_accuracy(model(xb), yb) for xb,yb in valid_dl]
    return round(torch.stack(accs).mean().item(), 4)

# %%
validate_epoch(linear1)

# %% [markdown]
# That's our starting point. Let's train for one epoch, and see if the accuracy improves:

# %%
lr = 1.
params = weights,bias
train_epoch(linear1, lr, params)
validate_epoch(linear1)

# %% [markdown]
# Then do a few more:

# %%
for i in range(20):
    train_epoch(linear1, lr, params)
    print(validate_epoch(linear1), end=' ')

# %% [markdown]
# Looking good! We're already about at the same accuracy as our "pixel similarity" approach, and we've created a general-purpose foundation we can build on. Our next step will be to create an object that will handle the SGD step for us. In PyTorch, it's called an *optimizer*.

# %% [markdown]
# ### Creating an Optimizer

# %% [markdown]
# Because this is such a general foundation, PyTorch provides some useful classes to make it easier to implement. The first thing we can do is replace our `linear1` function with PyTorch's `nn.Linear` module. A *module* is an object of a class that inherits from the PyTorch `nn.Module` class. Objects of this class behave identically to standard Python functions, in that you can call them using parentheses and they will return the activations of a model.
# 
# `nn.Linear` does the same thing as our `init_params` and `linear` together. It contains both the *weights* and *biases* in a single class. Here's how we replicate our model from the previous section:

# %%
linear_model = nn.Linear(28*28,1)

# %% [markdown]
# Every PyTorch module knows what parameters it has that can be trained; they are available through the `parameters` method:

# %%
w,b = linear_model.parameters()
w.shape,b.shape

# %% [markdown]
# We can use this information to create an optimizer:

# %%
class BasicOptim:
    def __init__(self,params,lr): self.params,self.lr = list(params),lr

    def step(self, *args, **kwargs):
        for p in self.params: p.data -= p.grad.data * self.lr

    def zero_grad(self, *args, **kwargs):
        for p in self.params: p.grad = None

# %% [markdown]
# We can create our optimizer by passing in the model's parameters:

# %%
opt = BasicOptim(linear_model.parameters(), lr)

# %% [markdown]
# Our training loop can now be simplified to:

# %%
def train_epoch(model):
    for xb,yb in dl:
        calc_grad(xb, yb, model)
        opt.step()
        opt.zero_grad()

# %% [markdown]
# Our validation function doesn't need to change at all:

# %%
validate_epoch(linear_model)

# %% [markdown]
# Let's put our little training loop in a function, to make things simpler:

# %%
def train_model(model, epochs):
    for i in range(epochs):
        train_epoch(model)
        print(validate_epoch(model), end=' ')

# %% [markdown]
# The results are the same as in the previous section:

# %%
train_model(linear_model, 20)

# %% [markdown]
# fastai provides the `SGD` class which, by default, does the same thing as our `BasicOptim`:

# %%
linear_model = nn.Linear(28*28,1)
opt = SGD(linear_model.parameters(), lr)
train_model(linear_model, 20)

# %% [markdown]
# fastai also provides `Learner.fit`, which we can use instead of `train_model`. To create a `Learner` we first need to create a `DataLoaders`, by passing in our training and validation `DataLoader`s:

# %%
dls = DataLoaders(dl, valid_dl)

# %% [markdown]
# To create a `Learner` without using an application (such as `vision_learner`) we need to pass in all the elements that we've created in this chapter: the `DataLoaders`, the model, the optimization function (which will be passed the parameters), the loss function, and optionally any metrics to print:

# %%
learn = Learner(dls, nn.Linear(28*28,1), opt_func=SGD,
                loss_func=mnist_loss, metrics=batch_accuracy)

# %% [markdown]
# Now we can call `fit`:

# %%
learn.fit(10, lr=lr)

# %% [markdown]
# As you can see, there's nothing magic about the PyTorch and fastai classes. They are just convenient pre-packaged pieces that make your life a bit easier! (They also provide a lot of extra functionality we'll be using in future chapters.)
# 
# With these classes, we can now replace our linear model with a neural network.

# %% [markdown]
# ## Adding a Nonlinearity

# %% [markdown]
# So far we have a general procedure for optimizing the parameters of a function, and we have tried it out on a very boring function: a simple linear classifier. A linear classifier is very constrained in terms of what it can do. To make it a bit more complex (and able to handle more tasks), we need to add something nonlinear between two linear classifiers—this is what gives us a neural network.
# 
# Here is the entire definition of a basic neural network:

# %%
def simple_net(xb): 
    res = xb@w1 + b1
    res = res.max(tensor(0.0))
    res = res@w2 + b2
    return res

# %% [markdown]
# That's it! All we have in `simple_net` is two linear classifiers with a `max` function between them.
# 
# Here, `w1` and `w2` are weight tensors, and `b1` and `b2` are bias tensors; that is, parameters that are initially randomly initialized, just like we did in the previous section:

# %%
w1 = init_params((28*28,30))
b1 = init_params(30)
w2 = init_params((30,1))
b2 = init_params(1)

# %% [markdown]
# The key point about this is that `w1` has 30 output activations (which means that `w2` must have 30 input activations, so they match). That means that the first layer can construct 30 different features, each representing some different mix of pixels. You can change that `30` to anything you like, to make the model more or less complex.
# 
# That little function `res.max(tensor(0.0))` is called a *rectified linear unit*, also known as *ReLU*. We think we can all agree that *rectified linear unit* sounds pretty fancy and complicated... But actually, there's nothing more to it than `res.max(tensor(0.0))`—in other words, replace every negative number with a zero. This tiny function is also available in PyTorch as `F.relu`:

# %%
plot_function(F.relu)


# %%
simple_net = nn.Sequential(
    nn.Linear(28*28,30),
    nn.ReLU(),
    nn.Linear(30,1)
)

# %% [markdown]
# `nn.Sequential` creates a module that will call each of the listed layers or functions in turn.
# 
# `nn.ReLU` is a PyTorch module that does exactly the same thing as the `F.relu` function. Most functions that can appear in a model also have identical forms that are modules. Generally, it's just a case of replacing `F` with `nn` and changing the capitalization. When using `nn.Sequential`, PyTorch requires us to use the module version. Since modules are classes, we have to instantiate them, which is why you see `nn.ReLU()` in this example. 
# 
# Because `nn.Sequential` is a module, we can get its parameters, which will return a list of all the parameters of all the modules it contains. Let's try it out! As this is a deeper model, we'll use a lower learning rate and a few more epochs.

# %%
learn = Learner(dls, simple_net, opt_func=SGD,
                loss_func=mnist_loss, metrics=batch_accuracy)

# %%
#hide_output
learn.fit(40, 0.1)

# %% [markdown]
# We're not showing the 40 lines of output here to save room; the training process is recorded in `learn.recorder`, with the table of output stored in the `values` attribute, so we can plot the accuracy over training as:

# %%
plt.plot(L(learn.recorder.values).itemgot(2));

# %% [markdown]
# And we can view the final accuracy:

# %%
learn.recorder.values[-1][2]

# %% [markdown]

# Here is what happens when we train an 18-layer model using the same approach we saw in <<chapter_intro>>:

# %%
dls = ImageDataLoaders.from_folder(path)
learn = vision_learner(dls, resnet18, pretrained=False,
                    loss_func=F.cross_entropy, metrics=accuracy)
learn.fit_one_cycle(1, 0.1)
learn.recorder.values[-1][2]

# %% [markdown]
# Nearly 100% accuracy! That's a big difference compared to our simple neural net. But as you'll learn in the remainder of this book, there are just a few little tricks you need to use to get such great results from scratch yourself. You already know the key foundational pieces. (Of course, even once you know all the tricks, you'll nearly always want to work with the pre-built classes provided by PyTorch and fastai, because they save you having to think about all the little details yourself.)



