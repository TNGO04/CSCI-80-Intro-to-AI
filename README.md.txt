I started off building a model without convolutional or pooling steps, training with a hidden layer of 128 nodes and a dropout rate of 0.5. I ran the training multiple times, and the accuracy I got from the runs are pretty low, between 0.4 and 0.5. 
40/40 - 1s - loss: 0.9302 - accuracy: 0.4738 - 562ms/epoch - 14ms/step
40/40 - 0s - loss: 0.9570 - accuracy: 0.4881 - 449ms/epoch - 11ms/step

I moved on and add a convolutional (3-by-3 filter learned 20 times) and pooling layers (2x2 Max Pooling) before the Neural Network. Since this step helps extract the features in the image, it improves the accuracy significantly. The training time per epoch also decreased since the Neural Network trains on less input nodes. 
40/40 - 0s - loss: 0.1531 - accuracy: 0.9548 - 385ms/epoch - 10ms/step
40/40 - 0s - loss: 0.2122 - accuracy: 0.9262 - 487ms/epoch - 12ms/step

I experimented with changing the convolutional filter from 3-by-3 to 5-by-5. This seems to yield slightly less accurate results and more training time. 
40/40 - 1s - loss: 0.1746 - accuracy: 0.9357 - 591ms/epoch - 15ms/step
40/40 - 1s - loss: 0.1503 - accuracy: 0.9381 - 631ms/epoch - 16ms/step

I experimented next by a convolutional layer (3x3 learned 20 times) and pooling layer (2x2 pooling) after an original convolutional layer (5x5 learned 20 times) and pooling layer (2x2 pooling). This yields the best result so far in terms of both higher accuracy and lower training time. I think the extra convolution and pooling step helps to extract more complex features from the source image after an elementary feature extraction, thus improving the algorithm.
40/40 - 0s - loss: 0.1201 - accuracy: 0.9563 - 373ms/epoch - 9ms/step
40/40 - 0s - loss: 0.0760 - accuracy: 0.9833 - 435ms/epoch - 11ms/step
40/40 - 0s - loss: 0.1416 - accuracy: 0.9619 - 371ms/epoch - 9ms/step

I try to improve the accuracy even more by adding another hidden layer before the 128-node layer. This does not significantly improve the accuracy or training time. 
40/40 - 0s - loss: 0.1398 - accuracy: 0.9532 - 378ms/epoch - 9ms/step
40/40 - 0s - loss: 0.1357 - accuracy: 0.9540 - 387ms/epoch - 10ms/step
40/40 - 0s - loss: 0.1810 - accuracy: 0.9579 - 409ms/epoch - 10ms/step

I try experimenting with the Dropout rate and decrease it to 0.3. The accurate increased to about 0.97, highest out of all the models I tried. The training time did not increase significantly. 
40/40 - 0s - loss: 0.0916 - accuracy: 0.9738 - 389ms/epoch - 10ms/step
40/40 - 0s - loss: 0.0853 - accuracy: 0.9762 - 410ms/epoch - 10ms/step
40/40 - 0s - loss: 0.0697 - accuracy: 0.9778 - 413ms/epoch - 10ms/step


