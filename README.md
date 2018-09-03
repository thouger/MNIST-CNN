# MNIST 
* The training set contains 60000 examples, and the test set 10000 examples.
* The images has the pixels of 28*28 and already transform to 784(=28*28) every row in file and array.So one row is one image.
* The labels values are 0 to 9. 

# cnn
1. the first convolutional layer:patch is [5,5,1,32],mean 5 * 5 size,32 num,1 is that images are grayscale,bias is 32,bias num = patch num.max_pool:[1,2,2,1]
2. the second convolutional layer:patch is [5,5,32,64],32 is come from the first layer output,mean 32 feature.bias is 64 same as patch[-1].max_pool:[1,2,2,1]
3. the thrid densely connected layer:w is [7 * 7 * 64,1024],bias is 1024.Activation function is y=wx+b
4 . prevent overfitting:dropout.Each of the nodes is either kept in the network with probability keep_prob or dropped with probability 1 - keep_prob.
5. the softmax layer:w is [1024,labels_count],b is labels_count.
6. predict use argmax() to get the heighest probability from one-hot vector.

* evaluate network performance:cross-entropy.
* minimise:ADAM optimiser.
* use stochastic trainning to train.

##### the dimensionality transfromation 
 
input layer>>>input:[None,784],output:[?,28,28,1]  
convolutional layer>>>input:[?,28,28,1],w:[5,5,1,32],output:[?,14,14,32]  
convolutional layer>>>input:[?,14,14,32],w:[5,5,32,64],output:[?,7,7,64]  
densely connected layer>>>input:[?,3136],w:[7*7*64,1024],output:[?,1024]  
dropout layer>>>input:[?,1024],output:[?,1024]  
output layer>>>input:[?,1024],w:[1024,10],output:[?,10]  
> the program are from "https://www.kaggle.com/kakauandme/tensorflow-deep-nn"
