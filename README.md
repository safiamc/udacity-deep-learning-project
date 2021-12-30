# Image Classification using AWS SageMaker

In this project, I used AWS Sagemaker to train an image-classification model on a dog breed data set provided by Udacity.

## Project Set Up
The data set was already split into train, validation, and test sets, each consisting of folders of images labelled with one of 133 different breeds of dog. I uploaded the datasets to S3.

## Hyperparameter Tuning
I chose to use a pretrained ResNet50 model, with two fully-connected layers. I chose to tune the following hyperparameters: batch size, learning rate, and number of epochs. In the future, I wouldn't include the number of epochs. I believe the results were heavily skewed towards the models with more epochs, almost regardless of the other hyperparameters.

I used the script `hpo.py` as the entry point for my tuning job, and I trained 8 jobs total. You can see that this took a long time!

![hyperparameter tuning job](https://github.com/safiamc/udacity-deep-learning-project/blob/main/Screenshot%20(17).png)

The best job ran for 5 epochs, had a batch size of 128, and a learning rate of approximately .001

![best job](https://github.com/safiamc/udacity-deep-learning-project/blob/main/Screenshot%20(18).png)

The worst job ran for 2 epochs, had a batch size of 32, and a learning rate of approximately .022

![worst job](https://github.com/safiamc/udacity-deep-learning-project/blob/main/Screenshot%20(19).png)

## Model Finetuning

Using the best hyperparameters, I finetuned the pretrained ResNet50 model. I used the script `train_model.py` as the entry point for my finetuning job.

## Debugging and Profiling
Within the script, I set a debugger hook to record the Cross Entropy Loss across training and validation through each epoch.

![cross entropy loss](https://github.com/safiamc/udacity-deep-learning-project/blob/main/Screenshot%20(22).png)

As you can see above, the loss decreases more or less uniformly, which is what we hope for! It looks like there isn't much of an advantage to training past 100 steps, so 5 epochs may be overkill. If I saw much more jagged lines, I would first try larger batch sizes, in the hopes that more images trained at a time would smooth out the calculated loss. If I saw loss increasing over time, I would go back to the hyperparameter tuning step, and perhaps start from a different pretrained model.

I also added a profiler to log performance metrics, such as low GPU utilization.

![rules summary](https://github.com/safiamc/udacity-deep-learning-project/blob/main/Screenshot%20(23).png)
![rules graph](https://github.com/safiamc/udacity-deep-learning-project/blob/main/Screenshot%20(24).png)

As you can see above, I had low GPU utilization, high GPU memory utilization, and high CPU utilization, which is why the finetuning job took 1 hour. I used a single instance of type "ml.g4dn.xlarge". In the future, I would increase the batch size and consider using more instances or larger/more powerful ones.

### Results
I don't believe it was necessary to train my model for 5 epochs, and I think training would have been quicker with a larger batch size. However, the model performed well in validation and testing, so I believe it is ready to deploy.

## Model Deployment
To deploy the model, I created an endpoint using the script `inference.py` as the entry point. I used a single instance of type ml.m5.large. The endpoint can then be queried to predict dog breed from images.

![endpoint](https://github.com/safiamc/udacity-deep-learning-project/blob/main/Screenshot%20(25).png)

## Endpoint Use
I used images of an Akita, a Bernese mountain dog, and a Greyhound as test images. I used the boto3 client to read the file names in 'dogImages/train' into a list of dog names, to check my inferences.

![Akita](https://upload.wikimedia.org/wikipedia/commons/7/78/Akita_inu.jpeg)
The largest number in the response vector to this image was in index 3, which corresponds to the Akita. The model correctly predicted the breed of this dog.

![Bernese](https://vetstreet-brightspot.s3.amazonaws.com/39/2750509e8d11e0a2380050568d634f/file/Bernese-Mtn-3-645mk062111.jpg)
The largest number in the response vector to this image was in index 22, which corresponds to the Bernese mountain dog. The model correctly predicted the breed of this dog too.

![Greyhound](https://2.bp.blogspot.com/_oX_iiKqvHUI/TI-Vs5fBzHI/AAAAAAAAAA8/hzI9XI7sJjI/s1600/greyhound-0005.jpg)
The largest number in the response vector to this image was in index 80, which corresponds to the Greyhound. The model correctly predicted the breed of this dog too!

When I was done testing, I deleted the endpoint.
## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.
