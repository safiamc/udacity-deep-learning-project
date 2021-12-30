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

As you can see above, I had low GPU utilization and high CPU utilization, which is why the finetuning job took 1 hour. I used a single instance of type "ml.g4dn.xlarge". In the future, I would increase the batch size and consider using more instances or larger/more powerful ones.

### Results
**TODO**: What are the results/insights did you get by profiling/debugging your model?

**TODO** Remember to provide the profiler html/pdf file in your submission.


## Model Deployment
**TODO**: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

**TODO** Remember to provide a screenshot of the deployed active endpoint in Sagemaker.

## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.
