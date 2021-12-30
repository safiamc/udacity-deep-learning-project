# Image Classification using AWS SageMaker

In this project, I used AWS Sagemaker to train an image-classification model on a dog breed data set provided by Udacity.

## Project Set Up
The data set was already split into train, validation, and test sets, each consisting of folders of images labelled with one of 133 different breeds of dog. I uploaded the datasets to S3.

## Hyperparameter Tuning
I chose to use a pretrained ResNet50 model, with two fully-connected layers. I chose to tune the following hyperparameters: batch size, learning rate, and number of epochs. In the future, I wouldn't include the number of epochs. I believe the results were heavily skewed towards the models with more epochs, almost regardless of the other hyperparameters.

I used the script 'hpo.py' as the entry point for my tuning job, and I trained 8 jobs total. You can see that this took a long time!

![hyperparameter tuning job](https://github.com/safiamc/udacity-deep-learning-project/blob/d04595f7fb8206b002ee9f43429c31e46e5a8361/Screenshot%20(17).png)

The best job ran for 5 epochs,had a batch size of 128, and a learning rate of approx. .001

![best job](https://github.com/safiamc/udacity-deep-learning-project/blob/d04595f7fb8206b002ee9f43429c31e46e5a8361/Screenshot%20(18).png)

The worst job ran for 2 epochs,had a batch size of 32, and a learning rate of approx. .022

![worst job](https://github.com/safiamc/udacity-deep-learning-project/blob/d04595f7fb8206b002ee9f43429c31e46e5a8361/Screenshot%20(19).png)

## Model Finetuning

Using the best hyperparameters, I finetuned the pretrained ResNet50 model. I used the script 'train_model.py' as the entry point for my finetuning job.

## Debugging and Profiling
**TODO**: Give an overview of how you performed model debugging and profiling in Sagemaker

### Results
**TODO**: What are the results/insights did you get by profiling/debugging your model?

**TODO** Remember to provide the profiler html/pdf file in your submission.


## Model Deployment
**TODO**: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

**TODO** Remember to provide a screenshot of the deployed active endpoint in Sagemaker.

## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.
