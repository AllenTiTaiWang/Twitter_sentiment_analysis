# Objectives

The learning objectives of this assignment are to:
1. build a neural network for a community task 
2. practice tuning model hyper-parameters

# Read about the CodaLab Competition

You will be participating in a class-wide competition.
The competition website is:

https://competitions.codalab.org/competitions/20980?secret_key=5cba3f22-a7d8-4744-8d07-85b01e604967

You should visit that site and read all the details of the competition, which
include the task definition, how your model will be evaluated, the format of
submissions to the competition, etc.

# Create a CodaLab account

You must create a CodaLab account and join the competition:
1. Visit the competition website.

2. In the upper right corner of the page, you should see a "Sign Up" button.
Click that and go through the process to create an account.
**Please use your @email.arizona.edu account when signing up.**
Your username will be displayed publicly on a leaderboard showing everyone's
scores.
**If you wish to remain anonymous, please select a username that does not reveal
your identity.**
Your instructor will still be able to match your score with your name via your
email address, but your email address will not be visible to other students. 

3. Return to the competition website and click the "Participate" tab, where you
should be able to request to be added to the competition.

4. Wait for your instructor to manually approve your request.
This may take a day or two. 

5. You should then be able to return to the "Participate" tab and see a
"Submit / View Results" option.
That means you are fully registered for the task.

# Clone the repository

Clone the repository created by GitHub Classroom to your local machine:
```
git clone https://github.com/ua-ista-457/graduate-project-<your-username>.git
```
Note that you do not need to create a separate branch as in previous assignments
(though you're welcome to if you so choose).
You are now ready to begin working on the assignment.

# Write your code

You should design a neural network model to perform the task described on the
CodaLab site.
You must create and train your neural network in the Keras framework that we
have been using in the class.
You should train and tune your model using the training and development data
that is already included in your GitHub Classroom repository.

**You may incorporate extra resources beyond this training data, but only if
you provide those same resources to all other students in the class by posting
the resource in the `#graduate-project` channel on the class's Slack workspace:
http://ua-ista457-fa19.slack.com**

There is some sample code in your repository from which you could start, but
you should feel free to delete that code entirely and start from scratch if
you prefer.

# Test your model predictions on the development set

To test the performance of your model, the only officially supported way is to
run your model on the development set (included in your GitHub Classroom
checkout), format your model predictions as instructed on the CodaLab site,
and upload your model's predictions on the "Participate" tab of the CodaLab
site.

Unofficially, you may make use of scikit-learn's `jaccard_similarity_score` to 
test your model locally.
But you are **strongly** encouraged to upload your model's development set
predictions to the CodaLab site many times to make sure you have all the
formatting correct.
Otherwise, you risk trying to debug formatting issues on the test set, when
the time to submit your model predictions is much more limited.

# Submit your model predictions on the test set

When the test phase of the competition begins (consult the CodaLab site for the
exact timing), the instructor will release the test data and update the CodaLab
site to expect predictions on the test set, rather than predictions on the
development set.
You should run your model on the test data, and upload your model predictions to
the CodaLab site.
 
# Grading

You will be graded first by your model's accuracy, and second on how well your
model ranks in the competition.
If your model achieves at least 0.440 accuracy on the test set, you will get
at least a B.
If your model achieves at least 0.500 accuracy on the test set, you will get
an A.
All models within the same letter grade will be distributed evenly across the
range, based on their rank.
So for example, the highest ranked model in the A range will get 100%, and the
lowest ranked model in the B range will get 80%.
