# Would-you-survive-the-Titanic?

A machine learning algorithm which predicts the probability of survival of an individual on the Titanic incident.
===========================================================================================================================================

Code written in Python, using the Tensorflow library. The dataset used for training can be found in the repository.

The process of prediction from the algorithm is made possible from the attributes alocated to the individuals from another python file which works using the answers given from the individuals on the questonnaire available on Google Forms
(The questionnaire can be foud here https://forms.gle/AmSNdhwBnLQTHbyW7)

It can be seen that the questionnaire does not contain the same number of questions as the attributes needed for the algorithm to make a prediction. That is because some of the attributes appear to have negligible effect on the final result, e.g. the port from which the passenger boarded. Also, a number of attributes such as the ticket price, the deck on which the passenger's cabin was located and other wealth based attributes were allocated using the answer from the annual income question.

In the case where the attribute had a negligible effect the attribute was allocated at random between the available options.

Α basic ui is available when running the application which allows the user to choose between a number of functions. In there someone can pick between using the individuals who have answered the questionnaire or manually "creating" a passenger.

It needs to be noted that someone needs to manually download the csv file from the google forms website and place it in the application's folder for the algorithm to work.
