The dataset is devided into 4 parts.

1. Phrase 1 for training.
2. Phrase 1 for testing.
3. Phrase 2 for training.
4. Phrase 2 for testing.
Phrase 2 is a lot larger than phrase 1.
The dataset was originally built by openai in the work [1].

The dataset is then organized and built in another way. In each one of the samples, there are:

instruction: the maths problem.
responses: the previous response(s) answering the problem.                 
next_response: the current response to be rated 1/-1/0.
answer: the correct answer of the maths problem.
is_human_response: if "True", then the "next_response" is a hint from human to direct the model continue answering, which doesn't need to be rated. 
is_solution: correctly finished or not
is_preferred_response: another human rating
rating: 1 for correct, 0 for ambiguous, -1 for wrong
error_reason: my meta-cognitive annotation :) giving illustration on why the step is wrong. For those who are correct or ambiguous, this should be "null".

The testsets don't have a error_reason column.

[1] Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker, B., Lee, T., Leike, J., Schulman, J., Sutskever, I., and Cobbe, K. (2023). Let's Verify Step by Step. *arXiv preprint arXiv:2305.20050*. 
