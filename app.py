"""
Assignment 2: Gradio dashboard.

You're using a script for this stage. If you'd rather use a notebook,
write app.ipynb instead and delete this file. main.py refuses to run if
it finds both app.py and app.ipynb, so exactly one of them must exist.

Replace this docstring and everything below it with your own code. When run
via uv run python main.py app, this file must build and launch a Gradio app
(a demo that calls .launch()) using only what your training stage already
produced. It must never train anything itself. Load the committed weights in
weights/, never whatever the training run left in runs/.

At minimum, the dashboard must let you:

- Upload an image and see the model's predictions at the current threshold.
- Move the two edges of your escalation band and watch the confusion matrix,
  the expected cost per 1,000 units, and the share of cases escalated to a
  human all change together.
- See the cost-against-threshold curve with your chosen operating point marked
  on it.
- See per-class performance, not just the overall number.

How you structure the code beyond that is your call to make and be able to
defend: file layout, function boundaries, naming.
"""
