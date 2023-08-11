# upstage/Llama-2-70b-instruct-v2 Cog model

This is an implementation of the [upstage/Llama-2-70b-instruct-v2](https://huggingface.co/upstage/Llama-2-70b-instruct-v2) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, download the pre-trained weights:

    cog run script/download-weights

Then, you can run predictions:

    cog predict -i prompt="### User:\nThomas is healthy, but he has to go to the hospital. What could be the reasons?\n\n### Assistant:\n"
