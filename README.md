# TheBloke/Upstage-Llama-2-70B-instruct-v2-GPTQ Cog model

This is an implementation of the [TheBloke/Upstage-Llama-2-70B-instruct-v2-GPTQ](https://huggingface.co/TheBloke/Upstage-Llama-2-70B-instruct-v2-GPTQ) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, download the pre-trained weights:

    cog run script/download-weights

Then, you can run predictions:

    cog predict -i prompt="Thomas is healthy, but he has to go to the hospital. What could be the reasons?"
