Before running we need to download intel's offline installers for linux as the docker container will require them.

intel-onednn-2026.0.2.46_offline.sh  
intel-deep-learning-essentials-2026.1.2.25_offline.sh

# Models
I created a share in my unraid on the cache called Models/qwen38/ and it is where I am storing the vision part and the llm model.
For intel I had seen i needed a special one that works with SYCL and those are at https://huggingface.co/bartowski/Qwen3.8-27B-GGUF
