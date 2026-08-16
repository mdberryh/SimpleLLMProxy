This container is the current basis that I use to host my llms. It is ROCM and LLAMA.CPP based. 
I use the ROCM because I was running into issues using Vulkan with two GPUs. ROCM I was able to use each GPU as an individual card.
If you also do this keep in mind the PCIe lanes to the second GPU slot. Using AMD 550 I was able to find a board that gave me
PCIe3 x2 or perhaps x4. Either way, it's not fast, but for llm inferencing that isn't splitting across the PCIe buss it works fine.
With AMD 9700 PROs I get 22-28 tokens/sec with qwen3.6 and qwen3.8. Using MTP it is much higher, and less consistent.
