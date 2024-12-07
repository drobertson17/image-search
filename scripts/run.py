from image_search.process import ProcessImages
from image_search.bedrock import BedrockLlamaMultiModeVLM, BedrockLlamaTextLLM

process = ProcessImages(
    subdirs=["Photos", "Pictures"],
    vlm=BedrockLlamaMultiModeVLM(),
    llm=BedrockLlamaTextLLM()
)

process.process()
print()