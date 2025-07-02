import paddlehub as hub
# from docarray import DocumentArray, Document
# 无api秘钥 无法调用

ernie_vilg_module = hub.Module(name='ernie_vilg')

result = ernie_vilg_module.generate_image(text_prompts=text_prompt, style=style, topk=6, output_dir='D:/pytorch_practice/Result')