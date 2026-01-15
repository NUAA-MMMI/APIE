# %% [markdown]
# # Entity 数据准备

# %%
import json

with open('data/origin/conll03/test.json', 'r') as f:
    dataList = [json.loads(line) for line in f.readlines()]
with open('data/APIE/conll2003/test.json', 'w') as f:
    for item in dataList:
        meta = {
            "text": item['text'],
            "standard":{"entities":[{"type": entity['type'],"text":entity['text']} for entity in item['entity']]}
        }
        f.write(json.dumps(meta, ensure_ascii=False) + '\n')
print(f"Converted {len(dataList)} items to Act-UIE format.")

# %%
import json

with open('data/origin/mrc_ace04/test.json', 'r') as f:
    dataList = [json.loads(line) for line in f.readlines()]
with open('data/APIE/ace04NER/test.json', 'w') as f:
    for item in dataList:
        meta = {
            "text": item['text'],
            "standard":{"entities":[{"type": entity['type'],"text":entity['text']} for entity in item['entity']]}
        }
        f.write(json.dumps(meta, ensure_ascii=False) + '\n')
print(f"Converted {len(dataList)} items to Act-UIE format.")

# %% [markdown]
# # Relation 数据准备

# %%
import json

with open('data/origin/conll04/test.json', 'r') as f:
    dataList = [json.loads(line) for line in f.readlines()]
with open('data/APIE/conll2004/test.json', 'w') as f:
    for item in dataList:
        meta = {
            "text": item['text'],
            "standard": {
                "entities":[{"type": entity['type'],"text":entity['text']} for entity in item['entity']],
                "relations": [{"head": relation["args"][0]['text'],
                              "tail": relation["args"][1]['text'],
                              "type": relation['type']} for relation in item['relation']]
                }
        }
        f.write(json.dumps(meta, ensure_ascii=False) + '\n')
        # print(meta)
print(f"Converted {len(dataList)} items to Act-UIE format.")

# %%
import json
import os

inputDir = 'data/origin/relation/scierc'
outputDir = 'data/APIE/scierc'

if not os.path.exists(outputDir):
    os.makedirs(outputDir)

with open(os.path.join(inputDir,'test.json'), 'r') as f:
    dataList = [json.loads(line) for line in f.readlines()]

with open(os.path.join(outputDir,'test.json'), 'w') as f:
    for item in dataList:
        meta = {
            "text": item['text'],
            "standard": {
                "entities":[{"type": entity['type'],"text":entity['text']} for entity in item['entity']],
                "relations": [{"head": relation["args"][0]['text'],
                              "tail": relation["args"][1]['text'],
                              "type": relation['type']} for relation in item['relation']]
                }
        }
        f.write(json.dumps(meta, ensure_ascii=False) + '\n')
        # print(meta)
print(f"Converted {len(dataList)} items to Act-UIE format.")


