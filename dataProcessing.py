# %% [markdown]
# # Entity 数据准备

# %%
import json

with open('data/Others/entity/conll03/test.json', 'r') as f:
    dataList = [json.loads(line) for line in f.readlines()]
with open('data/Act-UIE/conll2003/test.json', 'w') as f:
    for item in dataList:
        meta = {
            "text": item['text'],
            "standard":{"entities":[{"type": entity['type'],"text":entity['text']} for entity in item['entity']]}
        }
        f.write(json.dumps(meta, ensure_ascii=False) + '\n')
print(f"Converted {len(dataList)} items to Act-UIE format.")

# %%
import json

with open('data/Others/entity/mrc_ace04/test.json', 'r') as f:
    dataList = [json.loads(line) for line in f.readlines()]
with open('data/Act-UIE/ace04NER/test.json', 'w') as f:
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

with open('data/Others/relation/conll04/test.json', 'r') as f:
    dataList = [json.loads(line) for line in f.readlines()]
with open('data/Act-UIE/conll2004/test.json', 'w') as f:
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

inputDir = 'data/Others/relation/scierc'
outputDir = 'data/Act-UIE/scierc'

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


