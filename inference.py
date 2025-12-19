import json
import os
import random

from openai import OpenAI

from scripts.args import InferenceArgs
from models.adaptor import ModelAdaptor
from setting.loggerConfig import getLogger

import scripts.tools as tools

if __name__ == "__main__":
    # Parse command line arguments
    args = InferenceArgs().parse()
    # Convert the parsed arguments to a dictionary
    argsDict = vars(args)

    #* Initialize the model adaptor
    client = None
    if argsDict['client'] == "ollama":
        # default ollama client
        client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
    elif argsDict['client'] == "deepseek-api":
        # deepseek api client
        #! Note: Replace the API key with your actual DeepSeek API key
        client = OpenAI(api_key="sk-5cec141448304a66a835cffa12aabe32", base_url="https://api.deepseek.com")
    model = ModelAdaptor(model = argsDict['model'], client = client)

    #* Initialize logger
    inferencelogger = getLogger("inference")
    actuielogger = getLogger("actuie")

    inferencelogger.info("Starting generation process with arguments: %s", json.dumps(argsDict, indent=2))

    #* Load the test data and schema from the input directory
    with open(os.path.join(argsDict['inputFile']), "r") as f:
        dataList = [json.loads(line) for line in f.readlines()]
    if argsDict['pollSize'] > len(dataList) or argsDict['pollSize'] == -1:
        argsDict['pollSize'] = len(dataList)
    if argsDict['pollSize'] == -2:
        argsDict['pollSize'] = len(dataList)//10

    dataList = random.sample(dataList,argsDict['pollSize'])
    with open(os.path.join(argsDict['schema']), "r") as f:
        schema = json.loads(f.read())

    #* Generate responses based on the method specified in the arguments
    #? should have a method to uniformly generate the sample based on the schema

    if argsDict['method'] == "finalZSL":
        with open(os.path.join(argsDict['outputFile']), "w") as f:
            instruction = tools.getInstruction(schema, method=argsDict['method'])

            for data in dataList:
                inferencelogger.info("--------\nInstruction: %s", instruction)
                inferencelogger.info(f"Processing text: {data['text']}:")

                response = model.generate(instruction = instruction, text = data['text'])
                inferencelogger.info("Origion response: %s", response)
                response = tools.getJsonFormat(response, schema)
                inferencelogger.info("JsonFormat response: %s", response)

                if tools.validateJsonFormat(response, schema) == False:
                    inferencelogger.error("Invalid JsonFormat response: %s", response)
                    response = None
                inferencelogger.info("Final response: %s\n", response)
                
                data['response'] = response
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
                f.flush()
    if argsDict['method'] == "finalActPrompt":
        with open(os.path.join(argsDict['uncertaintyFile']), "r") as f:
            uncertaintyList = [json.loads(line) for line in f.readlines()]
        for data in uncertaintyList:
            data['uncertainty'] = data['normalizeUncertainty']['editUncertanity']
        uncertaintyList = sorted(uncertaintyList, key=lambda x: x['uncertainty'], reverse=True)
        uncertaintyList = uncertaintyList[:argsDict['shotSize']]
        actuielogger.info(f"Uncertainty sample list:{uncertaintyList}")
        sampleList = uncertaintyList

        with open(os.path.join(argsDict['outputFile']), "w") as f:
            instruction = tools.getInstruction(schema, method=argsDict['method'], sampleList=sampleList)

            inferencelogger.info("--------\nInstruction: %s", instruction)

            for data in dataList:
                inferencelogger.info(f"Processing text: {data['text']}:")
                response = model.generate(instruction = instruction, text = data['text'])
                inferencelogger.info("Origion response: %s", response)
                response = tools.getJsonFormat(response, schema)
                inferencelogger.info("JsonFormat response: %s", response)

                if tools.validateJsonFormat(response, schema) == False:
                    inferencelogger.error("Invalid JsonFormat response: %s", response)
                    response = None
                inferencelogger.info("Final response: %s\n", response)

                data['response'] = response
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
                f.flush()
    if argsDict['method'] == "finalRSL":
        with open(os.path.join(argsDict['outputFile']), "w") as f:
            sampleList = random.sample(dataList, argsDict['shotSize'])
            instruction = tools.getInstruction(schema, method=argsDict['method'], sampleList=sampleList)
            inferencelogger.info("--------\nInstruction: %s", instruction)

            for data in dataList:
                inferencelogger.info(f"Processing text: {data['text']}:")

                response = model.generate(instruction = instruction, text = data['text'])
                inferencelogger.info("Origion response: %s", response)
                response = tools.getJsonFormat(response, schema)
                inferencelogger.info("JsonFormat response: %s", response)

                if tools.validateJsonFormat(response, schema) == False:
                    inferencelogger.error("Invalid JsonFormat response: %s", response)
                    response = None
                inferencelogger.info("Final response: %s\n", response)

                data['response'] = response
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
                f.flush()
    if argsDict['method'] == "finalActUIE":
        with open(os.path.join(argsDict['uncertaintyFile']), "r") as f:
            uncertaintyList = [json.loads(line) for line in f.readlines()]
        for data in uncertaintyList:
            formatUncertanity = data['normalizeUncertainty']['formatUncertanity']
            contentUncertanity = data['normalizeUncertainty']['contentUncertanity']
            editUncertanity = data['normalizeUncertainty']['editUncertanity']
            data['uncertainty'] = argsDict['alpha'] * editUncertanity + argsDict['beta']* formatUncertanity + argsDict['gama'] * contentUncertanity
        uncertaintyList = sorted(uncertaintyList, key=lambda x: x['uncertainty'], reverse=True)
        uncertaintyList = uncertaintyList[:argsDict['shotSize']]
        actuielogger.info(f"Uncertainty sample list:{uncertaintyList}")
        sampleList = uncertaintyList

        with open(os.path.join(argsDict['outputFile']), "w") as f:
            instruction = tools.getInstruction(schema, method=argsDict['method'], sampleList=sampleList)

            inferencelogger.info("--------\nInstruction: %s", instruction)

            for data in dataList:
                inferencelogger.info(f"Processing text: {data['text']}:")
                response = model.generate(instruction = instruction, text = data['text'])
                inferencelogger.info("Origion response: %s", response)
                response = tools.getJsonFormat(response, schema)
                inferencelogger.info("JsonFormat response: %s", response)

                if tools.validateJsonFormat(response, schema) == False:
                    inferencelogger.error("Invalid JsonFormat response: %s", response)
                    response = None
                inferencelogger.info("Final response: %s\n", response)

                data['response'] = response
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
                f.flush()
    if argsDict['method'] == "finalKDSort":
        for data in dataList:
            data['ELD'] = tools.getELD(data)
        sortedData = sorted(dataList, key=lambda x: x['ELD'], reverse=True)
        sortedData = sortedData[:argsDict['shotSize']]
        instruction = tools.getInstruction(schema, method=argsDict['method'], sampleList=sortedData)
        inferencelogger.info("--------\nInstruction: %s", instruction)

        with open(os.path.join(argsDict['outputFile']), "w") as f:
            for data in dataList:
                inferencelogger.info(f"Processing text: {data['text']}:")

                response = model.generate(instruction = instruction, text = data['text'])
                inferencelogger.info("Origion response: %s", response)
                response = tools.getJsonFormat(response, schema)
                inferencelogger.info("JsonFormat response: %s", response)

                if tools.validateJsonFormat(response, schema) == False:
                    inferencelogger.error("Invalid JsonFormat response: %s", response)
                    response = None
                inferencelogger.info("Final response: %s\n", response)

                data['response'] = response
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
                f.flush()

    # #* Zero-shot learning method
    # if argsDict['method'] == "ZSL":
    #     with open(os.path.join(argsDict['outputFile']), "w") as f:
    #         instruction = tools.getInstruction(schema, method=argsDict['method'])

    #         for data in dataList:
    #             inferencelogger.info("--------\nInstruction: %s", instruction)
    #             inferencelogger.info(f"Processing text: {data['text']}:")

    #             response = model.generate(instruction = instruction, text = data['text'])
    #             inferencelogger.info("Origion response: %s", response)
    #             response = tools.getJsonFormat(response, schema)
    #             inferencelogger.info("JsonFormat response: %s", response)

    #             if tools.validateJsonFormat(response, schema) == False:
    #                 inferencelogger.error("Invalid JsonFormat response: %s", response)
    #                 response = None
    #                 continue
    #             inferencelogger.info("Final response: %s\n", response)
                
    #             data['response'] = response
    #             f.write(json.dumps(data, ensure_ascii=False) + "\n")
    #             f.flush()
    # #* Random Few-shot learning method
    # if argsDict['method'] == "RSL":
    #     with open(os.path.join(argsDict['outputFile']), "w") as f:

    #         for data in dataList:
    #             sampleList = random.sample(dataList, argsDict['shotSize'])
    #             instruction = tools.getInstruction(schema, method=argsDict['method'], sampleList=sampleList)

    #             inferencelogger.info("--------\nInstruction: %s", instruction)
    #             inferencelogger.info(f"Processing text: {data['text']}:")

    #             response = model.generate(instruction = instruction, text = data['text'])
    #             inferencelogger.info("Origion response: %s", response)
    #             response = tools.getJsonFormat(response, schema)
    #             inferencelogger.info("JsonFormat response: %s", response)

    #             if tools.validateJsonFormat(response, schema) == False:
    #                 inferencelogger.error("Invalid JsonFormat response: %s", response)
    #                 response = None
    #                 continue
    #             inferencelogger.info("Final response: %s\n", response)

    #             data['response'] = response
    #             f.write(json.dumps(data, ensure_ascii=False) + "\n")
    #             f.flush()
    # if argsDict['method'] == "ActUIE":
    #     with open(os.path.join(argsDict['uncertaintyFile']), "r") as f:
    #         uncertaintyList = [json.loads(line) for line in f.readlines()]
    #     for data in uncertaintyList:
    #         formatUncertanity = data['normalizeUncertainty']['formatUncertanity']
    #         contentUncertanity = data['normalizeUncertainty']['contentUncertanity']
    #         data['uncertainty'] = argsDict['alpha'] * formatUncertanity + argsDict['beta'] * contentUncertanity
    #     uncertaintyList = sorted(uncertaintyList, key=lambda x: x['uncertainty'], reverse=True)
    #     bestSize = int(len(uncertaintyList)*argsDict['frontArgs'])
    #     if bestSize < argsDict['shotSize']:
    #         argsDict['shotSize'] = bestSize
    #     uncertaintyList = uncertaintyList[:bestSize]
    #     actuielogger.info(f"Uncertainty sample list:{uncertaintyList}")

    #     with open(os.path.join(argsDict['outputFile']), "w") as f:
    #         for data in dataList:
    #             sampleList = random.sample(uncertaintyList, argsDict['shotSize'])
    #             instruction = tools.getInstruction(schema, method=argsDict['method'], sampleList=sampleList)

    #             inferencelogger.info("--------\nInstruction: %s", instruction)
    #             inferencelogger.info(f"Processing text: {data['text']}:")

    #             response = model.generate(instruction = instruction, text = data['text'])
    #             inferencelogger.info("Origion response: %s", response)
    #             response = tools.getJsonFormat(response, schema)
    #             inferencelogger.info("JsonFormat response: %s", response)

    #             if tools.validateJsonFormat(response, schema) == False:
    #                 inferencelogger.error("Invalid JsonFormat response: %s", response)
    #                 response = None
    #                 continue
    #             inferencelogger.info("Final response: %s\n", response)

    #             data['response'] = response
    #             f.write(json.dumps(data, ensure_ascii=False) + "\n")
    #             f.flush()
    # if argsDict['method'] == "textLength":

    #     sortedData = sorted(dataList, key=lambda x: len(x['text']), reverse=True)
    #     bestSize = int(len(sortedData)*argsDict['frontArgs'])
    #     if bestSize < argsDict['shotSize']:
    #         argsDict['shotSize'] = bestSize
    #     sortedData = sortedData[:bestSize]
    #     actuielogger.info(f"Uncertainty sample list:{sortedData}")

    #     with open(os.path.join(argsDict['outputFile']), "w") as f:
    #         for data in dataList:
    #             sampleList = random.sample(sortedData, argsDict['shotSize'])
    #             instruction = tools.getInstruction(schema, method=argsDict['method'], sampleList=sampleList)

    #             inferencelogger.info("--------\nInstruction: %s", instruction)
    #             inferencelogger.info(f"Processing text: {data['text']}:")

    #             response = model.generate(instruction = instruction, text = data['text'])
    #             inferencelogger.info("Origion response: %s", response)
    #             response = tools.getJsonFormat(response, schema)
    #             inferencelogger.info("JsonFormat response: %s", response)

    #             if tools.validateJsonFormat(response, schema) == False:
    #                 inferencelogger.error("Invalid JsonFormat response: %s", response)
    #                 response = None
    #                 continue
    #             inferencelogger.info("Final response: %s\n", response)

    #             data['response'] = response
    #             f.write(json.dumps(data, ensure_ascii=False) + "\n")
    #             f.flush()
    # if argsDict['method'] == "layerActUIE":

    #     with open(os.path.join(argsDict['uncertaintyFile']), "r") as f:
    #         uncertaintyList = [json.loads(line) for line in f.readlines()]
    #     for data in uncertaintyList:
    #         formatUncertanity = data['normalizeUncertainty']['formatUncertanity']
    #         contentUncertanity = data['normalizeUncertainty']['contentUncertanity']
    #         data['uncertainty'] = argsDict['alpha'] * formatUncertanity + argsDict['beta'] * contentUncertanity
    #     uncertaintyList = sorted(uncertaintyList, key=lambda x: x['uncertainty'], reverse=True)
    #     bestSize = int(len(uncertaintyList)*argsDict['frontArgs'])
    #     sampleList = []
    #     if bestSize < argsDict['shotSize']:
    #         argsDict['shotSize'] = bestSize
    #     for i in range(argsDict['shotSize']):
    #         sampleList.append(uncertaintyList[i*(len(uncertaintyList)//argsDict['shotSize'])])

    #     with open(os.path.join(argsDict['outputFile']), "w") as f:
    #         for data in dataList:
    #             instruction = tools.getInstruction(schema, method=argsDict['method'], sampleList=sampleList)

    #             inferencelogger.info("--------\nInstruction: %s", instruction)
    #             inferencelogger.info(f"Processing text: {data['text']}:")

    #             response = model.generate(instruction = instruction, text = data['text'])
    #             inferencelogger.info("Origion response: %s", response)
    #             response = tools.getJsonFormat(response, schema)
    #             inferencelogger.info("JsonFormat response: %s", response)

    #             if tools.validateJsonFormat(response, schema) == False:
    #                 inferencelogger.error("Invalid JsonFormat response: %s", response)
    #                 response = None
    #                 continue
    #             inferencelogger.info("Final response: %s\n", response)

    #             data['response'] = response
    #             f.write(json.dumps(data, ensure_ascii=False) + "\n")
    #             f.flush()

    # if argsDict['method'] == "ELDSort":
    #     for data in dataList:
    #         data['ELD'] = tools.getELD(data)
    #     sortedData = sorted(dataList, key=lambda x: x['ELD'], reverse=True)
    #     bestSize = int(len(sortedData)*argsDict['frontArgs'])
    #     if bestSize < argsDict['shotSize']:
    #         argsDict['shotSize'] = bestSize
    #     sortedData = sortedData[:bestSize]
    #     actuielogger.info(f"Uncertainty sample list:{sortedData}")

    #     with open(os.path.join(argsDict['outputFile']), "w") as f:
    #         for data in dataList:
    #             sampleList = random.sample(sortedData, argsDict['shotSize'])
    #             instruction = tools.getInstruction(schema, method=argsDict['method'], sampleList=sampleList)

    #             inferencelogger.info("--------\nInstruction: %s", instruction)
    #             inferencelogger.info(f"Processing text: {data['text']}:")

    #             response = model.generate(instruction = instruction, text = data['text'])
    #             inferencelogger.info("Origion response: %s", response)
    #             response = tools.getJsonFormat(response, schema)
    #             inferencelogger.info("JsonFormat response: %s", response)

    #             if tools.validateJsonFormat(response, schema) == False:
    #                 inferencelogger.error("Invalid JsonFormat response: %s", response)
    #                 response = None
    #                 continue
    #             inferencelogger.info("Final response: %s\n", response)

    #             data['response'] = response
    #             f.write(json.dumps(data, ensure_ascii=False) + "\n")
    #             f.flush()

    # if argsDict['method'] == "ActPrompt":
    #     with open(os.path.join(argsDict['uncertaintyFile']), "r") as f:
    #         uncertaintyList = [json.loads(line) for line in f.readlines()]
    #     for data in uncertaintyList:
    #         data['uncertainty'] = data['normalizeUncertainty']['editUncertanity']
    #     sortedData = sorted(uncertaintyList, key=lambda x: x['uncertainty'], reverse=True)
    #     bestSize = int(len(sortedData)*argsDict['frontArgs'])
    #     if bestSize < argsDict['shotSize']:
    #         argsDict['shotSize'] = bestSize
    #     sortedData = sortedData[:bestSize]
    #     actuielogger.info(f"Uncertainty sample list:{sortedData}")

    #     with open(os.path.join(argsDict['outputFile']), "w") as f:
    #         for data in dataList:
    #             sampleList = random.sample(sortedData, argsDict['shotSize'])
    #             instruction = tools.getInstruction(schema, method=argsDict['method'], sampleList=sampleList)

    #             inferencelogger.info("--------\nInstruction: %s", instruction)
    #             inferencelogger.info(f"Processing text: {data['text']}:")

    #             response = model.generate(instruction = instruction, text = data['text'])
    #             inferencelogger.info("Origion response: %s", response)
    #             response = tools.getJsonFormat(response, schema)
    #             inferencelogger.info("JsonFormat response: %s", response)

    #             if tools.validateJsonFormat(response, schema) == False:
    #                 inferencelogger.error("Invalid JsonFormat response: %s", response)
    #                 response = None
    #                 continue
    #             inferencelogger.info("Final response: %s\n", response)

    #             data['response'] = response
    #             f.write(json.dumps(data, ensure_ascii=False) + "\n")
    #             f.flush()

    # if argsDict['method'] == "topActUIE":
    #     with open(os.path.join(argsDict['uncertaintyFile']), "r") as f:
    #         uncertaintyList = [json.loads(line) for line in f.readlines()]
    #     for data in uncertaintyList:
    #         formatUncertanity = data['normalizeUncertainty']['formatUncertanity']
    #         contentUncertanity = data['normalizeUncertainty']['contentUncertanity']
    #         data['uncertainty'] = argsDict['alpha'] * formatUncertanity + argsDict['beta'] * contentUncertanity
    #     uncertaintyList = sorted(uncertaintyList, key=lambda x: x['uncertainty'], reverse=True)
    #     uncertaintyList = uncertaintyList[:argsDict['shotSize']]
    #     actuielogger.info(f"Uncertainty sample list:{uncertaintyList}")

    #     with open(os.path.join(argsDict['outputFile']), "w") as f:
    #         sampleList = random.sample(uncertaintyList, argsDict['shotSize'])
    #         instruction = tools.getInstruction(schema, method=argsDict['method'], sampleList=sampleList)

    #         inferencelogger.info("--------\nInstruction: %s", instruction)

    #         for data in dataList:
    #             inferencelogger.info(f"Processing text: {data['text']}:")
    #             response = model.generate(instruction = instruction, text = data['text'])
    #             inferencelogger.info("Origion response: %s", response)
    #             response = tools.getJsonFormat(response, schema)
    #             inferencelogger.info("JsonFormat response: %s", response)

    #             if tools.validateJsonFormat(response, schema) == False:
    #                 inferencelogger.error("Invalid JsonFormat response: %s", response)
    #                 response = None
    #                 continue
    #             inferencelogger.info("Final response: %s\n", response)

    #             data['response'] = response
    #             f.write(json.dumps(data, ensure_ascii=False) + "\n")
    #             f.flush()
    # if argsDict['method'] == "ActPromptUIE":
    #     with open(os.path.join(argsDict['uncertaintyFile']), "r") as f:
    #         uncertaintyList = [json.loads(line) for line in f.readlines()]
    #     for data in uncertaintyList:
    #         formatUncertanity = data['normalizeUncertainty']['formatUncertanity']
    #         contentUncertanity = data['normalizeUncertainty']['contentUncertanity']
    #         editUncertanity = data['normalizeUncertainty']['editUncertanity']
    #         data['allUncertainty'] = argsDict['alpha'] * formatUncertanity + argsDict['beta'] * contentUncertanity+ argsDict['gama'] * editUncertanity
    #     sortedData = sorted(uncertaintyList, key=lambda x: x['allUncertainty'], reverse=True)
    #     bestSize = int(len(sortedData)*argsDict['frontArgs'])
    #     if bestSize < argsDict['shotSize']:
    #         argsDict['shotSize'] = bestSize
    #     sortedData = sortedData[:bestSize]
    #     actuielogger.info(f"Uncertainty sample list:{sortedData}")

    #     with open(os.path.join(argsDict['outputFile']), "w") as f:
    #         for data in dataList:
    #             sampleList = random.sample(sortedData, argsDict['shotSize'])
    #             instruction = tools.getInstruction(schema, method=argsDict['method'], sampleList=sampleList)

    #             inferencelogger.info("--------\nInstruction: %s", instruction)
    #             inferencelogger.info(f"Processing text: {data['text']}:")

    #             response = model.generate(instruction = instruction, text = data['text'])
    #             inferencelogger.info("Origion response: %s", response)
    #             response = tools.getJsonFormat(response, schema)
    #             inferencelogger.info("JsonFormat response: %s", response)

    #             if tools.validateJsonFormat(response, schema) == False:
    #                 inferencelogger.error("Invalid JsonFormat response: %s", response)
    #                 response = None
    #                 continue
    #             inferencelogger.info("Final response: %s\n", response)

    #             data['response'] = response
    #             f.write(json.dumps(data, ensure_ascii=False) + "\n")
    #             f.flush()
    # if argsDict['method'] == "topActUIEPrompt":
    #     with open(os.path.join(argsDict['uncertaintyFile']), "r") as f:
    #         uncertaintyList = [json.loads(line) for line in f.readlines()]
    #     for data in uncertaintyList:
    #         formatUncertanity = data['normalizeUncertainty']['formatUncertanity']
    #         contentUncertanity = data['normalizeUncertainty']['contentUncertanity']
    #         editUncertanity = data['normalizeUncertainty']['editUncertanity']
    #         data['uncertainty'] = argsDict['alpha'] * formatUncertanity + argsDict['beta'] * contentUncertanity+ argsDict['gama'] * editUncertanity
    #     uncertaintyList = sorted(uncertaintyList, key=lambda x: x['uncertainty'], reverse=True)
    #     uncertaintyList = uncertaintyList[:argsDict['shotSize']]
    #     actuielogger.info(f"Uncertainty sample list:{uncertaintyList}")

    #     with open(os.path.join(argsDict['outputFile']), "w") as f:
    #         sampleList = random.sample(uncertaintyList, argsDict['shotSize'])
    #         instruction = tools.getInstruction(schema, method=argsDict['method'], sampleList=sampleList)

    #         inferencelogger.info("--------\nInstruction: %s", instruction)

    #         for data in dataList:
    #             inferencelogger.info(f"Processing text: {data['text']}:")
    #             response = model.generate(instruction = instruction, text = data['text'])
    #             inferencelogger.info("Origion response: %s", response)
    #             response = tools.getJsonFormat(response, schema)
    #             inferencelogger.info("JsonFormat response: %s", response)

    #             if tools.validateJsonFormat(response, schema) == False:
    #                 inferencelogger.error("Invalid JsonFormat response: %s", response)
    #                 response = None
    #                 continue
    #             inferencelogger.info("Final response: %s\n", response)

    #             data['response'] = response
    #             f.write(json.dumps(data, ensure_ascii=False) + "\n")
    #             f.flush()
    # if argsDict['method'] == "topActPrompt":
    #     with open(os.path.join(argsDict['uncertaintyFile']), "r") as f:
    #         uncertaintyList = [json.loads(line) for line in f.readlines()]
    #     for data in uncertaintyList:
    #         data['uncertainty'] = data['normalizeUncertainty']['editUncertanity']
    #     uncertaintyList = sorted(uncertaintyList, key=lambda x: x['uncertainty'], reverse=True)
    #     uncertaintyList = uncertaintyList[:argsDict['shotSize']]
    #     actuielogger.info(f"Uncertainty sample list:{uncertaintyList}")

    #     with open(os.path.join(argsDict['outputFile']), "w") as f:
    #         sampleList = random.sample(uncertaintyList, argsDict['shotSize'])
    #         instruction = tools.getInstruction(schema, method=argsDict['method'], sampleList=sampleList)

    #         inferencelogger.info("--------\nInstruction: %s", instruction)

    #         for data in dataList:
    #             inferencelogger.info(f"Processing text: {data['text']}:")
    #             response = model.generate(instruction = instruction, text = data['text'])
    #             inferencelogger.info("Origion response: %s", response)
    #             response = tools.getJsonFormat(response, schema)
    #             inferencelogger.info("JsonFormat response: %s", response)

    #             if tools.validateJsonFormat(response, schema) == False:
    #                 inferencelogger.error("Invalid JsonFormat response: %s", response)
    #                 response = None
    #                 continue
    #             inferencelogger.info("Final response: %s\n", response)

    #             data['response'] = response
    #             f.write(json.dumps(data, ensure_ascii=False) + "\n")
    #             f.flush()
    # if argsDict['method'] == "fewRSL":
    #     with open(os.path.join(argsDict['outputFile']), "w") as f:
    #         sampleList = random.sample(dataList, argsDict['shotSize'])
    #         instruction = tools.getInstruction(schema, method=argsDict['method'], sampleList=sampleList)
    #         inferencelogger.info("--------\nInstruction: %s", instruction)

    #         for data in dataList:
    #             inferencelogger.info(f"Processing text: {data['text']}:")

    #             response = model.generate(instruction = instruction, text = data['text'])
    #             inferencelogger.info("Origion response: %s", response)
    #             response = tools.getJsonFormat(response, schema)
    #             inferencelogger.info("JsonFormat response: %s", response)

    #             if tools.validateJsonFormat(response, schema) == False:
    #                 inferencelogger.error("Invalid JsonFormat response: %s", response)
    #                 response = None
    #                 continue
    #             inferencelogger.info("Final response: %s\n", response)

    #             data['response'] = response
    #             f.write(json.dumps(data, ensure_ascii=False) + "\n")
    #             f.flush()
    # if argsDict['method'] == "topELDSort":
    #     for data in dataList:
    #         data['ELD'] = tools.getELD(data)
    #     sortedData = sorted(dataList, key=lambda x: x['ELD'], reverse=True)
    #     sortedData = sortedData[:argsDict['shotSize']]
    #     instruction = tools.getInstruction(schema, method=argsDict['method'], sampleList=sortedData)
    #     inferencelogger.info("--------\nInstruction: %s", instruction)

    #     with open(os.path.join(argsDict['outputFile']), "w") as f:
    #         for data in dataList:
    #             inferencelogger.info(f"Processing text: {data['text']}:")

    #             response = model.generate(instruction = instruction, text = data['text'])
    #             inferencelogger.info("Origion response: %s", response)
    #             response = tools.getJsonFormat(response, schema)
    #             inferencelogger.info("JsonFormat response: %s", response)

    #             if tools.validateJsonFormat(response, schema) == False:
    #                 inferencelogger.error("Invalid JsonFormat response: %s", response)
    #                 response = None
    #                 continue
    #             inferencelogger.info("Final response: %s\n", response)

    #             data['response'] = response
    #             f.write(json.dumps(data, ensure_ascii=False) + "\n")
    #             f.flush()
    # if argsDict['method'] == "threeTop":
    #     with open(os.path.join(argsDict['uncertaintyFile']), "r") as f:
    #         uncertaintyList = [json.loads(line) for line in f.readlines()]
    #     for data in uncertaintyList:
    #         data['formatUncertanity'] = data['normalizeUncertainty']['formatUncertanity']
    #         data['contentUncertanity'] = data['normalizeUncertainty']['contentUncertanity']
    #         data['editUncertanity'] = data['normalizeUncertainty']['editUncertanity']
    #         # data['uncertainty'] = argsDict['alpha'] * formatUncertanity + argsDict['beta'] * contentUncertanity+ argsDict['gama'] * editUncertanity

    #     sampleList= []
    #     uncertaintyList = sorted(uncertaintyList, key=lambda x: x['formatUncertanity'], reverse=True)
    #     sampleList.append(uncertaintyList[0])
    #     uncertaintyList = sorted(uncertaintyList, key=lambda x: x['contentUncertanity'], reverse=True)
    #     sampleList.append(uncertaintyList[0])
    #     uncertaintyList = sorted(uncertaintyList, key=lambda x: x['editUncertanity'], reverse=True)
    #     sampleList.append(uncertaintyList[0])

    #     actuielogger.info(f"Uncertainty sample list:{sampleList}")

    #     with open(os.path.join(argsDict['outputFile']), "w") as f:
    #         instruction = tools.getInstruction(schema, method=argsDict['method'], sampleList=sampleList)

    #         inferencelogger.info("--------\nInstruction: %s", instruction)

    #         for data in dataList:
    #             inferencelogger.info(f"Processing text: {data['text']}:")
    #             response = model.generate(instruction = instruction, text = data['text'])
    #             inferencelogger.info("Origion response: %s", response)
    #             response = tools.getJsonFormat(response, schema)
    #             inferencelogger.info("JsonFormat response: %s", response)

    #             if tools.validateJsonFormat(response, schema) == False:
    #                 inferencelogger.error("Invalid JsonFormat response: %s", response)
    #                 response = None
    #                 continue
    #             inferencelogger.info("Final response: %s\n", response)

    #             data['response'] = response
    #             f.write(json.dumps(data, ensure_ascii=False) + "\n")
    #             f.flush()
    # if argsDict['method'] == "onlysample":
    #     with open(os.path.join(argsDict['uncertaintyFile']), "r") as f:
    #         uncertaintyList = [json.loads(line) for line in f.readlines()]
    #     for data in uncertaintyList:
    #         formatUncertanity = data['normalizeUncertainty']['formatUncertanity']
    #         contentUncertanity = data['normalizeUncertainty']['contentUncertanity']
    #         editUncertanity = data['normalizeUncertainty']['editUncertanity']
    #         data['uncertainty'] = argsDict['alpha'] * formatUncertanity + argsDict['beta'] * contentUncertanity+ argsDict['gama'] * editUncertanity
    #     uncertaintyList = sorted(uncertaintyList, key=lambda x: x['uncertainty'], reverse=True)
    #     uncertaintyList = uncertaintyList[:argsDict['shotSize']]
    #     actuielogger.info(f"Uncertainty sample list:{uncertaintyList}")

    #     with open(os.path.join(argsDict['outputFile']), "w") as f:
    #         sampleList = random.sample(uncertaintyList, argsDict['shotSize'])
    #         instruction = tools.getInstruction(schema, method=argsDict['method'], sampleList=sampleList)

    #         inferencelogger.info("--------\nInstruction: %s", instruction)

    #         for data in dataList:
    #             inferencelogger.info(f"Processing text: {data['text']}:")
    #             response = model.generate(instruction = instruction, text = data['text'])
    #             inferencelogger.info("Origion response: %s", response)
    #             response = tools.getJsonFormat(response, schema)
    #             inferencelogger.info("JsonFormat response: %s", response)

    #             if tools.validateJsonFormat(response, schema) == False:
    #                 inferencelogger.error("Invalid JsonFormat response: %s", response)
    #                 response = None
    #                 continue
    #             inferencelogger.info("Final response: %s\n", response)

    #             data['response'] = response
    #             f.write(json.dumps(data, ensure_ascii=False) + "\n")
    #             f.flush()
    