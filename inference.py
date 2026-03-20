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
        client = OpenAI(api_key="", base_url="https://api.deepseek.com")
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

    if not os.path.exists(os.path.join(os.path.dirname(argsDict['outputFile']))):
        os.makedirs(os.path.join(os.path.dirname(argsDict['outputFile'])), exist_ok=True)

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
    