import json
import random
import time

from openai import OpenAI

import scripts.tools as tools

from scripts.args import UncertaintyArgs
from models.adaptor import ModelAdaptor

from setting.loggerConfig import getLogger

if __name__ == "__main__":
    # Parse command line arguments
    args = UncertaintyArgs().parse()
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
    uncertaintyLogger = getLogger("uncertainty")

    uncertaintyLogger.info("Starting uncertainty process with arguments: %s", json.dumps(argsDict, indent=2))

    with open(argsDict['inputFile'], "r") as f:
        dataList = [json.loads(line) for line in f.readlines()]
    if argsDict['pollSize'] > len(dataList) or argsDict['pollSize'] == -1:
        argsDict['pollSize'] = len(dataList)
    if argsDict['pollSize'] == -2:
        argsDict['pollSize'] = len(dataList)//10
    dataList = random.sample(dataList,argsDict['pollSize'])
    with open(argsDict['schema'], "r") as f:
        schema = json.loads(f.read())

    with open(argsDict['uncertaintyFile'], "w") as f:
        for data in dataList:
            sampleList = random.sample(dataList, argsDict['shotSize'])
            instruction = tools.getInstruction(schema, method=argsDict['method'], sampleList=sampleList)

            uncertaintyLogger.info("--------\nInstruction: %s", instruction)
            uncertaintyLogger.info(f"Processing text: {data['text']}:")

            entitiesList = []
            relationsList = []
            eventsList = []
            responseList = []
            originResponseList = []


            formatUncertanity = 0
            contentUncertanity = 0

            for i in range(argsDict['responseSize']):
                uncertaintyLogger.info(f"Processing response: {i+1}")

                response = model.generate(instruction = instruction, text = data['text'])
                originResponseList.append(response)
                uncertaintyLogger.info("Origion response: %s", response)
                response = tools.getJsonFormat(response, schema)
                uncertaintyLogger.info("JsonFormat response: %s", response)

                if tools.validateJsonFormat(response, schema) == False:
                    uncertaintyLogger.error("Invalid JsonFormat response: %s", response)
                    response = None
                uncertaintyLogger.info("Final response: %s\n", response)

                if response != None:
                    if "NER" in schema['tasks']:
                        enitiesSet = tools.list2set(response['entities'])
                        uncertaintyLogger.info("Entities Set: %s", enitiesSet)
                        entitiesList.append(enitiesSet)
                    if "RE" in schema['tasks']:
                        relationsSet = tools.list2set(response['relations'])
                        uncertaintyLogger.info("Relations Set: %s", relationsSet)
                        relationsList.append(relationsSet)
                    if "EE" in schema['tasks']:
                        eventsSet = tools.list2set(response['events'])
                        uncertaintyLogger.info("Events Set: %s", eventsSet)
                        eventsList.append(eventsSet)
                else:
                    formatUncertanity += 1
                responseList.append(response)

            data['textUncertainty'] = tools.getTextUncertainty(originResponseList)

            similarity = {'entitySimilarity': 0, 'relationSimilarity': 0, 'eventSimilarity': 0}
            if "NER" in schema['tasks']:
                entitySimilarity = tools.similarity(entitiesList)
                similarity['entitySimilarity'] = entitySimilarity
                uncertaintyLogger.info("Entity Similarity: %s", entitySimilarity)

            if "RE" in schema['tasks']:
                relationSimilarity = tools.similarity(relationsList)
                similarity['relationSimilarity'] = relationSimilarity
                uncertaintyLogger.info("Relation Similarity: %s", relationSimilarity)

            if "EE" in schema['tasks']:
                eventSimilarity = tools.similarity(eventsList)
                similarity['eventSimilarity'] = eventSimilarity
                uncertaintyLogger.info("Event Similarity: %s", eventSimilarity)
            
            data['similarity'] = similarity
            data['formatUncertanity'] = formatUncertanity/argsDict['responseSize']
            data['contentUncertanity'] = -(similarity['entitySimilarity'] + similarity['relationSimilarity'] + similarity['eventSimilarity'])/len(similarity.keys())
            data['response'] = responseList
            data['originResponse'] = originResponseList
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
            f.flush()

        formatUncertanityList = [data['formatUncertanity'] for data in dataList]
        contentUncertanityList = [data['contentUncertanity'] for data in dataList]
        editUncertanityList = [data['textUncertainty'] for data in dataList]

        formatUncertanityList = tools.normalizeData(formatUncertanityList)
        contentUncertanityList = tools.normalizeData(contentUncertanityList)
        editUncertanityList = tools.normalizeData(editUncertanityList)

        with open(argsDict['uncertaintyFile'], "w") as f:
            for i in range(len(dataList)):
                dataList[i]['normalizeUncertainty'] = {
                    'formatUncertanity': formatUncertanityList[i],
                    'contentUncertanity': contentUncertanityList[i],
                    'editUncertanity': editUncertanityList[i]
                }
                f.write(json.dumps(dataList[i], ensure_ascii=False) + "\n")
                f.flush()
