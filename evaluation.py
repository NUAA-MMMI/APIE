import json
import os

from scripts.args import EvaluationArgs
from setting.loggerConfig import getLogger

import scripts.tools as tools

if __name__ == "__main__":
    # Parse command line arguments
    args = EvaluationArgs().parse()
    # Convert the parsed arguments to a dictionary
    argsDict = vars(args)

    #* Initialize logger
    evaluationLogger = getLogger("evaluation")

    evaluationLogger.info("Starting evaluation process with arguments: %s", json.dumps(argsDict, indent=2))


    #* Load the test data and schema from the input directory
    with open(argsDict["inputFile"], "r") as f:
        dataList = [json.loads(line) for line in f.readlines()]
    with open(argsDict['schema'], "r") as f:
        schema = json.loads(f.read())
    
    failNumber = 0
    with open(argsDict["outputFile"],"w") as f:
        for data in dataList:
            #! 假设：data包含正确的standard项，但response项不确定
            if data['response'] == None:
                data['response'] = {}
                failNumber += 1
                if "NER" in schema['tasks']:
                    data['response']['entitiesAccuracy'] = 0.0
                    data['response']['entitiesMicrof1'] = 0.0

                if "RE" in schema['tasks']:
                    data['response']['relationsAccuracy'] = 0.0
                    data['response']['relationsMicrof1'] = 0.0

                if "EE" in schema['tasks']:
                    data['response']['eventsAccuracy'] = 0.0
                    data['response']['eventsMicrof1'] = 0.0
            else:
                if "NER" in schema['tasks']:
                    standardEntities = tools.list2set(data['standard']['entities'])
                    responseEntities = tools.list2set(data['response']['entities'])
                    TP, FP, FN, TN = tools.indicator(standardEntities, responseEntities)

                    entitiesAccuracy = tools.accuracy(TP, FP, FN, TN)
                    entitiesMicrof1 = tools.f1score(TP, FP, FN)

                    data['response']['entitiesAccuracy'] = entitiesAccuracy
                    data['response']['entitiesMicrof1'] = entitiesMicrof1

                if "RE" in schema['tasks']:
                    standardRelations = tools.list2set(data['standard']['relations'])
                    responseRelations = tools.list2set(data['response']['relations'])
                    TP, FP, FN, TN = tools.indicator(standardRelations, responseRelations)

                    relationsAccuracy = tools.accuracy(TP, FP, FN, TN)
                    relationsMicrof1 = tools.f1score(TP, FP, FN)

                    data['response']['relationsAccuracy'] = relationsAccuracy
                    data['response']['relationsMicrof1'] = relationsMicrof1

                if "EE" in schema['tasks']:
                    standardEvents = tools.list2set(data['standard']['events'])
                    responseEvents = tools.list2set(data['response']['events'])
                    TP, FP, FN, TN = tools.indicator(standardEvents, responseEvents)

                    eventsAccuracy = tools.accuracy(TP, FP, FN, TN)
                    eventsMicrof1 = tools.f1score(TP, FP, FN)

                    data['response']['eventsAccuracy'] = eventsAccuracy
                    data['response']['eventsMicrof1'] = eventsMicrof1

            f.write(json.dumps(data, ensure_ascii=False) + "\n")
            f.flush()
    with open(argsDict['outputFile'], "r") as f:
        dataList = [json.loads(line) for line in f.readlines()]
        failRate = failNumber / len(dataList) if len(dataList) > 0 else 0.0
        argsDict['failRate'] = failRate
    
        if "NER" in schema['tasks']:
            entitiesAccuracyList = [data['response']['entitiesAccuracy'] for data in dataList]
            entitiesMicrof1List = [data['response']['entitiesMicrof1'] for data in dataList]
            averageEntityAccuracy = sum(entitiesAccuracyList) / len(entitiesAccuracyList) if entitiesAccuracyList else 0.0
            averageEntityMicrof1 = sum(entitiesMicrof1List) / len(entitiesMicrof1List) if entitiesMicrof1List else 0.0
            print(f"Entity Accuracy: {averageEntityAccuracy}, Entity Micro F1: {averageEntityMicrof1}")
            evaluationLogger.info(f"Entity Accuracy: {averageEntityAccuracy}, Entity Micro F1: {averageEntityMicrof1}")
            argsDict['averageEntityAccuracy'] = averageEntityAccuracy
            argsDict['averageEntityMicrof1'] = averageEntityMicrof1
        if "RE" in schema['tasks']:
            relationsAccuracyList = [data['response']['relationsAccuracy'] for data in dataList]
            relationsMicrof1List = [data['response']['relationsMicrof1'] for data in dataList]
            averageRelationAccuracy = sum(relationsAccuracyList) / len(relationsAccuracyList) if relationsAccuracyList else 0.0
            averageRelationMicrof1 = sum(relationsMicrof1List) / len(relationsMicrof1List) if relationsMicrof1List else 0.0
            print(f"Relation Accuracy: {averageRelationAccuracy}, Relation Micro F1: {averageRelationMicrof1}")
            evaluationLogger.info(f"Relation Accuracy: {averageRelationAccuracy}, Relation Micro F1: {averageRelationMicrof1}")
            argsDict['averageRelationAccuracy'] = averageRelationAccuracy
            argsDict['averageRelationMicrof1'] = averageRelationMicrof1
        if "EE" in schema['tasks']:
            eventsAccuracyList = [data['response']['eventsAccuracy'] for data in dataList]
            eventsMicrof1List = [data['response']['eventsMicrof1'] for data in dataList]
            averageEventAccuracy = sum(eventsAccuracyList) / len(eventsAccuracyList) if eventsAccuracyList else 0.0
            averageEventMicrof1 = sum(eventsMicrof1List) / len(eventsMicrof1List) if eventsMicrof1List else 0.0
            print(f"Event Accuracy: {averageEventAccuracy}, Event Micro F1: {averageEventMicrof1}")
            evaluationLogger.info(f"Event Accuracy: {averageEventAccuracy}, Event Micro F1: {averageEventMicrof1}")
            argsDict['averageEventAccuracy'] = averageEventAccuracy
            argsDict['averageEventMicrof1'] = averageEventMicrof1
    
    with open(argsDict['recordFile'], "a") as f:
        f.write(json.dumps(argsDict, ensure_ascii=False) + "\n")
        f.flush()