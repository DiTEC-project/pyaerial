# this script is taken from: https://github.com/TheophileBERTELOOT/ARM-AE/blob/master/ARMAE.py
# Berteloot, Théophile, Richard Khoury, and Audrey Durand. "Association rules mining with auto-encoders." International
# Conference on Intelligent Data Engineering and Automated Learning. Cham: Springer Nature Switzerland, 2024.

import torch
import numpy as np
from torch.autograd import Variable
from torch.nn import L1Loss
from torch.utils.data import DataLoader

from illustrative_experiments.algorithms.arm_ae_source.autoencoder import AutoEncoder
from illustrative_experiments.util.rule_quality import *

import copy
import time


class ARMAE:
    def __init__(self, dataSize, learningRate=1e-2, maxEpoch=2,
                 batchSize=128, hiddenSize='dataSize', likeness=0.5, columns=[], isLoadedModel=False,
                 IM=['support', 'confidence']):
        self.arm_ae_training_time = 0
        self.exec_time = None
        self.dataSize = dataSize
        self.learningRate = learningRate
        self.likeness = likeness
        self.IM = IM
        self.hiddenSize = hiddenSize
        self.isLoadedModel = isLoadedModel
        self.columns = columns
        if self.hiddenSize == 'dataSize':
            self.hiddenSize = self.dataSize
        self.maxEpoch = maxEpoch
        self.x = []
        self.y_ = []
        self.batchSize = batchSize
        self.model = AutoEncoder(self.dataSize)
        self.criterion = L1Loss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learningRate)
        self.dataset_coverage = []

        self.results = []

    def dataPreprocessing(self, d):
        self.columns = d.columns
        trainTensor = torch.tensor(d.values)
        dataLoader = DataLoader(trainTensor.float(), batch_size=self.batchSize, shuffle=True)
        x = torch.tensor([float('nan'), float('inf'), -float('inf'), 3.14])
        torch.nan_to_num(x, nan=0.0, posinf=0.0)
        return dataLoader

    def save(self, p):
        self.model.save(p)

    def load(self, encoderPath, decoderPath):
        self.model.load(encoderPath, decoderPath)

    def train(self, dataLoader):
        armae_training_start = time.time()
        for epoch in range(self.maxEpoch):
            for data in dataLoader:
                d = Variable(data)
                output = self.model.forward(d)
                self.y_ = output[0]
                self.x = d
                loss = self.criterion(output[0], d)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.arm_ae_training_time = time.time() - armae_training_start

    def computeMeasures(self, antecedent, consequent, data):
        # individual rule coverage of ARM-AE is not considered in the evaluation
        # instead, only the total data coverage of the final rules will be calculated
        measures = {"support": 0, "confidence": 0, "coverage": 0}
        self.dataset_coverage[antecedent] = 1

        if 'support' in self.IM:
            rules = copy.deepcopy(antecedent)
            rules.append(consequent)
            PAC = data[data.columns[rules]]
            PAC = np.sum(PAC, axis=1)
            PAC = PAC == len(rules)
            PAC = np.sum(PAC)
            PAC = PAC / len(data)
            support = round(PAC, 2)
            measures["support"] = support
        if 'confidence' in self.IM:
            PA = data[data.columns[antecedent]]
            PA = np.sum(PA, axis=1)
            PA = PA == len(antecedent)
            PA = np.sum(PA)
            PA = PA / len(data)
            if PA != 0:
                conf = PAC / PA
            else:
                conf = 0
            confidence = round(conf, 2)
            measures["confidence"] = confidence
        # the zhang's metric calculation is added to ARM-AE later on by us
        # if 'zhangs_metric' in self.IM:
        #     PA = data[data.columns[antecedent]]
        #     PA = np.sum(PA, axis=1)
        #     PA = PA == len(antecedent)
        #     PA = np.sum(PA)
        #     PC = data[data.columns[antecedent]]
        #     PC = np.sum(PC, axis=1)
        #     PC = PC == len(antecedent)
        #     PC = np.sum(PC)
        #     zhangs = calculate_zhangs_metric(support, (PA / len(data)), (PC / len(data)))
        #     measures["zhangs_metric"] = zhangs
        return measures

    def computeSimilarity(self, allAntecedents, antecedentsArray, nbantecedent):
        onlySameSize = [x for x in allAntecedents if len(x) >= len(antecedentsArray)]
        maxSimilarity = 0
        for antecedentIndex in range(len(onlySameSize)):
            antecedents = onlySameSize[antecedentIndex]
            similarity = 0
            for item in antecedents:
                if item in antecedentsArray:
                    similarity += 1
            similarity /= nbantecedent
            if similarity > maxSimilarity:
                maxSimilarity = similarity
        return maxSimilarity

    def generateRules(self, data, numberOfRules=2, nbAntecedent=2):
        timeCreatingRule = 0
        timeComputingMeasure = 0
        self.dataset_coverage = np.zeros(len(data.loc[0]))

        for consequent in range(self.dataSize):
            allAntecedents = []
            for j in range(numberOfRules):
                antecedentsArray = []
                for i in range(nbAntecedent):
                    t1 = time.time()
                    consequentArray = np.zeros(self.dataSize)
                    consequentArray[consequent] = 1
                    consequentArray[antecedentsArray] = 1
                    consequentArray = torch.tensor(consequentArray)
                    consequentArray = consequentArray.unsqueeze(0)
                    output = self.model(consequentArray.float())
                    output = output.cpu()
                    output = np.array(output.detach().numpy())
                    output = pd.DataFrame(output.reshape(self.dataSize, -1))
                    potentialAntecedentsArray = output[0].nlargest(len(data.loc[0]))
                    for antecedent in potentialAntecedentsArray.keys():
                        potentialAntecedents = copy.deepcopy(antecedentsArray)
                        potentialAntecedents.append(antecedent)
                        potentialAntecedents = sorted(potentialAntecedents)
                        if antecedent != consequent and antecedent not in antecedentsArray and self.computeSimilarity(
                                allAntecedents, potentialAntecedents, nbAntecedent) <= self.likeness:
                            antecedentsArray.append(antecedent)
                            break
                    t2 = time.time()
                    measures = self.computeMeasures(copy.deepcopy(antecedentsArray), consequent, data)
                    t3 = time.time()
                    ruleProperties = {"antecedents": list(sorted(copy.deepcopy(antecedentsArray))),
                                      "consequent": [consequent]}
                    ruleProperties = ruleProperties | measures
                    self.results.append(ruleProperties)
                    allAntecedents.append(sorted(copy.deepcopy(antecedentsArray)))
                    timeCreatingRule += t2 - t1
                    timeComputingMeasure += t3 - t2

        self.exec_time = timeCreatingRule

    def reformat_rules(self, data, data_columns):
        """
        Aerial+:
        ARM-AE generates rules in the form of indexes, e.g. feature1 implies feature2
        This method added to ARM-AE code to reformat the rules and use feature names instead
        In addition, it also calculates other rule quality measures besides support and confidence
        :param data_columns:
        :param data: non-formatted data (not one-hot encoded)
        :return:
        """
        formatted_rules = []
        for rule in self.results:
            # ignore the rules with 0 support value (as in the original paper)
            if rule['support'] == 0:
                continue
            # replace feature indexes in antecedent
            for index in range(len(rule['antecedents'])):
                rule['antecedents'][index] = data_columns[rule['antecedents'][index]]
            # replace feature indexes in consequent
            for index in range(len(rule['consequent'])):
                rule['consequent'][index] = data_columns[rule['consequent'][index]]
            rule_stats = {"antecedents": rule['antecedents'], "consequent": rule['consequent'],
                          "support": rule['support'], "confidence": rule['confidence']}
            formatted_rules.append(rule_stats)

        stats = mlxtend_calculate_average_rule_quality(formatted_rules)
        stats["coverage"] = sum(self.dataset_coverage) / len(data)
        return {
            "rule_count": len(formatted_rules),
            "average_support": stats['support'],
            "average_confidence": stats["confidence"],
            "data_coverage": stats["coverage"],
            "exec_time": self.exec_time + self.arm_ae_training_time
        }
