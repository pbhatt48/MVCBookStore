"""
A simple wrapper for linear regression.  (c) 2015 Tucker Balch
"""

import numpy as np
import random
import RTLearner


class BagLearner(object):
    # learner = bl.BagLearner(learner=al.ArbitraryLearner, kwargs={"argument1": 1, "argument2": 2}, bags=20, boost=False,
    #                         verbose=False)
    # learner.addEvidence(Xtrain, Ytrain)

    def __init__(self, learner=RTLearner,  bags=20, boost = False, verbose = False, kwargs= {"leaf_size":1}):
        # move along, these aren't the drones you're looking for
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.verbose = verbose
        learners = []
        for i in range(bags):
            learners.append(learner(**kwargs))
        self.learners = learners



    def author(self):
        return 'pbhatta3' # replace tb34 with your Georgia Tech username

    def addEvidence(self,dataX,dataY):
        #print("Add evidence of bagLearner")
        # num_of_rows = dataX.shape[0]
        # print("num of rows == ", num_of_rows)
        # num_of_items_for_bagging = int(0.6 * num_of_rows)
        # factor_index = np.random.randint(0,num_of_rows, num_of_items_for_bagging )
        # #factor_index = np.random.choice(dataX, num_of_items_for_bagging, replace=True)
        # bag_dataX = dataX[factor_index]
        # bag_dataY = dataY[factor_index]
        # print("factor_index ==", factor_index)
        # print("bag_dataX ==", bag_dataX)
        # print("bag_dataY ==", bag_dataY)

        for learner in self.learners:
            num_of_rows = dataX.shape[0]
            #print("num of rows == ", num_of_rows)
            #num_of_items_for_bagging = int(0.6 * num_of_rows)
            #factor_index = np.random.randint(0, num_of_rows, num_of_items_for_bagging)
            factor_index = np.random.randint(0, num_of_rows, num_of_rows)
            # factor_index = np.random.choice(dataX, num_of_items_for_bagging, replace=True)
            bag_dataX = dataX[factor_index]
            bag_dataY = dataY[factor_index]
            #print("factor_index ==", factor_index)
            #print("bag_dataX ==", bag_dataX)
            #print("bag_dataY ==", bag_dataY)
            learner.addEvidence(bag_dataX, bag_dataY)


    def query(self,points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        #print("POINTS == \n" , points)
        pred_y =[]
        for learner in self.learners:
            each_learner_predY= learner.query(points)
            #print("each_learner_predY ", each_learner_predY)
            pred_y.append(each_learner_predY)
        #print("FINAL PREDICTION ARRAY BEFORE MEAN == ", pred_y)
        pred_y_final = np.mean(pred_y, axis=0)
        #print("FINAL PREDICTION ARRAY AFTER MEAN == ", pred_y_final)
        return pred_y_final

if __name__=="__main__":
    print ("the secret clue is 'zzyzx'")

