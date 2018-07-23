"""
A simple wrapper for linear regression.  (c) 2015 Tucker Balch
"""

import numpy as np

class RTLearner(object):

    def __init__(self, leaf_size=1, verbose = False):
        # move along, these aren't the drones you're looking for
        self.leaf_size = leaf_size
        self.verbose = verbose

    def author(self):
        return 'pbhatta3' # replace tb34 with your Georgia Tech username

    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """

        #print("DATA X \n", dataX)
        #print("DATA Y \n", dataY)

        #my coding
        #self.build_tree(dataX, dataY)
        self.tree = self.build_tree(dataX, dataY)
        #print("FINAL TREE === \n " , self.tree)

    def query(self,points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        #print("POINTS == \n" , points)

        prediction_y = []
        for point in points:
            split_factor, split_val = point[0:2]
            tree_row_start = 0

            # if split_factor == -1:
            #     print("SPLIT FACTOR -1")
            #     return split_val
            #
            # else:
            tree_rows_length = self.tree.shape[0]
            while tree_row_start <= tree_rows_length:
                while not (self.tree[tree_row_start][0] == -1.0):
                    # print("PASS")
                    # get split val index and split val
                    split_val_index = int(self.tree[tree_row_start][0])
                    # print("split_val_index == ", split_val_index)
                    split_val = self.tree[tree_row_start][1]
                    # print("Node is :: split_val == ", split_val)
                    # print(" Our point array is   ", point, "and we are splitting in point[split_val_index] == ", point[int(split_val_index)])
                    if point[int(split_val_index)] <= split_val:
                        # print("Point split val is " , point[split_val_index], " is less than the Node split val : " , split_val )
                        tree_row_start += 1
                        # print("NEXT Iterating through out build tree :  Iteration at Tree row start == ", tree_row_start)
                    elif point[int(split_val_index)] > split_val:
                        # print(
                        # "Point split val is ", point[split_val_index], " is greater than the Node split val : ", split_val)
                        tree_row_start += int(self.tree[tree_row_start][3])
                        # print("NEXT Iterating through out build tree :  Iteration at Tree row start == ", tree_row_start)

                # print("Leaf node achieved.. now calculating--")
                # print("split_val_index == ", split_val_index)
                # print("split_val == ", split_val)
                # print("Tree row start == ", tree_row_start)
                if point[split_val_index] <= split_val:
                    preditedYVal = self.tree[tree_row_start][1]
                    # print("The predicted Y value  Left Tree== ", preditedYVal)
                    # print("Tree row start == ", tree_row_start)
                    prediction_y.append(preditedYVal)

                elif point[split_val_index] > split_val:
                    preditedYVal = self.tree[tree_row_start][1]
                    # print("The predicted Y value  Right Tree== ", preditedYVal)
                    # print("Tree row start == ", tree_row_start)
                    prediction_y.append(preditedYVal)


                break

                # split_val_index = self.tree[tree_row][1]
                # split_val = self.tree[tree_row][1][2]

        #print("The predicted Y value == \n" , prediction_y)

        #return (self.model_coefs[:-1] * points).sum(axis = 1) + self.model_coefs[-1]
        return prediction_y

    def build_tree(self,dataX, dataY, recursion=0):
        NAN = np.NAN

        #one way of finding unique
        # a = np.array([1, 1, 1, 1])
        # unique_dataY = np.unique(dataY).size
        # #print("UNIQUE == ", unique_dataY)
        # print("Length == ", unique_dataY)

        min_dataY = dataY.min()
        max_dataY = dataY.max()
        #print("min_dataY ==", min_dataY, "max_dataY", max_dataY)

        #print("left Tree SHAPE == ", dataX.shape)
        #print("Right Tree SHAPE == ", dataY.shape)
        #print("===========")

        if dataX.shape[0] == 1 :
            #print(" leaf Node dataX.shape[0] == 1")
            return [-1, dataY[0], NAN, NAN]
        elif min_dataY == max_dataY :
            #print(" leaf Node min_dataY == max_dataY")
            return [-1, dataY[0], NAN, NAN]
        elif dataX.shape[0] <= self.leaf_size :
            return [-1, np.mean(dataY), NAN, NAN]

        # elif dataX.shape[0] < 1 :
        #     print(" leaf Node dataX.shape[0] == 1")
        #     return [-1, dataY[0], NAN, NAN]

        split_index , split_val = self.random_splitval_index_and_value(dataX, dataY)
        data_column = dataX[:,split_index] #split value column
        #print("SPLIT COLUMN DATA \n" , data_column)
        #print("Split valuee == ", split_val)

        #edge case
        if split_val == np.max(data_column):
            #print("MAXIMUM REACH EDGE CASE")
            return [-1, dataY.mean(), NAN, NAN]

        left_split_Xdata_index = np.where(data_column <= split_val) #get indices of left dataX
        #left_split_Xdata = data_column[(data_column <= split_val)]
        left_split_Xdata = np.array(dataX[(data_column <= split_val)])
        #print("LEFT SPLIT DATA NewDataX \n", left_split_Xdata)
        left_split_Ydata = np.array(dataY[left_split_Xdata_index])
        #print("LEFT SPLIT DATA NewDataY \n", left_split_Ydata)

        right_split_Xdata_index = np.where(data_column > split_val)  # get indices of right dataX
        right_split_Xdata = np.array(dataX[(data_column > split_val)])
        #print("RIGHT SPLIT DATA NewDataX \n", right_split_Xdata)
        right_split_Ydata = np.array(dataY[right_split_Xdata_index])
        #print("RIGHT SPLIT DATA NewDataY \n", right_split_Ydata)

        #handle another edge case
        if len(right_split_Xdata) <=0 or len(left_split_Xdata) <=0:
            #print("ANOTHER MAXIMUM EDGE CASE")
            return [-1, dataY.mean(), NAN, NAN]

        recursion = recursion + 1
        #print("!!!!!!TREE BEGINS!!!!!!! RECURSION == ", recursion )


        #print(type(left_split_Xdata), type(left_split_Ydata))


        lefttree = self.build_tree(left_split_Xdata, left_split_Ydata, recursion=recursion)
        #print("LEFT TREE == \n", lefttree)

        #print("RIGHT TREE STARTS")
        righttree = self.build_tree(right_split_Xdata, right_split_Ydata, recursion = recursion)
        #print("RIGHT TREE == \n", righttree)

        #root=	[i,	SplitVal, 1, lefttree.shape[0] + 1]
        #i = "Rootx" + str(split_index)
        i = split_index
        #print("ROOOT == " , i)
        x = np.array(lefttree)
        #print("!!!!!!!!!!!!!!!!")
        lefttree = np.array(lefttree)
        righttree = np.array(righttree)
        #print("left Tree SHAPE == ", lefttree.shape, lefttree.ndim)
        #print("Right Tree SHAPE == ", righttree.shape, righttree.ndim)
        #print("TYPEEE == " , type(lefttree), type(righttree))

        #print(x)
        #print(type(lefttree))


        #compute the last value for right tree position
        if lefttree.ndim == 1 :
            right_tree_start = 2
        elif lefttree.ndim >1 :
            right_tree_start = lefttree.shape[0] + 1
        else:
            return [-1, dataY[0], NAN, NAN]

        root = np.array([[i, split_val, 1, right_tree_start]])
        #print("ROOOT ==== ", root)

        #print("ROOOT Shape ==== ", root.shape)
        #print("ROOT == \n", root)
        #print("LEFT TREE Shape == \n ", lefttree.shape)
        #print("LEFT TREE FINAL == \n ", lefttree)
        #print("RIGHT TREE Shape == \n ", righttree.shape)
        #print("Right TREE FINAL == \n ", righttree)
        #print("TREE TABLE=== \n", np.vstack((root, lefttree, righttree)).shape)
        return np.vstack((root,lefttree,righttree))
        #print("FINAL QUERY == \n", np.vstack((root,lefttree,righttree)))
        # returnVal = np.append(root,lefttree,righttree)
        #print("Return value \n ==", returnVal)
        # return returnVal





    def random_splitval_index_and_value(self, dataX, dataY):
        #print("We are in cal_cor_coef function!!")
        #print("DATAX == \n " , dataX)
        #print("DATAY == \n ", dataY)
        split_val_index = 0
        no_of_columns = int(dataX.shape[1])
        # print(no_of_columns +2)
        #print("dataX.ndim == ", dataX.ndim, "--- ", "dataX.shape ==", dataX.shape)
        #print("dataY.ndim == ", dataY.ndim, "--- ", "dataY.shape ==", dataY.shape)

        #split_val_index = np.argmax(correlation_array)
        split_val_index = np.random.randint(0, dataX.ndim)
        #split_val_index = random.randint(0, dataX.shape[0]-1)
        split_val_random_index = np.random.randint(0, dataX.ndim)

        #print("split_val_random_index == ", split_val_index, type(split_val_index))


        split_col_val = dataX[:,split_val_index]
        #print("Data column random == \n", split_col_val)

        #Not required as per the lecture
        # random_split_val1 = np.random.choice(split_col_val)
        # print("random_split_val1 ==", random_split_val1)
        # random_split_val2 = np.random.choice(split_col_val)
        # print("random_split_val2 ==", random_split_val2)
        # avg_random_vals = (random_split_val1+random_split_val2)/2


        split_val_col_sort = np.sort(split_col_val)
        # print("Split Val Column Sort == \n", split_val_col_sort)

        split_val_col_sort_median = np.median(split_val_col_sort)
        #print("Split Val Column Sort Median == \n", split_val_col_sort_median)

        # print("Correlation Array == ", correlation_array)
        #print("Split Val index ==", split_val_index)

        return split_val_index, split_val_col_sort_median


if __name__=="__main__":
    print ("the secret clue is 'zzyzx'")

