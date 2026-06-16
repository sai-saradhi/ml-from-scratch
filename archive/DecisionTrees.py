import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def preprocess_data(data):
    data["Pass"] = (data["GPA"] >= 2.0).astype(int)
    
    data["ParentalEducation"] = (
        data["ParentalEducation"]
        .map({
            "High School": 0,
            "Some College": 1,
            "Bachelor": 2,
            "Higher": 3
        })
        .fillna(4)
    )

    data["ParentalSupport"] = (
        data["ParentalSupport"]
        .map({
            "Low": 0,
            "Moderate": 1,
            "High": 2,
            "Very High": 3
        })
        .fillna(4)
    )

    data = pd.get_dummies(
        data,
        columns=["Ethnicity"],
        dtype=int
    )

    return data


def train_test_split_numpy(X, y, train_ratio=0.8):
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    X = X[indices]
    y = y[indices]

    split_index = int(len(X) * train_ratio)

    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]

    return X_train, X_test, y_train, y_test

class Node:
    
    def __init__(self, feature_idx, threshold, left = None, right = None, value = None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:

    def __init__(self, max_depth = 10 , min_samples_split = 5, root = None):
        self.root = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
    
    def entropy(self, p):
        unique_vals, counts = np.unique(p, return_counts=True)
        p = counts/len(p)
        
        return -np.sum(p * np.log2(p))


    def information_gain(self, parent, left, right):
        return self.entropy(parent) - (((len(left)/len(parent) * self.entropy(left))) + ((len(right)/len(parent) * self.entropy(right))))


    def generate_thresholds(self, feature):
        t = feature.flatten()
        t = np.sort(t)
        t = np.unique((t[:-1] + t[1:])/2)
        return t


    def split_data(self, X, y, feature, threshold):
        left_x = X[feature <= threshold]
        right_x = X[feature > threshold]
        left_y = y[feature <= threshold]
        right_y = y[feature > threshold]
        return left_y, right_y, left_x, right_x
            

    def best_split(self, X, y):
        best_ig = float('-inf')
        best_feature = None
        
        for feature_idx, x in enumerate(X.T):
            
            thresholds = self.generate_thresholds(x)
            
            for t in thresholds:
                ly, ry, lx, rx = self.split_data(X, y, x, t)
                
                if len(ly) == 0 or len(ry) == 0:
                    continue
                
                ig = self.information_gain(y, ly, ry)
                
                if best_ig < ig:
                    best_ig = ig
                    best_threshold = t
                    best_feature = feature_idx
                    left_x = lx
                    right_x = rx
                    left_y = ly
                    right_y = ry
        
        if best_feature is None or best_ig <= 0: 
            return None, None, None, None, None, None
        
        return best_feature, best_threshold, left_x, right_x,left_y, right_y
    
    
    def build_tree(self, X, y, depth = 1):
        
        if depth > self.max_depth :
            return Node(None,None,None,None,np.argmax(np.bincount(y.flatten())))
        
        if len(y) < self.min_samples_split:
            return Node(None,None,None,None,np.argmax(np.bincount(y.flatten())))
        
        if len(np.unique(y)) == 1:
            node = Node(None,None,None,None,y[0]) 
            return node
        
        best_feature, best_threshold, left_x, right_x,left_y, right_y = self.best_split(X, y)
        
        if best_feature is None:
            return Node(None,None,None,None,np.argmax(np.bincount(y.flatten())))
        
        node = Node(best_feature, best_threshold)
        
        node.left = self.build_tree(left_x, left_y,depth = depth + 1)
        node.right = self.build_tree(right_x, right_y,depth = depth + 1)
        
        return node
        

    def fit(self, X, y, feature_names):
        self.root = self.build_tree(X, y)
        self.feature_names = feature_names
        return self.root


    def tree(self):
        self._print_tree(self.root)

    def _print_tree(self, node, depth=0):

        if node is None:
            return

        indent = "    " * depth

        if node.value is not None:
            print(f"{indent}Predict: {node.value}")
            return

        print(f"{indent}{self.feature_names[node.feature_idx]} <= {node.threshold}")

        print(f"{indent}├── True")
        self._print_tree(node.left, depth + 1)

        print(f"{indent}└── False")
        self._print_tree(node.right, depth + 1)
    

    def predict(self, sample):
        pred = []
        for x in sample:
            curr = self.root
            while curr.value == None:
                if x[curr.feature_idx] <= curr.threshold:
                    curr = curr.left
                else:
                    curr = curr.right
            pred.append(int(curr.value))    
        
        return pred       


    def acc(self, y, p):
        return np.mean(y == p)

# Main code

np.random.seed(49)

data = pd.read_csv("data.csv")

data = preprocess_data(data)

X = data[
    [
        "StudyTimeWeekly",
        "Absences",
        "Age",
        "Tutoring",
        "Sports",
        "Extracurricular",
        "ParentalSupport",
        "Music"
    ]
]

feature_names = list(X.columns)

X = X.values

y = data["Pass"].values

X_train, X_test, y_train, y_test = train_test_split_numpy(X, y)

DecisionTreeClassifier = DecisionTree(max_depth = 7)

DecisionTreeClassifier.fit(X_train, y_train, feature_names,)
train_pred = DecisionTreeClassifier.predict(X_train)
test_pred = DecisionTreeClassifier.predict(X_test)

train_acc = DecisionTreeClassifier.acc(y_train, train_pred)
test_acc = DecisionTreeClassifier.acc(y_test, test_pred)

print("Train:", train_acc)
print("Test :", test_acc)