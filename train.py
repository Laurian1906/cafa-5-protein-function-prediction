import os
import numpy as np
import pandas as pd
from Bio import SeqIO
from goatools import obo_parser
from goatools.associations import read_ncbi_gene2go
from goatools.obo_parser import GODag
from goatools.semantic import TermCounts
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize

obo_file = os.path.join(os.getcwd(), "Train/go-basic.obo")
graph = obo_parser.GODag(obo_file)
godag = GODag("Train/go-basic.obo")
gene2go = read_ncbi_gene2go("gene2go", taxids=[9606])
termcounts = TermCounts(godag, gene2go)
# Converteste fisierul gene2go in format .csv
gene2go_df = pd.read_csv('gene2go', sep='\t', header=None, low_memory=False)
gene2go_df.columns = ["tax_id", "GeneID", "GO_ID", "Evidence", "Qualifier", "GO_term", "PubMed", "Category"]
gene2go_df = gene2go_df.drop(columns=["Qualifier", "PubMed"])
gene2go_df = gene2go_df.loc[(gene2go_df['tax_id'] == '9606') & (
        (gene2go_df['GO_term'] == 'biological_process') | (gene2go_df['GO_term'] == 'molecular_function') | (
        gene2go_df['GO_term'] == 'cellular_component'))]
gene2go_df.to_csv('train_data_2.csv', index=False)
fasta_file = "Train/train_sequences.fasta"
records = SeqIO.parse(fasta_file, "fasta")

data = []
for record in records:
    row = {'ID': record.id, 'Sequence': str(record.seq)}
    for key, val in record.annotations.items():
        row[key] = val
    os_string = record.description
    if "OS=" in os_string:
        os = record.description.split("OS=")[1].split(" ")[0]
    else:
        os = "NA"
    row['OS'] = os
    data.append(row)

df = pd.DataFrame(data)
df = df[['ID', 'OS', 'Sequence']]  # selectăm coloanele dorite
df.to_csv('train_data.csv', index=False)
trainTerms = pd.read_csv('Train/train_terms.tsv', delimiter='\t')
trainTerms.to_csv('train_data_3.csv', index=False)

# =======================================================================================

# Incarcarea datelor
X = pd.read_csv('train_data_2.csv')

# Maparea functiilor proteinei prin binarizarea tintei
gt = {'molecular_function': 0, 'cellular_component': 1, 'biological_process': 2}
X['GO_ID'] = X['GO_ID'].str.split(':').str[1]
X['GO_term'] = X['GO_term'].map(gt)
y = X['GO_term']
y.to_csv('target.csv', index=False)
X = X.drop(["Evidence", "Category", "GO_term"], axis=1)
X.to_csv('features.csv', index=False)
y = label_binarize(y, classes=[0, 1, 2])

# Divizarea setului de date in set de antrenare si de testare
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# training the model on training set
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Crearea predictiilor
y_pred = knn.predict(X_test)

# Graficul Precizie-Recall
precision = dict()
recall = dict()
n_classes = 3
for i in range(n_classes):
    precision[i], recall[i], _ = metrics.precision_recall_curve(y_test[:, i], y_pred[:, i])

# Calculare micro-average precizie si recall
precision["micro"], recall["micro"], _ = metrics.precision_recall_curve(y_test.ravel(), y_pred.ravel())

# Calculare macro-average precizie si recall
precision["macro"] = np.mean(list(precision.values()), axis=0)
recall["macro"] = np.mean(list(recall.values()), axis=0)

print("Precizie",precision["macro"])
print("Recall",recall["macro"])

plt.plot(recall["micro"], precision["micro"], label='micro-average Precision-recall curve', color='red')
plt.plot(recall["macro"], precision["macro"], label='macro-average Precision-recall curve', color='blue')

for i in range(n_classes):
    plt.plot(recall[i], precision[i],
             label='class {0} (area = {1:0.2f})'.format(i, metrics.auc(recall[i], precision[i])))

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve for multi-class classification')
plt.legend(loc="lower right")
plt.show()

accuracy = metrics.accuracy_score(y_test, y_pred)
print("kNN model accuracy:", accuracy)

# Incarcarea datelor pe care se va face predictia
X_pred = pd.read_csv('generated_data.csv')

# Executarea predictiilor
y_pred = knn.predict(X_pred)

# Crearea unui DataFrame pentru predictii
class_names = ['molecular_function', 'cellular_component', 'biological_process']
y_pred_df = pd.DataFrame(y_pred, columns=class_names)

# Combinarea X_pred si y_pred_df, adica DataFrame-ul cu predictiile si DataFrame-ul din care s-au facut predictiile
# pentru analiza ulterioara a predictiilor
result_df = pd.concat([X_pred[['tax_id', 'GeneID', 'GO_ID']], y_pred_df], axis=1)

# Salvarea rezultatelor intr-un fisier csv.
result_df.to_csv('predicted_functions.csv', index=False)

# Salvarea modelului
joblib.dump(knn, 'model.joblib')

