import numpy as np

text = np.loadtxt("extracted_files.txt",dtype=str)
list_processed = []
for t in text:
    print(t)
    t1 = t.split("/")
    list_processed.append(int(t1[-1].split("_")[0]))

ids_to_export = np.loadtxt("id_list.csv")
not_exported = []
for id_ in ids_to_export:
    found = False
    for id2_ in list_processed:
        if id_ ==id2_:
            found=True
    if found == False:
        not_exported.append(int(id_))
not_exported = np.round(np.array(not_exported,dtype=int),0)
np.savetxt("not_exported.txt", not_exported, fmt="%d")