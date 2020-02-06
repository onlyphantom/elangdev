import os

realpath = os.path.dirname(os.path.realpath(__file__))
stopwords_list_path = realpath + "/stopwords-list"
file_list = os.listdir(stopwords_list_path)

def combine_stopwords_files(combined_filename = "stopwords-id.txt"):
    # combine all txt files
    combined_stopwords = []

    for filename in file_list:
        stopwords = open(stopwords_list_path + "/" + filename).read().splitlines()
        combined_stopwords.extend(stopwords)
        
        #print(filename, str(len(stopwords)))

    # remove duplicate and sort
    unique_stopwords = sorted(set(combined_stopwords))

    # write to new txt file
    with open(realpath + "/" + combined_filename, "w") as file:
        file.write("\n".join(unique_stopwords))

    #print("TOTAL", len(combined_stopwords))
    #print("UNIQUE", len(unique_stopwords))

if __name__ == '__main__':
    combine_stopwords_files()
