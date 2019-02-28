from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import re


#Loading test and train files into an array.
def load_file(filename):
    line_list = []
    file = open(filename, "r", encoding="UTF8")
    for line in file:
        line_list.append(line)
    file.close()
    return line_list

train = load_file("training_set_data.dat") #train_data array is saved here.
test_data = load_file("training_set_data_test.dat") #test_data array is saved here in this variable.


#Taking the expected values out of the array.
def expected_values(list):
    expected = []
    for i in range(len(list)):
        review_expected = list[i][:2] #Takes out the first 2 letters from the train array.
        expected.append(review_expected)
    return expected

expected = expected_values(train) #expected out is saved in this variable.


#This function cleans the data that was earlier saved in the array.
def clean_data(list):
    stop = set(stopwords.words('english')) #Language for stop words is english
    clean_data = []
    Ls = LancasterStemmer() #Used to normalize a word
    punctuation = set(string.punctuation) #Used to remove punctuation
    punctuation.add("eof") #Expanding the punctuation library
    punctuation.add("<br />") #Expanding the punctuation library
    punctuation.add("+1")#Expanding the punctuation library
    punctuation.add("-1")#Expanding the punctuation library
    punctuation.add("--")#Expanding the punctuation library
    for i in range(len(list)):
        each_review = ""
        line = re.split("[-]|[^A-z]|\\s", list[i].lower()) #Used regular express
        for word in line:
            if (word not in stop and word not in punctuation and (len(word) > 3) and word.strip() != ''): #if word is not in
                Ls.stem(word) #normalize each word before create a string of whole comment
                each_review = each_review + " " + word #creating a whole comment
        clean_data.append(each_review)
    return clean_data #clean_data is an array that holds preprocessed data

clean = clean_data(train) #cleaned train data is saved here
clean_test = clean_data(test_data) #clean test data is saved here

#This is a KNN implementation. I have another implementation that I have commented out below. However, this is a fastest way I have discovered to compute KNN.
def KNN_implementation(cosine_List, file):
    k = 299
    cosine_sort = sorted(range(len(cosine_List[0])), key=cosine_List[0].__getitem__, reverse=True)[:k] #This line sorts the index of the highest values in the decending order
    positive = 0
    negative = 0
    half = k / 2 #To check if majority reaches then break our loop
    for i in range(k):
        if(expected[cosine_sort[i]] == "+1"): #Checking each index if they are positive or negative.
            positive = positive + 1
        else:
            negative = negative + 1
        if ((positive > half) or (negative > half)): #If positive of negative reaches more than half K then break the loop.
            break
    if (positive > negative): #If we get more positive values then negative then write +1 in the file
        file.write("+1\n")
    else: #if otherwise then write -1 in the file.
        file.write("-1\n")

#Taking the cleaned data and vectorizing it. After vectorizing finding a similarities among them.
def tfidf_vectorization():
    vectorize = TfidfVectorizer() #Used for vectorizing the data
    TrainDataTrans = vectorize.fit_transform(clean)
    TestDataTrans = vectorize.transform(clean_test)
    file = open("format_K299.dat", "w") #file to write our results
    for i in range(len(clean)):
        #Print statement goes here!!!!!
        cosine = cosine_similarity(TestDataTrans[i], TrainDataTrans) #Finding similarity in each test review with entire train data
        KNN_implementation(cosine, file) #Send the cosine similarity to a KNN function to compute my results.
    return "All Done!"

out = tfidf_vectorization()


#This was my initial implemental but it was extremely slow. Part of a reason was because I was passing complete test and train reviews to the cosine similarity function.
# Which was taking a long time. However, I was able to find a faster way to compute the same thing.
# def KNN_Implementation(cosine_list):
#     cosine_sorted = cosine_list.copy
#     top_similarity = []
#     saving_reviews = []
#     positive = 0
#     negative = 0
#     file = open("format.dat", "w")
#     for i in range(len(cosine_list)):
#         sort_each = sorted(cosine_sorted[i], reverse=True)
#         for j in range(10):
#             top_similarity.append((sort_each[j]))
#             for k in range(len(cosine_list[i])):
#                 if (top_similarity[i] == cosine_list[i][k]):
#                     saving_reviews.append(expected[k])
#                     if (expected[j] == "+1"):
#                         positive = positive + 1
#                     else:
#                         negative = negative + 1
#                     continue
#         if (positive > negative):
#             file.write("+1\n")
#         else:
#             file.write("-1\n")
#         positive = 0
#         negative = 0
#     return cosine_list