import nltk
import string
import sys
import os
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    fileDict = dict()
    # loop over file in directory, then read into dictionary
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), encoding='utf-8') as file:
            fileDict[filename] = file.read()
    return fileDict


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by converting all words to lowercase, and removing any
    punctuation or English stopwords.
    """

    # tokenize words
    prelimList = nltk.tokenize.word_tokenize(document)
    # convert list to lower case
    prelimList = [word.lower() for word in prelimList]

    # remove punctuation or english stopwords
    finalList = [word for word in prelimList if word not in string.punctuation
                 and word not in nltk.corpus.stopwords.words('english')]

    return finalList


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    wordCount = dict()

    # loop through documents
    for document in documents:
        # loop through unique word in document and
        # add 1 to count if appears in document
        for word in set(documents[document]):
            if word in wordCount.keys():
                wordCount[word] += 1
            else:
                wordCount[word] = 1

    nDoc = len(documents)
    # calculate idf and put in dictionary
    idfDict = {key: math.log(nDoc / value)
               for (key, value) in wordCount.items()}

    return idfDict


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    docDict = dict()

    for file in files:
        # initialize td-idf sum
        tfidfSum = 0
        for word in query:
            if word in idfs.keys():
                # calculate term frequency then add tf-idf to sum
                tf = files[file].count(word)
                tfidfSum += tf * idfs[word]
        docDict[file] = tfidfSum
    # sort documents in descending order based on tf-idf sum
    sortedFiles = sorted(docDict, key=docDict.get, reverse=True)

    # return n highest elements
    return sortedFiles[0:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """

    sentenceDict = dict()

    # loop through sentences
    for sentence in sentences:
        sentenceList = sentences[sentence]
        matchMeasure = 0
        wordCount = 0

        # loop through words in query
        for word in query:
            # if word in sentence, add idf to measure
            # also add the number of times word in sentence to wordcount
            if word in sentenceList and word in idfs.keys():
                matchMeasure += idfs[word]
                wordCount += sentenceList.count(word)

        queryDensity = wordCount/len(sentenceList)
        # add tuple of (match word measure, query density) as value to dict
        sentenceDict[sentence] = (matchMeasure, queryDensity)

    # get sorted keys, based on first matching word measure and then
    # query density, in descending order
    sortedSentence = sorted(sentenceDict.keys(), key=lambda
        k: (sentenceDict[k][0], sentenceDict[k][1]), reverse=True)
    # return the first n element in sorted sentence list
    return sortedSentence[0:n]


if __name__ == "__main__":
    main()
