import os
import random
import re
import sys
import numpy
from copy import deepcopy

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """

    probDist = {}

    # count number of page and links in current page
    nPage = len(corpus)
    nLinks = len(corpus[page])

    if nLinks > 0:
        # add probability of choosing a random page in corpus
        for pageKey in corpus:
            probDist[pageKey] = (1 - damping_factor) / nPage

        # add probability of choosing a page from the current page
        for currPage in corpus[page]:
            probDist[currPage] += damping_factor / nLinks
    else:
        # no link in the current page, choose random in corpus
        for pageKey in corpus:
            probDist[pageKey] = 1 / nPage

    return probDist


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # initialize count dictionary
    countDict = dict.fromkeys(corpus.keys(), 0)

    # randomly choose a starting page
    currPage = random.choice(list(corpus.keys()))
    countDict[currPage] += 1

    # samples n - 1 cases using surfer method
    for i in range(1, n):
        probDist = transition_model(corpus, currPage, damping_factor)
        # choose page randomly based on probability distribution
        currPage = random.choices(list(probDist.keys()), weights=list(probDist.values()), k=1)[0]
        countDict[currPage] += 1

    prDict = {}

    # calculate probability/pagerank
    for count in countDict:
        prDict[count] = countDict[count] / n

    return prDict


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    nPage = len(corpus)
    pageList = list(corpus.keys())

    # initialize pagerank dictionary with equal probability
    prDict = dict.fromkeys(pageList, 1 / nPage)

    # make a look-up dictionary where key is page
    # values are pages that points to key page
    reverseDict = dict()

    # initialize look-up dictionary
    for page in corpus:
        reverseDict[page] = set()

    # add entries to reverse look-up page
    for parentPage in corpus:
        # if page has no links, treat as if it contains link to all pages
        if len(corpus[parentPage]) == 0:
            for page in corpus:
                reverseDict[page].add(parentPage)

            continue

        for linkedPage in corpus[parentPage]:
            reverseDict[linkedPage].add(parentPage)

    oldDict = deepcopy(prDict)
    # calculate the probability of being chosen randomly from corpus
    fromCorpus = (1 - damping_factor) / nPage

    while True:
        # loop through all pages in corpus to update probability
        for page in pageList:
            # probability of click from corpus
            prDict[page] = fromCorpus

            # loop through parent page that points to current page
            # and update probability of click from parent page
            for parentPage in reverseDict[page]:
                # count number of links from parent
                # if none, treat as if pointing to all pages in corpus
                nLinks = len(corpus[parentPage])
                if nLinks == 0:
                    nLinks = nPage
                # add probability of being chosen by clicks from a parent page
                prDict[page] += damping_factor * oldDict[parentPage] / nLinks

        # if all entries changes by less than 0.001, break out of loop
        match = 0
        for page in pageList:
            if numpy.absolute(prDict[page] - oldDict[page]) < 0.001:
                match += 1

        if match == nPage:
            break

        # if not converged yet, update oldDict to store old distribution
        oldDict = deepcopy(prDict)

    return prDict


if __name__ == "__main__":
    main()
