import os
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_upstage import UpstageEmbeddings
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from pinecone import Pinecone
from langchain_upstage import ChatUpstage
from langchain import hub
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.schema import Document
from langchain.vectorstores import FAISS
import re
from datetime import datetime
import pytz
from langchain.schema.runnable import Runnable
from langchain.chains import RetrievalQAWithSourcesChain, RetrievalQA
from langchain.schema.runnable import RunnableSequence, RunnableMap
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from konlpy.tag import Okt
from collections import defaultdict
import Levenshtein
import numpy as np
from IPython.display import display, HTML
from rank_bm25 import BM25Okapi
from difflib import SequenceMatcher
from pymongo import MongoClient

# mongodb 연결, client로
client = MongoClient("mongodb://localhost:27017/")

db = client["test_db"]
collection = db["test_collection"]

image_urls = ["aaa", "bbb", "ccc", "ddd"]

data = {
	"title" :  "for testing",
	"transformed_title" : ["a", "b", "c", "d"],
	"image_urls" : image_urls
}

li = collection.find({"title" : "for testing"}, {"image_urls" : 1})
print(li)