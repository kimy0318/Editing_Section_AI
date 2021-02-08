# Editing_Section_AI
The project intends to use object detection and image labeling to create an AI that can identify what part of a footage should be edited. The AI intends to help newcomers understand how to edit. 

#Introduction
A key challenge in Machine Learning (ML) is to build agents that can identify complex human environments in response to spoken or written commands.
Today's models, including robots, are often able to explore complex environments, 
It is not yet possible to understand a series of behavioral expressions expressed in natural language such as  “the main character walks past the door quickly and approaches the person wearing glasses. After looking directly at his face for a moment, he looks confident and delivers the yellow envelope he brought.”

This challenge, called Visual and Language Navigation (VLN). requires a sophisticated understanding of spatial language.
This is a study of video editing technology using artificial intelligence. Video production is a series of sequential processes, and the part that requires the most time is later work. 
The editing process so far has had to be reviewed again all the footage taken and elaborated on distinguishing between necessary and unnecessary parts based on scenarios and conti. In particular, this part is very important in editing, which is the post production stage that produces the final results, so the director or producer will participate in editing directly. 
The necessary techniques to solve these problems require techniques to understand both images and languages at the same time. Natural language processing technology and image recognition processing technology have been studied separately so far, so it was necessary to have a technology to understand the verbal expression in the video at a contextual level. 
Against this backdrop, this study was conducted to confirm the applicability of visual and linguistic techniques to image editing.

#AI'S FUNCTION
Open CV
Open CV is the main computer program used to create the AI for this project. 
It was used for its function to detect and lable the objects within the frame. 

I. Object Detection
Objects in the footages will be detected using YOLO:Real Time-Object Detection coding.
Each objects will be labeled with a certain tag; ie. cars, phones, robots, etc.
"""
Start of:
Reading input video
"""

# Defining 'VideoCapture' object
video = cv2.VideoCapture('/Users/kimkwangil/Documents/VISION2020/김윤하_영상AI/movie/Scence1/DSCF0319.MP4')

writer = None

h, w = None, None

length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
print("video length:", length)
fps_ = video.get(cv2.CAP_PROP_FPS)
print("fps_:", fps_)
duration = length / fps_
print("duration:", duration)
minutes = int(duration / 60)
print("minutes:", minutes)
seconds = duration % 60
print("seconds:", seconds)
                        
II. Image Labeling
First, the keywords and topic words of the script will be extracted using NLP code.
The number of keywords that needs to be extracted can be cut down by only extracting the nouns. 

# Extraction of Keywords
from collections import OrderedDict
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import en_core_web_lg
nlp = en_core_web_lg.load()



import nltk
from pprint import pprint
import pandas as pd
from nltk.tokenize import sent_tokenize


class TextRank4Keyword():
    """Extract keywords from text"""
    
    def __init__(self):
        self.d = 0.85 # damping coefficient, usually is .85
        self.min_diff = 1e-5 # convergence threshold
        self.steps = 10 # iteration steps
        self.node_weight = None # save keywords and its weight

    
    def set_stopwords(self, stopwords):  
        """Set stop words"""
        for word in STOP_WORDS.union(set(stopwords)):
            lexeme = nlp.vocab[word]
            lexeme.is_stop = True
    
    def sentence_segment(self, doc, candidate_pos, lower):
        """Store those words only in cadidate_pos"""
        sentences = []
        for sent in doc.sents:
            selected_words = []
            for token in sent:
                # Store words only with cadidate POS tag
                if token.pos_ in candidate_pos and token.is_stop is False:
                    if lower is True:
                        selected_words.append(token.text.lower())
                        else:
                        selected_words.append(token.text)
            sentences.append(selected_words)
        return sentences
        
    def get_vocab(self, sentences):
        """Get all tokens"""
        vocab = OrderedDict()
        i = 0
        for sentence in sentences:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = i
                    i += 1
        return vocab
    
    def get_token_pairs(self, window_size, sentences):
        """Build token_pairs from windows in sentences"""
        token_pairs = list()
        for sentence in sentences:
            for i, word in enumerate(sentence):
                for j in range(i+1, i+window_size):
                    if j >= len(sentence):
                        break
                    pair = (word, sentence[j])
                    if pair not in token_pairs:
                        token_pairs.append(pair)
        return token_pairs
        
    def symmetrize(self, a):
        return a + a.T - np.diag(a.diagonal())
    
    def get_matrix(self, vocab, token_pairs):
        """Get normalized matrix"""
        # Build matrix
        vocab_size = len(vocab)
        g = np.zeros((vocab_size, vocab_size), dtype='float')
        for word1, word2 in token_pairs:
            i, j = vocab[word1], vocab[word2]
            g[i][j] = 1
            
Next, the keywords will be matched with the objects of the footage.
With the labeled objects from the previous coding, the keywords from the script can be matched with the objects from the frame. 

# Preparing text with label and confidence for current bounding box
            text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])], # >>>>>>>>>>> 여기의 레이블과 사니라오의 토픽이 같으면, 시간을 기록하거나 화면에 편집구간 표시할 것
                                                   confidences[i])
            print("text_box_current::::::::::", labels[int(class_numbers[i])])

            label_name = labels[int(class_numbers[i])] #save detected label of img

            print("nouns type >>>>>>>>>>:", type(nouns))
            print("label_name type>>>>>>>>>>>>:", type(label_name))



            # label_name 과 추출한 nouns 값이 같으면, 그 프레임의 시간을 txt파일로 저장한다.
            note = []
            saved_edit_point = []
            if nouns == label_name:

                saved_edit_point.append(frame_time)

            # Putting text with label and confidence on the original image
            cv2.putText(frame, text_box_current, (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)

        """
        End of:
        Drawing bounding boxes and labels
        """

        """
        Start of:
        Writing processed frame into the file
        """
The time frame created by the matching frames will indicate where the editing section should be.

#Limitation
Due to the high cost of image labeling, a free program called cocodata for this project. 
Only 80 objects was able to be labeled. 
The limiting number of objects that can be labeled meant that the process of identifying the editing sections was hindered.





